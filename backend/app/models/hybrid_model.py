"""
Hybrid Dual-Branch CNN Model for AI-Generated Image Detection

Architecture:
- Branch A (Semantic): EfficientNet-B0 pre-trained on ImageNet
  - Extracts high-level semantic features from RGB images
  - Fine-tuned for forensic classification
  
- Branch B (Forensic): Lightweight CNN with SRM-initialized first layer
  - Processes forensic feature maps (ELA + SRM + DCT)
  - Fixed SRM filters force focus on noise patterns
  
- Fusion: Squeeze-and-Excitation (SE) block for adaptive feature fusion
  - Learns channel-wise importance weights
  - Dynamically recalibrates features from both branches

- Classification: FC layers → Sigmoid → Binary (Real vs AI-Generated)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    
    Adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: channel-wise multiplication
        return x * y.expand_as(x)


class ForensicBranch(nn.Module):
    """
    Forensic Feature Branch (Branch B)
    
    A lightweight CNN that processes forensic feature maps.
    The first convolutional layer can be initialized with SRM filter
    weights to force the network to focus on noise residuals.
    """
    def __init__(self, in_channels: int = 9, feature_dim: int = 256):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # SE Block for channel attention
            SqueezeExcitation(256, reduction=16),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x


class HybridForensicModel(nn.Module):
    """
    Hybrid Dual-Branch Model for AI-Generated Image Detection
    
    Combines semantic features (EfficientNet-B0 on RGB) with
    forensic features (custom CNN on ELA+SRM+DCT) through
    adaptive SE-based fusion.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        
        # ===== Branch A: Semantic (EfficientNet-B0) =====
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.semantic_backbone = efficientnet_b0(weights=weights)
                print("[Model] Loaded pretrained EfficientNet-B0 weights")
            except Exception as e:
                print(f"[Model] Could not download pretrained weights: {e}")
                print("[Model] Using randomly initialized EfficientNet-B0")
                self.semantic_backbone = efficientnet_b0(weights=None)
        else:
            self.semantic_backbone = efficientnet_b0(weights=None)
        
        # Get the output feature dimension of EfficientNet-B0
        semantic_feat_dim = self.semantic_backbone.classifier[1].in_features  # 1280
        
        # Remove the original classifier
        self.semantic_backbone.classifier = nn.Identity()
        
        # Semantic projection
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # ===== Branch B: Forensic =====
        self.forensic_branch = ForensicBranch(in_channels=9, feature_dim=256)
        
        # ===== Fusion =====
        fusion_dim = 512 + 256  # 768
        
        # SE-based fusion (applied on concatenated features reshaped as 1D "channels")
        self.fusion_se = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 8, fusion_dim),
            nn.Sigmoid(),
        )
        
        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, rgb_input: torch.Tensor, forensic_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb_input: RGB image tensor, shape (B, 3, 224, 224)
            forensic_input: Forensic feature tensor, shape (B, 9, 224, 224)
        
        Returns:
            Logits tensor, shape (B, 1)
        """
        # Branch A: Semantic features
        semantic_features = self.semantic_backbone(rgb_input)  # (B, 1280)
        semantic_features = self.semantic_proj(semantic_features)  # (B, 512)
        
        # Branch B: Forensic features
        forensic_features = self.forensic_branch(forensic_input)  # (B, 256)
        
        # Fusion: Concatenate + SE attention
        combined = torch.cat([semantic_features, forensic_features], dim=1)  # (B, 768)
        
        # Apply SE-style channel attention on fused features
        attention_weights = self.fusion_se(combined)  # (B, 768)
        fused = combined * attention_weights  # Adaptive recalibration
        
        # Classification
        logits = self.classifier(fused)  # (B, 1)
        
        return logits
    
    def predict_proba(self, rgb_input: torch.Tensor, forensic_input: torch.Tensor) -> torch.Tensor:
        """Get probability of being AI-generated."""
        logits = self.forward(rgb_input, forensic_input)
        return torch.sigmoid(logits)
    
    def get_semantic_features(self, rgb_input: torch.Tensor) -> torch.Tensor:
        """Extract semantic features (for Grad-CAM)."""
        return self.semantic_backbone.features(rgb_input)


def create_model(pretrained: bool = True) -> HybridForensicModel:
    """Factory function to create a new HybridForensicModel."""
    model = HybridForensicModel(num_classes=1, pretrained=pretrained)
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_mb': total * 4 / (1024 * 1024),  # Assuming float32
    }
