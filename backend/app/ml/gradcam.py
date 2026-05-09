"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module

Generates visual explanations for the model's decisions by highlighting
the image regions that most influenced the classification.

This is crucial for:
1. Explainability - understanding what the model "sees"
2. Trust - proving the model focuses on meaningful forensic artifacts
3. Debugging - identifying if the model learned spurious correlations

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM implementation for the hybrid forensic model.
    
    Computes gradient-weighted class activation maps on the 
    EfficientNet backbone's last convolutional layer.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: HybridForensicModel instance
            target_layer: Target conv layer for CAM computation.
                         If None, uses the last conv layer of EfficientNet.
        """
        self.model = model
        self.model.eval()
        
        # Default to the last convolutional features of EfficientNet
        if target_layer is None:
            self.target_layer = model.semantic_backbone.features[-1]
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, rgb_input: torch.Tensor, forensic_input: torch.Tensor, 
                 target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            rgb_input: RGB tensor (1, 3, 224, 224)
            forensic_input: Forensic tensor (1, 9, 224, 224)
            target_class: Target class index (None = predicted class)
        
        Returns:
            Heatmap as numpy array (224, 224), values in [0, 1]
        """
        self.model.eval()
        rgb_input.requires_grad_(True)
        
        # Forward pass
        output = self.model(rgb_input, forensic_input)
        
        if target_class is None:
            target_class = (torch.sigmoid(output) > 0.5).long().item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        if target_class == 1:
            output.backward(retain_graph=True)
        else:
            (-output).backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients → weights
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = cv2.resize(cam, (224, 224))
        
        return cam
    
    def generate_overlay(self, rgb_input: torch.Tensor, forensic_input: torch.Tensor,
                         original_image: Image.Image, alpha: float = 0.5) -> Image.Image:
        """
        Generate Grad-CAM heatmap overlaid on the original image.
        
        Args:
            rgb_input: RGB tensor (1, 3, 224, 224)
            forensic_input: Forensic tensor (1, 9, 224, 224)
            original_image: Original PIL Image
            alpha: Overlay transparency (0 = original only, 1 = heatmap only)
        
        Returns:
            PIL Image with Grad-CAM overlay
        """
        # Generate heatmap
        heatmap = self.generate(rgb_input, forensic_input)
        
        # Resize original to match
        original_resized = original_image.resize((224, 224), Image.LANCZOS)
        original_array = np.array(original_resized)
        
        # Apply colormap (JET)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (original_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        
        return Image.fromarray(overlay)
