"""
Model Inference Engine - handles model loading & prediction.
Uses singleton pattern to avoid reloading the model on each request.
"""
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from app.config import MODEL_CHECKPOINT, IMAGE_SIZE, DEVICE
from app.models.hybrid_model import create_model, HybridForensicModel
from app.forensics.pipeline import extract_forensic_features
from app.ml.gradcam import GradCAM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

rgb_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class InferenceEngine:
    _instance = None
    _model = None
    _gradcam = None
    _device = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, checkpoint_path=None, device=None):
        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
        print(f"[InferenceEngine] Loading model on: {self._device}")
        self._model = create_model(pretrained=True)
        ckpt_path = Path(checkpoint_path) if checkpoint_path else MODEL_CHECKPOINT
        if ckpt_path.exists():
            print(f"[InferenceEngine] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self._model.load_state_dict(ckpt['model_state_dict'])
            else:
                self._model.load_state_dict(ckpt)
        else:
            print("[InferenceEngine] No checkpoint found — demo mode with pretrained EfficientNet")
        self._model.to(self._device)
        self._model.eval()
        self._gradcam = GradCAM(self._model)
        self._is_loaded = True
        print("[InferenceEngine] Model ready")

    @property
    def is_loaded(self):
        return self._is_loaded

    def predict(self, image: Image.Image) -> dict:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        start = time.time()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        forensic = extract_forensic_features(image)
        rgb_t = rgb_transform(image).unsqueeze(0).to(self._device)
        for_t = torch.from_numpy(forensic['forensic_tensor']).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(rgb_t, for_t)
            prob = torch.sigmoid(logits).item()
        try:
            gcam = self._gradcam.generate_overlay(
                rgb_t.clone().detach().requires_grad_(True), for_t.clone().detach(), image, alpha=0.5
            )
        except Exception as e:
            print(f"[InferenceEngine] Grad-CAM failed: {e}")
            gcam = image.resize((224, 224), Image.LANCZOS)
        is_ai = prob > 0.5
        prediction = "ai_generated" if is_ai else "real"
        confidence = prob if is_ai else (1 - prob)
        elapsed = (time.time() - start) * 1000
        return {
            'prediction': prediction, 'confidence': confidence,
            'confidence_percent': round(confidence * 100, 2), 'ai_probability': prob,
            'ela_score': forensic['ela_score'], 'srm_score': forensic['srm_score'],
            'dct_score': forensic['dct_score'], 'ela_image': forensic['ela_image'],
            'srm_image': forensic['srm_image'], 'dct_image': forensic['dct_image'],
            'gradcam_image': gcam, 'analysis_time_ms': round(elapsed, 2),
        }

engine = InferenceEngine()
