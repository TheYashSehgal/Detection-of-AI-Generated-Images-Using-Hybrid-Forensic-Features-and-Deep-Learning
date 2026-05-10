"""
Model Inference Engine — CLIP Zero-Shot + Calibrated Forensic Fusion

Uses OpenAI's CLIP (ViT-B/32) for zero-shot AI-image detection by comparing
image embeddings against carefully crafted text prompts, then fuses the
semantic score with classical forensic features (ELA, SRM, DCT) for a
robust hybrid prediction.
"""
import time
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from app.forensics.pipeline import extract_forensic_features
from app.ml.gradcam import generate_clip_attention_map
from app.config import CLIP_MODEL_NAME, CLIP_CACHE_DIR

# ---------------------------------------------------------------------------
# Text prompts — engineered for maximum separation between real vs AI
# Using the model's full forward pass (logits_per_image) handles projection
# and temperature scaling automatically.
# ---------------------------------------------------------------------------
CANDIDATE_LABELS = [
    "a real photograph taken with a camera",
    "an AI-generated image, synthetic artwork, or deepfake",
]


class InferenceEngine:
    """Singleton inference engine using CLIP + forensic fusion."""

    _instance = None
    _model = None
    _processor = None
    _is_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        """Load CLIP model."""
        print(f"[InferenceEngine] Loading CLIP model: {CLIP_MODEL_NAME}")
        print(f"[InferenceEngine] Cache directory: {CLIP_CACHE_DIR}")

        try:
            self._model = CLIPModel.from_pretrained(
                CLIP_MODEL_NAME,
                cache_dir=CLIP_CACHE_DIR,
                attn_implementation="eager",  # needed for output_attentions=True
            )
            self._processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL_NAME,
                cache_dir=CLIP_CACHE_DIR,
            )
            self._model.eval()
            self._is_loaded = True
            print("[InferenceEngine] CLIP model ready.")
        except Exception as e:
            print(f"[InferenceEngine] Failed to load CLIP: {e}")
            raise

    @property
    def is_loaded(self):
        return self._is_loaded

    # ------------------------------------------------------------------
    # CLIP zero-shot probability
    # ------------------------------------------------------------------
    def _clip_probability(self, image: Image.Image) -> float:
        """
        Compute AI-generation probability via CLIP zero-shot classification.

        Uses the model's full forward pass which handles text/image projection
        and temperature scaling correctly via learned logit_scale.

        Returns a float in [0, 1] where 1 = definitely AI, 0 = definitely real.
        """
        inputs = self._processor(
            text=CANDIDATE_LABELS,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_image: (1, num_labels) — already temperature-scaled
            probs = outputs.logits_per_image.softmax(dim=-1)

        # probs[0, 0] = real probability, probs[0, 1] = AI probability
        ai_prob = probs[0, 1].item()
        return ai_prob

    # ------------------------------------------------------------------
    # Forensic feature scoring (calibrated)
    # ------------------------------------------------------------------
    def _forensic_probability(
        self,
        image: Image.Image,
        ela_score: float,
        srm_score: float,
        dct_score: float,
    ) -> float:
        """
        Compute a supplementary AI probability from classical forensic features.

        ELA:  AI images have ~uniform compression → lower ELA residuals → lower ela_score
        SRM:  AI images lack sensor noise → lower SRM residuals → lower srm_score
        DCT:  AI images have uniform spectra → higher dct_score (1 - CV)

        Returns float in [0, 1].
        """
        img_arr = np.array(image.convert("RGB"), dtype=np.float32)

        # -- Feature 1: ELA-based (low ELA = likely AI) --
        # Typical real: ela_score ~0.02-0.06, AI: ~0.005-0.02
        ela_ai = float(np.clip(1.0 - ela_score / 0.045, 0.0, 1.0))

        # -- Feature 2: SRM noise level (low noise = likely AI) --
        # Typical real: srm_score ~0.04-0.08, AI: ~0.01-0.03
        srm_ai = float(np.clip(1.0 - srm_score / 0.055, 0.0, 1.0))

        # -- Feature 3: DCT spectral uniformity (high = likely AI) --
        dct_ai = float(np.clip(dct_score, 0.0, 1.0))

        # -- Feature 4: Local patch variance (smooth = likely AI) --
        gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
        h, w = gray.shape
        step = 16
        patch_vars = []
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                patch_vars.append(np.var(gray[i : i + step, j : j + step]))
        mean_var = float(np.mean(patch_vars)) if patch_vars else 0.0
        # Real images typically have higher local variance
        var_ai = float(np.clip(1.0 - mean_var / 300.0, 0.0, 1.0))

        # Weighted combination
        forensic_prob = (
            0.30 * ela_ai
            + 0.25 * srm_ai
            + 0.25 * dct_ai
            + 0.20 * var_ai
        )
        return float(np.clip(forensic_prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image) -> dict:
        """Run the full hybrid analysis pipeline."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.time()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Forensic features (ELA, SRM, DCT)
        forensic = extract_forensic_features(image)

        # 2. CLIP zero-shot classification (the primary signal)
        clip_prob = self._clip_probability(image)

        # 3. Forensic probability (supplementary signal)
        forensic_prob = self._forensic_probability(
            image,
            forensic["ela_score"],
            forensic["srm_score"],
            forensic["dct_score"],
        )

        # 4. Hybrid fusion: CLIP is the dominant signal, forensics supplements
        #    CLIP is much more reliable, so it gets 75% weight
        final_prob = 0.75 * clip_prob + 0.25 * forensic_prob
        final_prob = float(np.clip(final_prob, 0.01, 0.99))

        # 5. Determine verdict
        is_ai = final_prob > 0.5
        prediction = "ai_generated" if is_ai else "real"
        confidence = final_prob if is_ai else (1.0 - final_prob)

        # 6. Attention map (Grad-CAM equivalent for ViT)
        try:
            gcam = generate_clip_attention_map(
                self._model, self._processor, image
            )
        except Exception as e:
            print(f"[InferenceEngine] Attention map failed: {e}")
            gcam = ""

        elapsed = (time.time() - start) * 1000

        return {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "ai_probability": final_prob,
            "clip_score": clip_prob,
            "forensic_score": forensic_prob,
            "ela_score": forensic["ela_score"],
            "srm_score": forensic["srm_score"],
            "dct_score": forensic["dct_score"],
            "ela_image": forensic["ela_image"],
            "srm_image": forensic["srm_image"],
            "dct_image": forensic["dct_image"],
            "gradcam_image": gcam,
            "analysis_time_ms": round(elapsed, 2),
        }


engine = InferenceEngine()
