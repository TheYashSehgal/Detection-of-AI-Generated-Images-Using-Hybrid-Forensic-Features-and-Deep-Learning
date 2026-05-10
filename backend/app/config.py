"""
Configuration settings for the AI-Generated Image Detection application.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "samples"

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

# Model settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# Cache models on the G: drive to avoid C: disk space issues
CLIP_CACHE_DIR = os.getenv("HF_HOME", str(Path(r"g:\Academics\Final Year Project\.model_cache")))
IMAGE_SIZE = 224  # CLIP ViT-B/32 input size
DEVICE = os.getenv("DEVICE", "cpu")  # "cuda" or "cpu"

# Forensic settings
ELA_QUALITY = 90  # JPEG quality for ELA re-compression
SRM_NUM_FILTERS = 30  # Number of SRM high-pass filter kernels
DCT_BLOCK_SIZE = 8  # DCT block size

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:80").split(",")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Training settings (retained for academic reference)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
