"""
Error Level Analysis (ELA) Module

ELA detects regions of an image that have been modified by analyzing
JPEG compression artifacts. When an image is saved as JPEG, the entire
image is compressed uniformly. If a region has been modified (or generated
by AI), it will have a different compression level, which ELA reveals.
"""
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from app.config import ELA_QUALITY


def compute_ela(image: Image.Image, quality: int = ELA_QUALITY) -> tuple[Image.Image, float]:
    """
    Compute Error Level Analysis for an image.
    
    Process:
    1. Re-save the image as JPEG at a known quality level
    2. Compute pixel-wise difference between original and re-saved
    3. Scale differences for visibility
    
    Args:
        image: PIL Image in RGB mode
        quality: JPEG quality for re-compression (default: 90)
    
    Returns:
        tuple: (ELA image as PIL Image, ELA score as float 0-1)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Re-save to an in-memory buffer at known quality
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    
    # Compute pixel-wise absolute difference
    ela_image = ImageChops.difference(image, resaved)
    
    # Get the maximum difference across all channels for scaling
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    
    # Scale the difference image to make artifacts more visible
    scale_factor = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale_factor)
    
    # Compute ELA score (normalized mean intensity of difference)
    ela_array = np.array(ela_image, dtype=np.float32)
    ela_score = float(np.mean(ela_array) / 255.0)
    
    return ela_enhanced, ela_score


def compute_ela_tensor(image: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    Compute ELA and return as a normalized numpy tensor suitable for model input.
    
    Args:
        image: PIL Image
        target_size: Target spatial dimension
    
    Returns:
        numpy array of shape (3, target_size, target_size), normalized to [0, 1]
    """
    ela_img, _ = compute_ela(image)
    ela_img = ela_img.resize((target_size, target_size), Image.LANCZOS)
    ela_array = np.array(ela_img, dtype=np.float32) / 255.0
    # HWC -> CHW
    ela_tensor = np.transpose(ela_array, (2, 0, 1))
    return ela_tensor
