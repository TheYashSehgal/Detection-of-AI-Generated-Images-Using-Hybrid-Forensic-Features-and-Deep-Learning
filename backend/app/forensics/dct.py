"""
Discrete Cosine Transform (DCT) Spectral Analysis Module

DCT analysis examines the frequency domain characteristics of images.
AI-generated images often exhibit anomalous frequency distributions
compared to natural (camera-captured) images. This module performs
block-wise DCT to extract spectral features.

The DCT reveals compression artifacts and frequency patterns that are
characteristic signatures of different image generation methods.
"""
import numpy as np
from PIL import Image
from scipy.fft import dctn


def compute_dct_features(image: Image.Image, block_size: int = 8) -> tuple[Image.Image, float]:
    """
    Compute block-wise DCT spectral analysis.
    
    Process:
    1. Convert image to grayscale
    2. Divide into non-overlapping blocks
    3. Apply 2D DCT to each block
    4. Analyze spectral energy distribution
    
    Args:
        image: PIL Image
        block_size: Size of DCT blocks (default: 8, matching JPEG standard)
    
    Returns:
        tuple: (DCT spectral visualization as PIL Image, DCT anomaly score 0-1)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    
    # Convert to grayscale using luminance formula
    gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    
    h, w = gray.shape
    
    # Trim to exact multiples of block_size
    h_blocks = h // block_size
    w_blocks = w // block_size
    gray = gray[:h_blocks * block_size, :w_blocks * block_size]
    
    # Compute block-wise DCT
    spectral_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
    all_high_freq_energies = []
    all_low_freq_energies = []
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = gray[i*block_size:(i+1)*block_size, 
                        j*block_size:(j+1)*block_size]
            
            # Apply 2D DCT
            dct_block = dctn(block, type=2, norm='ortho')
            
            # Compute spectral energy
            energy = np.abs(dct_block)
            
            # Separate high and low frequency components
            # Low frequency: top-left quadrant
            low_freq = energy[:block_size//2, :block_size//2]
            # High frequency: bottom-right area
            high_freq = energy[block_size//2:, block_size//2:]
            
            low_e = np.mean(low_freq)
            high_e = np.mean(high_freq) if high_freq.size > 0 else 0
            
            all_low_freq_energies.append(low_e)
            all_high_freq_energies.append(high_e)
            
            # Ratio of high to low frequency energy
            if low_e > 0:
                spectral_map[i, j] = high_e / low_e
            else:
                spectral_map[i, j] = high_e
    
    # Compute DCT anomaly score
    # AI-generated images tend to have more uniform spectral distribution
    high_energies = np.array(all_high_freq_energies)
    low_energies = np.array(all_low_freq_energies)
    
    # Coefficient of variation of the spectral ratios
    if spectral_map.std() > 0:
        cv = spectral_map.std() / (spectral_map.mean() + 1e-8)
        # Lower CV suggests more uniform (AI-generated) spectral characteristics
        dct_score = float(1.0 - min(cv, 1.0))
    else:
        dct_score = 0.5
    
    # Create visualization
    vis = spectral_map - spectral_map.min()
    max_val = vis.max()
    if max_val > 0:
        vis = vis / max_val
    
    # Upscale to image size
    vis_upscaled = np.repeat(np.repeat(vis, block_size, axis=0), block_size, axis=1)
    
    # Apply a purple-green colormap for spectral visualization
    vis_rgb = np.zeros((*vis_upscaled.shape, 3), dtype=np.uint8)
    vis_rgb[:, :, 0] = (vis_upscaled * 180).astype(np.uint8)        # Red/Purple
    vis_rgb[:, :, 1] = ((1 - vis_upscaled) * 200).astype(np.uint8)  # Green
    vis_rgb[:, :, 2] = (vis_upscaled * 255).astype(np.uint8)        # Blue/Purple
    
    dct_image = Image.fromarray(vis_rgb, mode='RGB')
    
    return dct_image, dct_score


def compute_dct_tensor(image: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    Compute DCT features as a tensor for model input.
    
    Returns a multi-channel tensor with:
    - Channel 0: Full DCT magnitude spectrum
    - Channel 1: High-frequency energy map
    - Channel 2: Low-frequency energy map
    
    Args:
        image: PIL Image
        target_size: Target spatial dimension
    
    Returns:
        numpy array of shape (3, target_size, target_size)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_resized = image.resize((target_size, target_size), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    
    # Full image DCT
    full_dct = dctn(gray, type=2, norm='ortho')
    magnitude = np.log1p(np.abs(full_dct))
    
    # Normalize magnitude
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max
    
    # Create high-frequency mask
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Distance from DC component (top-left for DCT)
    dist = np.sqrt(y**2 + x**2)
    max_dist = np.sqrt(h**2 + w**2)
    
    # High frequency: far from DC
    high_mask = (dist > max_dist * 0.3).astype(np.float32)
    # Low frequency: close to DC
    low_mask = (dist <= max_dist * 0.3).astype(np.float32)
    
    high_freq = magnitude * high_mask
    low_freq = magnitude * low_mask
    
    tensor = np.stack([magnitude, high_freq, low_freq], axis=0)
    
    return tensor.astype(np.float32)
