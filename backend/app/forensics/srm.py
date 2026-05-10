"""
Spatial Rich Model (SRM) Noise Filters Module

SRM filters are high-pass filters originally designed for steganalysis
(detecting hidden data in images). They extract noise residuals from
images, revealing the subtle statistical fingerprints left by different
image sources. AI-generated images have distinctly different noise
patterns compared to camera-captured photographs.

Reference: Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images"
IEEE TIFS, 2012.
"""
import numpy as np
from PIL import Image


# Pre-defined SRM filter kernels (subset of the 30 most effective filters)
# These are fixed, non-learnable high-pass filters
SRM_KERNELS = []

def _build_srm_kernels():
    """Build the set of SRM high-pass filter kernels."""
    global SRM_KERNELS
    
    if len(SRM_KERNELS) > 0:
        return SRM_KERNELS
    
    kernels = []
    
    # 1st order residual filters (edge detectors in different directions)
    # Horizontal
    k1 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  1, -1,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k1)
    
    # Vertical
    k2 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  0,  1,  0,  0],
                   [ 0,  0, -1,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k2)
    
    # Diagonal
    k3 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  1,  0,  0,  0],
                   [ 0,  0, -1,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k3)
    
    # Anti-diagonal
    k4 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  1,  0],
                   [ 0,  0, -1,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k4)
    
    # 2nd order residual filters
    # Horizontal 2nd order
    k5 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  1, -2,  1,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k5)
    
    # Vertical 2nd order
    k6 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  0,  1,  0,  0],
                   [ 0,  0, -2,  0,  0],
                   [ 0,  0,  1,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k6)
    
    # 2nd order diagonal
    k7 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  1,  0,  0,  0],
                   [ 0,  0, -2,  0,  0],
                   [ 0,  0,  0,  1,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k7)
    
    # 3rd order residual filters
    # SQUARE 3x3
    k8 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0, -1,  2, -1,  0],
                   [ 0,  2, -4,  2,  0],
                   [ 0, -1,  2, -1,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k8)
    
    # EDGE 3x3 horizontal
    k9 = np.array([[ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0, -1,  3, -3,  1],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=np.float32)
    kernels.append(k9)
    
    # EDGE 3x3 vertical
    k10 = np.array([[0,  0,  0,  0,  0],
                    [0,  0, -1,  0,  0],
                    [0,  0,  3,  0,  0],
                    [0,  0, -3,  0,  0],
                    [0,  0,  1,  0,  0]], dtype=np.float32)
    kernels.append(k10)
    
    # 5x5 high-pass filters
    # KV filter (Ker-Veres)
    k11 = np.array([[-1,  2, -2,  2, -1],
                    [ 2, -6,  8, -6,  2],
                    [-2,  8,-12,  8, -2],
                    [ 2, -6,  8, -6,  2],
                    [-1,  2, -2,  2, -1]], dtype=np.float32)
    kernels.append(k11)
    
    # Laplacian variants
    k12 = np.array([[ 0,  0, -1,  0,  0],
                    [ 0,  0,  2,  0,  0],
                    [-1,  2, -4,  2, -1],
                    [ 0,  0,  2,  0,  0],
                    [ 0,  0, -1,  0,  0]], dtype=np.float32)
    kernels.append(k12)
    
    # Cross pattern
    k13 = np.array([[ 0,  0,  1,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 1, -2,  4, -2,  1],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  1,  0,  0]], dtype=np.float32)
    kernels.append(k13)
    
    # Sobel-like variants for noise
    k14 = np.array([[-1, -1,  0,  1,  1],
                    [-1, -1,  0,  1,  1],
                    [-1, -1,  0,  1,  1],
                    [-1, -1,  0,  1,  1],
                    [-1, -1,  0,  1,  1]], dtype=np.float32) / 10.0
    kernels.append(k14)
    
    k15 = np.array([[ 1,  1,  1,  1,  1],
                    [ 1,  1,  1,  1,  1],
                    [ 0,  0,  0,  0,  0],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1]], dtype=np.float32) / 10.0
    kernels.append(k15)
    
    # Min-max residual approximations
    for angle in range(0, 360, 45):
        rad = np.deg2rad(angle)
        k = np.zeros((5, 5), dtype=np.float32)
        k[2, 2] = -1
        dy, dx = int(round(np.sin(rad))), int(round(np.cos(rad)))
        k[2 + dy, 2 + dx] = 1
        kernels.append(k)
    
    # Normalize all kernels
    for i in range(len(kernels)):
        norm = np.sum(np.abs(kernels[i]))
        if norm > 0:
            kernels[i] = kernels[i] / norm
    
    SRM_KERNELS = kernels
    return kernels


def apply_srm_filters(image: Image.Image) -> tuple[Image.Image, float]:
    """
    Apply SRM high-pass filters to extract noise residuals.
    
    Args:
        image: PIL Image in RGB mode
    
    Returns:
        tuple: (SRM visualization as PIL Image, SRM noise score as float 0-1)
    """
    from scipy.ndimage import convolve
    
    kernels = _build_srm_kernels()
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply filters to each channel and accumulate residuals
    residual_map = np.zeros(img_array.shape[:2], dtype=np.float32)
    
    for kernel in kernels:
        for c in range(3):
            channel = img_array[:, :, c]
            filtered = convolve(channel, kernel, mode='reflect')
            residual_map += np.abs(filtered)
    
    # Normalize the residual map
    num_filters = len(kernels) * 3
    residual_map /= num_filters
    
    # Compute SRM noise score
    srm_score = float(np.mean(residual_map))
    
    # Create visualization (enhance contrast)
    vis = residual_map - residual_map.min()
    max_val = vis.max()
    if max_val > 0:
        vis = vis / max_val
    
    # Apply colormap for visualization (blue to red)
    vis_rgb = np.zeros((*vis.shape, 3), dtype=np.uint8)
    vis_rgb[:, :, 0] = (vis * 255).astype(np.uint8)  # Red channel
    vis_rgb[:, :, 1] = ((1 - 2 * np.abs(vis - 0.5)) * 128).astype(np.uint8)  # Green
    vis_rgb[:, :, 2] = ((1 - vis) * 255).astype(np.uint8)  # Blue channel
    
    srm_image = Image.fromarray(vis_rgb, mode='RGB')
    
    return srm_image, srm_score


def compute_srm_tensor(image: Image.Image, target_size: int = 224, num_output_channels: int = 3) -> np.ndarray:
    """
    Compute SRM residuals as a tensor for model input.
    
    Selects the top-N most informative SRM filter responses.
    
    Args:
        image: PIL Image
        target_size: Target spatial dimension
        num_output_channels: Number of output channels
    
    Returns:
        numpy array of shape (num_output_channels, target_size, target_size)
    """
    from scipy.ndimage import convolve
    
    kernels = _build_srm_kernels()
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_resized = image.resize((target_size, target_size), Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convert to grayscale for SRM
    gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    
    # Apply top filters and collect responses
    responses = []
    for kernel in kernels[:num_output_channels * 3]:
        filtered = convolve(gray, kernel, mode='reflect')
        responses.append(np.abs(filtered))
    
    # Sort by energy and pick top channels
    energies = [np.mean(r) for r in responses]
    top_indices = np.argsort(energies)[-num_output_channels:]
    
    tensor = np.stack([responses[i] for i in top_indices], axis=0)
    
    # Normalize each channel
    for c in range(tensor.shape[0]):
        ch_max = tensor[c].max()
        if ch_max > 0:
            tensor[c] = tensor[c] / ch_max
    
    return tensor.astype(np.float32)


def get_srm_kernel_weights() -> np.ndarray:
    """
    Get SRM kernels as a numpy array suitable for initializing
    a CNN convolutional layer.
    
    Returns:
        numpy array of shape (num_kernels, 1, 5, 5)
    """
    kernels = _build_srm_kernels()
    weights = np.stack(kernels, axis=0)  # (N, 5, 5)
    weights = weights[:, np.newaxis, :, :]  # (N, 1, 5, 5)
    return weights.astype(np.float32)
