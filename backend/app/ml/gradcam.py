"""
Attention-based Explainability for CLIP ViT models.

Since CLIP uses a Vision Transformer (not a CNN), traditional Grad-CAM doesn't
apply. Instead we use **attention rollout** — aggregating attention weights
across all transformer layers to produce a spatial attention map showing which
image regions influenced the prediction most.

Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", 2020
"""
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import torch


def _get_attention_maps(model, processor, image: Image.Image):
    """
    Extract attention maps from all layers of CLIP's vision transformer.

    Returns:
        List of attention matrices, each of shape (num_heads, num_patches+1, num_patches+1)
    """
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
        )

    # outputs.attentions is a tuple of (num_layers) tensors,
    # each of shape (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    return [att.squeeze(0).cpu().numpy() for att in attentions]


def attention_rollout(attention_maps, discard_ratio=0.1):
    """
    Compute attention rollout across all transformer layers.

    This aggregates attention from every layer, accounting for residual
    connections, to produce a single attention distribution over patches
    from the [CLS] token's perspective.

    Args:
        attention_maps: List of attention arrays (num_heads, seq_len, seq_len)
        discard_ratio: Fraction of lowest-attention heads to zero out

    Returns:
        1D array of attention weights over spatial patches (excluding CLS)
    """
    result = None

    for attention in attention_maps:
        # Average across heads
        att_heads_mean = np.mean(attention, axis=0)  # (seq_len, seq_len)

        # Optional: discard low-attention values for cleaner maps
        if discard_ratio > 0:
            flat = att_heads_mean.flatten()
            threshold = np.quantile(flat, discard_ratio)
            att_heads_mean = np.where(att_heads_mean < threshold, 0, att_heads_mean)

        # Re-normalize rows
        row_sums = att_heads_mean.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        att_heads_mean = att_heads_mean / row_sums

        # Add identity for residual connection
        att_with_residual = 0.5 * att_heads_mean + 0.5 * np.eye(att_heads_mean.shape[0])

        # Re-normalize
        row_sums = att_with_residual.sum(axis=-1, keepdims=True)
        att_with_residual = att_with_residual / row_sums

        if result is None:
            result = att_with_residual
        else:
            result = att_with_residual @ result

    # Extract attention from CLS token (index 0) to all patch tokens
    cls_attention = result[0, 1:]  # Exclude CLS-to-CLS

    return cls_attention


def generate_clip_attention_map(model, processor, image: Image.Image, alpha=0.5) -> str:
    """
    Generate an attention-based heatmap overlay (similar to Grad-CAM)
    for CLIP's Vision Transformer.

    Args:
        model: CLIP model
        processor: CLIP processor
        image: PIL Image
        alpha: Transparency of the heatmap overlay

    Returns:
        Base64-encoded JPEG string of the attention overlay
    """
    # Get attention maps from all layers
    attention_maps = _get_attention_maps(model, processor, image)

    # Compute rollout
    cls_attention = attention_rollout(attention_maps, discard_ratio=0.1)

    # Reshape to 2D grid
    # CLIP ViT-B/32 with 224x224 input: 7x7 = 49 patches
    num_patches = cls_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        # Fallback: just use what we can
        grid_size = int(np.ceil(np.sqrt(num_patches)))
        padded = np.zeros(grid_size * grid_size)
        padded[:num_patches] = cls_attention
        cls_attention = padded

    heatmap = cls_attention.reshape(grid_size, grid_size)

    # Normalize to [0, 1]
    heatmap = heatmap - heatmap.min()
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax

    # Resize to original image dimensions
    heatmap = cv2.resize(heatmap.astype(np.float32), (image.width, image.height))

    # Apply JET colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    orig_arr = np.array(image.convert("RGB"))
    superimposed = (heatmap_colored * alpha + orig_arr * (1 - alpha)).clip(0, 255).astype(np.uint8)

    vis_img = Image.fromarray(superimposed)

    # Encode to base64
    buf = BytesIO()
    vis_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
