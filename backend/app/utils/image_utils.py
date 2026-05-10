"""Image utility functions for loading, resizing, and encoding."""
import io, base64
import numpy as np
from PIL import Image

def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert('RGB')

def image_to_base64(image: Image.Image, format: str = "PNG", max_size: int = 512) -> str:
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_image(b64_string: str) -> Image.Image:
    data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(data)).convert('RGB')
