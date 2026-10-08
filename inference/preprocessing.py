"""
inference/preprocessing.py
===========================
Centralized Image Preprocessing & Sanitization Engine.
Single source of truth used across CLI prediction, Flask inference,
and automated evaluation.

Ensures:
- Exact alignment with training/validation transforms (Resize 256 + CenterCrop 224 + ImageNet norm)
- EXIF orientation auto-correction
- RGB conversion across all image modes (RGBA, CMYK, P, L, etc.)
- Thumbnail scaling for ultra-high-resolution smartphone captures
- Robust Windows path sanitization (stripping invisible Unicode, directional markers, smart quotes)
"""

import os
import sys
import re
from pathlib import Path
from io import BytesIO
from typing import Tuple, Union, Optional, Dict, Any

from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

# Project configuration import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg

# Preprocessing Constants
IMAGE_SIZE: int = getattr(cfg, 'IMAGE_SIZE', 224)
RESIZE_EDGE: int = getattr(cfg, 'RESIZE_EDGE', 256)
MEAN: list = getattr(cfg, 'MEAN', [0.485, 0.456, 0.406])
STD: list = getattr(cfg, 'STD', [0.229, 0.224, 0.225])
MAX_PRE_SCALE_DIM: int = 512
SUPPORTED_EXTENSIONS: set = getattr(cfg, 'IMAGE_EXTENSIONS', {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'})


def clean_image_path(raw_path: Union[str, Path]) -> Tuple[bool, str, str]:
    """
    Sanitize and validate an input image path from Windows CLI or environment.
    Removes invisible Unicode characters (LTR/RTL marks, zero-width spaces, BOM),
    stray quotation marks, normalizes path separators, and verifies file existence.

    Returns:
        (is_valid: bool, resolved_path: str, error_message: str)
    """
    if not raw_path:
        return False, "", "No image path provided."

    # Convert to string and strip surrounding whitespace
    cleaned = str(raw_path).strip()

    # Strip surrounding quotation marks (single, double, smart quotes)
    cleaned = cleaned.strip('\'"“”‘’')

    # Strip invisible Unicode directional and zero-width markers:
    # \u200e (LRM), \u200f (RLM), \u202a-\u202e (directional embeddings/overrides),
    # \u200b (zero-width space), \ufeff (BOM), \u00a0 (non-breaking space)
    cleaned = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00a0]', '', cleaned)

    # Strip again after removing invisible characters
    cleaned = cleaned.strip('\'"“”‘’ \t\r\n')

    if not cleaned:
        return False, "", "Image path is empty after sanitization."

    # Normalize and resolve absolute path
    norm_path = os.path.normpath(cleaned)
    abs_path = os.path.abspath(norm_path)

    # Check existence
    if not os.path.exists(abs_path):
        return False, abs_path, f"Image not found: {abs_path}"

    # Check if regular file
    if not os.path.isfile(abs_path):
        return False, abs_path, f"Path is not a regular file: {abs_path}"

    # Check extension
    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        return False, abs_path, f"Unsupported file extension '{ext}'. Supported: {supported_str}"

    return True, abs_path, ""


def load_image_safely(
    image_input: Union[str, Path, bytes, BytesIO, Image.Image]
) -> Tuple[bool, Optional[Image.Image], str, Dict[str, Any]]:
    """
    Safely load, auto-orient, and prepare any image input for neural network inference.
    Handles file paths, bytes, file-like streams, and existing PIL Image instances.

    Returns:
        (success: bool, image: Optional[PIL.Image], error_message: str, metadata: dict)
    """
    metadata: Dict[str, Any] = {
        "original_size": (0, 0),
        "original_mode": "",
        "format": "",
        "path": None,
    }

    raw_img: Optional[Image.Image] = None

    try:
        if isinstance(image_input, (str, Path)):
            valid, path_str, err = clean_image_path(image_input)
            if not valid:
                return False, None, err, metadata
            metadata["path"] = path_str
            raw_img = Image.open(path_str)

        elif isinstance(image_input, bytes):
            raw_img = Image.open(BytesIO(image_input))

        elif isinstance(image_input, BytesIO):
            image_input.seek(0)
            raw_img = Image.open(image_input)

        elif isinstance(image_input, Image.Image):
            raw_img = image_input

        elif hasattr(image_input, 'read'):
            # Werkzeug FileStorage or similar file-like stream
            image_input.seek(0)
            raw_img = Image.open(image_input)

        else:
            return False, None, f"Unsupported image input type: {type(image_input)}", metadata

        metadata["original_size"] = raw_img.size
        metadata["original_mode"] = raw_img.mode
        metadata["format"] = getattr(raw_img, 'format', '') or 'UNKNOWN'

        # 1. Apply EXIF orientation transposition
        oriented_img = ImageOps.exif_transpose(raw_img)

        # 2. Ensure standard RGB color space
        rgb_img = oriented_img.convert('RGB')

        return True, rgb_img, "", metadata

    except Exception as e:
        return False, None, f"Failed to load image: {str(e)}", metadata


def get_inference_transform() -> transforms.Compose:
    """Returns standard validation/inference transform matching training."""
    return transforms.Compose([
        transforms.Resize(RESIZE_EDGE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def preprocess_for_inference(
    image: Image.Image,
    device: Optional[torch.device] = None,
    apply_thumbnail: bool = True
) -> torch.Tensor:
    """
    Transforms a PIL Image into a normalized PyTorch tensor ready for model forward pass.

    Args:
        image: PIL Image in RGB mode.
        device: Target torch.device (defaults to config device).
        apply_thumbnail: If True, caps maximum dimension to MAX_PRE_SCALE_DIM
                         before resizing (matches ManifestDataset behavior).

    Returns:
        Tensor of shape (1, 3, 224, 224) on specified device.
    """
    img_copy = image.copy()

    # Pre-scale if image is huge (e.g. 12MP phone camera), preserving aspect ratio
    if apply_thumbnail and max(img_copy.size) > MAX_PRE_SCALE_DIM:
        img_copy.thumbnail((MAX_PRE_SCALE_DIM, MAX_PRE_SCALE_DIM), Image.Resampling.BILINEAR)

    transform = get_inference_transform()
    tensor = transform(img_copy).unsqueeze(0)

    target_device = device if device is not None else cfg.get_device()
    return tensor.to(target_device)
