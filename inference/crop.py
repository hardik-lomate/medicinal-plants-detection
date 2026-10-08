"""
inference/crop.py
=================
Botanical Foreground Localization & Safe Smart Cropping Engine.

Detects plant and leaf regions using vegetation chromatic indexing (Excess Green,
HSV botanical color thresholding, and morphological saliency) without requiring
any external neural model.

Features:
- Non-destructive: Leaves original image intact and provides cropped variant
- Margin safety: Adds 15-20% protective border around bounding boxes to prevent
  severing serrations, petioles, or leaf tips
- Conservative fallback: Reverts to original framing if foreground occupies < 10%
  or > 90% of image
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps


def detect_botanical_bbox(
    image: Image.Image,
    min_area_fraction: float = 0.10,
    max_area_fraction: float = 0.90,
    margin_expansion: float = 0.18
) -> Tuple[bool, Tuple[int, int, int, int], Dict[str, Any]]:
    """
    Locates the primary plant/leaf region using botanical color indexing
    and morphological saliency.

    Returns:
        (detected: bool, (x1, y1, x2, y2): Tuple[int, int, int, int], metadata: dict)
    """
    w, h = image.size
    total_area = w * h

    # Work on a lightweight downsampled representation for fast execution
    scale = min(1.0, 320.0 / max(w, h))
    thumb_w, thumb_h = max(32, int(w * scale)), max(32, int(h * scale))
    thumb = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

    img_np = np.array(thumb, dtype=np.float32)
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]

    # 1. Excess Green Index (ExG = 2*G - R - B)
    # Standard agricultural computer vision metric for foliage segmentation
    exg = 2.0 * g - r - b

    # 2. Plant vegetation mask (ExG thresholding + HSV green/yellow/dark foliage)
    denom = r + g + b + 1e-5
    norm_g = g / denom
    green_mask = (exg > 10.0) | (norm_g > 0.38)

    # 3. Handle dark foliage, purple/red-veined leaves, and shade
    # Contrast against neutral backgrounds (desaturated white/gray/brown surface)
    saturation = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) / (np.maximum(np.maximum(r, g), b) + 1e-5)
    foliage_mask = green_mask | ((saturation > 0.22) & (g > 30.0))

    foreground_count = int(np.sum(foliage_mask))
    foreground_fraction = foreground_count / (thumb_w * thumb_h)

    meta: Dict[str, Any] = {
        "foreground_fraction": float(round(foreground_fraction, 3)),
        "method": "chromatic_saliency",
        "crop_applied": False,
        "original_size": (w, h)
    }

    # If foreground is negligible or occupies virtually the entire frame, do not crop
    if foreground_fraction < min_area_fraction or foreground_fraction > max_area_fraction:
        meta["reason"] = "Foreground coverage outside safe cropping window (too sparse or already fills frame)."
        return False, (0, 0, w, h), meta

    # Find row and column bounding spans where foliage density is significant
    row_density = np.sum(foliage_mask, axis=1) / thumb_w
    col_density = np.sum(foliage_mask, axis=0) / thumb_h

    # Threshold for active rows/columns (at least 6% foliage density)
    active_rows = np.where(row_density > 0.06)[0]
    active_cols = np.where(col_density > 0.06)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        meta["reason"] = "Could not resolve coherent foreground bounds."
        return False, (0, 0, w, h), meta

    min_y, max_y = active_rows[0], active_rows[-1]
    min_x, max_x = active_cols[0], active_cols[-1]

    # Convert thumbnail coordinates back to original resolution
    x1 = int(min_x / scale)
    y1 = int(min_y / scale)
    x2 = int((max_x + 1) / scale)
    y2 = int((max_y + 1) / scale)

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h

    if bbox_area / total_area < min_area_fraction:
        meta["reason"] = "Bounding box too small to represent primary specimen."
        return False, (0, 0, w, h), meta

    # Add generous margin expansion to prevent clipping leaf tips, petioles, or serrations
    margin_x = int(bbox_w * margin_expansion)
    margin_y = int(bbox_h * margin_expansion)

    final_x1 = max(0, x1 - margin_x)
    final_y1 = max(0, y1 - margin_y)
    final_x2 = min(w, x2 + margin_x)
    final_y2 = min(h, y2 + margin_y)

    meta["crop_applied"] = True
    meta["raw_bbox"] = (x1, y1, x2, y2)
    meta["expanded_bbox"] = (final_x1, final_y1, final_x2, final_y2)
    meta["cropped_area_fraction"] = float(round((final_x2 - final_x1) * (final_y2 - final_y1) / total_area, 3))

    return True, (final_x1, final_y1, final_x2, final_y2), meta


def get_smart_cropped_image(
    image: Image.Image,
    min_area_fraction: float = 0.12,
    max_area_fraction: float = 0.88
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Applies botanical foreground localization and returns the cropped PIL Image.
    If no salient foreground region can be safely isolated, returns the original image.

    Returns:
        (processed_image: PIL.Image, metadata: dict)
    """
    detected, bbox, meta = detect_botanical_bbox(
        image,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction
    )

    if detected and meta.get("crop_applied"):
        cropped = image.crop(bbox)
        return cropped, meta

    return image.copy(), meta
