"""
inference/quality.py
====================
Comprehensive Image Quality Analyzer for Botanical Computer Vision.

Inspects images for:
- Resolution inadequacy (< 60px min edge, or < 6,400 total pixels)
- Excessive motion/optical blur via Laplacian edge variance
- Severe underexposure (dark images with lost shadow detail)
- Severe overexposure / specular glare (washed out highlights)
- Flat/blank images (uniform walls, lens caps, low Shannon entropy)
- Extreme low contrast (gray fog / haze)

Provides a normalized continuous quality score (0.0 to 1.0) and actionable feedback.
"""

import math
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageStat, ImageFilter


@dataclass
class QualityResult:
    is_acceptable: bool
    quality_score: float  # 0.0 to 1.0
    brightness: float     # Mean luminance (0-255)
    contrast: float       # Stddev of luminance (0-128)
    sharpness: float      # Laplacian variance metric
    entropy: float        # Shannon entropy in bits (0-8)
    width: int
    height: int
    issues: List[str]     # Severe issues that reject the image
    warnings: List[str]   # Mild issues that permit inference with caveat
    user_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_image_entropy(gray_image: Image.Image) -> float:
    """
    Computes Shannon entropy of the 8-bit luminance histogram.
    Blank/flat images have entropy < 2.5; rich natural foliage has entropy 5.5 - 7.5.
    """
    hist = gray_image.histogram()
    total_pixels = sum(hist)
    if total_pixels == 0:
        return 0.0

    entropy = 0.0
    for count in hist:
        if count > 0:
            p = count / total_pixels
            entropy -= p * math.log2(p)
    return float(round(entropy, 3))


def calculate_sharpness(gray_image: Image.Image) -> float:
    """
    Estimates image sharpness using the variance of the Laplacian filter.
    Higher values represent sharper edges and finer vein details.
    """
    try:
        # 3x3 Laplacian kernel in Pillow
        laplacian_kernel = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1,
            offset=0
        )
        filtered = gray_image.filter(laplacian_kernel)
        stat = ImageStat.Stat(filtered)
        variance = stat.var[0]
        return float(round(variance, 2))
    except Exception:
        # Fallback to simple edge detection
        edges = gray_image.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges)
        return float(round(stat.var[0], 2))


def check_image_quality(
    image: Image.Image,
    min_dimension: int = 60,
    min_pixels: int = 6400,
    blur_threshold: float = 7.0,
    min_brightness: float = 14.0,
    max_brightness: float = 246.0,
    min_contrast: float = 9.0,
    min_entropy: float = 2.8
) -> QualityResult:
    """
    Evaluates input image quality before running through the neural network.
    Uses conservative thresholds to prevent false rejections of real-world mobile photos
    while catching truly unusable, blank, or corrupted inputs.

    Returns:
        QualityResult with is_acceptable flag and detailed diagnostic metrics.
    """
    width, height = image.size
    total_pixels = width * height
    issues: List[str] = []
    warnings: List[str] = []

    # 1. Resolution Check
    if width < min_dimension or height < min_dimension or total_pixels < min_pixels:
        issues.append(f"Resolution is too low ({width}x{height}px). Minimum required is {min_dimension}x{min_dimension}px.")

    # Convert to grayscale for illumination & texture analysis
    gray = image.convert('L')
    stat = ImageStat.Stat(gray)
    brightness = float(round(stat.mean[0], 2))
    contrast = float(round(stat.stddev[0], 2))

    # 2. Brightness & Exposure Checks
    if brightness < min_brightness:
        issues.append("Image is severely underexposed (too dark) to resolve leaf veins or margins.")
    elif brightness < 30.0:
        warnings.append("Image is dimly lit. Identification accuracy may improve with better lighting.")

    if brightness > max_brightness and contrast < 12.0:
        issues.append("Image is severely overexposed or washed out by harsh glare.")
    elif brightness > 220.0:
        warnings.append("High brightness detected. Watch out for washed-out leaf details.")

    # 3. Contrast Check
    if contrast < min_contrast and total_pixels >= min_pixels:
        issues.append("Image has almost no contrast (nearly uniform monochrome gray).")
    elif contrast < 18.0:
        warnings.append("Low tonal contrast detected between foreground and background flora.")

    # 4. Entropy / Visual Content Check
    entropy = calculate_image_entropy(gray)
    if entropy < min_entropy and total_pixels >= min_pixels:
        issues.append("Image contains virtually no visual structure (flat blank surface, wall, or lens cap).")

    # 5. Blur / Sharpness Check
    sharpness = calculate_sharpness(gray)
    if sharpness < blur_threshold and total_pixels >= 40000 and len(issues) == 0:
        issues.append("Image is severely blurry with lack of distinguishable edge or vein structure.")
    elif sharpness < 15.0:
        warnings.append("Soft focus detected. Holding the camera steady produces optimal results.")

    # Compute continuous composite quality score (0.0 to 1.0)
    # Norm components:
    # Sharpness: 0 to 60 -> 0.0 to 1.0
    s_norm = min(1.0, max(0.0, sharpness / 60.0))
    # Contrast: 0 to 60 -> 0.0 to 1.0
    c_norm = min(1.0, max(0.0, contrast / 50.0))
    # Brightness center distance from 128: 0 to 128 -> 1.0 to 0.0
    b_norm = max(0.0, 1.0 - abs(brightness - 128.0) / 115.0)
    # Entropy: 0 to 8 -> 0.0 to 1.0
    e_norm = min(1.0, max(0.0, entropy / 7.0))

    composite_score = float(round(0.35 * s_norm + 0.25 * c_norm + 0.20 * b_norm + 0.20 * e_norm, 3))
    if issues:
        composite_score = min(composite_score, 0.25)

    is_acceptable = len(issues) == 0

    if not is_acceptable:
        user_message = " ".join(issues)
    elif warnings:
        user_message = f"Quality acceptable (score: {composite_score:.2f}). Note: " + "; ".join(warnings)
    else:
        user_message = f"Optimal photograph quality (score: {composite_score:.2f})."

    return QualityResult(
        is_acceptable=is_acceptable,
        quality_score=composite_score,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        entropy=entropy,
        width=width,
        height=height,
        issues=issues,
        warnings=warnings,
        user_message=user_message
    )
