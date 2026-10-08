"""
inference/tta.py
================
Test-Time Augmentation (TTA) & Multi-Crop Inference Engine.

Improves real-world botanical classification accuracy and robustness without retraining.
Evaluates multiple biologically valid photographic perspectives and lighting transforms,
aggregating predictions via weighted probability blending and measuring cross-transform stability.

Supported TTA Modes:
- 'off'      : 1 forward pass (Standard baseline)
- 'fast'     : 3 forward passes (Original, Horizontal Flip, Smart Crop) — Default for fast web requests
- 'standard' : 5 forward passes (Original, H-Flip, 180-Rot, Smart Crop, Scaled Crop)
- 'thorough' : 7 forward passes (Adds contrast & subtle lighting perturbations)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps

from inference.preprocessing import preprocess_for_inference, get_inference_transform
from inference.crop import get_smart_cropped_image


def generate_tta_variants(
    image: Image.Image,
    mode: str = "standard"
) -> List[Tuple[str, Image.Image, float]]:
    """
    Generates biologically plausible image variants for test-time augmentation.
    Returns list of (transform_name, transformed_image, voting_weight).
    """
    variants: List[Tuple[str, Image.Image, float]] = []

    # 1. Base canonical image (Highest weighting)
    variants.append(("original", image.copy(), 1.0))

    if mode == "off" or mode is None:
        return variants

    # 2. Horizontal Flip (Natural bilateral symmetry of botanical specimens)
    hflip_img = ImageOps.mirror(image)
    variants.append(("horizontal_flip", hflip_img, 0.85))

    # 3. Smart Botanical Foreground Crop
    smart_cropped, crop_meta = get_smart_cropped_image(image)
    if crop_meta.get("crop_applied"):
        variants.append(("smart_crop", smart_cropped, 0.90))

    if mode in ["standard", "thorough"]:
        # 4. 180-Degree Rotation / Vertical Flip (Leaves exhibit arbitrary gravity orientation in photos)
        rot_img = image.rotate(180, expand=False, resample=Image.BILINEAR)
        variants.append(("rotate_180", rot_img, 0.75))

        # 5. Mild Center Zoom / Scale (0.9 zoom crop)
        w, h = image.size
        crop_margin_x = int(w * 0.05)
        crop_margin_y = int(h * 0.05)
        if crop_margin_x > 2 and crop_margin_y > 2:
            zoomed = image.crop((crop_margin_x, crop_margin_y, w - crop_margin_x, h - crop_margin_y))
            variants.append(("scale_zoom", zoomed, 0.80))

    if mode == "thorough":
        # 6. Subtle Contrast Adjustment (+10% contrast for vein structure clarity)
        enhancer_c = ImageEnhance.Contrast(image)
        contrast_img = enhancer_c.enhance(1.10)
        variants.append(("contrast_boost", contrast_img, 0.70))

        # 7. Subtle Brightness Adjustment (-5% to recover highlights)
        enhancer_b = ImageEnhance.Brightness(image)
        dim_img = enhancer_b.enhance(0.95)
        variants.append(("brightness_recovery", dim_img, 0.70))

    return variants


def predict_with_tta(
    model: nn.Module,
    class_names: List[str],
    image: Image.Image,
    mode: str = "fast",
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Executes Test-Time Augmentation inference on an input image.

    Args:
        model: Trained PyTorch classifier in eval mode.
        class_names: List of class strings indexed by output logits.
        image: PIL Image in RGB format.
        mode: TTA mode ('off', 'fast', 'standard', 'thorough').
        device: Torch execution device.
        temperature: Logit temperature scaling factor (default 1.0).
        top_k: Number of highest-ranked candidate classes to return.

    Returns:
        dict containing:
        - aggregated_probs: np.ndarray (shape: num_classes)
        - top_predictions: list of dicts with 'name', 'confidence', 'raw_class'
        - stability_score: float (fraction of variants matching top-1 prediction)
        - variant_details: list of individual variant top-1 predictions
        - variant_count: int
    """
    variants = generate_tta_variants(image, mode=mode)
    num_classes = len(class_names)
    target_device = device if device is not None else next(model.parameters()).device

    variant_results = []
    weighted_prob_sum = np.zeros(num_classes, dtype=np.float64)
    total_weight = 0.0

    model.eval()
    with torch.no_grad():
        for name, var_img, weight in variants:
            tensor = preprocess_for_inference(var_img, device=target_device)
            logits = model(tensor)

            # Apply temperature calibration to logits
            scaled_logits = logits / max(1e-4, temperature)
            probs = torch.softmax(scaled_logits, dim=1)[0].cpu().numpy()

            weighted_prob_sum += probs * weight
            total_weight += weight

            top_idx = int(np.argmax(probs))
            top_class = class_names[top_idx]
            top_conf = float(probs[top_idx])

            variant_results.append({
                "transform": name,
                "weight": weight,
                "top_class": top_class,
                "top_confidence": round(top_conf, 4)
            })

    # Normalized weighted average probability distribution
    aggregated_probs = weighted_prob_sum / max(1e-6, total_weight)

    # Sort classes by ensemble probability
    sorted_indices = aggregated_probs.argsort()[::-1]
    top_predictions = [
        {
            "name": class_names[idx],
            "raw_class": class_names[idx],
            "confidence": float(round(float(aggregated_probs[idx]), 4))
        }
        for idx in sorted_indices[:top_k]
    ]

    # Calculate prediction stability across variants
    ensemble_top1 = top_predictions[0]["name"]
    agreeing_variants = sum(1 for v in variant_results if v["top_class"] == ensemble_top1)
    stability_score = float(round(agreeing_variants / len(variant_results), 3))

    return {
        "aggregated_probs": aggregated_probs,
        "top_predictions": top_predictions,
        "stability_score": stability_score,
        "variant_details": variant_results,
        "variant_count": len(variants),
        "ensemble_top1": ensemble_top1
    }
