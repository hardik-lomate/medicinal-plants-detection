"""
backend/model.py
=================
Unified Model Loading, Multi-Stage Inference, and Database Integration Interface.

Primary Architecture:
- Exact-parity baseline inference matching trained model evaluations
- Authoritative prediction preservation (model argmax is authoritative)
- Aspect-ratio & EXIF-safe image loading
- Optional diagnostic signals (Grad-CAM, image quality, embedding similarity)
- Botanical name normalization & SQLite monograph retrieval
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg

# Import centralized modular components
from inference.preprocessing import (
    clean_image_path,
    load_image_safely,
    preprocess_for_inference,
    IMAGE_SIZE,
    RESIZE_EDGE,
    MEAN,
    STD,
    SUPPORTED_EXTENSIONS
)
from inference.quality import check_image_quality as check_img_qual
from inference.engine import (
    predict_baseline,
    predict_pipeline,
    get_loaded_models,
    normalize_plant_name,
    get_plant_name_mapping
)

# Exported decision thresholds for backward compatibility
CONFIDENCE_THRESHOLD = cfg.CONFIDENCE_THRESHOLD
PREDICTION_THRESHOLD = getattr(cfg, 'PREDICTION_THRESHOLD', 0.60)
MARGIN_THRESHOLD = cfg.MARGIN_THRESHOLD
UNKNOWN_THRESHOLD = cfg.UNKNOWN_THRESHOLD
IMAGE_TYPE_CONFIDENCE = getattr(cfg, 'IMAGE_TYPE_CONFIDENCE', 0.65)


def check_image_quality(image: Image.Image) -> tuple[bool, str]:
    """
    Backward-compatible quality check adapter returning (is_acceptable, message).
    """
    res = check_img_qual(image)
    return res.is_acceptable, res.user_message


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Backward-compatible tensor transformation.
    """
    return preprocess_for_inference(image)


def init_model():
    """Warm up and load all trained models at application startup."""
    get_loaded_models()


def predict_baseline_image(
    image_input: Union[str, Path, bytes, Image.Image, Any],
    forced_type: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Step 1 Pure Baseline Inference interface.
    Executes single forward pass without secondary heuristics.
    """
    return predict_baseline(
        image_input=image_input,
        forced_type=forced_type,
        debug=debug
    )


def predict_image(
    image_input: Union[str, Path, bytes, Image.Image, Any],
    forced_type: Optional[str] = None,
    debug: bool = False,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Unified prediction interface for web server, CLI, and real-world evaluation.
    Uses authoritative baseline classification with diagnostic explainability.
    """
    return predict_pipeline(
        image_input=image_input,
        forced_type=forced_type,
        tta_mode="off",
        generate_explanation=True,
        debug=debug,
        threshold=threshold
    )


# Backwards compatibility aliases
predict = predict_image
