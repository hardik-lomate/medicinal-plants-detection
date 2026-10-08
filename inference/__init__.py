"""
inference package
=================
Modular, production-grade inference engine for medicinal plant detection.
"""

from inference.preprocessing import (
    load_image_safely,
    clean_image_path,
    preprocess_for_inference,
    IMAGE_SIZE,
    MEAN,
    STD
)
from inference.quality import check_image_quality, QualityResult
from inference.crop import get_smart_cropped_image, detect_botanical_bbox
from inference.tta import predict_with_tta, generate_tta_variants
from inference.calibration import DEFAULT_CALIBRATOR, ConfidenceCalibrator
from inference.embedding import extract_feature_embedding, compute_embedding_similarity
from inference.explainability import generate_gradcam_overlay
from inference.ood import DEFAULT_DECISION_ENGINE, UncertaintyDecisionEngine
from inference.engine import predict_pipeline, get_loaded_models, normalize_plant_name, get_plant_name_mapping

__all__ = [
    "predict_pipeline",
    "get_loaded_models",
    "normalize_plant_name",
    "get_plant_name_mapping",
    "load_image_safely",
    "clean_image_path",
    "preprocess_for_inference",
    "check_image_quality",
    "QualityResult",
    "get_smart_cropped_image",
    "predict_with_tta",
    "DEFAULT_CALIBRATOR",
    "extract_feature_embedding",
    "compute_embedding_similarity",
    "generate_gradcam_overlay",
    "DEFAULT_DECISION_ENGINE",
    "IMAGE_SIZE",
    "MEAN",
    "STD",
]
