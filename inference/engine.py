"""
inference/engine.py
===================
Unified Inference Engine for Medicinal Plant Detection.

Orchestrates the entire multi-stage computer-vision and pharmacological pipeline:
1. Robust image loading, orientation correction, and path sanitization
2. Image quality verification (sharpness, exposure, contrast, entropy)
3. Stage 1: Image-type classification (Leaf vs Whole Plant) with calibrated routing
4. Stage 2: Specialized species classification with Test-Time Augmentation (TTA)
5. Temperature-scaled probability calibration and Top-K distribution analysis
6. Deep feature embedding extraction and centroid cosine similarity verification
7. Grad-CAM visual attention heatmap generation
8. Multi-signal out-of-distribution (OOD) and uncertainty gating
9. Canonical class normalization and SQLite database monograph retrieval
10. Standardized, backward-compatible JSON reporting
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg
from training.model_factory import create_model

from inference.preprocessing import (
    load_image_safely,
    clean_image_path,
    preprocess_for_inference,
    IMAGE_SIZE,
    MEAN,
    STD
)
from inference.quality import check_image_quality, QualityResult
from inference.tta import predict_with_tta
from inference.calibration import DEFAULT_CALIBRATOR
from inference.embedding import compute_embedding_similarity
from inference.explainability import generate_gradcam_overlay
from inference.ood import DEFAULT_DECISION_ENGINE

# Cached model singletons & mappings
_LOADED_MODELS: Dict[str, Any] = {}
_PLANT_NAME_MAP: Optional[Dict[str, Any]] = None
_DEVICE = cfg.get_device()


def get_plant_name_mapping() -> Dict[str, Any]:
    """Loads and caches models/class_mappings/plant_name_mapping.json."""
    global _PLANT_NAME_MAP
    if _PLANT_NAME_MAP is not None:
        return _PLANT_NAME_MAP

    mapping_path = getattr(cfg, 'PLANT_NAME_MAPPING_JSON', cfg.CLASS_MAPPING_DIR / "plant_name_mapping.json")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                _PLANT_NAME_MAP = json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load plant_name_mapping.json: {e}")
            _PLANT_NAME_MAP = {}
    else:
        _PLANT_NAME_MAP = {}

    return _PLANT_NAME_MAP


def normalize_plant_name(raw_name: str) -> tuple[str, str, str]:
    """
    Map raw model prediction class name to canonical display name,
    botanical scientific name, and database lookup key.

    Returns:
        (display_name: str, scientific_name: str, database_key: str)
    """
    if not raw_name:
        return "Unknown Plant", "", ""

    mapping = get_plant_name_mapping()

    # Exact match in mapping
    if raw_name in mapping:
        info = mapping[raw_name]
        return info.get("display_name", raw_name), info.get("scientific_name", ""), info.get("database_key", raw_name)

    # Case-insensitive match in mapping
    raw_lower = raw_name.lower().replace("_", " ").strip()
    for key, info in mapping.items():
        if key.lower().replace("_", " ").strip() == raw_lower or info.get("normalized_name", "").lower() == raw_lower:
            return info.get("display_name", raw_name), info.get("scientific_name", ""), info.get("database_key", key)

    # Fallback to humanized display name
    clean_display = raw_name.replace("_", " ").title()
    return clean_display, "", raw_name


def load_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Dynamically load any trained model checkpoint with its metadata."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)
        backbone = ckpt.get('model_name', 'efficientnet_b0')
        num_classes = ckpt.get('num_classes', len(ckpt.get('class_names', [])))
        class_names = ckpt.get('class_names', [])

        model = create_model(backbone, num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(_DEVICE)
        model.eval()

        return {
            'model': model,
            'class_names': class_names,
            'class_to_index': ckpt.get('class_to_index', {c: i for i, c in enumerate(class_names)}),
            'version': ckpt.get('model_version', 'v3'),
            'config': ckpt.get('config', {}),
            'path': str(checkpoint_path),
        }
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def get_loaded_models() -> Dict[str, Any]:
    """Initializes and caches models on first access."""
    global _LOADED_MODELS
    if _LOADED_MODELS:
        return _LOADED_MODELS

    cache = {
        'image_type': None,
        'leaf': None,
        'whole_plant': None,
        'legacy': None,
    }

    # 1. Image Type Model
    if os.path.exists(cfg.IMAGE_TYPE_MODEL_PATH):
        cache['image_type'] = load_checkpoint(cfg.IMAGE_TYPE_MODEL_PATH)
        if cache['image_type']:
            print(f"[OK] Loaded Image-Type Model: {cfg.IMAGE_TYPE_MODEL_PATH.name}")

    # 2. Leaf Model
    if os.path.exists(cfg.LEAF_MODEL_PATH):
        cache['leaf'] = load_checkpoint(cfg.LEAF_MODEL_PATH)
        if cache['leaf']:
            print(f"[OK] Loaded Leaf Model ({len(cache['leaf']['class_names'])} classes): {cfg.LEAF_MODEL_PATH.name}")

    # 3. Whole-Plant Model
    if os.path.exists(cfg.PLANT_MODEL_PATH):
        cache['whole_plant'] = load_checkpoint(cfg.PLANT_MODEL_PATH)
        if cache['whole_plant']:
            print(f"[OK] Loaded Whole-Plant Model ({len(cache['whole_plant']['class_names'])} classes): {cfg.PLANT_MODEL_PATH.name}")

    # 4. Legacy Model (fallback)
    legacy_path = os.path.join(PROJECT_ROOT, 'models', 'medicinal_plant_model.pth')
    if os.path.exists(legacy_path):
        cache['legacy'] = load_checkpoint(legacy_path)
        if cache['legacy']:
            print(f"[OK] Loaded Legacy Fallback Model: {legacy_path}")

    _LOADED_MODELS = cache
    return _LOADED_MODELS


def predict_pipeline(
    image_input: Union[str, Path, bytes, Image.Image, Any],
    forced_type: Optional[str] = None,
    tta_mode: str = "fast",
    generate_explanation: bool = True,
    debug: bool = False,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Unified inference pipeline called by Flask API, CLI scripts, and benchmark evaluation.

    Args:
        image_input: Image path, raw bytes, or PIL Image.
        forced_type: Optional override ('leaf' or 'whole_plant').
        tta_mode: Test-time augmentation mode ('off', 'fast', 'standard', 'thorough').
        generate_explanation: If True, generates Grad-CAM attention heatmap.
        debug: If True, includes diagnostic telemetry in output.
        threshold: Prediction threshold override (defaults to config).

    Returns:
        Structured JSON-ready dictionary conforming to API contracts.
    """
    models = get_loaded_models()
    debug_log: Dict[str, Any] = {} if debug else {}

    # Step 1: Safe image loading & orientation correction
    success, image, load_err, img_meta = load_image_safely(image_input)
    if not success or image is None:
        return {
            "success": False,
            "status": "bad_image",
            "input_type": "unknown",
            "input_type_confidence": 0.0,
            "plant_name": None,
            "scientific_name": "",
            "confidence": 0.0,
            "margin": 0.0,
            "prediction_stability": 0.0,
            "image_quality": 0.0,
            "is_medicinal": False,
            "medicinal_information_available": False,
            "top_predictions": [],
            "message": load_err or "Invalid image file.",
            "explanation_available": False,
            "debug_info": {"error": load_err, "input_meta": img_meta} if debug else None
        }

    img_w, img_h = image.size
    if debug:
        debug_log["1_image_path"] = img_meta.get("path") or "<in-memory>"
        debug_log["2_image_dimensions"] = f"{img_w}x{img_h}"
        debug_log["image_mode"] = image.mode

    # Step 2: Image Quality Analysis
    quality = check_image_quality(image)
    if debug:
        debug_log["image_quality_score"] = quality.quality_score
        debug_log["sharpness"] = quality.sharpness
        debug_log["brightness"] = quality.brightness
        debug_log["contrast"] = quality.contrast
        debug_log["entropy"] = quality.entropy

    if not quality.is_acceptable:
        return {
            "success": True,
            "status": "bad_image",
            "input_type": "unknown",
            "input_type_confidence": 0.0,
            "plant_name": None,
            "scientific_name": "",
            "confidence": 0.0,
            "margin": 0.0,
            "prediction_stability": 0.0,
            "image_quality": quality.quality_score,
            "quality_issues": quality.issues,
            "is_medicinal": False,
            "medicinal_information_available": False,
            "top_predictions": [],
            "message": quality.user_message,
            "explanation_available": False,
            "debug_info": debug_log if debug else None
        }

    # Step 3: Stage 1 — Image-Type Classification (Leaf vs Whole Plant)
    type_confidence = 1.0
    if forced_type in ["leaf", "whole_plant"]:
        detected_type = forced_type
        selected_model_key = forced_type
    elif models.get('image_type') is not None:
        type_model_entry = models['image_type']
        type_res = predict_with_tta(
            type_model_entry['model'],
            type_model_entry['class_names'],
            image,
            mode="off",
            temperature=DEFAULT_CALIBRATOR.get_temperature("image_type"),
            device=_DEVICE
        )
        top_type_pred = type_res["top_predictions"][0]
        type_confidence = top_type_pred["confidence"]

        if type_confidence >= getattr(cfg, 'IMAGE_TYPE_CONFIDENCE', 0.65):
            detected_type = top_type_pred["name"]
            selected_model_key = detected_type
        else:
            detected_type = "uncertain"
            selected_model_key = "both_evaluated"
    else:
        detected_type = "leaf"
        selected_model_key = "leaf"

    if debug:
        debug_log["3_image_type_prediction"] = detected_type
        debug_log["4_image_type_confidence"] = f"{type_confidence:.1%}"
        debug_log["5_selected_model"] = selected_model_key

    # Step 4: Stage 2 — Specialized Species Classification with TTA
    active_type = detected_type
    species_res = None

    if forced_type in ["leaf", "whole_plant"] and models.get(forced_type):
        active_type = forced_type
        species_res = predict_with_tta(
            models[active_type]['model'],
            models[active_type]['class_names'],
            image,
            mode=tta_mode,
            temperature=DEFAULT_CALIBRATOR.get_temperature(active_type),
            device=_DEVICE
        )

    if species_res is None:
        if detected_type == "whole_plant" and models.get('whole_plant'):
            active_type = "whole_plant"
            species_res = predict_with_tta(
                models['whole_plant']['model'],
                models['whole_plant']['class_names'],
                image,
                mode=tta_mode,
                temperature=DEFAULT_CALIBRATOR.get_temperature("whole_plant"),
                device=_DEVICE
            )
        elif detected_type == "leaf" and models.get('leaf'):
            active_type = "leaf"
            species_res = predict_with_tta(
                models['leaf']['model'],
                models['leaf']['class_names'],
                image,
                mode=tta_mode,
                temperature=DEFAULT_CALIBRATOR.get_temperature("leaf"),
                device=_DEVICE
            )
        elif models.get('leaf') and models.get('whole_plant'):
            # Stage 1 router was uncertain: evaluate both models and compare margin
            leaf_res = predict_with_tta(
                models['leaf']['model'],
                models['leaf']['class_names'],
                image,
                mode="fast",
                temperature=DEFAULT_CALIBRATOR.get_temperature("leaf"),
                device=_DEVICE
            )
            plant_res = predict_with_tta(
                models['whole_plant']['model'],
                models['whole_plant']['class_names'],
                image,
                mode="fast",
                temperature=DEFAULT_CALIBRATOR.get_temperature("whole_plant"),
                device=_DEVICE
            )
            leaf_margin = DEFAULT_CALIBRATOR.compute_margin(leaf_res["top_predictions"])
            plant_margin = DEFAULT_CALIBRATOR.compute_margin(plant_res["top_predictions"])

            if leaf_res["top_predictions"][0]["confidence"] + leaf_margin >= plant_res["top_predictions"][0]["confidence"] + plant_margin:
                species_res = leaf_res
                active_type = "leaf"
            else:
                species_res = plant_res
                active_type = "whole_plant"
        elif models.get('leaf'):
            active_type = "leaf"
            species_res = predict_with_tta(
                models['leaf']['model'],
                models['leaf']['class_names'],
                image,
                mode=tta_mode,
                temperature=DEFAULT_CALIBRATOR.get_temperature("leaf"),
                device=_DEVICE
            )
        elif models.get('whole_plant'):
            active_type = "whole_plant"
            species_res = predict_with_tta(
                models['whole_plant']['model'],
                models['whole_plant']['class_names'],
                image,
                mode=tta_mode,
                temperature=DEFAULT_CALIBRATOR.get_temperature("whole_plant"),
                device=_DEVICE
            )
        elif models.get('legacy'):
            active_type = "leaf"
            species_res = predict_with_tta(
                models['legacy']['model'],
                models['legacy']['class_names'],
                image,
                mode="off",
                temperature=1.0,
                device=_DEVICE
            )
        else:
            return {
                "success": False,
                "status": "bad_image",
                "input_type": "unknown",
                "input_type_confidence": 0.0,
                "plant_name": None,
                "scientific_name": "",
                "confidence": 0.0,
                "margin": 0.0,
                "prediction_stability": 0.0,
                "image_quality": quality.quality_score,
                "is_medicinal": False,
                "medicinal_information_available": False,
                "top_predictions": [],
                "message": "No trained botanical models available on server.",
                "explanation_available": False,
                "debug_info": debug_log if debug else None
            }

    top_preds = species_res["top_predictions"]
    stability = species_res["stability_score"]
    top1 = top_preds[0]
    top1_raw_class = top1["name"]
    top1_conf = top1["confidence"]

    margin = DEFAULT_CALIBRATOR.compute_margin(top_preds)
    conf_tier, conf_desc = DEFAULT_CALIBRATOR.interpret_confidence(top1_conf, margin)

    # Step 5: Feature Embedding & Centroid Cosine Similarity
    target_model_entry = models.get(active_type) or models.get('legacy')
    embedding_sim = 0.50
    embedding_details = {}
    if target_model_entry:
        input_tensor = preprocess_for_inference(image, device=_DEVICE)
        embedding_sim, embedding_details = compute_embedding_similarity(
            target_model_entry['model'],
            target_model_entry['class_names'],
            active_type,
            input_tensor,
            top1_raw_class,
            device=_DEVICE
        )

    if debug:
        debug_log["6_model_class_count"] = len(target_model_entry['class_names'])
        debug_log["7_top_5_plant_predictions"] = [
            f"{p['name']}: {p['confidence']:.1%}" for p in top_preds[:5]
        ]
        debug_log["prediction_stability"] = f"{stability:.1%}"
        debug_log["margin"] = f"{margin:.1%}"
        debug_log["embedding_similarity"] = f"{embedding_sim:.3f}"

    # Step 6: Grad-CAM Explainability Heatmap
    explanation_available = False
    explanation_image = None
    if generate_explanation and target_model_entry:
        try:
            target_class_idx = target_model_entry['class_to_index'].get(top1_raw_class, 0)
            _, data_uri, _ = generate_gradcam_overlay(
                target_model_entry['model'],
                image,
                target_class_idx=target_class_idx,
                alpha=0.45,
                device=_DEVICE
            )
            explanation_available = True
            explanation_image = data_uri
        except Exception as e:
            if debug:
                debug_log["gradcam_error"] = str(e)

    # Step 7: Class Normalization & Database Lookup
    display_name, scientific_name, db_key = normalize_plant_name(top1_raw_class)
    if debug:
        debug_log["8_normalized_plant_name"] = display_name
        debug_log["9_database_lookup_key"] = db_key

    from backend.database import get_plant_info
    plant_info = get_plant_info(db_key)
    has_db_match = plant_info is not None
    is_medicinal = bool(plant_info and plant_info.get("is_medicinal", False))
    has_med_info = bool(is_medicinal and len(plant_info.get("medicinal_uses", [])) > 0) if plant_info else False

    if plant_info and plant_info.get("scientific_name"):
        scientific_name = plant_info.get("scientific_name")

    # Format top predictions with display and scientific names
    formatted_top_preds = []
    for pred in top_preds[:5]:
        d_name, s_name, _ = normalize_plant_name(pred["name"])
        formatted_top_preds.append({
            "name": d_name,
            "raw_class": pred["name"],
            "scientific_name": s_name,
            "confidence": pred["confidence"]
        })

    # Step 8: Multi-Signal Uncertainty & OOD Decision
    decision = DEFAULT_DECISION_ENGINE.evaluate(
        confidence=top1_conf,
        margin=margin,
        stability_score=stability,
        quality_score=quality.quality_score,
        quality_acceptable=quality.is_acceptable,
        embedding_similarity=embedding_sim,
        has_database_match=has_db_match,
        has_medicinal_info=has_med_info,
        plant_display_name=display_name,
        threshold_override=threshold
    )

    if debug:
        debug_log["10_database_match_result"] = "FOUND" if has_db_match else "NOT_FOUND"
        debug_log["11_final_confidence"] = f"{top1_conf:.1%}"
        debug_log["12_final_returned_status"] = decision.status
        debug_log["decision_reason"] = decision.decision_reason

    # Step 9: Assemble final standardized result
    base_response = {
        "success": True,
        "status": decision.status,
        "input_type": active_type,
        "input_type_confidence": float(round(type_confidence, 4)),
        "plant_name": display_name if decision.is_confident else None,
        "scientific_name": scientific_name if decision.is_confident else "",
        "confidence": float(round(top1_conf, 4)),
        "margin": float(round(margin, 4)),
        "confidence_tier": conf_tier,
        "confidence_interpretation": conf_desc,
        "prediction_stability": float(round(stability, 4)),
        "image_quality": float(round(quality.quality_score, 4)),
        "embedding_similarity": float(round(embedding_sim, 4)),
        "is_medicinal": is_medicinal if decision.is_confident else False,
        "medicinal_information_available": has_med_info if decision.is_confident else False,
        "database_match": has_db_match,
        "message": decision.user_message,
        "top_predictions": formatted_top_preds[:3],
        "explanation_available": explanation_available,
        "explanation_image": explanation_image if explanation_available else None,
        "debug_info": debug_log if debug else None
    }

    # Populate monograph fields if available
    if plant_info and decision.is_confident:
        base_response["description"] = plant_info.get("description", "")
        base_response["medicinal_uses"] = plant_info.get("medicinal_uses", [])
        base_response["traditional_uses"] = plant_info.get("traditional_uses", [])
        base_response["parts_used"] = plant_info.get("parts_used", [])
        base_response["precautions"] = plant_info.get("precautions", [])
        base_response["sources"] = plant_info.get("sources", [])
    else:
        base_response["description"] = "Plant recognized by botanical visual classifier." if decision.is_confident else ""
        base_response["medicinal_uses"] = []
        base_response["traditional_uses"] = []
        base_response["parts_used"] = []
        base_response["precautions"] = []
        base_response["sources"] = []

    return base_response
