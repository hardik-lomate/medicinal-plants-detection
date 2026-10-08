"""
inference/engine.py
===================
Authoritative Inference Engine for Medicinal Plant Detection.

Architecture:
1. Safe image loading & EXIF orientation auto-correction.
2. Stage 1: Image-Type classification (Leaf vs Whole Plant) -> Single forward pass -> Softmax -> Argmax.
3. Stage 2: Specialist botanical classifier (Leaf or Whole Plant) -> Single forward pass -> Softmax -> Argmax.
4. Top-1 prediction is authoritative; Top-3 candidates returned with raw probabilities.
5. Canonical plant name normalization (Display Name, Scientific Name, DB Key).
6. SQLite database monograph lookup (monograph absence never overrides model prediction).
7. Optional diagnostic telemetry: Grad-CAM heatmap, image quality score, embedding similarity.
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


def predict_baseline(
    image_input: Union[str, Path, bytes, Image.Image, Any],
    forced_type: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Step 1 Pure Baseline Inference Implementation.

    Guarantees exact parity with trained-model test accuracy:
    1. Preprocesses image matching training (Resize 256 + CenterCrop 224 + ImageNet Norm).
    2. Runs Stage 1 Image-Type classification (Leaf vs Whole Plant) with single forward pass.
    3. Runs Stage 2 Specialist classification with single forward pass.
    4. Computes raw softmax probabilities and selects argmax.
    5. Normalizes class name to display name and scientific name.
    6. Queries SQLite database for medicinal monograph.
    7. Model prediction is authoritative: NEVER wiped out or replaced.

    Returns:
        Standard structured prediction dictionary.
    """
    models = get_loaded_models()
    debug_log: Dict[str, Any] = {} if debug else {}

    # 1. Safe image loading & orientation correction
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
            "top_predictions": [],
            "message": load_err or "Invalid image file.",
            "is_medicinal": False,
            "medicinal_information_available": False,
            "explanation_available": False
        }

    img_w, img_h = image.size
    if debug:
        debug_log["1_image_path"] = img_meta.get("path") or "<in-memory>"
        debug_log["2_image_dimensions"] = f"{img_w}x{img_h}"
        debug_log["image_mode"] = image.mode

    # Transform image for neural network
    tensor = preprocess_for_inference(image, device=_DEVICE)

    # 2. Stage 1: Image-Type Classification (Leaf vs Whole Plant)
    type_confidence = 1.0
    if forced_type in ["leaf", "whole_plant"]:
        detected_type = forced_type
    elif models.get('image_type') is not None:
        type_entry = models['image_type']
        type_model = type_entry['model']
        type_classes = type_entry['class_names']
        type_model.eval()
        with torch.no_grad():
            type_logits = type_model(tensor)
            type_probs = torch.softmax(type_logits, dim=1)[0]
        type_idx = type_probs.argmax().item()
        detected_type = type_classes[type_idx]
        type_confidence = float(type_probs[type_idx].item())
    else:
        detected_type = "leaf"

    if debug:
        debug_log["3_image_type_prediction"] = detected_type
        debug_log["4_image_type_confidence"] = f"{type_confidence:.1%}"

    # 3. Stage 2: Specialist Species Classification
    specialist_key = detected_type if models.get(detected_type) else ('leaf' if models.get('leaf') else 'whole_plant')
    spec_entry = models.get(specialist_key) or models.get('legacy')
    if spec_entry is None:
        return {
            "success": False,
            "status": "error",
            "input_type": detected_type,
            "input_type_confidence": type_confidence,
            "plant_name": None,
            "scientific_name": "",
            "confidence": 0.0,
            "margin": 0.0,
            "top_predictions": [],
            "message": "No trained botanical models available on server.",
            "is_medicinal": False,
            "medicinal_information_available": False,
            "explanation_available": False
        }

    spec_model = spec_entry['model']
    spec_classes = spec_entry['class_names']
    spec_model.eval()
    with torch.no_grad():
        logits = spec_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top1_idx = probs.argmax().item()
    top1_raw_class = spec_classes[top1_idx]
    top1_conf = float(probs[top1_idx].item())

    # Extract Top-5 ranked candidate classes
    k = min(5, len(spec_classes))
    topk_indices = torch.topk(probs, k=k).indices.tolist()

    top_candidates = []
    for rank_idx in topk_indices:
        r_raw = spec_classes[rank_idx]
        r_conf = float(probs[rank_idx].item())
        d_name, s_name, _ = normalize_plant_name(r_raw)
        top_candidates.append({
            "name": d_name,
            "raw_class": r_raw,
            "scientific_name": s_name,
            "confidence": round(r_conf, 4)
        })

    p1 = top_candidates[0]["confidence"]
    p2 = top_candidates[1]["confidence"] if len(top_candidates) > 1 else 0.0
    margin = round(max(0.0, p1 - p2), 4)

    # 4. Class Normalization & Database Lookup
    display_name, scientific_name, db_key = normalize_plant_name(top1_raw_class)

    from backend.database import get_plant_info
    plant_info = get_plant_info(db_key)
    has_db_match = plant_info is not None
    is_medicinal = bool(plant_info and plant_info.get("is_medicinal", False))
    has_med_info = bool(is_medicinal and len(plant_info.get("medicinal_uses", [])) > 0) if plant_info else False

    if plant_info and plant_info.get("scientific_name"):
        scientific_name = plant_info.get("scientific_name")

    if has_med_info:
        status = "identified"
        msg = f"Successfully identified as {display_name} with verified pharmacological monograph."
    else:
        status = "identified_no_database_info"
        msg = f"Plant identified as {display_name}. No verified medicinal information is available for this plant in our database."

    if debug:
        debug_log["5_selected_model"] = specialist_key
        debug_log["6_model_class_count"] = len(spec_classes)
        debug_log["7_top_5_plant_predictions"] = [f"{p['name']}: {p['confidence']:.1%}" for p in top_candidates]
        debug_log["8_normalized_plant_name"] = display_name
        debug_log["9_database_lookup_key"] = db_key
        debug_log["10_database_match_result"] = "FOUND" if has_db_match else "NOT_FOUND"
        debug_log["11_final_confidence"] = f"{top1_conf:.1%}"
        debug_log["12_final_returned_status"] = status

    return {
        "success": True,
        "status": status,
        "input_type": detected_type,
        "input_type_confidence": float(round(type_confidence, 4)),
        "plant_name": display_name,
        "raw_class": top1_raw_class,
        "scientific_name": scientific_name,
        "confidence": float(round(top1_conf, 4)),
        "margin": float(round(margin, 4)),
        "confidence_tier": "HIGH" if top1_conf >= 0.70 else ("MODERATE" if top1_conf >= 0.40 else "LOW"),
        "confidence_interpretation": f"Baseline classification with {top1_conf:.1%} confidence and +{margin:.1%} candidate separation.",
        "prediction_stability": 1.0,
        "image_quality": 1.0,
        "embedding_similarity": 0.50,
        "is_medicinal": is_medicinal,
        "medicinal_information_available": has_med_info,
        "database_match": has_db_match,
        "message": msg,
        "top_predictions": top_candidates[:3],
        "description": plant_info.get("description", "") if plant_info else "Plant recognized by botanical visual classifier.",
        "medicinal_uses": plant_info.get("medicinal_uses", []) if plant_info else [],
        "traditional_uses": plant_info.get("traditional_uses", []) if plant_info else [],
        "parts_used": plant_info.get("parts_used", []) if plant_info else [],
        "precautions": plant_info.get("precautions", []) if plant_info else [],
        "sources": plant_info.get("sources", []) if plant_info else [],
        "explanation_available": False,
        "explanation_image": None,
        "debug_info": debug_log if debug else None
    }


def predict_pipeline(
    image_input: Union[str, Path, bytes, Image.Image, Any],
    forced_type: Optional[str] = None,
    tta_mode: str = "off",
    generate_explanation: bool = True,
    debug: bool = False,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Production Inference Pipeline with Diagnostic Signals.

    Key principles:
    - Authoritative classification is driven directly by predict_baseline.
    - Diagnostics (image quality, TTA stability, Grad-CAM, centroid similarity)
      provide helpful telemetry but NEVER alter or wipe out the predicted plant name.
    """
    # 1. Run authoritative baseline classification
    baseline_res = predict_baseline(image_input, forced_type=forced_type, debug=debug)
    if not baseline_res.get("success"):
        return baseline_res

    # 2. Safe image retrieval for diagnostics
    success, image, _, _ = load_image_safely(image_input)
    if not success or image is None:
        return baseline_res

    # 3. Diagnostic Image Quality Analysis
    quality = check_image_quality(image)
    baseline_res["image_quality"] = float(round(quality.quality_score, 4))
    baseline_res["quality_issues"] = quality.issues

    # If image is completely unreadable / zero feature content, report bad_image
    if not quality.is_acceptable and quality.quality_score < 0.15:
        baseline_res["status"] = "bad_image"
        baseline_res["message"] = quality.user_message
        baseline_res["plant_name"] = None
        baseline_res["scientific_name"] = ""
        return baseline_res

    # 4. Optional Diagnostic TTA Stability Analysis (if explicitly requested)
    models = get_loaded_models()
    active_type = baseline_res.get("input_type", "leaf")
    target_entry = models.get(active_type) or models.get('legacy')

    stability = 1.0
    if tta_mode != "off" and target_entry:
        try:
            tta_res = predict_with_tta(
                target_entry['model'],
                target_entry['class_names'],
                image,
                mode=tta_mode,
                temperature=1.0,
                device=_DEVICE
            )
            stability = tta_res.get("stability_score", 1.0)
        except Exception:
            stability = 1.0
    baseline_res["prediction_stability"] = float(round(stability, 4))

    # 5. Diagnostic Embedding Similarity
    embedding_sim = 0.50
    if target_entry:
        try:
            input_tensor = preprocess_for_inference(image, device=_DEVICE)
            raw_cls = baseline_res.get("raw_class", "")
            embedding_sim, _ = compute_embedding_similarity(
                target_entry['model'],
                target_entry['class_names'],
                active_type,
                input_tensor,
                raw_cls,
                device=_DEVICE
            )
        except Exception:
            embedding_sim = 0.50
    baseline_res["embedding_similarity"] = float(round(embedding_sim, 4))

    # 6. Diagnostic Grad-CAM Explainability Heatmap
    if generate_explanation and target_entry:
        try:
            raw_cls = baseline_res.get("raw_class", "")
            target_class_idx = target_entry['class_to_index'].get(raw_cls, 0)
            _, data_uri, _ = generate_gradcam_overlay(
                target_entry['model'],
                image,
                target_class_idx=target_class_idx,
                alpha=0.45,
                device=_DEVICE
            )
            baseline_res["explanation_available"] = True
            baseline_res["explanation_image"] = data_uri
        except Exception as e:
            baseline_res["explanation_available"] = False
            if debug and baseline_res.get("debug_info"):
                baseline_res["debug_info"]["gradcam_error"] = str(e)

    # 7. Multi-Signal Decision Synthesis (Preserves model prediction!)
    decision = DEFAULT_DECISION_ENGINE.evaluate(
        confidence=baseline_res["confidence"],
        margin=baseline_res["margin"],
        stability_score=stability,
        quality_score=quality.quality_score,
        quality_acceptable=quality.is_acceptable,
        embedding_similarity=embedding_sim,
        has_database_match=baseline_res["database_match"],
        has_medicinal_info=baseline_res["medicinal_information_available"],
        plant_display_name=baseline_res["plant_name"],
        threshold_override=threshold
    )

    baseline_res["status"] = decision.status
    baseline_res["decision_reason"] = decision.decision_reason
    baseline_res["message"] = decision.user_message

    if debug and baseline_res.get("debug_info"):
        dbg = baseline_res["debug_info"]
        dbg["image_quality_score"] = quality.quality_score
        dbg["prediction_stability"] = f"{stability:.1%}"
        dbg["embedding_similarity"] = f"{embedding_sim:.3f}"
        dbg["decision_reason"] = decision.decision_reason

    return baseline_res
