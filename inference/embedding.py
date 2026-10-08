"""
inference/embedding.py
======================
Feature Embedding Extraction & Centroid Cosine Similarity Engine.

Extracts 1280-dimensional penultimate representations from the trained EfficientNet-B0
backbone (prior to the linear classification head) and evaluates cosine similarity
against learned class centroids.

Serves as a secondary, non-destructive verification signal for:
- Detecting out-of-distribution / non-plant inputs
- Verifying whether a predicted class genuinely shares feature manifold geometry
- Distinguishing ambiguous leaf morphologies
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg
from inference.preprocessing import preprocess_for_inference

# In-memory centroid cache
_CENTROIDS_CACHE: Dict[str, torch.Tensor] = {}


def extract_feature_embedding(
    model: nn.Module,
    tensor: torch.Tensor,
    normalize: bool = True
) -> np.ndarray:
    """
    Extracts the 1280-dimensional feature embedding from EfficientNet-B0.

    Args:
        model: Trained EfficientNet-B0 model.
        tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        normalize: If True, L2-normalizes the vector to unit sphere.

    Returns:
        1D numpy array of shape (1280,).
    """
    model.eval()
    with torch.no_grad():
        features = model.features(tensor)
        pooled = model.avgpool(features)
        embedding = torch.flatten(pooled, 1)[0]

        if normalize:
            norm = torch.norm(embedding, p=2)
            embedding = embedding / max(1e-8, norm.item())

        return embedding.cpu().numpy()


def get_class_centroids(
    model: nn.Module,
    class_names: List[str],
    model_type: str,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Retrieves or constructs L2-normalized class centroids in feature space (1280-D).
    First checks if a precomputed centroid tensor exists on disk.
    If not, derives prototype vectors directly from the trained linear classifier weights:
        W = classifier.1.1.weight (shape: num_classes, 1280)
    which represent the optimal linear prototype directions learned during training.

    Returns:
        torch.Tensor of shape (num_classes, 1280), unit-normalized.
    """
    global _CENTROIDS_CACHE
    if model_type in _CENTROIDS_CACHE:
        return _CENTROIDS_CACHE[model_type]

    target_device = device if device is not None else next(model.parameters()).device
    centroid_file = cfg.CLASS_MAPPING_DIR / f"centroids_{model_type}.pt"

    if centroid_file.exists():
        try:
            centroids = torch.load(centroid_file, map_location=target_device, weights_only=False)
            if centroids.shape == (len(class_names), 1280):
                _CENTROIDS_CACHE[model_type] = centroids
                return centroids
        except Exception:
            pass

    # Derive prototype directions directly from the classifier head weights
    # In EfficientNet create_model: model.classifier[1] is nn.Sequential(Dropout, Linear(1280, num_classes))
    linear_layer = None
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            linear_layer = module
            break

    if linear_layer is not None:
        weights = linear_layer.weight.data.clone().to(target_device)  # shape (num_classes, 1280)
        norms = torch.norm(weights, p=2, dim=1, keepdim=True)
        centroids = weights / torch.clamp(norms, min=1e-8)
    else:
        # Fallback to random unit vectors if linear layer cannot be inspected
        centroids = torch.randn(len(class_names), 1280, device=target_device)
        centroids = centroids / torch.norm(centroids, p=2, dim=1, keepdim=True)

    _CENTROIDS_CACHE[model_type] = centroids

    # Save to disk for fast caching
    try:
        os.makedirs(cfg.CLASS_MAPPING_DIR, exist_ok=True)
        torch.save(centroids, centroid_file)
    except Exception:
        pass

    return centroids


def compute_embedding_similarity(
    model: nn.Module,
    class_names: List[str],
    model_type: str,
    input_tensor: torch.Tensor,
    predicted_class: str,
    device: Optional[torch.device] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes cosine similarity between query image embedding and candidate class centroids.

    Returns:
        (top1_similarity: float, details: dict)
    """
    target_device = device if device is not None else next(model.parameters()).device
    query_emb = extract_feature_embedding(model, input_tensor, normalize=True)
    query_tensor = torch.tensor(query_emb, device=target_device).unsqueeze(0)  # (1, 1280)

    centroids = get_class_centroids(model, class_names, model_type, device=target_device)  # (C, 1280)

    # Cosine similarities to all classes: (1, 1280) x (1280, C) -> (1, C)
    similarities = torch.mm(query_tensor, centroids.t())[0].cpu().numpy()

    top_idx = class_names.index(predicted_class) if predicted_class in class_names else int(np.argmax(similarities))
    top1_sim = float(round(float(similarities[top_idx]), 4))

    max_sim_idx = int(np.argmax(similarities))
    max_sim_class = class_names[max_sim_idx]
    max_sim = float(round(float(similarities[max_sim_idx]), 4))

    return top1_sim, {
        "target_class_similarity": top1_sim,
        "max_similarity_class": max_sim_class,
        "max_similarity_value": max_sim,
        "embedding_norm": float(round(float(np.linalg.norm(query_emb)), 4)),
        "is_aligned": bool(top1_sim >= 0.40)
    }
