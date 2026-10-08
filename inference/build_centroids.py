"""
inference/build_centroids.py
============================
Precomputes empirical class feature centroids for the Leaf (78 classes)
and Whole-Plant (40 classes) EfficientNet-B0 classifiers without retraining.

Uses the validation split of each dataset to compute true class prototype vectors
in the 1280-dimensional feature space, saving them to:
  models/class_mappings/centroids_leaf.pt
  models/class_mappings/centroids_whole_plant.pt
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg
from training.model_factory import create_model
from training.manifest_dataset import ManifestDataset
from inference.embedding import extract_feature_embedding


def compute_model_centroids(
    model_path: Path,
    manifest_path: Path,
    output_path: Path,
    device: torch.device
):
    print(f"\nComputing centroids for: {model_path.name}")
    if not model_path.exists() or not manifest_path.exists():
        print(f"[SKIP] Model or manifest missing ({model_path}, {manifest_path})")
        return

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    class_names = ckpt.get('class_names', [])
    class_to_idx = ckpt.get('class_to_index', {c: i for i, c in enumerate(class_names)})
    num_classes = len(class_names)

    model = create_model('efficientnet_b0', num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Use validation split for rapid, clean prototype extraction
    ds = ManifestDataset(manifest_path, split='val', class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    class_accum = defaultdict(list)
    total_processed = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.features(images)
            pooled = model.avgpool(features)
            embeddings = torch.flatten(pooled, 1)  # (B, 1280)
            norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            norm_embs = (embeddings / torch.clamp(norms, min=1e-8)).cpu().numpy()

            for emb, lbl in zip(norm_embs, labels.numpy()):
                class_accum[int(lbl)].append(emb)
                total_processed += 1

    print(f"Processed {total_processed} images across {len(class_accum)} classes.")

    # Compute unit-normalized centroid vector for each class
    centroid_matrix = np.zeros((num_classes, 1280), dtype=np.float32)
    linear_weights = None
    for m in model.classifier.modules():
        if isinstance(m, torch.nn.Linear):
            linear_weights = m.weight.data.clone().cpu().numpy()
            break

    for c in range(num_classes):
        if c in class_accum and len(class_accum[c]) > 0:
            mean_vec = np.mean(class_accum[c], axis=0)
            norm = np.linalg.norm(mean_vec)
            centroid_matrix[c] = mean_vec / max(1e-8, norm)
        elif linear_weights is not None:
            # Fallback to normalized linear weight for classes with 0 val samples
            w = linear_weights[c]
            centroid_matrix[c] = w / max(1e-8, np.linalg.norm(w))
        else:
            centroid_matrix[c] = np.random.randn(1280).astype(np.float32)
            centroid_matrix[c] /= np.linalg.norm(centroid_matrix[c])

    tensor_centroids = torch.tensor(centroid_matrix, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_centroids, output_path)
    print(f"[OK] Centroids saved to: {output_path} (shape: {tuple(tensor_centroids.shape)})")


def main():
    device = cfg.get_device()
    print("=" * 60)
    print("BUILDING DEEP FEATURE CENTROIDS FOR PROTOTYPE VERIFICATION")
    print("=" * 60)

    # 1. Leaf Model Centroids
    compute_model_centroids(
        model_path=cfg.LEAF_MODEL_PATH,
        manifest_path=cfg.LEAF_MANIFEST_CSV,
        output_path=cfg.CLASS_MAPPING_DIR / "centroids_leaf.pt",
        device=device
    )

    # 2. Whole Plant Model Centroids
    compute_model_centroids(
        model_path=cfg.PLANT_MODEL_PATH,
        manifest_path=cfg.PLANT_MANIFEST_CSV,
        output_path=cfg.CLASS_MAPPING_DIR / "centroids_whole_plant.pt",
        device=device
    )
    print("\nCentroid generation complete.")


if __name__ == "__main__":
    main()
