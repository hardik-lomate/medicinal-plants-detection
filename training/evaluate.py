"""
training/evaluate.py
====================
Multi-Model Evaluation Suite:
1. Leaf Model on leaf_manifest.csv (test split)
2. Whole-Plant Model on whole_plant_manifest.csv (test split)
3. Image-Type Model on image_type_manifest.csv (test split)
4. Real-world challenging images (real_world_test/)

Computes:
- Top-1 and Top-3 accuracy
- Precision, Recall, F1 (macro & weighted)
- Per-class confusion matrix
- Saves results to models/evaluation_results/

Usage:
    python training/evaluate.py [--model leaf|whole_plant|image_type|all]
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import ProjectConfig as cfg
from training.model_factory import create_model
from training.manifest_dataset import ManifestDataset


def evaluate_manifest_model(model_path, manifest_path, title, output_dir, device):
    print(f"\n{'='*70}")
    print(f"EVALUATING: {title}")
    print(f"Model   : {model_path}")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*70}")

    model_path = Path(model_path)
    manifest_path = Path(manifest_path)

    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None
    if not manifest_path.exists():
        print(f"  [SKIP] Manifest not found: {manifest_path}")
        return None

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    backbone = ckpt.get('model_name', 'efficientnet_b0')
    class_names = ckpt.get('class_names', [])
    num_classes = len(class_names)
    class_to_idx = ckpt.get('class_to_index', {c: i for i, c in enumerate(class_names)})

    model = create_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_ds = ManifestDataset(manifest_path, split='test', class_to_idx=class_to_idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"  Test samples: {len(test_ds)} across {num_classes} classes")

    y_true = []
    y_pred = []
    y_scores = []
    top3_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            top1 = probs.argmax(dim=1).cpu().numpy()
            y_pred.extend(top1)
            y_true.extend(labels.numpy())
            y_scores.extend(probs.cpu().numpy())

            # Top-3
            _, top3_idx = probs.topk(min(3, num_classes), dim=1)
            for true_l, t3 in zip(labels, top3_idx.cpu()):
                if true_l.item() in t3:
                    top3_correct += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        print("  [WARNING] No test samples evaluated.")
        return None

    top1_acc = (y_pred == y_true).mean()
    top3_acc = top3_correct / len(y_true)

    # Per-class metrics
    per_class = {}
    for c_idx, c_name in enumerate(class_names):
        tp = int(((y_pred == c_idx) & (y_true == c_idx)).sum())
        fp = int(((y_pred == c_idx) & (y_true != c_idx)).sum())
        fn = int(((y_pred != c_idx) & (y_true == c_idx)).sum())
        total_actual = int((y_true == c_idx).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c_name] = {
            'samples': total_actual,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }

    macro_f1 = np.mean([v['f1'] for v in per_class.values()])
    weights = [v['samples'] for v in per_class.values()]
    weighted_f1 = np.average([v['f1'] for v in per_class.values()], weights=weights) if sum(weights) > 0 else 0.0

    print(f"\n  RESULTS: {title}")
    print(f"  Top-1 Accuracy : {top1_acc:.2%} ({int(top1_acc * len(y_true))}/{len(y_true)})")
    print(f"  Top-3 Accuracy : {top3_acc:.2%}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Weighted F1    : {weighted_f1:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'title': title,
        'model_path': str(model_path),
        'manifest_path': str(manifest_path),
        'evaluated_at': datetime.now().isoformat(),
        'num_samples': len(y_true),
        'top1_accuracy': round(float(top1_acc), 4),
        'top3_accuracy': round(float(top3_acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'weighted_f1': round(float(weighted_f1), 4),
        'per_class': per_class,
    }

    res_json = output_dir / "metrics.json"
    with open(res_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Saved metrics: {res_json}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['leaf', 'whole_plant', 'image_type', 'all'])
    args = parser.parse_args()

    device = cfg.get_device()
    print("=" * 70)
    print("MEDICINAL PLANT DETECTION — MODEL EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device   : {device}")
    print("=" * 70)

    cfg.ensure_dirs()

    if args.model in ['image_type', 'all']:
        evaluate_manifest_model(
            model_path=cfg.IMAGE_TYPE_MODEL_PATH,
            manifest_path=cfg.IMAGE_TYPE_MANIFEST_CSV,
            title="Image-Type Classifier (LEAF vs WHOLE_PLANT)",
            output_dir=cfg.EVAL_IMAGE_TYPE_DIR,
            device=device,
        )

    if args.model in ['whole_plant', 'all']:
        evaluate_manifest_model(
            model_path=cfg.PLANT_MODEL_PATH,
            manifest_path=cfg.PLANT_MANIFEST_CSV,
            title="Whole-Plant Classifier (40 Classes)",
            output_dir=cfg.EVAL_PLANT_DIR,
            device=device,
        )

    if args.model in ['leaf', 'all']:
        evaluate_manifest_model(
            model_path=cfg.LEAF_MODEL_PATH,
            manifest_path=cfg.LEAF_MANIFEST_CSV,
            title="Leaf Classifier (78 Classes)",
            output_dir=cfg.EVAL_LEAF_DIR,
            device=device,
        )

    print("\n" + "=" * 70)
    print("ALL REQUESTED EVALUATIONS COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
