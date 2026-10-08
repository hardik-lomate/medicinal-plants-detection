"""
training/evaluate_baseline_inference.py
========================================
Definitive, empirical baseline evaluation script.
Evaluates pure single-pass forward inference (preprocess -> model -> softmax -> argmax)
on the exact test splits of the dataset manifests:
1. Image-Type classifier (leaf vs whole_plant) - image_type_manifest.csv
2. Whole-Plant classifier (40 classes)        - whole_plant_manifest.csv
3. Leaf classifier (78 classes)               - leaf_manifest.csv

Computes:
- Top-1 and Top-3 Accuracy
- Macro F1 and Weighted F1
- Total samples, correct, and incorrect counts
- Per-class accuracy and sample counts
- Common confusion pairs (for botanical error analysis)
- Saves full empirical audit to reports/baseline_evaluation.json

Usage:
    python training/evaluate_baseline_inference.py [--model all|leaf|whole_plant|image_type]
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps

from config import ProjectConfig as cfg
from training.model_factory import create_model
from training.manifest_dataset import ManifestDataset


def evaluate_single_model(
    model_name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    label_field: str = 'normalized_class',
    device: torch.device = None,
    batch_size: int = 32
) -> dict:
    if device is None:
        device = cfg.get_device()

    print(f"\n{'='*70}")
    print(f"EVALUATING BASELINE INFERENCE: {model_name.upper()}")
    print(f"Model Checkpoint : {checkpoint_path}")
    print(f"Manifest Path    : {manifest_path}")
    print(f"Device           : {device}")
    print(f"{'='*70}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone = ckpt.get('model_name', 'efficientnet_b0')
    class_names = ckpt.get('class_names', [])
    num_classes = len(class_names)
    class_to_idx = ckpt.get('class_to_index', {c: i for i, c in enumerate(class_names)})

    # Instantiate model and load state
    model = create_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Build dataset matching training/eval pipeline
    test_ds = ManifestDataset(
        manifest_path=manifest_path,
        split='test',
        class_to_idx=class_to_idx,
        label_field=label_field
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    total_samples = len(test_ds)
    print(f"  Test specimens : {total_samples} across {num_classes} classes")

    y_true = []
    y_pred = []
    y_scores = []
    top3_correct = 0
    confusion_pairs = Counter()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            top3 = torch.topk(probs, k=min(3, num_classes), dim=1).indices

            for i in range(len(labels)):
                gt = labels[i].item()
                pr = preds[i].item()
                y_true.append(gt)
                y_pred.append(pr)
                y_scores.append(probs[i, pr].item())

                if gt in top3[i].tolist():
                    top3_correct += 1

                if gt != pr:
                    gt_name = class_names[gt]
                    pr_name = class_names[pr]
                    confusion_pairs[(gt_name, pr_name)] += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_count = int(np.sum(y_true == y_pred))
    incorrect_count = total_samples - correct_count
    top1_accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    top3_accuracy = top3_correct / total_samples if total_samples > 0 else 0.0

    # Per-class metrics
    per_class_stats = {}
    precisions = []
    recalls = []
    f1s = []

    for c_idx, c_name in enumerate(class_names):
        tp = int(np.sum((y_pred == c_idx) & (y_true == c_idx)))
        fp = int(np.sum((y_pred == c_idx) & (y_true != c_idx)))
        fn = int(np.sum((y_pred != c_idx) & (y_true == c_idx)))
        support = int(np.sum(y_true == c_idx))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        per_class_stats[c_name] = {
            "samples": support,
            "correct": tp,
            "accuracy": round(tp / support, 4) if support > 0 else 0.0,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Weighted F1
    total_support = sum(per_class_stats[c]["samples"] for c in class_names)
    weighted_f1 = (
        float(sum(per_class_stats[c]["f1"] * per_class_stats[c]["samples"] for c in class_names) / total_support)
        if total_support > 0 else 0.0
    )

    # Top confused pairs
    most_common_confusions = [
        {"true_class": pair[0], "predicted_class": pair[1], "count": count}
        for pair, count in confusion_pairs.most_common(10)
    ]

    print(f"\nRESULTS FOR {model_name.upper()}:")
    print(f"  Total Samples      : {total_samples}")
    print(f"  Correct (Top-1)    : {correct_count} ({top1_accuracy:.2%})")
    print(f"  Incorrect          : {incorrect_count}")
    print(f"  Top-3 Accuracy     : {top3_correct}/{total_samples} ({top3_accuracy:.2%})")
    print(f"  Macro F1           : {macro_f1:.4f}")
    print(f"  Weighted F1        : {weighted_f1:.4f}")
    if most_common_confusions:
        print(f"  Top Confusion Pairs: {most_common_confusions[:3]}")

    return {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "num_classes": num_classes,
        "total_samples": total_samples,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "top1_accuracy": round(top1_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top_confusion_pairs": most_common_confusions,
        "per_class": per_class_stats,
        "evaluated_at": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Inference Evaluator")
    parser.add_argument("--model", choices=["all", "image_type", "whole_plant", "leaf"], default="all")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    models_to_run = (
        ["image_type", "whole_plant", "leaf"] if args.model == "all" else [args.model]
    )

    model_configs = {
        "image_type": {
            "checkpoint": cfg.IMAGE_TYPE_MODEL_PATH,
            "manifest": cfg.IMAGE_TYPE_MANIFEST_CSV,
            "label_field": "class_name",
        },
        "whole_plant": {
            "checkpoint": cfg.PLANT_MODEL_PATH,
            "manifest": cfg.PLANT_MANIFEST_CSV,
            "label_field": "normalized_class",
        },
        "leaf": {
            "checkpoint": cfg.LEAF_MODEL_PATH,
            "manifest": cfg.LEAF_MANIFEST_CSV,
            "label_field": "normalized_class",
        },
    }

    for m in models_to_run:
        conf = model_configs[m]
        res = evaluate_single_model(
            model_name=m,
            checkpoint_path=conf["checkpoint"],
            manifest_path=conf["manifest"],
            label_field=conf["label_field"],
            batch_size=args.batch_size
        )
        results[m] = res

    # Save summary report
    out_file = reports_dir / "baseline_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Baseline evaluation report saved to: {out_file}")


if __name__ == "__main__":
    main()
