"""
training/generate_inference_comparison.py
=========================================
Generates the quantitative A/B report comparing:
- Baseline Inference (Method A: single-pass forward pass, uncalibrated argmax)
VS
- Modified / Gated Inference (Method B: temperature scaling + confidence/margin rejection gating)

Outputs:
- reports/inference_comparison.json
- reports/inference_comparison.csv
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import ProjectConfig as cfg
from training.model_factory import create_model
from training.manifest_dataset import ManifestDataset
from inference.ood import UncertaintyDecisionEngine


def evaluate_comparison_for_model(
    model_name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    label_field: str = 'normalized_class',
    temperature: float = 1.25,
    max_eval_samples: int = 500
) -> dict:
    device = torch.device('cpu')
    print(f"\n--- Comparing {model_name.upper()} (up to {max_eval_samples} samples) ---")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt['class_names']
    num_classes = len(class_names)
    c2i = ckpt['class_to_index']

    model = create_model('efficientnet_b0', num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ds = ManifestDataset(manifest_path, split='test', class_to_idx=c2i, label_field=label_field)
    total_available = len(ds.samples)
    eval_count = min(max_eval_samples, total_available)
    step = max(1, total_available // eval_count)
    indices = list(range(0, total_available, step))[:eval_count]

    # Pre-gating engine representing the regression logic
    gating_engine = UncertaintyDecisionEngine(
        min_prediction_confidence=0.22,
        min_margin_threshold=0.06,
        min_stability_threshold=0.40,
        min_embedding_similarity=0.30
    )

    base_correct = 0
    mod_correct = 0
    base_top3_correct = 0
    mod_top3_correct = 0

    regressions = 0     # Was correct in baseline, became incorrect / rejected in modified
    improvements = 0    # Was incorrect in baseline, became correct in modified
    changed_preds = 0

    y_true = []
    base_preds = []
    mod_preds = []

    for i, idx in enumerate(indices):
        fpath, gt = ds.samples[idx]
        img, _ = ds[idx]
        tensor = img.unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs_base = torch.softmax(logits, dim=1)[0]
            probs_mod = torch.softmax(logits / temperature, dim=1)[0]

        # Method A: Baseline
        base_pred = probs_base.argmax().item()
        base_top3 = torch.topk(probs_base, k=min(3, num_classes)).indices.tolist()

        # Method B: Modified with gating
        mod_pred_idx = probs_mod.argmax().item()
        mod_conf = probs_mod[mod_pred_idx].item()
        mod_top3 = torch.topk(probs_mod, k=min(3, num_classes)).indices.tolist()
        p2 = probs_mod[mod_top3[1]].item() if len(mod_top3) > 1 else 0.0
        mod_margin = mod_conf - p2

        # Simulate strict gating wipe
        dec = gating_engine.evaluate(
            confidence=mod_conf,
            margin=mod_margin,
            stability_score=0.85,
            quality_score=0.85,
            quality_acceptable=True,
            embedding_similarity=0.35,
            has_database_match=True,
            has_medicinal_info=True,
            plant_display_name="specimen"
        )

        mod_pred = mod_pred_idx if dec.is_confident else -1  # -1 represents rejected / None

        y_true.append(gt)
        base_preds.append(base_pred)
        mod_preds.append(mod_pred)

        is_base_correct = (base_pred == gt)
        is_mod_correct = (mod_pred == gt)

        if is_base_correct:
            base_correct += 1
        if gt in base_top3:
            base_top3_correct += 1

        if is_mod_correct:
            mod_correct += 1
        if mod_pred != -1 and gt in mod_top3:
            mod_top3_correct += 1

        if base_pred != mod_pred:
            changed_preds += 1

        if is_base_correct and not is_mod_correct:
            regressions += 1
        elif not is_base_correct and is_mod_correct:
            improvements += 1

    n = len(indices)
    base_acc = base_correct / n
    mod_acc = mod_correct / n
    diff = mod_acc - base_acc

    base_top3_acc = base_top3_correct / n
    mod_top3_acc = mod_top3_correct / n

    # Compute F1
    base_f1s = []
    mod_f1s = []
    y_true_arr = np.array(y_true)
    base_arr = np.array(base_preds)
    mod_arr = np.array(mod_preds)

    for c in range(num_classes):
        tp_b = np.sum((base_arr == c) & (y_true_arr == c))
        fp_b = np.sum((base_arr == c) & (y_true_arr != c))
        fn_b = np.sum((base_arr != c) & (y_true_arr == c))
        f1_b = 2 * tp_b / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
        base_f1s.append(f1_b)

        tp_m = np.sum((mod_arr == c) & (y_true_arr == c))
        fp_m = np.sum((mod_arr == c) & (y_true_arr != c))
        fn_m = np.sum((mod_arr != c) & (y_true_arr == c))
        f1_m = 2 * tp_m / (2 * tp_m + fp_m + fn_m) if (2 * tp_m + fp_m + fn_m) > 0 else 0.0
        mod_f1s.append(f1_m)

    base_macro_f1 = float(np.mean(base_f1s))
    mod_macro_f1 = float(np.mean(mod_f1s))

    print(f"  Samples Evaluated   : {n}")
    print(f"  Baseline Accuracy   : {base_acc:.2%}")
    print(f"  Modified Accuracy   : {mod_acc:.2%}")
    print(f"  Difference          : {diff:+.2%}")
    print(f"  Regressions (Lost)  : {regressions}")
    print(f"  Improvements (Gained): {improvements}")

    return {
        "model": model_name,
        "samples_evaluated": n,
        "baseline_accuracy": round(base_acc, 4),
        "modified_accuracy": round(mod_acc, 4),
        "difference": round(diff, 4),
        "baseline_macro_f1": round(base_macro_f1, 4),
        "modified_macro_f1": round(mod_macro_f1, 4),
        "baseline_top3": round(base_top3_acc, 4),
        "modified_top3": round(mod_top3_acc, 4),
        "number_of_changed_predictions": changed_preds,
        "number_of_regressions": regressions,
        "number_of_improvements": improvements,
        "verdict": "REVERT_TO_BASELINE" if diff < 0 else "RETAIN_MODIFICATION"
    }


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparisons = {}

    configs = [
        ("image_type", cfg.IMAGE_TYPE_MODEL_PATH, cfg.IMAGE_TYPE_MANIFEST_CSV, "class_name", 1.10, 200),
        ("whole_plant", cfg.PLANT_MODEL_PATH, cfg.PLANT_MANIFEST_CSV, "normalized_class", 1.20, 200),
        ("leaf", cfg.LEAF_MODEL_PATH, cfg.LEAF_MANIFEST_CSV, "normalized_class", 1.25, 200),
    ]

    for name, ckpt, man, lbl_field, temp, cnt in configs:
        res = evaluate_comparison_for_model(
            model_name=name,
            checkpoint_path=ckpt,
            manifest_path=man,
            label_field=lbl_field,
            temperature=temp,
            max_eval_samples=cnt
        )
        comparisons[name] = res

    # Save JSON report
    json_path = reports_dir / "inference_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)
    print(f"\n[OK] Saved A/B comparison report: {json_path}")

    # Save CSV report
    csv_path = reports_dir / "inference_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "samples_evaluated", "baseline_accuracy", "modified_accuracy",
            "difference", "baseline_macro_f1", "modified_macro_f1",
            "baseline_top3", "modified_top3", "number_of_changed_predictions",
            "number_of_regressions", "number_of_improvements", "verdict"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparisons.values():
            writer.writerow(row)
    print(f"[OK] Saved A/B comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
