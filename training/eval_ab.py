"""
training/eval_ab.py
===================
Quantitative A/B Benchmark Evaluation Script.
Directly compares Baseline Inference (Method A) vs. Upgraded Inference (Method B)
WITHOUT retraining any neural network weights.

Metrics Computed:
- Top-1 Accuracy (%)
- Top-3 Accuracy (%)
- OOD / Non-Plant Rejection Rate (%)
- Prediction Stability / Consistency (%)
- Average Latency per Image (ms)
- Detailed Comparative Performance Summary

Usage:
    python training/eval_ab.py [--samples 50] [--tta fast]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from config import ProjectConfig as cfg
from training.model_factory import create_model
from training.manifest_dataset import ManifestDataset

from inference.engine import predict_pipeline
from inference.preprocessing import (
    load_image_safely,
    preprocess_for_inference,
    IMAGE_SIZE,
    RESIZE_EDGE,
    MEAN,
    STD
)
from inference.quality import check_image_quality
from inference.tta import predict_with_tta
from inference.calibration import DEFAULT_CALIBRATOR


# ---------------------------------------------------------------------------
# Method A: Baseline Inference Implementation
# ---------------------------------------------------------------------------
def baseline_predict_image(
    model: nn.Module,
    class_names: List[str],
    image_path: Path,
    device: torch.device
) -> Dict[str, Any]:
    """
    Baseline Method A:
    - Raw center crop without quality checking
    - Single forward pass (no TTA)
    - Raw uncalibrated softmax (no temperature scaling)
    - Always returns argmax class (no OOD rejection)
    """
    t0 = time.perf_counter()
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(RESIZE_EDGE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        tensor = transform(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    latency_ms = (time.perf_counter() - t0) * 1000.0

    sorted_indices = probs.argsort()[::-1]
    top_preds = [
        {"name": class_names[idx], "confidence": float(probs[idx])}
        for idx in sorted_indices[:5]
    ]

    return {
        "status": "identified",
        "top_class": top_preds[0]["name"],
        "confidence": top_preds[0]["confidence"],
        "top_3_classes": [p["name"] for p in top_preds[:3]],
        "latency_ms": latency_ms
    }


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------
def run_ab_benchmark(max_samples: int = 50, tta_mode: str = "fast"):
    device = cfg.get_device()
    print("=" * 70)
    print("A/B INFERENCE BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Max Samples   : {max_samples} test specimens")
    print(f"TTA Mode      : {tta_mode}")
    print("=" * 70)

    # Load Leaf Model Checkpoint
    if not cfg.LEAF_MODEL_PATH.exists():
        print(f"[ERROR] Leaf model checkpoint missing: {cfg.LEAF_MODEL_PATH}")
        sys.exit(1)

    leaf_ckpt = torch.load(cfg.LEAF_MODEL_PATH, map_location=device, weights_only=False)
    leaf_classes = leaf_ckpt.get('class_names', [])
    num_classes = len(leaf_classes)
    leaf_c2i = leaf_ckpt.get('class_to_index', {c: i for i, c in enumerate(leaf_classes)})

    baseline_model = create_model('efficientnet_b0', num_classes=num_classes, pretrained=False)
    baseline_model.load_state_dict(leaf_ckpt['model_state_dict'])
    baseline_model = baseline_model.to(device)
    baseline_model.eval()

    # Load Test Dataset
    manifest_path = cfg.LEAF_MANIFEST_CSV
    if not manifest_path.exists():
        print(f"[ERROR] Test manifest missing: {manifest_path}")
        sys.exit(1)

    test_ds = ManifestDataset(manifest_path, split='test', class_to_idx=leaf_c2i)
    total_test = len(test_ds.samples)
    print(f"Found {total_test} total test samples. Subsampling {min(max_samples, total_test)} for benchmark...")

    # Sample test items
    indices = np.linspace(0, total_test - 1, min(max_samples, total_test), dtype=int)
    test_items = [test_ds.samples[i] for i in indices]

    # Metrics Accumulators
    metrics_a = {
        "top1_correct": 0,
        "top3_correct": 0,
        "confidences": [],
        "latencies": [],
        "rejected": 0
    }
    metrics_b = {
        "top1_correct": 0,
        "top3_correct": 0,
        "confidences": [],
        "latencies": [],
        "uncertain": 0,
        "stabilities": []
    }

    print("\nExecuting side-by-side evaluation on plant test set...")
    for idx, (fpath, true_lbl_idx) in enumerate(test_items, 1):
        true_class = leaf_classes[true_lbl_idx]
        img_p = Path(fpath)
        if not img_p.exists():
            continue

        # --- Method A (Baseline) ---
        res_a = baseline_predict_image(baseline_model, leaf_classes, img_p, device)
        a_pred = res_a["top_class"]
        a_top3 = res_a["top_3_classes"]
        metrics_a["confidences"].append(res_a["confidence"])
        metrics_a["latencies"].append(res_a["latency_ms"])
        if a_pred == true_class:
            metrics_a["top1_correct"] += 1
        if true_class in a_top3:
            metrics_a["top3_correct"] += 1

        # --- Method B (Upgraded) ---
        t0 = time.perf_counter()
        res_b = predict_pipeline(
            img_p,
            forced_type="leaf",
            tta_mode=tta_mode,
            generate_explanation=False,
            debug=False
        )
        latency_b = (time.perf_counter() - t0) * 1000.0

        top_preds_b = res_b.get("top_predictions", [])
        b_pred = top_preds_b[0]["raw_class"] if top_preds_b else ""
        b_top3 = [p["raw_class"] for p in top_preds_b[:3]]

        metrics_b["confidences"].append(res_b.get("confidence", 0.0))
        metrics_b["latencies"].append(latency_b)
        metrics_b["stabilities"].append(res_b.get("prediction_stability", 1.0))

        if res_b.get("status") in ["uncertain", "unknown", "bad_image"]:
            metrics_b["uncertain"] += 1

        if b_pred == true_class:
            metrics_b["top1_correct"] += 1
        if true_class in b_top3:
            metrics_b["top3_correct"] += 1

        if idx % 10 == 0 or idx == len(test_items):
            print(f"  Processed {idx}/{len(test_items)} items...", flush=True)

    # -----------------------------------------------------------------------
    # Non-Plant / OOD Challenge Test
    # -----------------------------------------------------------------------
    print("\nExecuting OOD / Non-Plant Rejection Challenge...")
    ood_images = []
    # 1. Textured synthetic non-plant
    ood1 = PROJECT_ROOT / "real_world_test" / "non_plant_object.jpg"
    if ood1.exists():
        ood_images.append(ood1)

    # 2. Blank gray noise
    ood2 = PROJECT_ROOT / "scratch" / "unsupported_non_plant.jpg"
    if not ood2.exists():
        ood2.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 200), (128, 128, 128)).save(ood2)
    ood_images.append(ood2)

    ood_rejected_a = 0
    ood_rejected_b = 0

    for ood_p in ood_images:
        res_a = baseline_predict_image(baseline_model, leaf_classes, ood_p, device)
        # Baseline always returns identified (cannot reject)
        if res_a["status"] != "identified":
            ood_rejected_a += 1

        res_b = predict_pipeline(ood_p, tta_mode="off", generate_explanation=False)
        if res_b["status"] in ["unknown", "bad_image"]:
            ood_rejected_b += 1

    # -----------------------------------------------------------------------
    # Summary Table Computation
    # -----------------------------------------------------------------------
    n = len(test_items)
    acc_top1_a = (metrics_a["top1_correct"] / n) * 100.0
    acc_top3_a = (metrics_a["top3_correct"] / n) * 100.0
    avg_conf_a = float(np.mean(metrics_a["confidences"])) * 100.0
    avg_lat_a = float(np.mean(metrics_a["latencies"]))

    acc_top1_b = (metrics_b["top1_correct"] / n) * 100.0
    acc_top3_b = (metrics_b["top3_correct"] / n) * 100.0
    avg_conf_b = float(np.mean(metrics_b["confidences"])) * 100.0
    avg_lat_b = float(np.mean(metrics_b["latencies"]))
    avg_stab_b = float(np.mean(metrics_b["stabilities"])) * 100.0

    ood_total = len(ood_images)
    ood_rate_a = (ood_rejected_a / ood_total) * 100.0 if ood_total else 0.0
    ood_rate_b = (ood_rejected_b / ood_total) * 100.0 if ood_total else 100.0

    print("\n" + "=" * 70)
    print("A/B BENCHMARK QUANTITATIVE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Performance Metric':<32} {'Method A (Baseline)':<20} {'Method B (Upgraded)':<20}")
    print("-" * 72)
    print(f"{'Top-1 Accuracy':<32} {acc_top1_a:>18.2f}% {acc_top1_b:>18.2f}%")
    print(f"{'Top-3 Accuracy':<32} {acc_top3_a:>18.2f}% {acc_top3_b:>18.2f}%")
    print(f"{'Average Prediction Confidence':<32} {avg_conf_a:>18.2f}% {avg_conf_b:>18.2f}% (Calibrated)")
    print(f"{'Prediction Stability (TTA)':<32} {'N/A':>19} {avg_stab_b:>18.2f}%")
    print(f"{'OOD/Non-Plant Rejection Rate':<32} {ood_rate_a:>18.2f}% {ood_rate_b:>18.2f}%")
    print(f"{'Average Latency per Specimen':<32} {avg_lat_a:>17.1f}ms {avg_lat_b:>17.1f}ms")
    print("=" * 70)

    # Save results to JSON
    out_dir = cfg.EVAL_DIR / "ab_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "samples_evaluated": n,
        "tta_mode": tta_mode,
        "baseline_method_a": {
            "top1_accuracy": round(acc_top1_a, 2),
            "top3_accuracy": round(acc_top3_a, 2),
            "avg_confidence": round(avg_conf_a, 2),
            "ood_rejection_rate": round(ood_rate_a, 2),
            "avg_latency_ms": round(avg_lat_a, 1)
        },
        "upgraded_method_b": {
            "top1_accuracy": round(acc_top1_b, 2),
            "top3_accuracy": round(acc_top3_b, 2),
            "avg_confidence": round(avg_conf_b, 2),
            "prediction_stability": round(avg_stab_b, 2),
            "ood_rejection_rate": round(ood_rate_b, 2),
            "avg_latency_ms": round(avg_lat_b, 1)
        }
    }
    report_file = out_dir / "ab_results.json"
    import json
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Benchmark report saved to: {report_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Benchmark Evaluator")
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples to evaluate")
    parser.add_argument("--tta", choices=["off", "fast", "standard"], default="fast", help="TTA mode for Method B")
    args = parser.parse_args()

    run_ab_benchmark(max_samples=args.samples, tta_mode=args.tta)
