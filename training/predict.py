"""
training/predict.py
====================
Robust Individual Image Prediction CLI Script with Step 16 Debug Diagnostic Mode.

Supports:
    python training/predict.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg"
    python training/predict.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg" --debug
    python training/predict.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg" --baseline
    python training/predict.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg" --type leaf
"""

import sys
import os
import argparse
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from backend.model import predict_image, predict_baseline_image, clean_image_path


def main():
    parser = argparse.ArgumentParser(description="Medicinal Plant Detection Inference CLI")
    parser.add_argument("pos_image", nargs="?", default=None, help="Path to input plant image")
    parser.add_argument("--image", "-i", dest="opt_image", default=None, help="Explicit path to input plant image")
    parser.add_argument("--type", "-t", dest="input_type", choices=["leaf", "whole_plant", "auto"], default=None,
                        help="Optional override for plant type")
    parser.add_argument("--baseline", action="store_true", help="Force direct Step 1 predict_baseline evaluation")
    parser.add_argument("--tta", choices=["off", "fast", "standard"], default="off", help="TTA mode")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--debug", action="store_true", help="Enable Step 16 debug prediction mode")

    args = parser.parse_args()

    raw_path = args.opt_image or args.pos_image
    if not raw_path:
        print("[ERROR] No image specified.")
        print("Usage: python training/predict.py \"C:\\path\\to\\image.jpg\" [--debug]")
        sys.exit(1)

    # 1. Clean and validate image path
    is_valid, resolved_path, error_msg = clean_image_path(raw_path)

    # 2. Print initial pre-prediction inspection block
    print(f"\nImage path:       {resolved_path or raw_path}")
    print(f"Image exists:     {is_valid}")

    if not is_valid:
        print(f"[ERROR] {error_msg}")
        sys.exit(1)

    try:
        with Image.open(resolved_path) as img:
            img_format = img.format or os.path.splitext(resolved_path)[1].lstrip('.').upper()
            img_w, img_h = img.size
            print(f"Image format:     {img_format}")
            print(f"Image dimensions: {img_w}x{img_h}")
    except Exception as e:
        print(f"Image format:     Unknown")
        print(f"Image dimensions: Unknown ({e})")

    # 3. Run baseline prediction first
    forced_type = args.input_type if args.input_type in ["leaf", "whole_plant"] else None
    baseline_result = predict_baseline_image(resolved_path, forced_type=forced_type, debug=args.debug)

    if args.baseline:
        result = baseline_result
    else:
        from inference.engine import predict_pipeline
        result = predict_pipeline(
            resolved_path,
            forced_type=forced_type,
            tta_mode=args.tta,
            generate_explanation=True,
            debug=args.debug,
            threshold=args.threshold
        )

    # 4. STEP 16: Required Debug Diagnostic Mode
    if args.debug:
        top3_strs = [f"{p['name']} ({p['confidence']:.1%})" for p in result.get("top_predictions", [])[:3]]
        dbg = result.get("debug_info") or {}

        print(f"\n{'='*60}")
        print("DEBUG PREDICTION MODE (Step 16 Diagnostics)")
        print(f"{'='*60}")
        print(f"Image:                 {resolved_path}")
        print(f"Image type:            {result.get('input_type', 'N/A')}")
        print(f"Image quality:         {result.get('image_quality', 1.0):.2f}")
        print(f"Model used:            {dbg.get('5_selected_model', result.get('input_type', 'specialist'))}")
        print(f"Predicted class:       {result.get('plant_name') or 'N/A'}")
        print(f"Top-1 probability:     {result.get('confidence', 0.0):.1%}")
        print(f"Top-3 predictions:     {', '.join(top3_strs)}")
        print(f"Prediction margin:     {result.get('margin', 0.0):.1%}")
        print(f"TTA result if enabled: {args.tta} (Stability: {result.get('prediction_stability', 1.0):.1%})")
        print(f"Baseline result:       {baseline_result.get('plant_name')} ({baseline_result.get('confidence', 0.0):.1%})")
        print(f"Final result:          {result.get('plant_name')} ({result.get('confidence', 0.0):.1%})")
        print(f"{'='*60}")

    # 5. Format user-facing output
    status = result.get("status", "error")
    print(f"\n{'='*60}")
    print("PLANT IDENTIFICATION RESULT")
    print(f"{'='*60}")
    print(f"Status:             {status.upper()}")
    print(f"Input Type:         {result.get('input_type', 'N/A').upper()} ({result.get('input_type_confidence', 0.0):.1%})")
    print(f"Plant:              {result.get('plant_name') or 'N/A'}")
    print(f"Scientific Name:    {result.get('scientific_name') or 'N/A'}")
    print(f"Confidence:         {result.get('confidence', 0.0):.1%}")
    print(f"Separation Margin:  {result.get('margin', 0.0):.1%}")
    print(f"Medicinal Status:   {'Verified Monograph Available' if result.get('medicinal_information_available') else 'No Verified Monograph'}")
    print(f"Message:            {result.get('message', '')}")

    # Top Candidate Predictions
    print(f"\nTop Candidate Matches:")
    for i, pred in enumerate(result.get("top_predictions", []), 1):
        bar = "█" * int(pred["confidence"] * 30)
        print(f"  {i}. {pred['name']:<24} {pred['confidence']:>6.1%}  {bar}")

    # Detailed Monograph (if available)
    if result.get("medicinal_information_available"):
        print(f"\n{'─'*60}")
        print("VERIFIED MEDICINAL MONOGRAPH")
        print(f"{'─'*60}")
        if result.get("description"):
            print(f"Description:\n  {result['description']}\n")
        if result.get("medicinal_uses"):
            print("Evidence-Supported Medicinal Uses:")
            for u in result["medicinal_uses"]:
                print(f"  • {u}")
        if result.get("traditional_uses"):
            print("\nTraditional & Ayurvedic Applications:")
            for u in result["traditional_uses"]:
                print(f"  • {u}")
        if result.get("parts_used"):
            print(f"\nParts Utilized: {', '.join(result['parts_used'])}")
        if result.get("precautions"):
            print("\nPrecautions & Contraindications:")
            for p in result["precautions"]:
                print(f"  ⚠️ {p}")
        if result.get("sources"):
            print("\nAuthoritative References:")
            for s in result["sources"]:
                print(f"  📚 {s}")

    print("=" * 60)


if __name__ == "__main__":
    main()
