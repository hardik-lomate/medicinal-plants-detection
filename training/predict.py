"""
training/predict.py
====================
Robust Individual Image Prediction CLI Script.

Supports:
    python training/predict.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg"
    python training/predict.py --image "C:\\Users\\hardi\\Downloads\\tulsi.jpg" --debug
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

from backend.model import predict_image, clean_image_path


def main():
    parser = argparse.ArgumentParser(description="Medicinal Plant Detection Inference CLI")
    parser.add_argument("pos_image", nargs="?", default=None, help="Path to input plant image")
    parser.add_argument("--image", "-i", dest="opt_image", default=None, help="Explicit path to input plant image")
    parser.add_argument("--type", "-t", dest="input_type", choices=["leaf", "whole_plant", "auto"], default=None,
                        help="Optional override for plant type")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--debug", action="store_true", help="Enable 12-point pipeline diagnostic output")

    args = parser.parse_args()

    raw_path = args.opt_image or args.pos_image
    if not raw_path:
        print("[ERROR] No image specified.")
        print("Usage: python training/predict.py \"C:\\path\\to\\image.jpg\" [--debug]")
        sys.exit(1)

    # 1. Clean and validate image path
    is_valid, resolved_path, error_msg = clean_image_path(raw_path)

    # 2. Print required image pre-prediction inspection block
    print(f"\nImage path:       {resolved_path or raw_path}")
    print(f"Image exists:     {is_valid}")

    if not is_valid:
        print(f"[ERROR] {error_msg}")
        sys.exit(1)

    try:
        with Image.open(resolved_path) as img:
            img_format = img.format or os.path.splitext(resolved_path)[1].lstrip('.').upper()
            img_w, img_h = img.size
            print(f"Image type:       {img_format}")
            print(f"Image dimensions: {img_w}x{img_h}")
    except Exception as e:
        print(f"Image type:       Unknown")
        print(f"Image dimensions: Unknown ({e})")

    # 3. Run prediction pipeline
    forced_type = args.input_type if args.input_type in ["leaf", "whole_plant"] else None
    result = predict_image(resolved_path, forced_type=forced_type, debug=args.debug, threshold=args.threshold)

    # 4. If debug, display the 12 diagnostic points
    if args.debug and result.get("debug_info"):
        dbg = result["debug_info"]
        print(f"\n{'='*60}")
        print("PIPELINE DEBUG DIAGNOSTICS (12 Checkpoints)")
        print(f"{'='*60}")
        print(f"1. Image path:               {dbg.get('1_image_path', 'N/A')}")
        print(f"2. Image dimensions:         {dbg.get('2_image_dimensions', 'N/A')} ({dbg.get('image_mode', 'N/A')})")
        print(f"3. Image-type prediction:    {dbg.get('3_image_type_prediction', 'N/A')}")
        print(f"4. Image-type confidence:    {dbg.get('4_image_type_confidence', 'N/A')}")
        print(f"5. Selected model:           {dbg.get('5_selected_model', 'N/A')}")
        print(f"6. Model class count:        {dbg.get('6_model_class_count', 'N/A')}")
        print(f"7. Top-5 plant predictions:  {', '.join(dbg.get('7_top_5_plant_predictions', []))}")
        print(f"8. Normalized plant name:    {dbg.get('8_normalized_plant_name', 'N/A')}")
        print(f"9. Database lookup key:      {dbg.get('9_database_lookup_key', 'N/A')}")
        print(f"10. Database match result:   {dbg.get('10_database_match_result', 'N/A')}")
        print(f"11. Final confidence:        {dbg.get('11_final_confidence', 'N/A')}")
        print(f"12. Final returned status:   {dbg.get('12_final_returned_status', 'N/A')}")
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
