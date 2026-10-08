"""
training/test_real_world.py
============================
Dedicated command-line real-world plant test script with comprehensive debug inspection.

Usage:
    python training/test_real_world.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg"
    python training/test_real_world.py "C:\\Users\\hardi\\Downloads\\tulsi.jpg" --debug
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

from backend.model import clean_image_path
from inference.engine import predict_pipeline


def main():
    parser = argparse.ArgumentParser(description="Real-World Plant Prediction Test Runner")
    parser.add_argument("image", nargs="?", default=None, help="Path to input plant image")
    parser.add_argument("--image", "-i", dest="opt_image", default=None, help="Explicit path to input plant image")
    parser.add_argument("--type", "-t", dest="input_type", choices=["leaf", "whole_plant", "auto"], default=None,
                        help="Optional override for plant type")
    parser.add_argument("--tta", choices=["off", "fast", "standard", "thorough"], default="fast",
                        help="Test-time augmentation mode")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--debug", action="store_true", help="Enable comprehensive pipeline diagnostic output")

    args = parser.parse_args()

    raw_path = args.opt_image or args.image
    if not raw_path:
        print("[ERROR] No image specified.")
        print("Usage: python training/test_real_world.py \"C:\\path\\to\\image.jpg\" [--debug]")
        sys.exit(1)

    is_valid, resolved_path, error_msg = clean_image_path(raw_path)
    if not is_valid:
        print(f"[ERROR] {error_msg}")
        sys.exit(1)

    forced_type = args.input_type if args.input_type in ["leaf", "whole_plant"] else None
    result = predict_pipeline(
        resolved_path,
        forced_type=forced_type,
        tta_mode=args.tta,
        generate_explanation=args.debug,
        debug=args.debug,
        threshold=args.threshold
    )

    if args.debug and result.get("debug_info"):
        dbg = result["debug_info"]
        print(f"\n{'='*65}")
        print("PIPELINE DEBUG DIAGNOSTICS (Inference Telemetry)")
        print(f"{'='*65}")
        print(f"1. Image path:                 {dbg.get('1_image_path', resolved_path)}")
        print(f"2. Image dimensions:           {dbg.get('2_image_dimensions', 'N/A')} ({dbg.get('image_mode', 'RGB')})")
        print(f"3. Image quality score:        {result.get('image_quality', 0.0):.2f}/1.00 (Sharpness: {dbg.get('sharpness', 'N/A')}, Entropy: {dbg.get('entropy', 'N/A')})")
        print(f"4. Detected image type:        {dbg.get('3_image_type_prediction', 'N/A')}")
        print(f"5. Image-type confidence:      {dbg.get('4_image_type_confidence', 'N/A')}")
        print(f"6. Selected classifier:        {dbg.get('5_selected_model', 'N/A')} ({dbg.get('6_model_class_count', 0)} classes)")
        print(f"7. Preprocessing info:         Resize(256) -> CenterCrop(224) -> ImageNet Norm [EXIF auto-transposed]")
        print(f"8. TTA mode & stability:       mode={args.tta}, stability={result.get('prediction_stability', 1.0):.1%}")
        print(f"9. Top-5 plant predictions:    {', '.join(dbg.get('7_top_5_plant_predictions', []))}")
        print(f"10. Top-1 / Top-2 margin:      {result.get('margin', 0.0):.1%} ({result.get('confidence_tier', 'N/A')})")
        print(f"11. Embedding similarity:      {result.get('embedding_similarity', 0.0):.3f} (Class prototype alignment)")
        print(f"12. Normalized plant name:     {dbg.get('8_normalized_plant_name', 'N/A')}")
        print(f"13. Database lookup key:       {dbg.get('9_database_lookup_key', 'N/A')}")
        print(f"14. Database match status:     {dbg.get('10_database_match_result', 'N/A')}")
        print(f"15. Medicinal monograph info:  {'AVAILABLE' if result.get('medicinal_information_available') else 'NOT AVAILABLE'}")
        print(f"16. Final confidence:          {result.get('confidence', 0.0):.1%}")
        print(f"17. Final returned status:     {result.get('status', 'N/A')}")
        print(f"18. Decision reason:           {dbg.get('decision_reason', 'N/A')}")
        gradcam_status = "Generated (Base64 JPEG encoded)" if result.get("explanation_available") else "Not generated"
        print(f"19. Grad-CAM visualizer:       {gradcam_status}")
        print(f"{'='*65}\n")

    status = result.get("status", "error")
    input_type = result.get("input_type", "unknown").upper()
    type_conf = result.get("input_type_confidence", 0.0)
    plant_name = result.get("plant_name")
    sci_name = result.get("scientific_name", "")
    plant_conf = result.get("confidence", 0.0)
    has_med = result.get("medicinal_information_available", False)

    print("=" * 55)
    print("REAL-WORLD PLANT PREDICTION")
    print("=" * 55)
    print(f"\nImage:\n{resolved_path}")
    print(f"\nImage type:\n{input_type}")
    print(f"\nImage-type confidence:\n{type_conf:.1%}")

    if status == "uncertain":
        print(f"\nPlant:\nUncertain")
        print(f"\nPlant confidence:\n{plant_conf:.1%}")
        print(f"\nPrediction stability:\n{result.get('prediction_stability', 0.0):.1%}")
        print(f"\nMessage:\n{result.get('message')}")
        print("\nTop Candidate Matches:")
        for i, pred in enumerate(result.get("top_predictions", []), 1):
            print(f"  #{i} {pred['name']} ({pred['confidence']:.1%})")
        print("\n" + "=" * 55)
        return

    if status == "unknown":
        print(f"\nStatus:\nUNKNOWN (Out-Of-Distribution / Non-Plant Object)")
        print(f"\nMessage:\n{result.get('message')}")
        if result.get("top_predictions"):
            print("\nNearest Specimen Prototypes:")
            for i, pred in enumerate(result["top_predictions"], 1):
                print(f"  #{i} {pred['name']} ({pred['confidence']:.1%})")
        print("\n" + "=" * 55)
        return

    if status == "bad_image":
        print(f"\nStatus:\nBAD_IMAGE (Quality Check Failed)")
        print(f"\nQuality Score:\n{result.get('image_quality', 0.0):.2f}/1.00")
        print(f"\nMessage:\n{result.get('message')}")
        print("\n" + "=" * 55)
        return

    print(f"\nPlant:\n{plant_name}")
    print(f"\nScientific name:\n{sci_name if sci_name else 'N/A'}")
    print(f"\nPlant confidence:\n{plant_conf:.1%}")
    print(f"\nConfidence tier:\n{result.get('confidence_tier', 'N/A')}")
    print(f"\nSeparation margin:\n+{result.get('margin', 0.0):.1%}")
    print(f"\nPrediction stability:\n{result.get('prediction_stability', 1.0):.1%}")

    if status == "identified" and has_med:
        print("\nDatabase:\nFOUND")
        print("\nMedicinal information:\nAVAILABLE")

        if result.get("medicinal_uses"):
            print("\nMedicinal uses:")
            for u in result["medicinal_uses"]:
                print(f"• {u}")

        if result.get("traditional_uses"):
            print("\nTraditional uses:")
            for u in result["traditional_uses"]:
                print(f"• {u}")

        if result.get("parts_used"):
            print(f"\nParts used:\n{', '.join(result['parts_used'])}")

        if result.get("precautions"):
            print("\nPrecautions:")
            for p in result["precautions"]:
                print(f"⚠️ {p}")

        if result.get("sources"):
            print("\nSources:")
            for s in result["sources"]:
                print(f"📚 {s}")

    elif status == "identified_no_database_info":
        db_status = "FOUND (Botanical Only)" if result.get("description") and "no verified" in result.get("description", "").lower() else "NOT FOUND"
        print(f"\nDatabase:\n{db_status}")
        print("\nMedicinal information:\nNOT AVAILABLE")
        print(f"\nMessage:\n{result.get('message')}")

    else:
        print(f"\nDatabase:\nNOT FOUND")
        print(f"\nMessage:\n{result.get('message', 'Plant identified, but detailed information is not available in the current database.')}")

    if result.get("top_predictions"):
        print("\nTop Candidate Matches:")
        for i, pred in enumerate(result["top_predictions"], 1):
            print(f"  #{i} {pred['name']} ({pred['confidence']:.1%})")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
