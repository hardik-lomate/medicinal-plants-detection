"""
Stage 2: Dataset Download and Inspection Script
================================================
This script helps download the Indian Medicinal Leaf Image Dataset from Kaggle
and inspects its structure, class names, image counts, and quality.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get your Kaggle API token:
       - Go to https://www.kaggle.com/settings
       - Click "Create New Token" under the API section
       - This downloads kaggle.json
       - Place it at: C:\\Users\\<username>\\.kaggle\\kaggle.json
    3. Run this script: python training/download_and_inspect.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import Counter

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DATASET_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

# Kaggle dataset identifier
KAGGLE_DATASET = "warcoder/indian-medicinal-leaf-image-dataset"

# Classes we want for the first version (14 classes: 10 medicinal + 4 general)
SELECTED_CLASSES = {
    # Medicinal plants
    "Aloevera": {"type": "medicinal", "display_name": "Aloe Vera"},
    "Amla": {"type": "medicinal", "display_name": "Amla"},
    "Ashwagandha": {"type": "medicinal", "display_name": "Ashwagandha"},
    "Brahmi": {"type": "medicinal", "display_name": "Brahmi"},
    "Curry Leaf": {"type": "medicinal", "display_name": "Curry Leaf"},
    "Hibiscus": {"type": "medicinal", "display_name": "Hibiscus"},
    "Lemon grass": {"type": "medicinal", "display_name": "Lemon Grass"},
    "Mint": {"type": "medicinal", "display_name": "Mint"},
    "Neem": {"type": "medicinal", "display_name": "Neem"},
    "Tulasi": {"type": "medicinal", "display_name": "Tulasi"},
    # General plants (non-medicinal in our database)
    "Mango": {"type": "general", "display_name": "Mango"},
    "Guava": {"type": "general", "display_name": "Guava"},
    "Jasmine": {"type": "general", "display_name": "Jasmine"},
    "Bamboo": {"type": "general", "display_name": "Bamboo"},
}


def download_dataset():
    """Download dataset from Kaggle using the Kaggle API."""
    print("=" * 60)
    print("STEP 1: Downloading Dataset from Kaggle")
    print("=" * 60)

    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("\nERROR: kaggle package not installed.")
        print("Run: pip install kaggle")
        print("\nAlso ensure your Kaggle API token (kaggle.json) is placed at:")
        print(f"  Windows: C:\\Users\\{os.getenv('USERNAME')}\\.kaggle\\kaggle.json")
        sys.exit(1)

    # Check if dataset already exists
    if DATASET_RAW_DIR.exists() and any(DATASET_RAW_DIR.iterdir()):
        print(f"\nDataset directory already contains files: {DATASET_RAW_DIR}")
        response = input("Re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return

    # Create directory
    DATASET_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {KAGGLE_DATASET}")
    print(f"Destination: {DATASET_RAW_DIR}")
    print("This may take several minutes (~9 GB)...\n")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATASET_RAW_DIR),
            unzip=True
        )
        print("\n✅ Dataset downloaded and extracted successfully!")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"  Extract the zip file into: {DATASET_RAW_DIR}")
        sys.exit(1)


def find_dataset_root(base_dir):
    """
    Find the actual root directory containing class folders.
    Kaggle downloads sometimes have nested directory structures.
    """
    base_dir = Path(base_dir)

    # Look for common folder names in the dataset
    # The dataset has two main folders:
    #   "Medicinal Leaf dataset" (80 classes)
    #   "Medicinal plant dataset" (40 classes)
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        # Check if this directory has subdirectories that look like class folders
        if len(dirs) >= 30:  # At least 30 subdirectories suggests we found the class-level
            # Verify these subdirectories contain image files
            sample_dir = root_path / dirs[0]
            if sample_dir.is_dir():
                image_files = list(sample_dir.glob("*.jpg")) + \
                              list(sample_dir.glob("*.jpeg")) + \
                              list(sample_dir.glob("*.png"))
                if len(image_files) > 0:
                    return root_path

    # If not found by heuristic, list what's available
    print("\nDirectory structure found:")
    for item in sorted(base_dir.rglob("*")):
        if item.is_dir():
            depth = len(item.relative_to(base_dir).parts)
            if depth <= 2:
                print(f"  {'  ' * depth}{item.name}/")

    return None


def inspect_dataset():
    """Inspect the downloaded dataset structure."""
    print("\n" + "=" * 60)
    print("STEP 2: Inspecting Dataset Structure")
    print("=" * 60)

    if not DATASET_RAW_DIR.exists():
        print(f"\n❌ Dataset directory not found: {DATASET_RAW_DIR}")
        print("Please download the dataset first.")
        return None

    # Find all directories in the raw dataset
    print(f"\nScanning: {DATASET_RAW_DIR}")

    # List top-level contents
    top_level = sorted([d for d in DATASET_RAW_DIR.iterdir()])
    print(f"\nTop-level contents ({len(top_level)} items):")
    for item in top_level:
        if item.is_dir():
            sub_count = sum(1 for _ in item.iterdir() if _.is_dir())
            print(f"  📁 {item.name}/ ({sub_count} subdirectories)")
        else:
            print(f"  📄 {item.name} ({item.stat().st_size / 1024:.1f} KB)")

    # Find the leaf dataset folder (80 classes)
    leaf_dir = None
    plant_dir = None

    for d in DATASET_RAW_DIR.rglob("*"):
        if d.is_dir():
            name_lower = d.name.lower()
            if "leaf" in name_lower and "dataset" in name_lower:
                leaf_dir = d
            elif "plant" in name_lower and "dataset" in name_lower:
                plant_dir = d

    # If specific folders not found, try to find the root with class folders
    if leaf_dir is None:
        print("\n⚠️ Could not find 'Medicinal Leaf dataset' folder by name.")
        print("Searching for the directory with the most class-like subdirectories...")
        leaf_dir = find_dataset_root(DATASET_RAW_DIR)

    results = {}

    if leaf_dir:
        print(f"\n{'='*60}")
        print(f"LEAF DATASET: {leaf_dir}")
        print(f"{'='*60}")
        results["leaf"] = inspect_class_directory(leaf_dir)

    if plant_dir:
        print(f"\n{'='*60}")
        print(f"PLANT DATASET: {plant_dir}")
        print(f"{'='*60}")
        results["plant"] = inspect_class_directory(plant_dir)

    if not leaf_dir and not plant_dir:
        print("\n⚠️ Could not automatically identify dataset folders.")
        print("Please check the directory structure manually.")
        print(f"Look inside: {DATASET_RAW_DIR}")
        # Try to be helpful - show all directories with their subdirectory counts
        for d in sorted(DATASET_RAW_DIR.iterdir()):
            if d.is_dir():
                sub_dirs = [sd for sd in d.iterdir() if sd.is_dir()]
                if len(sub_dirs) > 5:
                    results[d.name] = inspect_class_directory(d)

    return results


def inspect_class_directory(class_dir):
    """Inspect a directory containing class subdirectories."""
    class_dir = Path(class_dir)
    class_info = {}

    # Get all class directories
    class_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
    print(f"\nTotal classes found: {len(class_dirs)}")
    print(f"\n{'Class Name':<30} {'Images':>8} {'Formats':>20}")
    print("-" * 62)

    total_images = 0
    image_formats = Counter()

    for cd in class_dirs:
        # Count images in this class
        images = []
        formats = Counter()
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp", "*.webp"]:
            found = list(cd.glob(ext))
            images.extend(found)
            if found:
                formats[ext.replace("*", "").upper()] += len(found)

        count = len(images)
        total_images += count
        format_str = ", ".join(f"{k}:{v}" for k, v in formats.items())

        # Mark selected classes
        marker = " ✅" if cd.name in SELECTED_CLASSES else ""
        print(f"  {cd.name:<28} {count:>8}   {format_str}{marker}")

        class_info[cd.name] = {
            "count": count,
            "formats": dict(formats),
            "path": str(cd),
        }

        for k, v in formats.items():
            image_formats[k] += v

    print("-" * 62)
    print(f"  {'TOTAL':<28} {total_images:>8}")
    print(f"\nImage format distribution: {dict(image_formats)}")

    # Check which of our selected classes are present
    print(f"\n{'='*60}")
    print("SELECTED CLASSES VERIFICATION")
    print(f"{'='*60}")

    available_classes = set(ci for ci in class_info.keys())

    found = []
    not_found = []
    for class_name, info in SELECTED_CLASSES.items():
        if class_name in available_classes:
            count = class_info[class_name]["count"]
            found.append((class_name, info["display_name"], info["type"], count))
            print(f"  ✅ {class_name:<20} ({info['type']:<10}) — {count} images")
        else:
            not_found.append((class_name, info["display_name"], info["type"]))
            # Try case-insensitive match
            matches = [c for c in available_classes if c.lower() == class_name.lower()]
            if matches:
                actual = matches[0]
                count = class_info[actual]["count"]
                found.append((actual, info["display_name"], info["type"], count))
                print(f"  ✅ {actual:<20} ({info['type']:<10}) — {count} images (matched as '{actual}')")
                not_found.pop()
            else:
                print(f"  ❌ {class_name:<20} ({info['type']:<10}) — NOT FOUND")

    if not_found:
        print(f"\n⚠️ {len(not_found)} selected classes not found!")
        print("Available classes that might be alternatives:")
        for cls in sorted(available_classes):
            if cls not in [f[0] for f in found]:
                count = class_info[cls]["count"]
                print(f"    - {cls} ({count} images)")

    # Class distribution analysis
    if found:
        counts = [f[3] for f in found]
        print(f"\n{'='*60}")
        print("CLASS DISTRIBUTION (Selected Classes)")
        print(f"{'='*60}")
        print(f"  Min images per class: {min(counts)}")
        print(f"  Max images per class: {max(counts)}")
        print(f"  Mean images per class: {sum(counts)/len(counts):.1f}")
        print(f"  Total selected images: {sum(counts)}")

        # Warn about class imbalance
        if max(counts) > 3 * min(counts):
            print(f"\n  ⚠️ CLASS IMBALANCE DETECTED!")
            print(f"  Ratio (max/min): {max(counts)/min(counts):.1f}x")
            print(f"  Consider using data augmentation or weighted sampling.")

    return class_info


def check_image_quality(class_dir, num_samples=3):
    """Check sample images for quality issues."""
    print(f"\n{'='*60}")
    print("STEP 3: Image Quality Check (Samples)")
    print(f"{'='*60}")

    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed. Run: pip install Pillow")
        return

    class_dir = Path(class_dir)
    class_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])

    for cd in class_dirs[:5]:  # Check first 5 classes as samples
        images = list(cd.glob("*.jpg")) + list(cd.glob("*.jpeg")) + list(cd.glob("*.png")) + \
                 list(cd.glob("*.JPG")) + list(cd.glob("*.JPEG")) + list(cd.glob("*.PNG"))

        if not images:
            continue

        print(f"\n  Class: {cd.name}")
        for img_path in images[:num_samples]:
            try:
                img = Image.open(img_path)
                print(f"    {img_path.name}: {img.size[0]}x{img.size[1]}, mode={img.mode}, "
                      f"format={img.format}, size={img_path.stat().st_size/1024:.1f}KB")
            except Exception as e:
                print(f"    {img_path.name}: ❌ ERROR - {e}")


def save_inspection_report(results):
    """Save the inspection results to a JSON file."""
    report_path = PROJECT_ROOT / "dataset" / "inspection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": KAGGLE_DATASET,
        "selected_classes": {k: v for k, v in SELECTED_CLASSES.items()},
        "inspection_results": {}
    }

    if results:
        for key, class_info in results.items():
            report["inspection_results"][key] = {
                "total_classes": len(class_info),
                "total_images": sum(info["count"] for info in class_info.values()),
                "classes": {name: info["count"] for name, info in class_info.items()},
            }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Inspection report saved to: {report_path}")


def main():
    print("=" * 60)
    print("MEDICINAL PLANT DETECTION — DATASET DOWNLOAD & INSPECTION")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir:  {DATASET_RAW_DIR}")

    # Step 1: Download
    download_dataset()

    # Step 2: Inspect
    results = inspect_dataset()

    # Step 3: Save report
    if results:
        save_inspection_report(results)

        # Check image quality for the leaf dataset
        for key in results:
            if "leaf" in key.lower() or len(results[key]) >= 30:
                # Find the actual directory
                for d in DATASET_RAW_DIR.rglob("*"):
                    if d.is_dir() and d.name.lower().replace(" ", "") == key.lower().replace(" ", ""):
                        check_image_quality(d)
                        break
                else:
                    # Try the first result
                    if results[key]:
                        first_class = list(results[key].values())[0]
                        parent = Path(first_class["path"]).parent
                        check_image_quality(parent)
                break

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the class list and image counts above")
    print("  2. If any selected classes are missing, we'll find alternatives")
    print("  3. Run the dataset preparation script (Stage 3)")


if __name__ == "__main__":
    main()
