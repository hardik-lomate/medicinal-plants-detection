"""
training/inspect_local_dataset.py
===================================
Comprehensive inspection of the Indian Medicinal Leaves dataset.
Inspects both Datasets_Leaf and Datasets_Plant without copying files.

Usage:
    python training/inspect_local_dataset.py

Outputs:
    models/dataset_analysis/dataset_report.json
    models/dataset_analysis/dataset_report.csv
"""

import os
import sys
import json
import csv
import hashlib
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

from config import ProjectConfig as cfg

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARNING] Pillow not installed. Detailed image size/mode inspection disabled.")


def compute_md5(filepath, chunk_size=65536):
    h = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def get_image_info(filepath):
    """Checks corruption, dimensions, and color mode."""
    if not PIL_OK:
        return None, None, None, True, None
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            mode = img.mode
            img.verify()
        # Verify invalidates file pointer, re-open to ensure valid header
        with Image.open(filepath) as img:
            _ = img.size
        return w, h, mode, True, None
    except Exception as e:
        return None, None, None, False, str(e)


# Normalization dictionary to match plant names across leaf and whole plant datasets
NAME_ALIASES = {
    'brahmi': 'bhrami',
    'bhrami': 'bhrami',
    'curry_leaf': 'curry',
    'curry': 'curry',
    'lemon_grass': 'lemongrass',
    'lemongrass': 'lemongrass',
    'gauva': 'guava',
    'guava': 'guava',
    'amruta_balli': 'amruthaballi',
    'amruthaballi': 'amruthaballi',
    'doddapatre': 'doddpathre',
    'doddpathre': 'doddpathre',
    'pappaya': 'papaya',
    'papaya': 'papaya',
    'pomoegranate': 'pomegranate',
    'pomegranate': 'pomegranate',
    'nagadali': 'common rue(naagdalli)',
    'common rue(naagdalli)': 'common rue(naagdalli)',
    'tulasi': 'tulsi',
    'tulsi': 'tulsi',
    'ashoka': 'ashoka',
}


def normalize_class_name(name):
    norm = name.lower().replace('_', ' ').replace('-', ' ').strip()
    return NAME_ALIASES.get(norm, norm)


def inspect_dataset_group(group_name, group_path):
    group_path = Path(group_path)
    print(f"\n{'='*70}")
    print(f"INSPECTING: {group_name}")
    print(f"Path: {group_path}")
    print(f"{'='*70}")

    if not group_path.exists():
        print(f"  [ERROR] Path does not exist: {group_path}")
        return None

    class_dirs = sorted([d for d in group_path.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class folders.")

    classes_data = {}
    all_hashes = {}
    duplicates = []
    total_valid = 0
    total_corrupted = 0
    unsupported_files = []
    all_widths = []
    all_heights = []
    all_modes = defaultdict(int)

    for cdir in class_dirs:
        cname = cdir.name
        img_files = []
        unsup_files = []

        for root, _, files in os.walk(cdir):
            for fn in files:
                ext = Path(fn).suffix.lower()
                full_p = Path(root) / fn
                if ext in cfg.IMAGE_EXTENSIONS:
                    img_files.append(full_p)
                else:
                    unsup_files.append(str(full_p))

        unsupported_files.extend(unsup_files)

        widths = []
        heights = []
        modes = defaultdict(int)
        corrupted = []

        print(f"  Checking {cname} ({len(img_files)} images)...", end=" ", flush=True)

        for img_p in img_files:
            w, h, mode, is_ok, err = get_image_info(img_p)
            if not is_ok:
                corrupted.append({"file": str(img_p), "error": err})
                total_corrupted += 1
                continue

            widths.append(w)
            heights.append(h)
            all_widths.append(w)
            all_heights.append(h)
            modes[mode] += 1
            all_modes[mode] += 1

            # Hash for duplicate detection
            fhash = compute_md5(img_p)
            if fhash:
                if fhash in all_hashes:
                    duplicates.append({
                        "file1": str(all_hashes[fhash]),
                        "file2": str(img_p),
                        "class": cname,
                        "hash": fhash
                    })
                else:
                    all_hashes[fhash] = img_p

        valid_count = len(widths)
        total_valid += valid_count

        avg_w = round(sum(widths) / len(widths), 1) if widths else 0
        avg_h = round(sum(heights) / len(heights), 1) if heights else 0
        min_w = min(widths) if widths else 0
        max_w = max(widths) if widths else 0
        min_h = min(heights) if heights else 0
        max_h = max(heights) if heights else 0

        classes_data[cname] = {
            "folder": str(cdir),
            "total_files": len(img_files),
            "valid_images": valid_count,
            "corrupted": len(corrupted),
            "corrupted_files": corrupted,
            "unsupported_files": len(unsup_files),
            "modes": dict(modes),
            "min_width": min_w,
            "max_width": max_w,
            "avg_width": avg_w,
            "min_height": min_h,
            "max_height": max_h,
            "avg_height": avg_h,
        }
        print(f"Valid: {valid_count}, Corrupted: {len(corrupted)}")

    empty_classes = [c for c, d in classes_data.items() if d['valid_images'] == 0]
    low_data_classes = [(c, d['valid_images']) for c, d in classes_data.items() if 0 < d['valid_images'] < cfg.MIN_IMAGES_PER_CLASS]

    print(f"\n  Summary for {group_name}:")
    print(f"  Total classes       : {len(classes_data)}")
    print(f"  Total valid images  : {total_valid}")
    print(f"  Total corrupted     : {total_corrupted}")
    print(f"  Unsupported files   : {len(unsupported_files)}")
    print(f"  Duplicate images    : {len(duplicates)}")
    print(f"  Empty folders       : {empty_classes}")
    print(f"  Classes with < {cfg.MIN_IMAGES_PER_CLASS} images: {low_data_classes}")
    if all_widths:
        print(f"  Widths (px)         : min={min(all_widths)}, max={max(all_widths)}, avg={round(sum(all_widths)/len(all_widths), 1)}")
        print(f"  Heights (px)        : min={min(all_heights)}, max={max(all_heights)}, avg={round(sum(all_heights)/len(all_heights), 1)}")
    print(f"  Color modes         : {dict(all_modes)}")

    return {
        "group_name": group_name,
        "path": str(group_path),
        "num_classes": len(classes_data),
        "total_valid_images": total_valid,
        "total_corrupted": total_corrupted,
        "total_unsupported": len(unsupported_files),
        "total_duplicates": len(duplicates),
        "empty_classes": empty_classes,
        "low_data_classes": [c for c, _ in low_data_classes],
        "duplicate_pairs": duplicates,
        "color_modes": dict(all_modes),
        "dimensions": {
            "min_width": min(all_widths) if all_widths else 0,
            "max_width": max(all_widths) if all_widths else 0,
            "avg_width": round(sum(all_widths)/len(all_widths), 1) if all_widths else 0,
            "min_height": min(all_heights) if all_heights else 0,
            "max_height": max(all_heights) if all_heights else 0,
            "avg_height": round(sum(all_heights)/len(all_heights), 1) if all_heights else 0,
        },
        "classes": classes_data
    }


def compare_datasets(leaf_res, plant_res):
    print(f"\n{'='*70}")
    print("CROSS-DATASET ALIGNMENT & OVERLAP ANALYSIS")
    print(f"{'='*70}")

    leaf_classes = set(leaf_res["classes"].keys())
    plant_classes = set(plant_res["classes"].keys())

    leaf_norm = {normalize_class_name(c): c for c in leaf_classes}
    plant_norm = {normalize_class_name(c): c for c in plant_classes}

    common_norms = sorted(set(leaf_norm.keys()) & set(plant_norm.keys()))
    leaf_only_norms = sorted(set(leaf_norm.keys()) - set(plant_norm.keys()))
    plant_only_norms = sorted(set(plant_norm.keys()) - set(plant_norm.keys()))

    common_pairs = [{"normalized": n, "leaf_name": leaf_norm[n], "plant_name": plant_norm[n]} for n in common_norms]
    leaf_only = [leaf_norm[n] for n in leaf_only_norms]
    plant_only = [plant_norm[n] for n in sorted(set(plant_norm.keys()) - set(leaf_norm.keys()))]

    # Cross-dataset duplicate hash check
    print("Checking for duplicate images shared across both datasets...")
    leaf_hashes = {}
    for cname, cdata in leaf_res["classes"].items():
        cdir = Path(cdata["folder"])
        for root, _, files in os.walk(cdir):
            for fn in files:
                p = Path(root) / fn
                if p.suffix.lower() in cfg.IMAGE_EXTENSIONS:
                    h = compute_md5(p)
                    if h:
                        leaf_hashes[h] = str(p)

    cross_duplicates = []
    for cname, cdata in plant_res["classes"].items():
        cdir = Path(cdata["folder"])
        for root, _, files in os.walk(cdir):
            for fn in files:
                p = Path(root) / fn
                if p.suffix.lower() in cfg.IMAGE_EXTENSIONS:
                    h = compute_md5(p)
                    if h and h in leaf_hashes:
                        cross_duplicates.append({
                            "leaf_file": leaf_hashes[h],
                            "plant_file": str(p),
                            "hash": h
                        })

    print(f"  Shared classes count : {len(common_pairs)}")
    print(f"  Leaf-only classes    : {len(leaf_only)}")
    print(f"  Plant-only classes   : {len(plant_only)}")
    print(f"  Cross-dataset duplicate image count: {len(cross_duplicates)}")

    return {
        "common_classes": common_pairs,
        "leaf_only_classes": leaf_only,
        "plant_only_classes": plant_only,
        "cross_dataset_duplicates": cross_duplicates,
    }


def main():
    print("=" * 70)
    print("MEDICINAL PLANT DETECTION — LOCAL DATASET INSPECTION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"1. Dataset Root: {cfg.DATASET_ROOT}")
    print(f"2. Leaf Dataset: {cfg.LEAF_DATASET_DIR}")
    print(f"3. Plant Dataset: {cfg.PLANT_DATASET_DIR}")

    cfg.ensure_dirs()

    leaf_res = inspect_dataset_group("Datasets_Leaf", cfg.LEAF_DATASET_DIR)
    plant_res = inspect_dataset_group("Datasets_Plant", cfg.PLANT_DATASET_DIR)

    if not leaf_res or not plant_res:
        print("[ERROR] Failed to inspect one or both dataset directories.")
        sys.exit(1)

    comp_res = compare_datasets(leaf_res, plant_res)

    # Save full JSON report
    report_dict = {
        "generated_at": datetime.now().isoformat(),
        "dataset_root": str(cfg.DATASET_ROOT),
        "leaf_dataset": leaf_res,
        "plant_dataset": plant_res,
        "comparison": comp_res,
    }

    report_json_path = cfg.DATASET_REPORT_JSON
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"\n[OK] JSON report saved to: {report_json_path}")

    # Save CSV report
    report_csv_path = cfg.DATASET_REPORT_CSV
    with open(report_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_group", "class_name", "valid_images", "corrupted", "unsupported",
            "min_width", "max_width", "avg_width",
            "min_height", "max_height", "avg_height", "modes"
        ])
        for grp_name, grp_res in [("Datasets_Leaf", leaf_res), ("Datasets_Plant", plant_res)]:
            for cname, cdata in grp_res["classes"].items():
                writer.writerow([
                    grp_name, cname, cdata["valid_images"], cdata["corrupted"], cdata["unsupported_files"],
                    cdata["min_width"], cdata["max_width"], cdata["avg_width"],
                    cdata["min_height"], cdata["max_height"], cdata["avg_height"],
                    str(cdata["modes"])
                ])
    print(f"[OK] CSV report saved to: {report_csv_path}")

    print("\n" + "=" * 70)
    print("INSPECTION EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    main()
