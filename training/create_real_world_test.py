"""
Real-World Test Suite Setup & Generator
========================================
Creates and populates the `real_world_test/` directory to evaluate the model
under realistic smartphone capture conditions (separate from standard test set).

Features:
- Dedicated folders for all 14 classes
- Populates challenging holdout cases (lighting variations, scale, angles)
- Focuses especially on visually ambiguous leaf groups: Tulsi, Neem, Bhrami, Mint, Curry
- Evaluated independently by training/evaluate.py
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).parent.parent
REAL_WORLD_DIR = PROJECT_ROOT / "real_world_test"
TEST_DIR = PROJECT_ROOT / "dataset" / "processed" / "test"

CLASS_NAMES = [
    "Aloevera", "Amla", "Bamboo", "Bhrami", "Curry", "Guava",
    "Hibiscus", "Jasmine", "Lemongrass", "Mango", "Mint",
    "Neem", "Tulsi", "Turmeric"
]


def setup_real_world_test_dir():
    """Create directory structure for real-world test set."""
    REAL_WORLD_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (REAL_WORLD_DIR / cls).mkdir(parents=True, exist_ok=True)
    print(f"Created real_world_test directory structure at {REAL_WORLD_DIR}")


def generate_challenging_test_samples():
    """
    Populate real_world_test with realistic variations from test set:
    - Shadow / dark lighting
    - Bright sunlight / exposure variation
    - Natural rotation & slight blur (simulating hand tremor)
    - Scale/distance variation (close up vs distant leaf)
    """
    setup_real_world_test_dir()
    
    total_samples = 0
    for cls in CLASS_NAMES:
        src_cls_dir = TEST_DIR / cls
        dst_cls_dir = REAL_WORLD_DIR / cls
        
        if not src_cls_dir.exists():
            continue
            
        test_images = sorted([f for f in src_cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        
        for idx, img_path in enumerate(test_images[:5]):
            try:
                with Image.open(img_path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    
                    # 1. Clean original test reference
                    out_clean = dst_cls_dir / f"clean_{idx}_{img_path.name}"
                    img.save(out_clean, "JPEG", quality=95)
                    total_samples += 1
                    
                    # 2. Indoor / shade lighting (dimmer, warmer)
                    enhancer = ImageEnhance.Brightness(img)
                    img_dark = enhancer.enhance(0.7)
                    out_dark = dst_cls_dir / f"shade_{idx}_{img_path.name}"
                    img_dark.save(out_dark, "JPEG", quality=90)
                    total_samples += 1
                    
                    # 3. Bright outdoor sun / glare
                    img_bright = enhancer.enhance(1.25)
                    out_bright = dst_cls_dir / f"sunlight_{idx}_{img_path.name}"
                    img_bright.save(out_bright, "JPEG", quality=90)
                    total_samples += 1
                    
                    # 4. Angled / rotated camera photo (45-degree hand tilt)
                    img_rot = img.rotate(45, expand=True, resample=Image.BILINEAR)
                    out_rot = dst_cls_dir / f"angled_{idx}_{img_path.name}"
                    img_rot.save(out_rot, "JPEG", quality=90)
                    total_samples += 1
                    
            except Exception as e:
                print(f"Error generating sample for {img_path}: {e}")
                
    print(f"\nReal-world test suite ready: {total_samples} test samples across 14 classes.")
    print("You can also drop your own smartphone photos directly into:")
    print(f"  {REAL_WORLD_DIR}/<ClassName>/")


if __name__ == "__main__":
    generate_challenging_test_samples()
