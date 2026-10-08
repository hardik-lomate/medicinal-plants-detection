"""
training/train_all.py
======================
Unified sequential runner for training all three models:
  1. Image-Type Classifier (LEAF vs WHOLE_PLANT)
  2. Whole-Plant Classifier (40 classes)
  3. Leaf Classifier (78 classes)

Runs sequentially to respect RAM/VRAM limits on Windows.

Usage:
    python training/train_all.py [--image_type_epochs 8] [--plant_epochs 15] [--leaf_epochs 15]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_image_type import train_image_type
from training.train_whole_plant import train_whole_plant
from training.train_leaf import train_leaf


def train_all(image_type_epochs=3, plant_epochs=5, leaf_epochs=5, batch_size=64):
    start_time = datetime.now()
    print("=" * 80, flush=True)
    print("MEDICINAL PLANT DETECTION — FULL THREE-MODEL SEQUENTIAL TRAINING PIPELINE", flush=True)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    # 1. Image Type
    print("\n>>> STAGE 1/3: Training Image-Type Classifier (LEAF vs WHOLE_PLANT)...", flush=True)
    type_ckpt = train_image_type(epochs=image_type_epochs, warmup_epochs=1, batch_size=batch_size)

    # 2. Whole Plant
    print("\n>>> STAGE 2/3: Training Whole-Plant Classifier (40 Classes)...", flush=True)
    plant_ckpt = train_whole_plant(epochs=plant_epochs, warmup_epochs=1, batch_size=batch_size)

    # 3. Leaf
    print("\n>>> STAGE 3/3: Training Leaf Classifier (78 Classes)...", flush=True)
    leaf_ckpt = train_leaf(epochs=leaf_epochs, warmup_epochs=1, batch_size=batch_size)

    total_time = datetime.now() - start_time
    print("\n" + "=" * 80, flush=True)
    print("ALL THREE MODELS TRAINED SUCCESSFULLY", flush=True)
    print(f"Total time taken: {total_time}", flush=True)
    print(f"  1. Image-Type Model : {type_ckpt['test_acc']:.2%} test acc -> models/image_type_model.pth", flush=True)
    print(f"  2. Whole-Plant Model: {plant_ckpt['test_acc']:.2%} test acc -> models/whole_plant_model.pth", flush=True)
    print(f"  3. Leaf Model       : {leaf_ckpt['test_acc']:.2%} test acc -> models/leaf_model.pth", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_type_epochs', type=int, default=3)
    parser.add_argument('--plant_epochs', type=int, default=5)
    parser.add_argument('--leaf_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train_all(
        image_type_epochs=args.image_type_epochs,
        plant_epochs=args.plant_epochs,
        leaf_epochs=args.leaf_epochs,
        batch_size=args.batch_size
    )

