"""
training/train_leaf.py
======================
Trains a dedicated classifier for leaf photographs (78 classes).
Uses leaf_manifest.csv directly from external dataset without copying.

Saves to: models/leaf_model.pth

Usage:
    python training/train_leaf.py [--epochs 15] [--batch_size 32] [--backbone efficientnet_b0]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import ProjectConfig as cfg
from training.model_factory import create_model, freeze_backbone, unfreeze_backbone, get_classifier_params
from training.manifest_dataset import build_dataloaders, load_class_names_from_manifest


def compute_smoothed_weights(dataset, num_classes):
    labels = [s[1] for s in dataset.samples]
    counts = Counter(labels)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights[c] = total / (num_classes * cnt)
    weights = np.power(weights, 0.5)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_leaf(epochs=15, warmup_epochs=2, batch_size=32, lr=1e-3, lr_backbone=1e-4,
               backbone="efficientnet_b0"):
    device = cfg.get_device()
    print("=" * 70)
    print("STAGE: TRAIN LEAF CLASSIFIER (78 CLASSES)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device   : {device}")
    print(f"Backbone : {backbone}")
    print(f"Manifest : {cfg.LEAF_MANIFEST_CSV}")
    print("=" * 70)

    if not cfg.LEAF_MANIFEST_CSV.exists():
        print(f"[ERROR] Manifest not found: {cfg.LEAF_MANIFEST_CSV}")
        print("Run `python training/prepare_dataset.py` first.")
        sys.exit(1)

    class_names, class_to_idx = load_class_names_from_manifest(
        cfg.LEAF_MANIFEST_CSV, label_field='normalized_class'
    )
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from manifest.")

    loaders = build_dataloaders(
        manifest_path=cfg.LEAF_MANIFEST_CSV,
        class_to_idx=class_to_idx,
        label_field='normalized_class',
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS
    )

    class_weights = compute_smoothed_weights(loaders['train'].dataset, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.10)

    model = create_model(backbone, num_classes=num_classes, pretrained=True).to(device)

    # Phase 1: Warmup
    print(f"\n--- Phase 1: Warmup classifier head ({warmup_epochs} epochs) ---")
    freeze_backbone(model, backbone)
    head_params, _ = get_classifier_params(model, backbone)
    optimizer = optim.AdamW(head_params, lr=lr, weight_decay=1e-4)

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for b_idx, (images, labels) in enumerate(loaders['train'], 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if b_idx % 25 == 0 or b_idx == len(loaders['train']):
                print(f"  Warmup Batch {b_idx}/{len(loaders['train'])} - Loss: {train_loss/total:.4f}, Acc: {correct/total:.2%}", flush=True)
        print(f"  Warmup Epoch {epoch}/{warmup_epochs} Finished - Loss: {train_loss/total:.4f} | Acc: {correct/total:.2%}", flush=True)

    # Phase 2: Fine-tuning
    print(f"\n--- Phase 2: Fine-Tuning Full Backbone ({epochs} epochs) ---", flush=True)
    unfreeze_backbone(model)
    head_params, backbone_params = get_classifier_params(model, backbone)
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_backbone * 5}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_weights = None
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for b_idx, (images, labels) in enumerate(loaders['train'], 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            if b_idx % 25 == 0 or b_idx == len(loaders['train']):
                print(f"  Epoch {epoch} Batch {b_idx}/{len(loaders['train'])} - Loss: {train_loss/train_total:.4f}, Acc: {train_correct/train_total:.2%}", flush=True)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loaders['val']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{epochs:2d} [{elapsed:.1f}s] - Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2%}")

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ⭐ New best validation accuracy: {best_val_acc:.2%}")

    if best_weights:
        model.load_state_dict(best_weights)

    # Test set evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total if test_total > 0 else 0.0
    print(f"\nFinal Leaf Test Accuracy: {test_acc:.2%} ({test_correct}/{test_total})")

    cfg.ensure_dirs()
    checkpoint = {
        'model_name': backbone,
        'model_version': cfg.LEAF_MODEL_VERSION,
        'num_classes': num_classes,
        'class_names': class_names,
        'class_to_index': class_to_idx,
        'index_to_class': {i: c for i, c in enumerate(class_names)},
        'image_size': cfg.IMAGE_SIZE,
        'mean': cfg.MEAN,
        'std': cfg.STD,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
        'margin_threshold': cfg.MARGIN_THRESHOLD,
        'training_dataset': 'leaf_manifest',
        'trained_at': datetime.now().isoformat(),
        'model_state_dict': model.state_dict(),
        'history': history,
    }

    torch.save(checkpoint, cfg.LEAF_MODEL_PATH)
    print(f"\n[OK] Leaf model saved: {cfg.LEAF_MODEL_PATH}")
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    args = parser.parse_args()

    train_leaf(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        backbone=args.backbone
    )
