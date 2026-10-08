"""
Stage 4: Model Training Script — Robust EfficientNet-B0 Transfer Learning
==========================================================================
Trains an EfficientNet-B0 model with:
- Authoritative class mapping
- Aspect-ratio-preserving preprocessing (Resize 256 + CenterCrop 224)
- Rotation-invariant and scale-invariant augmentation (RandomResizedCrop + 360deg rotation)
- Smoothed class weighting with label smoothing (prevents Tulsi/Bhrami penalty asymmetry)
- Checkpoint metadata versioning (plant_classifier_v2)
- Differential learning rates and early stopping

Usage:
    python training/train.py
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Authoritative class names list
CLASS_NAMES = [
    "Aloevera",
    "Amla",
    "Bamboo",
    "Bhrami",
    "Curry",
    "Guava",
    "Hibiscus",
    "Jasmine",
    "Lemongrass",
    "Mango",
    "Mint",
    "Neem",
    "Tulsi",
    "Turmeric",
]

# ============================================================
# Configuration
# ============================================================

class Config:
    """All training hyperparameters in one place."""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATASET_DIR = PROJECT_ROOT / "dataset" / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

    # Model & Version
    MODEL_VERSION = "plant_classifier_v2"
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True

    # Image settings (Aspect-ratio preserving standard)
    IMAGE_SIZE = 224
    RESIZE_EDGE = 256
    MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    STD = [0.229, 0.224, 0.225]

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    WARMUP_EPOCHS = 2     # Train classifier head first with backbone frozen
    LEARNING_RATE = 1e-3  # For classifier head
    LR_BACKBONE = 1e-4    # For fine-tuning backbone
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1 # Prevents overconfidence on identical leaves

    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001

    # LR Scheduling
    LR_PATIENCE = 3
    LR_FACTOR = 0.5

    # Data loading
    NUM_WORKERS = 0       # Windows compatibility
    PIN_MEMORY = torch.cuda.is_available()

    # Reproducibility
    SEED = 42

    # Calibrated inference thresholds
    CONFIDENCE_THRESHOLD = 0.65
    MARGIN_THRESHOLD = 0.15

    @classmethod
    def get_device(cls):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    """
    Create aspect-ratio-preserving image transforms.

    Training transforms:
    - RandomResizedCrop(224): preserves natural leaf aspect ratios across zoom/scale
    - RandomRotation(180) + Flips: ensures 360-degree rotation invariance
    - Gentle ColorJitter: simulates realistic phone exposure/shadow variations
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD),
    ])

    # Validation/test: aspect-ratio-preserving Resize(256) + CenterCrop(224)
    val_test_transform = transforms.Compose([
        transforms.Resize(Config.RESIZE_EDGE),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD),
    ])

    return train_transform, val_test_transform


def load_datasets():
    """Load train/val/test datasets using ImageFolder."""
    train_transform, val_test_transform = get_transforms()

    train_dir = Config.DATASET_DIR / "train"
    val_dir = Config.DATASET_DIR / "val"
    test_dir = Config.DATASET_DIR / "test"

    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not d.exists():
            print(f"[ERROR] {name} directory not found: {d}")
            print("Run training/prepare_dataset.py first!")
            sys.exit(1)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_test_transform)

    # Verify class mapping matches authoritative CLASS_NAMES
    assert train_dataset.classes == CLASS_NAMES, (
        f"Mismatch in dataset classes! Expected {CLASS_NAMES}, got {train_dataset.classes}"
    )

    print(f"\n[OK] Dataset loaded:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes ({len(train_dataset.classes)}): {train_dataset.classes}")

    return train_dataset, val_dataset, test_dataset


def compute_smoothed_class_weights(dataset):
    """
    Compute square-root smoothed class weights.
    Prevents disproportionate penalty differences between Tulsi (123) and Bhrami (71)
    while still supporting minority classes (Lemongrass, Turmeric).
    """
    class_counts = np.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1

    max_count = np.max(class_counts)
    # Square-root smoothing: w = (max / count) ** 0.5, clamped to [0.85, 2.5]
    weights = np.sqrt(max_count / class_counts)
    weights = np.clip(weights, 0.85, 2.5)
    weights = torch.FloatTensor(weights)

    print(f"\nSmoothed class weights (balanced):")
    for cls, count, w in zip(dataset.classes, class_counts, weights):
        print(f"  {cls:<15}: count={int(count):<4} weight={float(w):.3f}")

    return weights


def create_model(num_classes, device):
    """
    Create EfficientNet-B0 with ImageNet pretrained weights.
    Replace classifier head with our 14 classes.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[OK] Model: EfficientNet-B0 (torchvision, pretrained={Config.PRETRAINED})")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    model = model.to(device)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with real-time responsive progress display."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()
    total_batches = len(dataloader)

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        curr_acc = 100.0 * correct / max(total, 1)
        elapsed = time.time() - t0
        rate = (batch_idx + 1) / max(elapsed, 1e-3)
        eta_sec = (total_batches - (batch_idx + 1)) / max(rate, 1e-3)
        bar_len = 16
        filled = int(bar_len * (batch_idx + 1) / total_batches)
        bar = "=" * filled + (">" if filled < bar_len else "") + "." * (bar_len - filled - (1 if filled < bar_len else 0))
        print(f"\r    [{bar}] Batch {batch_idx+1:02d}/{total_batches} ({100.*(batch_idx+1)/total_batches:3.0f}%) | Loss: {loss.item():.4f} | Acc: {curr_acc:4.1f}% | ETA: {eta_sec:2.0f}s", end="", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print()
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model on unseen dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            print(f"\r    Validating... [{batch_idx+1}/{total_batches}]", end="", flush=True)

    print("\r" + " " * 45 + "\r", end="", flush=True)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


class EarlyStopping:
    """Early stopping on validation loss."""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  [INFO] Early stopping triggered (no improvement for {self.patience} epochs)")


def main():
    print("=" * 70)
    print("MEDICINAL PLANT DETECTION — ROBUST MODEL TRAINING")
    print(f"Version: {Config.MODEL_VERSION} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    set_seed(Config.SEED)
    device = Config.get_device()
    print(f"\nDevice: {device}")

    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_datasets()
    num_classes = len(CLASS_NAMES)

    class_weights = compute_smoothed_class_weights(train_dataset).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    model = create_model(num_classes, device)

    # Loss with smoothed weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING)

    # Phase 1: Warmup - Freeze backbone features for first WARMUP_EPOCHS
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = optim.AdamW(classifier_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    print(f"\nPhase 1: Warmup training classifier head ({Config.WARMUP_EPOCHS} epochs, backbone frozen)...")
    for epoch in range(Config.WARMUP_EPOCHS):
        print(f"\nWarmup Epoch {epoch+1}/{Config.WARMUP_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # Phase 2: Unfreeze entire model for fine-tuning
    print("\nPhase 2: Fine-tuning full network with differential learning rates...")
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": Config.LR_BACKBONE},
        {"params": classifier_params, "lr": Config.LEARNING_RATE * 0.5},
    ], weight_decay=Config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
    )

    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    training_start = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc or (abs(val_acc - best_val_acc) < 0.01 and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

            checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                "model_version": Config.MODEL_VERSION,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": CLASS_NAMES,
                "num_classes": num_classes,
                "config": {
                    "model_name": Config.MODEL_NAME,
                    "image_size": Config.IMAGE_SIZE,
                    "resize_edge": Config.RESIZE_EDGE,
                    "mean": Config.MEAN,
                    "std": Config.STD,
                    "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                    "margin_threshold": Config.MARGIN_THRESHOLD,
                },
            }, checkpoint_path)
            print(f"  [SAVED] Best checkpoint saved (val_acc={val_acc:.4f}, val_loss={val_loss:.4f})")

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break

    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — Total time: {total_time/60:.1f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} | Best Loss: {best_val_loss:.4f}")
    print(f"{'='*70}")

    # Save final model
    final_model_path = Config.MODEL_DIR / "medicinal_plant_model.pth"
    torch.save({
        "model_version": Config.MODEL_VERSION,
        "model_state_dict": best_model_state,
        "class_names": CLASS_NAMES,
        "num_classes": num_classes,
        "config": {
            "model_name": Config.MODEL_NAME,
            "image_size": Config.IMAGE_SIZE,
            "resize_edge": Config.RESIZE_EDGE,
            "mean": Config.MEAN,
            "std": Config.STD,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
            "margin_threshold": Config.MARGIN_THRESHOLD,
        },
        "training_history": history,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "trained_at": datetime.now().isoformat(),
    }, final_model_path)
    print(f"\n[OK] Saved final model checkpoint to: {final_model_path}")

    # Save training history
    with open(Config.MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save authoritative class mapping
    class_mapping = {
        "model_version": Config.MODEL_VERSION,
        "class_names": CLASS_NAMES,
        "index_to_class": {i: name for i, name in enumerate(CLASS_NAMES)},
        "class_to_index": {name: i for i, name in enumerate(CLASS_NAMES)},
    }
    with open(Config.MODEL_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"[OK] Authoritative class mapping saved to: {Config.MODEL_DIR / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
