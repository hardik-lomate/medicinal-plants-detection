"""
training/manifest_dataset.py
=============================
PyTorch Dataset that loads images from a CSV manifest file.
No raw dataset copying needed — images are loaded directly from
the external dataset path at training time.

Usage:
    from training.manifest_dataset import ManifestDataset, build_dataloaders
"""

import csv
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(cfg.RESIZE_EDGE),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


class ManifestDataset(Dataset):
    """
    Loads images from a CSV manifest.
    Each row: filepath, class_name, normalized_class, dataset_type, split
    The class label is derived from `normalized_class` (or `class_name` for image_type model).
    """

    def __init__(self, manifest_path, split, class_to_idx, transform=None,
                 label_field='normalized_class', max_samples_per_class=None):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.class_to_idx = class_to_idx
        self.label_field = label_field
        self.max_samples_per_class = max_samples_per_class

        if transform is None:
            self.transform = get_train_transforms() if split == 'train' else get_val_transforms()
        else:
            self.transform = transform

        self.samples = []
        self._load_manifest()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {self.manifest_path}')

        from collections import defaultdict
        class_counts = defaultdict(int)

        with open(self.manifest_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] != self.split:
                    continue
                label_str = row[self.label_field]
                if label_str not in self.class_to_idx:
                    continue  # skip classes not in this model's class list
                if self.max_samples_per_class and class_counts[label_str] >= self.max_samples_per_class:
                    continue
                filepath = Path(row['filepath'])
                if not filepath.exists():
                    continue  # skip missing files silently
                self.samples.append((str(filepath), self.class_to_idx[label_str]))
                class_counts[label_str] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            with Image.open(filepath) as img:
                img = ImageOps.exif_transpose(img)
                if max(img.size) > 512:
                    img.thumbnail((512, 512), Image.Resampling.BILINEAR)
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            black = Image.new('RGB', (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), (0, 0, 0))
            if self.transform:
                black = self.transform(black)
            return black, label


def build_dataloaders(manifest_path, class_to_idx, label_field='normalized_class',
                      batch_size=None, num_workers=None, max_samples_per_class=None):
    """
    Build train/val/test DataLoaders from a manifest CSV.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    bs = batch_size or cfg.BATCH_SIZE
    nw = num_workers if num_workers is not None else cfg.NUM_WORKERS

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = ManifestDataset(
            manifest_path, split, class_to_idx,
            label_field=label_field,
            max_samples_per_class=max_samples_per_class
        )
        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=cfg.PIN_MEMORY if hasattr(cfg, 'PIN_MEMORY') else torch.cuda.is_available(),
        )
        print(f'  {split:<6} DataLoader: {len(ds):>6} samples, {len(loaders[split]):>4} batches', flush=True)

    return loaders



def load_class_names_from_manifest(manifest_path, label_field='normalized_class'):
    """
    Derive the sorted list of unique class names from a manifest CSV.
    This ensures class_to_idx is consistent with what the manifest contains.
    """
    manifest_path = Path(manifest_path)
    classes = set()
    with open(manifest_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.add(row[label_field])
    sorted_classes = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    return sorted_classes, class_to_idx
