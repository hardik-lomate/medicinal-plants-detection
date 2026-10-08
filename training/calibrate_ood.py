"""
training/calibrate_ood.py
==========================
Calibrate OOD threshold by measuring energy score, entropy, and max-softmax
across real plant images (positives) and synthetic non-plant images (negatives).

Run: python training/calibrate_ood.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.engine import get_loaded_models
from inference.preprocessing import load_image_safely, preprocess_for_inference

models = get_loaded_models()


def measure_signals(img, model, n_classes):
    tensor = preprocess_for_inference(img)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)[0]
        probs = torch.softmax(logits, dim=0)
        energy = torch.logsumexp(logits, dim=0).item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_prob = probs.max().item()
        norm_entropy = entropy / np.log(n_classes)
    return max_prob, norm_entropy, energy


leaf_model = models['leaf']['model']
leaf_classes = models['leaf']['class_names']
n_leaf = len(leaf_classes)

print("=== PLANT IMAGES (should NOT be rejected) ===")
rw = PROJECT_ROOT / 'real_world_test'
plant_max_probs = []
for d in sorted(rw.iterdir()):
    if d.is_dir() and d.name not in ['leaf', 'whole_plant', 'negatives', 'positives']:
        imgs = list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) + list(d.glob('*.png'))
        for img_path in imgs[:5]:
            ok, img, _, _ = load_image_safely(str(img_path))
            if ok:
                max_p, norm_e, energy = measure_signals(img, leaf_model, n_leaf)
                plant_max_probs.append(max_p)
                print(f"  {d.name}/{img_path.name[:25]:25s}: max_softmax={max_p:.4f}, norm_ent={norm_e:.4f}, energy={energy:.3f}")

print()
print("=== SYNTHETIC OOD IMAGES (should be rejected) ===")
ood_max_probs = []

synthetics = [
    ("Solid blue", Image.new('RGB', (300, 300), color=(80, 120, 200))),
    ("Solid red", Image.new('RGB', (300, 300), color=(200, 50, 50))),
    ("Solid green", Image.new('RGB', (300, 300), color=(50, 200, 50))),
    ("White", Image.new('RGB', (300, 300), color=(255, 255, 255))),
    ("Black", Image.new('RGB', (300, 300), color=(0, 0, 0))),
    ("Gray", Image.new('RGB', (300, 300), color=(128, 128, 128))),
]

# Random noise
noise = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
synthetics.append(("Random noise", Image.fromarray(noise)))

# Color gradients
grad = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    grad[i, :, 0] = int(255 * i / 300)
    grad[:, i, 1] = int(255 * i / 300)
synthetics.append(("Color gradient", Image.fromarray(grad)))

# Horizontal stripes
stripe = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    stripe[i, :] = [255, 0, 0] if (i // 30) % 2 == 0 else [0, 0, 255]
synthetics.append(("Stripe pattern", Image.fromarray(stripe)))

# Checkerboard
check = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    for j in range(300):
        check[i, j] = [255, 255, 255] if (i // 30 + j // 30) % 2 == 0 else [0, 0, 0]
synthetics.append(("Checkerboard", Image.fromarray(check.astype(np.uint8))))

# Text on white
text_img = Image.new('RGB', (400, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(text_img)
draw.text((20, 50), "INDIA", fill=(0, 0, 0))
draw.text((20, 100), "Country Map", fill=(50, 50, 50))
draw.text((20, 150), "Geographic Area", fill=(30, 30, 30))
synthetics.append(("Text/map text", text_img))

# Logo-like shapes
logo = Image.new('RGB', (300, 300), color=(255, 255, 255))
draw2 = ImageDraw.Draw(logo)
draw2.ellipse([50, 50, 250, 250], fill=(0, 80, 200), outline=(0, 0, 0), width=5)
draw2.rectangle([100, 100, 200, 200], fill=(255, 255, 255))
synthetics.append(("Logo circle", logo))

for name, img in synthetics:
    max_p, norm_e, energy = measure_signals(img, leaf_model, n_leaf)
    ood_max_probs.append(max_p)
    print(f"  {name:25s}: max_softmax={max_p:.4f}, norm_ent={norm_e:.4f}, energy={energy:.3f}")

print()
print("=== THRESHOLD ANALYSIS ===")
if plant_max_probs:
    print(f"Plant images  — min max_softmax: {min(plant_max_probs):.4f}, mean: {np.mean(plant_max_probs):.4f}")
if ood_max_probs:
    print(f"OOD images    — max max_softmax: {max(ood_max_probs):.4f}, mean: {np.mean(ood_max_probs):.4f}")

# Find safe threshold: max(ood) + margin below min(plant)
if plant_max_probs and ood_max_probs:
    safe_threshold = (min(plant_max_probs) + max(ood_max_probs)) / 2
    print(f"Suggested threshold (midpoint): {safe_threshold:.4f}")
    print(f"Conservative threshold (max_ood + 10%): {max(ood_max_probs) * 1.1:.4f}")
