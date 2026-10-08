"""
training/test_ood_rejection.py
================================
Verifies the OOD rejection pipeline:
- Plant images must produce status 'identified' or 'identified_no_database_info'
- Non-plant images must produce status 'unknown'

Run: python training/test_ood_rejection.py
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reload engine fresh
import inference.engine as engine_module
engine_module._LOADED_MODELS = {}  # reset cache

from inference.engine import predict_baseline

print("=" * 70)
print("OOD REJECTION VERIFICATION TEST")
print("=" * 70)

passed = 0
failed = 0
results = []


def test_image(img_or_path, label, expected_status_set):
    """Run predict_baseline and check if status is in expected_status_set."""
    global passed, failed
    if isinstance(img_or_path, str):
        result = predict_baseline(img_or_path, debug=False)
    else:
        result = predict_baseline(img_or_path, debug=False)

    status = result.get("status")
    confidence = result.get("confidence", 0)
    plant_name = result.get("plant_name")
    entropy = result.get("normalized_entropy", 0)

    ok = status in expected_status_set

    # Critical check: if status is unknown, plant_name MUST be None
    if status == "unknown" and plant_name is not None:
        ok = False
        label += " [CRITICAL: plant_name not None on unknown!]"

    # Critical check: if status is identified*, plant_name MUST be set
    if status in ("identified", "identified_no_database_info") and plant_name is None:
        ok = False
        label += " [CRITICAL: plant_name is None on identified!]"

    symbol = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    print(f"[{symbol}] {label}")
    print(f"       status={status}, conf={confidence:.3f}, entropy={entropy:.3f}, plant={plant_name}")
    if not ok:
        print(f"       EXPECTED one of: {expected_status_set}")
    print()

    return ok, status, confidence


print("\n--- REAL PLANT IMAGES (must identify, never return 'unknown') ---\n")
rw = PROJECT_ROOT / 'real_world_test'
plant_statuses = []
for d in sorted(rw.iterdir()):
    if d.is_dir() and d.name not in ['leaf', 'whole_plant', 'negatives', 'positives']:
        imgs = list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) + list(d.glob('*.png'))
        for img_path in imgs[:3]:
            ok, status, conf = test_image(
                str(img_path),
                f"{d.name}/{img_path.name[:30]}",
                {"identified", "identified_no_database_info"}
            )
            plant_statuses.append((d.name, status, conf, ok))

print("\n--- SYNTHETIC OOD IMAGES (must return 'unknown') ---\n")
ood_images = []

# Solid colors
for name, color in [("Solid blue", (80, 120, 200)), ("Solid red", (200, 50, 50)),
                    ("Solid white", (255, 255, 255)), ("Solid black", (0, 0, 0)),
                    ("Solid gray", (128, 128, 128))]:
    ood_images.append((name, Image.new('RGB', (300, 300), color=color)))

# Random noise
noise = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
ood_images.append(("Random noise", Image.fromarray(noise)))

# Gradients and patterns
grad = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    grad[i, :, 0] = int(255 * i / 300)
    grad[:, i, 1] = int(255 * i / 300)
ood_images.append(("Color gradient", Image.fromarray(grad)))

stripe = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    stripe[i, :] = [255, 0, 0] if (i // 30) % 2 == 0 else [0, 0, 255]
ood_images.append(("Stripe pattern", Image.fromarray(stripe)))

# Text/map
text_img = Image.new('RGB', (400, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(text_img)
draw.text((20, 50), "INDIA", fill=(0, 0, 0))
draw.text((20, 100), "Country Map", fill=(50, 50, 50))
draw.text((20, 150), "Geographic Data", fill=(30, 30, 30))
ood_images.append(("Text / map", text_img))

# Logo-like
logo = Image.new('RGB', (300, 300), color=(255, 255, 255))
draw2 = ImageDraw.Draw(logo)
draw2.ellipse([50, 50, 250, 250], fill=(0, 80, 200), outline=(0, 0, 0), width=5)
draw2.rectangle([100, 100, 200, 200], fill=(255, 255, 255))
ood_images.append(("Logo circle", logo))

# Checkerboard
check = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    for j in range(300):
        check[i, j] = [255, 255, 255] if (i // 30 + j // 30) % 2 == 0 else [0, 0, 0]
ood_images.append(("Checkerboard", Image.fromarray(check.astype(np.uint8))))

ood_statuses = []
for name, img in ood_images:
    ok, status, conf = test_image(img, name, {"unknown"})
    ood_statuses.append((name, status, conf, ok))

print("=" * 70)
print("SUMMARY")
print("=" * 70)
total_plant = len([s for s in plant_statuses])
total_ood = len([s for s in ood_statuses])
plant_correct = sum(1 for _, _, _, ok in plant_statuses if ok)
ood_correct = sum(1 for _, _, _, ok in ood_statuses if ok)

print(f"Plant images:  {plant_correct}/{total_plant} correctly identified ({100*plant_correct/max(total_plant,1):.0f}%)")
print(f"OOD images:    {ood_correct}/{total_ood} correctly rejected ({100*ood_correct/max(total_ood,1):.0f}%)")
print(f"Total PASS:    {passed}/{passed+failed}")
print(f"Total FAIL:    {failed}/{passed+failed}")
print()
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} TESTS FAILED")
    plant_fails = [(n, s, c) for n, s, c, ok in plant_statuses if not ok]
    ood_fails = [(n, s, c) for n, s, c, ok in ood_statuses if not ok]
    if plant_fails:
        print("\nFailed plant images (false rejects):")
        for n, s, c in plant_fails:
            print(f"  {n}: status={s}, conf={c:.3f}")
    if ood_fails:
        print("\nFailed OOD images (false accepts):")
        for n, s, c in ood_fails:
            print(f"  {n}: status={s}, conf={c:.3f}")
