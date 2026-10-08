"""Quick API smoke test via requests"""
import requests
from PIL import Image, ImageDraw
from io import BytesIO

API = 'http://localhost:5000/predict'

def post_image(path_or_img, name='test.jpg'):
    if isinstance(path_or_img, str):
        with open(path_or_img, 'rb') as f:
            resp = requests.post(API, files={'image': (name, f, 'image/jpeg')})
    else:
        buf = BytesIO()
        path_or_img.save(buf, format='JPEG')
        buf.seek(0)
        resp = requests.post(API, files={'image': (name, buf, 'image/jpeg')})
    return resp.json()

# 1. Real plant
r = post_image('real_world_test/Neem/angled_0_1006.jpg', 'neem.jpg')
print("Neem:", r['status'], r['plant_name'], r['confidence'])
assert r['status'] in ('identified', 'identified_no_database_info'), f"FAIL: {r['status']}"
assert r['plant_name'] is not None
print("  -> PASS")

# 2. Map/text image (OOD)
img = Image.new('RGB', (400, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((20, 50), 'INDIA', fill=(0, 0, 0))
draw.text((20, 100), 'Country Map', fill=(50, 50, 50))
r2 = post_image(img, 'map.jpg')
print("Map:", r2['status'], r2['plant_name'], r2['confidence'])
assert r2['status'] == 'unknown', f"FAIL: expected unknown, got {r2['status']}"
assert r2['plant_name'] is None, f"FAIL: plant_name={r2['plant_name']}"
print("  -> PASS")

# 3. Solid blue (OOD)
img3 = Image.new('RGB', (300, 300), color=(80, 120, 200))
r3 = post_image(img3, 'blue.jpg')
print("Blue solid:", r3['status'], r3['plant_name'])
assert r3['status'] == 'unknown'
print("  -> PASS")

# 4. Neem via predict CLI
import subprocess, sys
result = subprocess.run(
    [sys.executable, 'training/predict.py', 'real_world_test/Neem/angled_0_1006.jpg'],
    capture_output=True, text=True, cwd='.'
)
output = result.stdout
print("\nCLI test (Neem):")
print(output[:400])

print("\nALL API TESTS PASSED")
