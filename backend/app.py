"""
Flask Backend — Medicinal Plant Detection API
===============================================
Main application server with:
- Multi-status response handling ('identified', 'uncertain', 'unknown', 'bad_image')
- Top-K alternatives and margin reporting
- Aspect-ratio & EXIF-safe inference
- Educational safety disclaimers
- Safe botanical and medicinal database lookups

Usage:
    python backend/app.py
"""

import os
import sys
import uuid

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.model import predict, init_model, get_loaded_models
from backend.database import get_plant_info, init_db, populate_db

# App Setup
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

DISCLAIMER = (
    "This identification is for educational and informational purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Never consume or apply any wild plant based solely on an automated identification."
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def serve_frontend():
    """Serve the main frontend page."""
    return app.send_static_file('index.html')


@app.route('/health')
def health_check():
    """Health check endpoint."""
    models = get_loaded_models()
    leaf_classes = models['leaf']['class_names'] if models.get('leaf') else []
    plant_classes = models['whole_plant']['class_names'] if models.get('whole_plant') else []
    legacy_classes = models['legacy']['class_names'] if models.get('legacy') else []

    all_supported = sorted(set(leaf_classes) | set(plant_classes) | set(legacy_classes))
    return jsonify({
        "status": "healthy",
        "models_status": {k: (v is not None) for k, v in models.items()},
        "num_supported_classes": len(all_supported),
        "supported_classes": all_supported[:40],
    }), 200



@app.route('/plants')
def list_supported_plants():
    """Returns catalog of all supported plant species and monograph availability."""
    from backend.database import _get_combined_meta, get_all_plants
    db_plants = {p['common_name'].lower(): p for p in get_all_plants()}
    meta_dict = _get_combined_meta()

    unique_species = []
    seen = set()
    for key, meta in meta_dict.items():
        name = meta.get('display_name', key)
        if name in seen:
            continue
        seen.add(name)
        db_entry = db_plants.get(name.lower())
        unique_species.append({
            "common_name": name,
            "scientific_name": meta.get("scientific_name", ""),
            "is_medicinal": meta.get("is_medicinal", False),
            "has_medicinal_info": bool(db_entry and db_entry.get("medicinal_uses")),
            "leaf_available": meta.get("leaf_available", False),
            "whole_plant_available": meta.get("whole_plant_available", False)
        })
    unique_species.sort(key=lambda x: x['common_name'])
    return jsonify({
        "success": True,
        "total_species": len(unique_species),
        "plants": unique_species
    })


@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Accept image and optional input_type parameter, run inference through
    two-stage quality-checked model pipeline, and return structured result.
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided. Please select an image."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Please upload a JPG, PNG, WEBP, or BMP image."
        }), 400

    forced_type = request.form.get('input_type', None)
    if forced_type not in ['leaf', 'whole_plant']:
        forced_type = None

    filepath = None
    try:
        # Save temporary file
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run unified prediction pipeline
        result = predict(filepath, forced_type=forced_type)
        status = result.get("status", "error")
        margin = result.get("margin", 0.0)

        # Server-side debug logging
        print(f"[PREDICT] File: {file.filename} | Status: {status} | Top: {result.get('plant_name')} ({result.get('confidence')}) | Margin: {margin}")

        response_payload = dict(result)
        response_payload["success"] = True
        response_payload["disclaimer"] = DISCLAIMER
        response_payload["alternatives"] = result.get("top_predictions", [])
        response_payload["prediction"] = result
        response_payload["uncertain"] = bool(status in ["uncertain", "unknown", "bad_image"])
        response_payload["explanation_available"] = result.get("explanation_available", False)
        response_payload["explanation_image"] = result.get("explanation_image")
        response_payload["medicinal_information"] = {
            "description": result.get("description", ""),
            "medicinal_uses": result.get("medicinal_uses", []),
            "traditional_uses": result.get("traditional_uses", []),
            "parts_used": result.get("parts_used", []),
            "precautions": result.get("precautions", []),
            "sources": result.get("sources", []),
        }

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


if __name__ == '__main__':
    print("=" * 60)
    print("MEDICINAL PLANT DETECTION — SERVER")
    print("=" * 60)

    init_db()
    populate_db()
    init_model()

    print(f"\nServer ready at http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
