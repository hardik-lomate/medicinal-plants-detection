# PlantScope — Botanical Computer Vision & Medicinal Plant Identification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![EfficientNet-B0](https://img.shields.io/badge/Model-EfficientNet--B0-brightgreen.svg)](https://arxiv.org/abs/1905.11946)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-accuracy, production-grade botanical computer vision and pharmacological monograph retrieval system. Built with multi-stage PyTorch EfficientNet-B0 transfer learning, Test-Time Augmentation (TTA), post-hoc temperature calibration, feature embedding similarity, and explainable Grad-CAM visual attention mapping.

---

## System Architecture

```text
                                  Real-World Photograph
                                            │
                                            ▼
                             Image Quality Verification
                    (Resolution, Blur, Exposure, Contrast, Entropy)
                                            │
                                            ▼
                             Stage 1: Image-Type Router
                             (EfficientNet-B0: 98.86% Acc)
                                     /             \
                                    /               \
                          [Leaf Specimen]      [Whole Plant]
                                 /                     \
                                ▼                       ▼
                      Stage 2A: Leaf Model    Stage 2B: Whole Plant
                       (78 Species Classes)    (40 Species Classes)
                                \                       /
                                 \                     /
                                  ▼                   ▼
                           Test-Time Augmentation (TTA)
                           & Temperature-Scaled Calibration
                                            │
                                            ▼
                           Deep Feature Centroid Cosine Check
                             (1280-D Manifold Verification)
                                            │
                                            ▼
                             Multi-Signal Decision Engine
                      ('identified' | 'uncertain' | 'unknown' | 'bad_image')
                                            │
                                            ▼
                         Grad-CAM Attention Heatmap Generation
                                            │
                                            ▼
                           SQLite Pharmacological Retrieval
                         (54 Verified Medicinal Monographs)
                                            │
                                            ▼
                       PlantScope Scientific Web Interface
```

---

## Key Features

1. **Multi-Stage Neural Cascade**:
   - **Stage 1 (Router)**: Distinguishes leaf close-ups vs. whole plant / shrub habits (**98.86% test accuracy**).
   - **Stage 2A (Leaf Classifier)**: Identifies **78 botanical leaf species** (**96.53% top-3 accuracy**).
   - **Stage 2B (Whole-Plant Classifier)**: Identifies **40 whole plant species** (**98.59% top-3 accuracy**).
2. **Test-Time Augmentation (TTA)**:
   - Evaluates multi-perspective geometric variants (bilateral flip, botanical smart crop, 180° rotation) and aggregates probabilities via weighted voting.
   - Measures **cross-transform prediction stability** (TTA consistency score).
3. **Multi-Signal Out-Of-Distribution (OOD) Gating**:
   - Synthesizes top-1 confidence, candidate separation margin, TTA stability, image quality, and 1280-D feature embedding cosine similarity to reject non-plant objects and arbitrary photographs with **100% rejection rate** on benchmark challenges.
4. **Grad-CAM Visual Attention**:
   - Hooks into the final convolutional layer of EfficientNet-B0 to produce a 2D gradient-weighted attention heatmap showing the exact morphological regions (veins, margins, serrations) driving the classification.
5. **Pharmacological Database Integration**:
   - 54 peer-reviewed medicinal monographs stored in SQLite with evidence-supported uses, traditional Ayurvedic applications, parts utilized, precautions, contraindications, and authoritative literature citations.
6. **Scientific Web Interface (PlantScope)**:
   - Designed with an editorial, scientific aesthetic using **Inter** (sans-serif) and **Lora** (serif italic for binomials).
   - Zero emojis, clean SVG line icons, interactive species catalog, live camera viewfinder, and toast notification system.

---

## Empirical A/B Benchmark Results

Evaluated side-by-side using `training/eval_ab.py`:

| Performance Metric | Method A (Baseline) | Method B (PlantScope Upgraded) | Delta / Impact |
| :--- | :---: | :---: | :--- |
| **Top-1 Accuracy** | 76.00% | **80.00%** | **+4.00%** (TTA multi-view voting) |
| **Top-3 Accuracy** | 88.00% | **96.00%** | **+8.00%** (Candidate ranking) |
| **Average Prediction Confidence** | 61.54% (Overconfident) | **45.50%** (Calibrated) | Temperature scaling ($T=1.25$) |
| **Prediction Stability** | N/A | **98.00%** | Consistent across transforms |
| **OOD / Non-Plant Rejection** | 0.00% (Hallucinated plants) | **100.00%** (Rejected) | Zero false plant predictions |
| **Average Latency per Specimen** | 134.6 ms | **508.8 ms** | Fast 3-pass CPU inference |

---

## Project Structure

```text
medicinal-plants-detection/
├── backend/
│   ├── app.py                  # Flask REST API server
│   ├── model.py                # Backward-compatible model interface
│   ├── database.py             # SQLite database operations
│   ├── plant_data.py           # 54 verified pharmacological monographs
│   └── plants.db               # SQLite database file
├── frontend/
│   ├── index.html              # PlantScope application UI
│   ├── style.css               # Design system & CSS tokens
│   └── script.js               # Client application logic & SVG icon library
├── inference/                  # Modular production inference package
│   ├── __init__.py             # Unified exports
│   ├── preprocessing.py        # Centralized image transforms & sanitization
│   ├── quality.py              # Sharpness, exposure, contrast, entropy checker
│   ├── crop.py                 # Botanical foreground localization & smart cropping
│   ├── tta.py                  # Test-time augmentation & stability engine
│   ├── calibration.py          # Post-hoc temperature calibration & margin analysis
│   ├── embedding.py            # 1280-D feature extraction & centroid cosine similarity
│   ├── ood.py                  # Multi-signal uncertainty & OOD rejection engine
│   ├── explainability.py       # Native Grad-CAM visual attention heatmaps
│   ├── engine.py               # Central inference engine for Flask and CLI
│   └── build_centroids.py      # Empirical class centroid builder
├── models/
│   ├── image_type_model.pth    # Stage 1 router checkpoint (15.6 MB)
│   ├── leaf_model.pth          # Stage 2A leaf classifier checkpoint (16.0 MB)
│   ├── whole_plant_model.pth   # Stage 2B whole plant checkpoint (15.8 MB)
│   ├── class_mappings/         # Class JSONs and precomputed centroids
│   └── evaluation_results/     # Performance metrics and confusion data
├── training/
│   ├── predict.py              # CLI prediction runner
│   ├── test_real_world.py      # 19-point debug inspection runner
│   ├── eval_ab.py              # Quantitative A/B benchmark evaluation script
│   ├── model_factory.py        # Swappable backbone factory
│   └── manifest_dataset.py     # High-throughput manifest DataLoader
├── config.py                   # Central configuration & paths
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git exclusions (19GB dataset excluded)
```

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/medicinal-plants-detection.git
cd medicinal-plants-detection
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
python backend/app.py
```
Open **`http://localhost:5000`** in your browser to access the PlantScope interface.

### 3. Run Command-Line Prediction

```bash
# Standard prediction
python training/predict.py "path/to/leaf.jpg"

# Comprehensive 19-point diagnostic inspection
python training/test_real_world.py "path/to/leaf.jpg" --debug
```

### 4. Run the Quantitative A/B Benchmark

```bash
python training/eval_ab.py --samples 30 --tta fast
```

---

## REST API Specification

### `POST /predict`
Accepts a multipart form upload containing an image file.

**Request:**
- `image`: Image file (JPEG, PNG, WebP)
- `input_type` *(optional)*: `leaf` | `whole_plant` (override auto-detection)

**Response:**
```json
{
  "success": true,
  "status": "identified",
  "plant_name": "Neem (Indian Lilac)",
  "scientific_name": "Azadirachta indica",
  "input_type": "leaf",
  "input_type_confidence": 0.986,
  "confidence": 0.6037,
  "margin": 0.5122,
  "confidence_tier": "MODERATE",
  "prediction_stability": 1.0,
  "image_quality": 0.895,
  "database_match": true,
  "medicinal_information_available": true,
  "explanation_available": true,
  "explanation_image": "data:image/jpeg;base64,...",
  "top_predictions": [
    { "name": "Neem (Indian Lilac)", "confidence": 0.6037 },
    { "name": "Thumbe (Dronapushpi)", "confidence": 0.0915 },
    { "name": "Eucalyptus (Nilgiri)", "confidence": 0.0432 }
  ],
  "medicinal_information": {
    "medicinal_uses": [...],
    "traditional_uses": [...],
    "parts_used": [...],
    "precautions": [...],
    "sources": [...]
  }
}
```

---

## Advisory & Safety Disclaimer

This system is developed strictly for **botanical education and research purposes**. Automated computer-vision predictions are probabilistic and must never be used as a substitute for professional clinical advice, medical diagnosis, or unsupervised foraging.

---

## License

This project is licensed under the MIT License.
