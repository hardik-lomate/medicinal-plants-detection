"""
training/normalize_classes.py
==============================
Generates class normalization mappings for both datasets:
  models/class_mappings/leaf_classes.json
  models/class_mappings/whole_plant_classes.json
  models/class_mappings/image_type_classes.json
  models/class_mappings/combined_plant_metadata.json

Usage:
    python training/normalize_classes.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig as cfg

# Raw folder name -> normalized display name
NORMALIZATION_MAP = {
    # Leaf dataset
    'Bhrami': 'Bhrami',
    'ashoka': 'Ashoka',
    'Citron lime (herelikai)': 'Citron_Lime',
    'Common rue(naagdalli)': 'Common_Rue',
    'Palak(Spinach)': 'Palak_Spinach',
    # Plant dataset
    'Brahmi': 'Bhrami',
    'Curry_Leaf': 'Curry',
    'Lemon_grass': 'Lemongrass',
    'Gauva': 'Guava',
    'Amruta_Balli': 'Amruthaballi',
    'Doddapatre': 'Doddapatre',
    'Pappaya': 'Papaya',
    'Pomegranate': 'Pomegranate',
    'Nagadali': 'Common_Rue',
    'Tulasi': 'Tulsi',
    'Ashoka': 'Ashoka',
    'Betel_Nut': 'Betel_Nut',
}

# Scientific names for known plants (verified botanical binomials only)
SCIENTIFIC_NAMES = {
    'Aloevera': 'Aloe barbadensis miller',
    'Amla': 'Phyllanthus emblica',
    'Amruthaballi': 'Tinospora cordifolia',
    'Arali': 'Nerium oleander',
    'Ashoka': 'Saraca asoca',
    'Ashwagandha': 'Withania somnifera',
    'Avacado': 'Persea americana',
    'Bamboo': 'Bambusoideae',
    'Basale': 'Basella alba',
    'Beans': 'Phaseolus vulgaris',
    'Betel': 'Piper betle',
    'Betel_Nut': 'Areca catechu',
    'Bhrami': 'Bacopa monnieri',
    'Bringaraja': 'Eclipta prostrata',
    'Castor': 'Ricinus communis',
    'Catharanthus': 'Catharanthus roseus',
    'Citron_Lime': 'Citrus medica',
    'Coffee': 'Coffea arabica',
    'Common_Rue': 'Ruta graveolens',
    'Coriender': 'Coriandrum sativum',
    'Curry': 'Murraya koenigii',
    'Doddapatre': 'Coleus amboinicus',
    'Doddpathre': 'Coleus amboinicus',
    'Drumstick': 'Moringa oleifera',
    'Ekka': 'Calotropis gigantea',
    'Eucalyptus': 'Eucalyptus globulus',
    'Ganike': 'Solanum nigrum',
    'Gasagase': 'Papaver somniferum',
    'Geranium': 'Pelargonium graveolens',
    'Ginger': 'Zingiber officinale',
    'Guava': 'Psidium guajava',
    'Henna': 'Lawsonia inermis',
    'Hibiscus': 'Hibiscus rosa-sinensis',
    'Honge': 'Pongamia pinnata',
    'Insulin': 'Costus igneus',
    'Jackfruit': 'Artocarpus heterophyllus',
    'Jasmine': 'Jasminum sambac',
    'Lemon': 'Citrus limon',
    'Lemongrass': 'Cymbopogon citratus',
    'Mango': 'Mangifera indica',
    'Mint': 'Mentha arvensis',
    'Neem': 'Azadirachta indica',
    'Nooni': 'Morinda citrifolia',
    'Onion': 'Allium cepa',
    'Papaya': 'Carica papaya',
    'Pepper': 'Piper nigrum',
    'Pomegranate': 'Punica granatum',
    'Rose': 'Rosa',
    'Sapota': 'Manilkara zapota',
    'Tamarind': 'Tamarindus indica',
    'Tomato': 'Solanum lycopersicum',
    'Tulsi': 'Ocimum tenuiflorum',
    'Turmeric': 'Curcuma longa',
}

MEDICINAL_PLANTS = {
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Ashoka', 'Ashwagandha',
    'Bhrami', 'Bringaraja', 'Castor', 'Catharanthus', 'Citron_Lime',
    'Common_Rue', 'Coriender', 'Curry', 'Doddapatre', 'Doddpathre', 'Drumstick',
    'Ekka', 'Eucalyptus', 'Ganike', 'Gasagase', 'Ginger', 'Guava',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemongrass', 'Mint', 'Neem', 'Nelavembu', 'Nooni', 'Papaya',
    'Pepper', 'Pomegranate', 'Sapota', 'Tamarind', 'Tulsi', 'Turmeric',
}


def normalize(raw_name):
    return NORMALIZATION_MAP.get(raw_name, raw_name)


def build_class_list(ds_path, dataset_type):
    ds_path = Path(ds_path)
    classes = []
    for folder in sorted(ds_path.iterdir()):
        if not folder.is_dir():
            continue
        raw = folder.name
        norm = normalize(raw)
        entry = {
            'raw_folder_name': raw,
            'normalized_name': norm,
            'scientific_name': SCIENTIFIC_NAMES.get(norm, ''),
            'is_medicinal': norm in MEDICINAL_PLANTS,
            'source_dataset': dataset_type,
        }
        classes.append(entry)
    return classes


def build_combined_metadata(leaf_classes, plant_classes):
    leaf_norm = {c['normalized_name']: c for c in leaf_classes}
    plant_norm = {c['normalized_name']: c for c in plant_classes}
    all_norm = sorted(set(leaf_norm) | set(plant_norm))

    combined = []
    for norm in all_norm:
        leaf_entry = leaf_norm.get(norm)
        plant_entry = plant_norm.get(norm)
        entry = {
            'normalized_name': norm,
            'display_name': norm.replace('_', ' '),
            'scientific_name': SCIENTIFIC_NAMES.get(norm, ''),
            'is_medicinal': norm in MEDICINAL_PLANTS,
            'leaf_available': leaf_entry is not None,
            'leaf_folder': leaf_entry['raw_folder_name'] if leaf_entry else None,
            'whole_plant_available': plant_entry is not None,
            'plant_folder': plant_entry['raw_folder_name'] if plant_entry else None,
            'database_available': False,
        }
        combined.append(entry)
    return combined


def main():
    cfg.ensure_dirs()
    print("=" * 60)
    print("MEDICINAL PLANT DETECTION — CLASS NORMALIZATION")
    print("=" * 60)

    print(f"\nBuilding leaf class list from: {cfg.LEAF_DATASET_DIR}")
    leaf_classes = build_class_list(cfg.LEAF_DATASET_DIR, 'leaf')
    print(f"  Found {len(leaf_classes)} leaf classes.")

    print(f"\nBuilding plant class list from: {cfg.PLANT_DATASET_DIR}")
    plant_classes = build_class_list(cfg.PLANT_DATASET_DIR, 'whole_plant')
    print(f"  Found {len(plant_classes)} plant classes.")

    image_type_classes = [
        {"class_index": 0, "class_name": "leaf", "source": "Datasets_Leaf"},
        {"class_index": 1, "class_name": "whole_plant", "source": "Datasets_Plant"},
    ]

    combined = build_combined_metadata(leaf_classes, plant_classes)
    print(f"\nTotal unique combined species: {len(combined)}")

    leaf_sorted = sorted(set(c['normalized_name'] for c in leaf_classes))
    plant_sorted = sorted(set(c['normalized_name'] for c in plant_classes))

    leaf_out = {
        "generated_at": datetime.now().isoformat(),
        "num_classes": len(leaf_sorted),
        "class_names": leaf_sorted,
        "class_details": leaf_classes,
        "index_to_class": {i: c for i, c in enumerate(leaf_sorted)},
        "class_to_index": {c: i for i, c in enumerate(leaf_sorted)},
    }

    plant_out = {
        "generated_at": datetime.now().isoformat(),
        "num_classes": len(plant_sorted),
        "class_names": plant_sorted,
        "class_details": plant_classes,
        "index_to_class": {i: c for i, c in enumerate(plant_sorted)},
        "class_to_index": {c: i for i, c in enumerate(plant_sorted)},
    }

    image_type_out = {
        "generated_at": datetime.now().isoformat(),
        "num_classes": 2,
        "class_names": cfg.IMAGE_TYPE_CLASSES,
        "class_details": image_type_classes,
        "index_to_class": {i: c for i, c in enumerate(cfg.IMAGE_TYPE_CLASSES)},
        "class_to_index": {c: i for i, c in enumerate(cfg.IMAGE_TYPE_CLASSES)},
    }

    with open(cfg.LEAF_CLASSES_JSON, 'w', encoding='utf-8') as f:
        json.dump(leaf_out, f, indent=2)
    print(f"[OK] Saved: {cfg.LEAF_CLASSES_JSON}")

    with open(cfg.PLANT_CLASSES_JSON, 'w', encoding='utf-8') as f:
        json.dump(plant_out, f, indent=2)
    print(f"[OK] Saved: {cfg.PLANT_CLASSES_JSON}")

    with open(cfg.IMAGE_TYPE_CLASSES_JSON, 'w', encoding='utf-8') as f:
        json.dump(image_type_out, f, indent=2)
    print(f"[OK] Saved: {cfg.IMAGE_TYPE_CLASSES_JSON}")

    with open(cfg.COMBINED_METADATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print(f"[OK] Saved: {cfg.COMBINED_METADATA_JSON}")

    print("\nClass normalization completed successfully.")


if __name__ == '__main__':
    main()
