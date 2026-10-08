import os
import sys
import sqlite3
import json

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.plant_data import PLANT_DATA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'backend', 'plants.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the plants table if it doesn't exist and ensures v3 schema columns."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            common_name TEXT UNIQUE,
            scientific_name TEXT,
            is_medicinal BOOLEAN,
            description TEXT,
            medicinal_uses TEXT,
            traditional_uses TEXT,
            parts_used TEXT,
            precautions TEXT,
            sources TEXT,
            available_in_whole_plant_dataset BOOLEAN DEFAULT 0,
            available_in_leaf_dataset BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()

    # Ensure new columns exist for existing databases
    c.execute("PRAGMA table_info(plants)")
    existing_cols = {row[1] for row in c.fetchall()}
    if 'available_in_whole_plant_dataset' not in existing_cols:
        c.execute('ALTER TABLE plants ADD COLUMN available_in_whole_plant_dataset BOOLEAN DEFAULT 0')
    if 'available_in_leaf_dataset' not in existing_cols:
        c.execute('ALTER TABLE plants ADD COLUMN available_in_leaf_dataset BOOLEAN DEFAULT 0')
    conn.commit()
    conn.close()


def populate_db():
    """Populates the database with plant data from plant_data.py."""
    init_db()
    conn = get_db_connection()
    c = conn.cursor()
    
    for plant in PLANT_DATA:
        try:
            c.execute('''
                INSERT INTO plants (
                    common_name, scientific_name, is_medicinal, description,
                    medicinal_uses, traditional_uses, parts_used, precautions, sources
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plant['common_name'],
                plant['scientific_name'],
                plant['is_medicinal'],
                plant['description'],
                json.dumps(plant.get('medicinal_uses', [])),
                json.dumps(plant.get('traditional_uses', [])),
                json.dumps(plant.get('parts_used', [])),
                json.dumps(plant.get('precautions', [])),
                json.dumps(plant.get('sources', []))
            ))
        except sqlite3.IntegrityError:
            # Plant already exists, update it
            c.execute('''
                UPDATE plants SET
                    scientific_name = ?,
                    is_medicinal = ?,
                    description = ?,
                    medicinal_uses = ?,
                    traditional_uses = ?,
                    parts_used = ?,
                    precautions = ?,
                    sources = ?
                WHERE common_name = ?
            ''', (
                plant['scientific_name'],
                plant['is_medicinal'],
                plant['description'],
                json.dumps(plant.get('medicinal_uses', [])),
                json.dumps(plant.get('traditional_uses', [])),
                json.dumps(plant.get('parts_used', [])),
                json.dumps(plant.get('precautions', [])),
                json.dumps(plant.get('sources', [])),
                plant['common_name']
            ))
            
    conn.commit()
    conn.close()

NAME_ALIASES = {
    "brahmi": "Bhrami",
    "bhrami": "Bhrami",
    "curry leaf": "Curry",
    "curry_leaf": "Curry",
    "curry": "Curry",
    "lemongrass": "Lemongrass",
    "lemon grass": "Lemongrass",
    "lemon_grass": "Lemongrass",
    "tulasi": "Tulsi",
    "tulsi": "Tulsi",
    "aloe vera": "Aloevera",
    "aloe_vera": "Aloevera",
    "aloevera": "Aloevera",
}

_combined_meta_cache = None

def _get_combined_meta():
    global _combined_meta_cache
    if _combined_meta_cache is None:
        meta_path = os.path.join(PROJECT_ROOT, 'models', 'class_mappings', 'combined_plant_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding='utf-8') as f:
                    data = json.load(f)
                _combined_meta_cache = {}
                for entry in data:
                    _combined_meta_cache[entry['normalized_name'].lower()] = entry
                    _combined_meta_cache[entry['display_name'].lower()] = entry
                    if entry.get('leaf_folder'):
                        _combined_meta_cache[entry['leaf_folder'].lower()] = entry
                    if entry.get('plant_folder'):
                        _combined_meta_cache[entry['plant_folder'].lower()] = entry
            except Exception:
                _combined_meta_cache = {}
        else:
            _combined_meta_cache = {}
    return _combined_meta_cache


def get_plant_info(plant_name):
    """
    Retrieves plant information from SQLite database.
    If the plant exists in SQLite, returns verified medicinal details.
    If the plant is in the expanded botanical dataset but lacks verified medicinal entries,
    returns botanical metadata with medicinal_information_available=False.
    """
    if not plant_name:
        return None

    # Normalize name
    clean_name = plant_name.strip()
    lookup_name = NAME_ALIASES.get(clean_name.lower().replace("_", " "), clean_name)
    lookup_name = NAME_ALIASES.get(clean_name.lower(), lookup_name)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM plants WHERE common_name = ? COLLATE NOCASE', (lookup_name,))
    row = c.fetchone()
    
    if not row:
        # Try direct match with clean_name
        c.execute('SELECT * FROM plants WHERE common_name = ? COLLATE NOCASE', (clean_name,))
        row = c.fetchone()

    conn.close()
    
    if row:
        plant_dict = dict(row)
        for field in ['medicinal_uses', 'traditional_uses', 'parts_used', 'precautions', 'sources']:
            if plant_dict.get(field):
                try:
                    plant_dict[field] = json.loads(plant_dict[field])
                except:
                    plant_dict[field] = []
            else:
                plant_dict[field] = []
        plant_dict['medicinal_information_available'] = bool(
            plant_dict.get('is_medicinal') and len(plant_dict.get('medicinal_uses', [])) > 0
        )
        return plant_dict

    # Fallback to combined plant metadata for species in the expanded dataset
    meta_dict = _get_combined_meta()
    lookup_key = clean_name.lower().replace("_", " ").strip()
    norm_key = clean_name.lower().strip()
    meta = meta_dict.get(norm_key) or meta_dict.get(lookup_key)

    if meta:
        return {
            "common_name": meta.get("display_name", clean_name),
            "scientific_name": meta.get("scientific_name", ""),
            "is_medicinal": meta.get("is_medicinal", False),
            "description": "Plant identified, but no verified medicinal information is available in our database.",
            "medicinal_uses": [],
            "traditional_uses": [],
            "parts_used": [],
            "precautions": [],
            "sources": [],
            "available_in_whole_plant_dataset": meta.get("whole_plant_available", False),
            "available_in_leaf_dataset": meta.get("leaf_available", False),
            "medicinal_information_available": False,
            "message": "Plant identified, but no verified medicinal information is available in our database."
        }

    return None


def get_all_plants():
    """Retrieves all plants from the database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM plants')
    rows = c.fetchall()
    conn.close()
    
    plants = []
    for row in rows:
        plant_dict = dict(row)
        # Parse JSON fields
        for field in ['medicinal_uses', 'traditional_uses', 'parts_used', 'precautions', 'sources']:
            if plant_dict.get(field):
                try:
                    plant_dict[field] = json.loads(plant_dict[field])
                except:
                    plant_dict[field] = []
        plants.append(plant_dict)
    return plants

# Initialize and populate DB on module load if DB doesn't exist
if not os.path.exists(DB_PATH):
    populate_db()
