"""
config.py — Central Project Configuration
==========================================
Single source of truth for all paths, dataset locations, model paths,
and training hyperparameters. Every training and backend script imports from here.

To point the project at a different dataset location, change DATASET_ROOT below,
or set the environment variable:

    set MEDICINAL_DATASET_ROOT=C:\\path\\to\\your\\dataset

Usage:
    from config import ProjectConfig as cfg
    print(cfg.DATASET_ROOT)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent

_default_dataset_root = Path(r"C:\Users\hardi\Downloads\medicinal plants detection\dataset\Indian Medicinal Leaves Image Datasets")

DATASET_ROOT = Path(
    os.environ.get("MEDICINAL_DATASET_ROOT", str(_default_dataset_root))
)

LEAF_DATASET_SUBFOLDER = "Datasets_Leaf"
PLANT_DATASET_SUBFOLDER = "Datasets_Plant"


class ProjectConfig:
    DATASET_ROOT: Path = DATASET_ROOT
    LEAF_DATASET_DIR: Path = DATASET_ROOT / LEAF_DATASET_SUBFOLDER
    PLANT_DATASET_DIR: Path = DATASET_ROOT / PLANT_DATASET_SUBFOLDER

    MODEL_DIR: Path = PROJECT_ROOT / "models"
    ARCHIVE_DIR: Path = MODEL_DIR / "archive"

    LEGACY_MODEL_PATH: Path = MODEL_DIR / "medicinal_plant_model.pth"
    LEAF_MODEL_PATH: Path = MODEL_DIR / "leaf_model.pth"
    PLANT_MODEL_PATH: Path = MODEL_DIR / "whole_plant_model.pth"
    IMAGE_TYPE_MODEL_PATH: Path = MODEL_DIR / "image_type_model.pth"

    CHECKPOINT_DIR: Path = MODEL_DIR / "checkpoints"
    CHECKPOINT_LEAF_DIR: Path = CHECKPOINT_DIR / "leaf"
    CHECKPOINT_PLANT_DIR: Path = CHECKPOINT_DIR / "whole_plant"
    CHECKPOINT_IMAGE_TYPE_DIR: Path = CHECKPOINT_DIR / "image_type"

    CLASS_MAPPING_DIR: Path = MODEL_DIR / "class_mappings"
    LEAF_CLASSES_JSON: Path = CLASS_MAPPING_DIR / "leaf_classes.json"
    PLANT_CLASSES_JSON: Path = CLASS_MAPPING_DIR / "whole_plant_classes.json"
    IMAGE_TYPE_CLASSES_JSON: Path = CLASS_MAPPING_DIR / "image_type_classes.json"
    COMBINED_METADATA_JSON: Path = CLASS_MAPPING_DIR / "combined_plant_metadata.json"

    ANALYSIS_DIR: Path = MODEL_DIR / "dataset_analysis"
    LEAF_MANIFEST_CSV: Path = ANALYSIS_DIR / "leaf_manifest.csv"
    PLANT_MANIFEST_CSV: Path = ANALYSIS_DIR / "whole_plant_manifest.csv"
    IMAGE_TYPE_MANIFEST_CSV: Path = ANALYSIS_DIR / "image_type_manifest.csv"
    DATASET_REPORT_JSON: Path = ANALYSIS_DIR / "dataset_report.json"
    DATASET_REPORT_CSV: Path = ANALYSIS_DIR / "dataset_report.csv"

    EVAL_DIR: Path = MODEL_DIR / "evaluation_results"
    EVAL_LEAF_DIR: Path = EVAL_DIR / "leaf"
    EVAL_PLANT_DIR: Path = EVAL_DIR / "whole_plant"
    EVAL_IMAGE_TYPE_DIR: Path = EVAL_DIR / "image_type"

    REAL_WORLD_DIR: Path = PROJECT_ROOT / "real_world_test"
    REAL_WORLD_LEAF_DIR: Path = REAL_WORLD_DIR / "leaf"
    REAL_WORLD_PLANT_DIR: Path = REAL_WORLD_DIR / "whole_plant"

    IMAGE_SIZE: int = 224
    RESIZE_EDGE: int = 256
    MEAN: list = [0.485, 0.456, 0.406]
    STD: list = [0.229, 0.224, 0.225]

    IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0
    SEED: int = 42
    NUM_EPOCHS: int = 20
    WARMUP_EPOCHS: int = 2
    LEARNING_RATE: float = 1e-3
    LR_BACKBONE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.1
    PATIENCE: int = 5
    LR_PATIENCE: int = 3
    LR_FACTOR: float = 0.5
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    MIN_IMAGES_PER_CLASS: int = 20

    PLANT_NAME_MAPPING_JSON: Path = CLASS_MAPPING_DIR / "plant_name_mapping.json"

    CONFIDENCE_THRESHOLD: float = 0.65
    PREDICTION_THRESHOLD: float = float(os.environ.get("PREDICTION_THRESHOLD", "0.15"))
    MARGIN_THRESHOLD: float = 0.15
    UNKNOWN_THRESHOLD: float = 0.20
    IMAGE_TYPE_CONFIDENCE: float = 0.65

    DEFAULT_BACKBONE: str = "efficientnet_b0"

    LEAF_MODEL_VERSION: str = "leaf_classifier_v3"
    PLANT_MODEL_VERSION: str = "whole_plant_classifier_v3"
    IMAGE_TYPE_MODEL_VERSION: str = "image_type_classifier_v1"

    IMAGE_TYPE_CLASSES: list = ["leaf", "whole_plant"]

    @classmethod
    def get_device(cls):
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def validate_dataset_paths(cls) -> dict:
        result = {}
        for key, path in [("leaf", cls.LEAF_DATASET_DIR), ("plant", cls.PLANT_DATASET_DIR)]:
            exists = path.exists() and path.is_dir()
            result[f"{key}_ok"] = exists
            if not exists:
                result[f"{key}_error"] = f"Directory not found: {path}"
            else:
                result[f"{key}_path"] = str(path)
        return result

    @classmethod
    def ensure_dirs(cls):
        dirs = [
            cls.MODEL_DIR, cls.ARCHIVE_DIR,
            cls.CHECKPOINT_LEAF_DIR, cls.CHECKPOINT_PLANT_DIR, cls.CHECKPOINT_IMAGE_TYPE_DIR,
            cls.CLASS_MAPPING_DIR,
            cls.ANALYSIS_DIR,
            cls.EVAL_LEAF_DIR, cls.EVAL_PLANT_DIR, cls.EVAL_IMAGE_TYPE_DIR,
            cls.REAL_WORLD_LEAF_DIR, cls.REAL_WORLD_PLANT_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = ProjectConfig
    print("=" * 60)
    print("MEDICINAL PLANT DETECTION - PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"PROJECT_ROOT    : {PROJECT_ROOT}")
    print(f"DATASET_ROOT    : {cfg.DATASET_ROOT}")
    print(f"LEAF DATASET    : {cfg.LEAF_DATASET_DIR}")
    print(f"PLANT DATASET   : {cfg.PLANT_DATASET_DIR}")
    print()
    validation = cfg.validate_dataset_paths()
    print("Dataset Validation:")
    for k, v in validation.items():
        print(f"  {k}: {v}")
    print()
    cfg.ensure_dirs()
    print("All output directories ensured.")

    print()
    validation = cfg.validate_dataset_paths()
    print("Dataset Validation:")
    for k, v in validation.items():
        print(f"  {k}: {v}")
    print()
    cfg.ensure_dirs()
    print("All output directories ensured.")
