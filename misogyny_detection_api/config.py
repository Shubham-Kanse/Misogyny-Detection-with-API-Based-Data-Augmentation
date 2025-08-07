# File: misogyny_detection_api/services/config.py

from pathlib import Path

# === Base Paths ===
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent / "misogyny_detection_api"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# === File Paths ===
FINAL_DATASET_PATH = DATA_DIR / "final_labels.csv"
AUGMENTED_DATASET_PATH = PROJECT_ROOT / "data" / "augmented"
MODEL_PATH = MODEL_DIR / "misogyny_model.pt"
LOW_CONF_LOG = DATA_DIR / "low_confidence_log.csv"
MODEL_LOG_PATH = MODEL_DIR / "model_performance_log.csv"

# === Retraining Lock File ===
RETRAIN_LOCK_PATH = MODEL_DIR / "retrain.lock"

# === Model Configuration ===
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128
NUM_LABELS = 2

# === Training Hyperparameters ===
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
EPOCHS = 3

# === Thresholds and Metadata ===
CONFIDENCE_THRESHOLD = 0.75
VERSION = "v0.1"

# === Augmentation Trigger Config ===
AUGMENT_TRIGGER_THRESHOLD = 1
AUGMENT_ENDPOINT = "http://localhost:8000/augment"

# === Augmented Data Output ===
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_data.csv"

# === Augmentation Technique Toggles ===
USE_PROMPT_BASED = True
USE_SYNONYM_REPLACEMENT = True
USE_RANDOM_NOISE = True
USE_LABEL_VERIFICATION = True
