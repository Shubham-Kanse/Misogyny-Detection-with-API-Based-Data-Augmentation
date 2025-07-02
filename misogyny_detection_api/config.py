# File: misogyny_detection_api/services/config.py

from pathlib import Path

# === Base Paths ===
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent / "misogyny_detection_api"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# === File Paths ===
FINAL_DATASET_PATH = DATA_DIR / "final_labels.csv"
MODEL_PATH = MODEL_DIR / "misogyny_model.pt"

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
