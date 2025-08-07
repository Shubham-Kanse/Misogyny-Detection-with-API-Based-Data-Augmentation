# File: services/retrainer.py

import os
import subprocess
from datetime import datetime
from misogyny_detection_api.config import AUGMENTED_DATASET_PATH, RETRAIN_LOCK_PATH
from pathlib import Path

def get_latest_augmented_dataset() -> Path:
    all_versions = list(AUGMENTED_DATASET_PATH.glob("augmented_dataset_v*.csv"))
    if not all_versions:
        raise FileNotFoundError("No augmented datasets found.")
    latest = sorted(all_versions, key=lambda x: int(x.stem.split('_v')[-1]))[-1]
    return latest

def trigger_retraining():
    if RETRAIN_LOCK_PATH.exists():
        raise RuntimeError("Retraining already in progress.")

    try:
        # Lock
        with open(RETRAIN_LOCK_PATH, "w") as f:
            f.write(f"Started at: {datetime.now().isoformat()}")

        latest_dataset = get_latest_augmented_dataset()
        env = os.environ.copy()
        env["AUGMENTED_VERSION"] = latest_dataset.stem.split("_v")[-1]

        print("🚀 Starting training subprocess...")
        subprocess.Popen([
            "python", "-m", "misogyny_detection_api.retraining_pipeline.train_model"
        ], env=env)

    finally:
        # Do NOT remove the lock here! It will be removed at the end of training
        pass
