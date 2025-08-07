# File: services/retrainer.py

import subprocess
from pathlib import Path

RETRAIN_SCRIPT_PATH = Path("misogyny_detection_api/retraining_pipeline/train_model.py")

def run_retraining() -> dict:
    """
    Runs the retraining pipeline script as a subprocess and returns the result.
    """
    if not RETRAIN_SCRIPT_PATH.exists():
        return {"success": False, "error": "train_model.py not found."}

    try:
        result = subprocess.run(
            ["python", str(RETRAIN_SCRIPT_PATH)],
            check=True,
            capture_output=True,
            text=True
        )
        return {
            "success": True,
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr
        }
