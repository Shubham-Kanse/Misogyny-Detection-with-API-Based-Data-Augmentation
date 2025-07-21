# services/augmenter.py

import os
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from misogyny_detection_api.config import LOW_CONF_LOG, DATA_DIR

SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_data.csv"

def generate_variants(text):
    """
    Stub augmentation logic – you can replace with LLM or TF-IDF later.
    """
    return [
        f"{text} (rephrased mildly)",
        f"{text} (paraphrased differently)"
    ]

def augment_inputs():
    """
    Reads low-confidence samples and generates augmented variants.
    Saves them to synthetic_data.csv.
    Returns number of new samples added.
    """
    if not LOW_CONF_LOG.exists():
        raise FileNotFoundError("No low-confidence log found.")

    # Load and extract unique texts
    df = pd.read_csv(LOW_CONF_LOG)
    unique_texts = df["text"].dropna().drop_duplicates().tolist()

    # Generate synthetic data
    synthetic_rows = []
    for original_text in unique_texts:
        for variant in generate_variants(original_text):
            synthetic_rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "original": original_text,
                "augmented": variant,
                "label": 1  # Because we're reinforcing misogynistic patterns
            })

    # Write to synthetic_data.csv
    if SYNTHETIC_DATA_PATH.exists():
        existing_df = pd.read_csv(SYNTHETIC_DATA_PATH)
        combined_df = pd.concat([existing_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        combined_df = pd.DataFrame(synthetic_rows)

    combined_df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    return len(synthetic_rows)
