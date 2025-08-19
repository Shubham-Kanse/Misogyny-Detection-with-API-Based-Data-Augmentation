# File: services/augmentor.py

import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional, Dict

from misogyny_detection_api.config import (
    LOW_CONF_LOG, DATA_DIR, SYNTHETIC_DATA_PATH,
    AUGMENTED_DATASET_PATH, MODEL_DIR,
    USE_PROMPT_BASED, USE_SYNONYM_REPLACEMENT,
    USE_RANDOM_NOISE, USE_LABEL_VERIFICATION, RETRAIN_ENDPOINT
)
from misogyny_detection_api.services.llm_prompting import classify_label, generate_prompt_variants
from misogyny_detection_api.services.augment_strategies import apply_synonym_replacement, apply_typo_noise

MODEL_LOG_PATH = MODEL_DIR / "model_performance_log.csv"


def _augment_one(original_text: str, force_label: Optional[int] = None) -> List[Dict]:
    """
    Generate up to 5 variants for a single input text using LLM, synonym replacement, and noise.
    Includes clear logging for each augmentation.
    """
    rows: List[Dict] = []

    # --- Label verification / forcing ---
    if force_label in (0, 1):
        label = force_label
        rationale = "Forced label"
    else:
        label = 1  # default assumption
        rationale = "Default assumption"
        if USE_LABEL_VERIFICATION:
            try:
                label, rationale = classify_label(original_text)
            except Exception as e:
                print(f"[WARN] Label verification failed, defaulting to {label}: {e}")

    label_text = "misogynistic" if label == 1 else "nonmisogynistic"

    # --- Logging header for this input ---
    print("\n" + "=" * 80)
    print(f"[INPUT] {original_text}")
    print(f"[LABEL] {label_text} → {rationale}")

    # 1) Original input
    rows.append({
        "body": original_text,
        "level_1": label_text,
        "misogynistic_binary_label": label,
        "technique": "triggering"
    })

    # 2) Prompt-based LLM variants
    if USE_PROMPT_BASED and len(rows) < 5:
        try:
            prompt_variants = generate_prompt_variants(original_text, max_variants=2) or []
            for v in prompt_variants[:2]:
                rows.append({
                    "body": v,
                    "level_1": label_text,
                    "misogynistic_binary_label": label,
                    "technique": "prompt_llm"
                })
                print(f"   + [prompt_llm] {v}")
                if len(rows) >= 5:
                    break
        except Exception as e:
            print(f"[ERROR] Prompt variant generation failed: {e}")

    # 3) Synonym replacement
    if USE_SYNONYM_REPLACEMENT and len(rows) < 5:
        try:
            synonym_variants = apply_synonym_replacement(original_text, max_replacements=2) or []
            if synonym_variants:
                rows.append({
                    "body": synonym_variants[0],
                    "level_1": label_text,
                    "misogynistic_binary_label": label,
                    "technique": "synonym"
                })
                print(f"   + [synonym] {synonym_variants[0]}")
        except Exception as e:
            print(f"[ERROR] Synonym replacement failed: {e}")

    # 4) Noise injection
    if USE_RANDOM_NOISE and len(rows) < 5:
        try:
            noise_variants = apply_typo_noise(original_text, num_typos=2) or []
            if noise_variants:
                rows.append({
                    "body": noise_variants[0],
                    "level_1": label_text,
                    "misogynistic_binary_label": label,
                    "technique": "noise"
                })
                print(f"   + [noise] {noise_variants[0]}")
        except Exception as e:
            print(f"[ERROR] Noise injection failed: {e}")

    return rows[:5]


def generate_augmented_samples_for_text(text: str, force_label: Optional[int] = None) -> List[Dict]:
    """Preview API: generate augmented samples for a single text (no file writes, no retrain)."""
    if not text or not isinstance(text, str):
        raise ValueError("A non-empty 'text' string is required.")
    return _augment_one(text, force_label=force_label)


def augment_inputs():
    """
    Batch pipeline:
    - Reads LOW_CONF_LOG
    - Cleans duplicate headers and normalizes text
    - Generates synthetic data for unique texts
    - Saves synthetic_data.csv
    - Appends into versioned augmented_dataset_vN.csv
    - Triggers /retrain
    - Cleans logs and temp files
    """
    if not LOW_CONF_LOG.exists():
        raise FileNotFoundError("No low-confidence log found.")

    # Read CSV safely
    df = pd.read_csv(LOW_CONF_LOG, on_bad_lines="skip")

    # Drop duplicate headers
    if "text" not in df.columns:
        raise ValueError("Low-confidence log missing 'text' column.")

    df = df[df["text"].str.lower() != "text"]

    if df.empty:
        raise ValueError("Low-confidence log is empty after cleaning.")

    # Normalize + deduplicate
    unique_texts = (
        df["text"]
        .dropna()
        .apply(lambda x: str(x).strip())
        .drop_duplicates()
        .tolist()
    )

    print(f"\n🚀 Starting augmentation for {len(unique_texts)} unique sentences...\n")
    augmented_rows = []

    for i, original_text in enumerate(unique_texts, start=1):
        print(f"\n[PROCESSING] {i}/{len(unique_texts)}")
        rows = _augment_one(original_text)
        augmented_rows.extend(rows)

    if not augmented_rows:
        print("❌ No augmentations created.")
        return 0

    # --- Save synthetic dataset ---
    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"[✔] Synthetic data saved to: {SYNTHETIC_DATA_PATH}")

    # --- Versioned dataset append ---
    try:
        existing_versions = list(AUGMENTED_DATASET_PATH.glob("augmented_dataset_v*.csv"))
        existing_versions.sort()
        if existing_versions:
            latest_file = existing_versions[-1]
            df_combined = pd.read_csv(latest_file)
            print(f"📎 Appending to existing: {latest_file.name}")
        else:
            print("📁 No existing augmented version found. Creating fresh one...")
            cleaned_path = DATA_DIR / "final_labels_cleaned.csv"
            df_combined = pd.read_csv(cleaned_path)
            df_combined["technique"] = "original"
            df_combined = df_combined[["body", "level_1", "misogynistic_binary_label", "technique"]]

        final_df = pd.concat([df_combined, df_augmented], ignore_index=True)
        next_version = len(existing_versions) + 1
        versioned_file = AUGMENTED_DATASET_PATH / f"augmented_dataset_v{next_version}.csv"
        final_df.to_csv(versioned_file, index=False)

        print(f"✅ Augmented dataset saved as: {versioned_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to generate versioned dataset: {e}")
        return 0

    # --- Trigger retrain ---
    try:
        print("🔁 Calling retrain API...")
        response = requests.post(RETRAIN_ENDPOINT)
        if response.status_code == 200:
            print("✅ Retrain API triggered successfully.")
        else:
            print(f"[ERROR] Retrain API failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[ERROR] Failed to call retrain API: {e}")

    # --- Cleanup ---
    try:
        LOW_CONF_LOG.unlink()
        print("🧹 Cleared low_confidence_log.csv for next cycle.")
    except Exception as e:
        print(f"[WARN] Could not clear low_confidence_log.csv: {e}")

    try:
        SYNTHETIC_DATA_PATH.unlink()
        print("🧹 Cleared synthetic_data.csv for next cycle.")
    except Exception as e:
        print(f"[WARN] Could not clear synthetic_data.csv: {e}")

    print(f"\n✅ Augmentation completed: {len(unique_texts)} sentences → {len(augmented_rows)} synthetic samples")
    return len(augmented_rows)
