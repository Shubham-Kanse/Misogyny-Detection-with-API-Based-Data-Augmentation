# File: services/augmentor.py

import os
import pandas as pd
from pathlib import Path
import requests

from misogyny_detection_api.config import (
    LOW_CONF_LOG, DATA_DIR, SYNTHETIC_DATA_PATH,
    AUGMENTED_DATASET_PATH, MODEL_DIR,
    USE_PROMPT_BASED, USE_SYNONYM_REPLACEMENT,
    USE_RANDOM_NOISE, USE_LABEL_VERIFICATION
)
from misogyny_detection_api.services.llm_prompting import classify_label, generate_prompt_variants
from misogyny_detection_api.services.augment_strategies import apply_synonym_replacement, apply_typo_noise

MODEL_LOG_PATH = MODEL_DIR / "model_performance_log.csv"

def augment_inputs():
    if not LOW_CONF_LOG.exists():
        raise FileNotFoundError("No low-confidence log found.")

    df = pd.read_csv(LOW_CONF_LOG)
    if df.empty or "text" not in df.columns:
        raise ValueError("Low-confidence log is empty or malformed.")

    unique_texts = df["text"].dropna().drop_duplicates().tolist()
    augmented_rows = []

    for original_text in unique_texts:
        # === Step 1: Label verification ===
        label = 1  # default
        if USE_LABEL_VERIFICATION:
            label, rationale = classify_label(original_text)
            print(f"[LLM LABEL] {original_text} → {label} ({rationale})")

        # === Step 2: Add original input as 'triggering' row
        augmented_rows.append({
            "body": original_text,
            "level_1": "misogynistic" if label == 1 else "nonmisogynistic",
            "misogynistic_binary_label": label,
            "technique": "triggering"
        })

        # === Step 3: Prompt-based LLM variants (2)
        if USE_PROMPT_BASED:
            try:
                prompt_variants = generate_prompt_variants(original_text, max_variants=2)
                for variant in prompt_variants[:2]:
                    augmented_rows.append({
                        "body": variant,
                        "level_1": "misogynistic" if label == 1 else "nonmisogynistic",
                        "misogynistic_binary_label": label,
                        "technique": "prompt_llm"
                    })
            except Exception as e:
                print(f"[ERROR] Prompt variant generation failed: {e}")

        # === Step 4: Synonym replacement (1)
        if USE_SYNONYM_REPLACEMENT:
            try:
                synonym_variants = apply_synonym_replacement(original_text, max_replacements=2)
                if synonym_variants:
                    augmented_rows.append({
                        "body": synonym_variants[0],
                        "level_1": "misogynistic" if label == 1 else "nonmisogynistic",
                        "misogynistic_binary_label": label,
                        "technique": "synonym"
                    })
            except Exception as e:
                print(f"[ERROR] Synonym replacement failed: {e}")

        # === Step 5: Noise injection (1)
        if USE_RANDOM_NOISE:
            try:
                noise_variants = apply_typo_noise(original_text, num_typos=2)
                if noise_variants:
                    augmented_rows.append({
                        "body": noise_variants[0],
                        "level_1": "misogynistic" if label == 1 else "nonmisogynistic",
                        "misogynistic_binary_label": label,
                        "technique": "noise"
                    })
            except Exception as e:
                print(f"[ERROR] Noise injection failed: {e}")

    # === Save synthetic dataset ===
    if not augmented_rows:
        print("❌ No augmentations created.")
        return 0

    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"[✔] Synthetic data saved to: {SYNTHETIC_DATA_PATH}")

    # === Append synthetic data to latest augmented_dataset_vN.csv ===
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

    # === Trigger retrain API ===
    try:
        print("🔁 Calling retrain API...")
        response = requests.post("http://localhost:8000/retrain")
        if response.status_code == 200:
            print("✅ Retrain API triggered successfully.")
        else:
            print(f"[ERROR] Retrain API failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[ERROR] Failed to call retrain API: {e}")

    # === Clean-up
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

    return len(augmented_rows)
