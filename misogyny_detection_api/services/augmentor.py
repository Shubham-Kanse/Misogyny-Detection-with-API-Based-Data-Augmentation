# File: services/augmentor.py

import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil

from misogyny_detection_api.config import (
    LOW_CONF_LOG, DATA_DIR, SYNTHETIC_DATA_PATH, AUGMENTED_DATASET_PATH,
    USE_PROMPT_BASED, USE_SYNONYM_REPLACEMENT, USE_RANDOM_NOISE, USE_LABEL_VERIFICATION
)
from misogyny_detection_api.services.llm_prompting import classify_label, generate_prompt_variants
from misogyny_detection_api.services.augment_strategies import apply_synonym_replacement, apply_typo_noise

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
        label = 1
        if USE_LABEL_VERIFICATION:
            label, rationale = classify_label(original_text)
            print(f"[LLM LABEL] {original_text} → {label} ({rationale})")

        # === Step 2: Prompt-based LLM variants (2)
        if USE_PROMPT_BASED:
            try:
                prompt_variants = generate_prompt_variants(original_text, max_variants=2)
                for variant in prompt_variants[:2]:
                    augmented_rows.append({
                        "original": original_text,
                        "augmented": variant,
                        "label": label,
                        "technique": "prompt_llm"
                    })
            except Exception as e:
                print(f"[ERROR] Prompt variant generation failed: {e}")

        # === Step 3: Synonym replacement (1)
        if USE_SYNONYM_REPLACEMENT:
            try:
                synonym_variants = apply_synonym_replacement(original_text, max_replacements=2)
                if synonym_variants:
                    augmented_rows.append({
                        "original": original_text,
                        "augmented": synonym_variants[0],
                        "label": label,
                        "technique": "synonym"
                    })
            except Exception as e:
                print(f"[ERROR] Synonym replacement failed: {e}")

        # === Step 4: Noise injection (1)
        if USE_RANDOM_NOISE:
            try:
                noise_variants = apply_typo_noise(original_text, num_typos=2)
                if noise_variants:
                    augmented_rows.append({
                        "original": original_text,
                        "augmented": noise_variants[0],
                        "label": label,
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

    # === Now create versioned augmented dataset ===
    try:
        # Load original cleaned data (baseline)
        cleaned_path = DATA_DIR / "final_labels_cleaned.csv"
        df_cleaned = pd.read_csv(cleaned_path)

        df_cleaned["technique"] = "original"
        df_cleaned = df_cleaned[["body", "level_1", "misogynistic_binary_label", "technique"]]

        # Prepare synthetic data
        df_augmented = pd.read_csv(SYNTHETIC_DATA_PATH)
        df_synth = pd.DataFrame({
            "body": df_augmented["augmented"],
            "level_1": df_augmented["label"].apply(lambda x: "misogynistic" if x == 1 else "nonmisogynistic"),
            "misogynistic_binary_label": df_augmented["label"],
            "technique": df_augmented["technique"]
        })

        # Combine and save
        final_df = pd.concat([df_cleaned, df_synth], ignore_index=True)

        AUGMENTED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
        existing_versions = list(AUGMENTED_DATASET_PATH.glob("augmented_dataset_v*.csv"))
        next_version = len(existing_versions) + 1
        versioned_file = AUGMENTED_DATASET_PATH / f"augmented_dataset_v{next_version}.csv"
        final_df.to_csv(versioned_file, index=False)

        print(f"✅ Augmented dataset saved as: {versioned_file.name}")

    except Exception as e:
        print(f"[ERROR] Failed to generate versioned dataset: {e}")

    # === Clear low_confidence_log ===
    try:
        LOW_CONF_LOG.unlink()
        print("🧹 Cleared low_confidence_log.csv for next cycle.")
    except Exception as e:
        print(f"[WARN] Could not clear low_confidence_log.csv: {e}")

    return len(augmented_rows)
