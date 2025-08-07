# File: misogyny_detection_api/scripts/clean_final_labels.py

import pandas as pd
from misogyny_detection_api.config import FINAL_DATASET_PATH, DATA_DIR

# Define output path
CLEANED_OUTPUT_PATH = DATA_DIR / "final_labels_cleaned.csv"

# === Load raw final dataset ===
print("🔹 Loading original dataset...")
df = pd.read_csv(FINAL_DATASET_PATH)

# === Clean and standardize ===
df["body"] = df["body"].astype(str).str.strip()
df["level_1"] = df["level_1"].astype(str).str.strip().str.lower()

df["misogynistic_binary_label"] = df["level_1"].apply(
    lambda x: 0 if x == "nonmisogynistic" else 1
)

df_cleaned = df[["body", "level_1", "misogynistic_binary_label"]]

# === Save cleaned file ===
df_cleaned.to_csv(CLEANED_OUTPUT_PATH, index=False)
print(f"✅ Cleaned dataset saved to: {CLEANED_OUTPUT_PATH}")
print(f"📊 Total samples: {len(df_cleaned)}")
