# File: retraining_pipeline/train_model.py

import os
import re
import emoji
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from misogyny_detection_api.config import (
    AUGMENTED_DATASET_PATH, MODEL_DIR,
    MODEL_NAME, MAX_SEQ_LEN, NUM_LABELS,
    LEARNING_RATE, BATCH_SIZE, EPOCHS
)

# === Step 1: Load the latest augmented dataset ===
dataset_files = sorted(list(AUGMENTED_DATASET_PATH.glob("augmented_dataset_v*.csv")))
if not dataset_files:
    raise FileNotFoundError("No augmented dataset found in path.")
dataset_path = dataset_files[-1]
df = pd.read_csv(dataset_path)
print(f"📂 Using training data: {dataset_path.name}")

# === Step 2: Preprocessing ===
def preprocess(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", str(text).lower())
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

df = df.dropna(subset=["body", "misogynistic_binary_label"])
df["processed_text"] = df["body"].apply(preprocess)

# === Step 3: Contextual TF-IDF Expansion ===
from sklearn.feature_extraction.text import TfidfVectorizer
def extract_keywords(doc_text, top_n=3):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit([doc_text])
        scores = tfidf.transform([doc_text]).toarray().flatten()
        features = tfidf.get_feature_names_out()
        return [features[i] for i in scores.argsort()[::-1][:top_n] if scores[i] > 0]
    except:
        return []

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
df["keywords"] = df["processed_text"].apply(lambda x: extract_keywords(x, top_n=3))
df["augmented_text"] = df.apply(lambda row: row["processed_text"] + " " + " ".join(row["keywords"]), axis=1)

# === Step 4: Train-val split ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["misogynistic_binary_label"])
print(f"📊 Train size: {len(train_df)}, Val size: {len(val_df)}")

train_texts = train_df["augmented_text"].tolist()
train_labels = train_df["misogynistic_binary_label"].tolist()
val_texts = val_df["augmented_text"].tolist()
val_labels = val_df["misogynistic_binary_label"].tolist()

# === Step 5: Tokenization ===
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors='pt')

# === Step 6: Dataset ===
class MisogynyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = MisogynyDataset(train_enc, train_labels)
val_dataset = MisogynyDataset(val_enc, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Step 7: Model setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, EPOCHS * len(train_loader))

# === Step 8: Version management ===
def get_next_version(path: Path, prefix: str, suffix: str) -> str:
    files = list(path.glob(f"{prefix}*{suffix}"))
    nums = [int(f.stem.replace(prefix, "").replace(suffix, "").strip("v")) for f in files if f.stem.replace(prefix, "").replace(suffix, "").strip("v").isdigit()]
    next_v = max(nums) + 1 if nums else 1
    return f"{prefix}v{next_v}{suffix}"

model_name = get_next_version(MODEL_DIR, "misogyny_model_", ".pt")
log_dir = Path("misogyny_detection_api/data/logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / get_next_version(log_dir, "training_log_", ".txt")

# === Step 9: Training ===
print("🚀 Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1}/{EPOCHS} — Avg Loss: {total_loss / len(train_loader):.4f}")

# === Step 10: Evaluation ===
def evaluate(model, loader):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
            trues += labels.cpu().numpy().tolist()
            probs += torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()

    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average='binary')
    try:
        auc = roc_auc_score(trues, probs)
    except:
        auc = 0.0
    return acc, prec, rec, f1, auc

acc, prec, rec, f1, auc = evaluate(model, val_loader)
print(f"📊 Evaluation — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# === Step 11: Save model + log ===
model_path = MODEL_DIR / model_name
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved: {model_path.name}")

with open(log_file, "w") as f:
    f.write(f"Model: {model_path.name}\n")
    f.write(f"Data: {dataset_path.name}\n")
    f.write(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}\n")
print(f"📝 Training log saved: {log_file.name}")
