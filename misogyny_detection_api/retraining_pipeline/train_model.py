import pandas as pd
import numpy as np
import torch
import re
import emoji

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from misogyny_detection_api.config import (
    FINAL_DATASET_PATH, MODEL_PATH,
    MODEL_NAME, MAX_SEQ_LEN, NUM_LABELS,
    LEARNING_RATE, BATCH_SIZE, EPOCHS
)

# === 1. Load CSV ===
print("🔹 Loading dataset...")
df = pd.read_csv(FINAL_DATASET_PATH)

TEXT_COLUMN = 'body'
RAW_LABEL_COLUMN = 'level_1'
LABEL_COLUMN = 'misogynistic_binary_label'

# === 2. Clean and convert ===
df[TEXT_COLUMN] = df[TEXT_COLUMN].replace(r'^\s*$', np.nan, regex=True)
df.dropna(subset=[TEXT_COLUMN, RAW_LABEL_COLUMN], inplace=True)
df[LABEL_COLUMN] = df[RAW_LABEL_COLUMN].apply(lambda x: 0 if str(x).strip().lower() == 'nonmisogynistic' else 1).astype(int)

# === 3. Preprocess Text ===
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['processed_text'] = df[TEXT_COLUMN].apply(preprocess_text)

# === 4. Extract TF-IDF keywords ===
def extract_keywords(doc_text, top_n=3):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit([doc_text])
        features = tfidf.get_feature_names_out()
        scores = tfidf.transform([doc_text]).toarray().flatten()
        sorted_indices = scores.argsort()[::-1]
        return [features[i] for i in sorted_indices[:top_n] if scores[i] > 0]
    except:
        return []

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def contextual_expansion(text, keywords):
    if not keywords:
        return text
    return text + " " + " ".join(keywords)

df['keywords'] = df['processed_text'].apply(lambda x: extract_keywords(x, top_n=3))
df['augmented_text'] = df.apply(lambda row: contextual_expansion(row['processed_text'], row['keywords']), axis=1)

# === 5. Train/Val Split (safe handling) ===
use_manual_split = False
if 'split' in df.columns:
    df['split'] = df['split'].astype(str).str.strip().str.lower()
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    if len(train_df) == 0 or len(val_df) == 0:
        print("⚠️ 'split' exists but no data in 'train' or 'val' → falling back to 80/20 split.")
        use_manual_split = True
else:
    use_manual_split = True

if use_manual_split:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN])

print(f"📊 Train samples: {len(train_df)}, Val samples: {len(val_df)}")

train_texts = train_df['augmented_text'].tolist()
train_labels = train_df[LABEL_COLUMN].tolist()

val_texts = val_df['augmented_text'].tolist()
val_labels = val_df[LABEL_COLUMN].tolist()

# === 6. Tokenization ===
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN, return_tensors='pt')

# === 7. Dataset ===
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

# === 8. Model Init ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, EPOCHS * len(train_loader))

# === 9. Train Loop ===
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
    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} completed. Avg Loss: {avg_loss:.4f}")

# === 10. Evaluate ===
def evaluate(model, loader):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
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
    print(f"\n📊 Evaluation:\nAcc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

evaluate(model, val_loader)

# === 11. Save Model ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")
