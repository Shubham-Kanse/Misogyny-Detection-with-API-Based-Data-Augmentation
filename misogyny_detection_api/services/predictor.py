# services/predictor.py

import csv
import requests
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from misogyny_detection_api.services.model_loader import load_model, load_tokenizer
from misogyny_detection_api.config import (
    MODEL_NAME, VERSION, CONFIDENCE_THRESHOLD, LOW_CONF_LOG,
    AUGMENT_TRIGGER_THRESHOLD, AUGMENT_ENDPOINT
)

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_text(text: str):
    """
    Predict misogyny using BERT model.
    - Logs low-confidence inputs (no duplicates)
    - Auto-triggers augmentation when log reaches threshold
    """
    # Tokenize input
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Model inference
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    # Format result
    predicted_class = prediction.item()
    label = "Misogynistic" if predicted_class == 1 else "Non-misogynistic"
    is_misogynistic = predicted_class == 1
    confidence_percentage = round(confidence.item() * 100, 2)
    confidence_score_str = f"{confidence_percentage}%"
    sanitized_text = text.strip()

    # === Logging & Auto-Trigger ===
    try:
        LOW_CONF_LOG.parent.mkdir(parents=True, exist_ok=True)

        # Read all existing texts
        existing_texts = set()
        if LOW_CONF_LOG.exists():
            with open(LOW_CONF_LOG, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_texts = {row["text"].strip() for row in reader}

        # Append new entry only if unique and confidence < threshold
        if sanitized_text not in existing_texts and confidence.item() < CONFIDENCE_THRESHOLD:
            with open(LOW_CONF_LOG, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                file_empty = file.tell() == 0
                if file_empty:
                    writer.writerow(["timestamp", "text", "predicted_class", "confidence", "model_version"])
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    sanitized_text,
                    predicted_class,
                    round(confidence.item(), 4),
                    VERSION
                ])
            existing_texts.add(sanitized_text)  # Update the set for triggering

        # Auto-trigger augmentation if threshold reached
        if len(existing_texts) >= AUGMENT_TRIGGER_THRESHOLD:
            print(f"[INFO] {len(existing_texts)} low-confidence entries found. Triggering /augment...")
            try:
                response = requests.post(AUGMENT_ENDPOINT)
                if response.status_code == 200:
                    print("[SUCCESS] Augmentation triggered.")
                else:
                    print(f"[ERROR] Augment API failed. Status: {response.status_code}")
            except Exception as e:
                print(f"[ERROR] Failed to call /augment: {e}")

    except Exception as e:
        print(f"[LOGGING ERROR] Failed to log input or trigger augment: {e}")

    # Return the prediction result
    return {
        "input": text,
        "is_misogynistic": is_misogynistic,
        "label": label,
        "confidence_score": confidence_score_str,
        "predicted_class": predicted_class,
        "model": MODEL_NAME,
        "version": VERSION
    }
