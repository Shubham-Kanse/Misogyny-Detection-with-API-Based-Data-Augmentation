import csv
import threading
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


def _log_and_trigger_augmentation(text: str, predicted_class: int, confidence: float):
    """
    Logs low-confidence inputs and triggers augmentation if threshold is met.
    This runs in the background to avoid blocking the API response.
    """
    sanitized_text = text.strip()

    try:
        LOW_CONF_LOG.parent.mkdir(parents=True, exist_ok=True)

        # Load existing entries
        existing_texts = set()
        if LOW_CONF_LOG.exists():
            with open(LOW_CONF_LOG, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_texts = {row["text"].strip() for row in reader}

        # Only log if text is new and low-confidence
        if sanitized_text not in existing_texts and confidence < CONFIDENCE_THRESHOLD:
            with open(LOW_CONF_LOG, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                file_empty = file.tell() == 0
                if file_empty:
                    writer.writerow(["timestamp", "text", "predicted_class", "confidence", "model_version"])
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    sanitized_text,
                    predicted_class,
                    round(confidence, 4),
                    VERSION
                ])
            existing_texts.add(sanitized_text)

        # Trigger augmentation if enough samples exist
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
        print(f"[LOGGING ERROR] Failed to log or trigger augmentation: {e}")


def predict_text(text: str):
    """
    Run the trained BERT model on input text to classify misogyny.
    Responds instantly, logs and triggers augmentation in background.
    """
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    predicted_class = prediction.item()
    confidence_val = confidence.item()
    confidence_score_str = f"{round(confidence_val * 100, 2)}%"
    label = "Misogynistic" if predicted_class == 1 else "Non-misogynistic"
    is_misogynistic = predicted_class == 1

    # Trigger logging and augmentation in background
    threading.Thread(
        target=_log_and_trigger_augmentation,
        args=(text, predicted_class, confidence_val)
    ).start()

    return {
        "input": text,
        "is_misogynistic": is_misogynistic,
        "label": label,
        "confidence_score": confidence_score_str,
        "predicted_class": predicted_class,
        "model": MODEL_NAME,
        "version": VERSION
    }
