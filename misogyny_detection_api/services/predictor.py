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
    Runs in the background to avoid blocking the API response.
    """
    sanitized_text = text.strip()

    try:
        LOW_CONF_LOG.parent.mkdir(parents=True, exist_ok=True)

        # Load existing entries safely
        existing_texts = set()
        if LOW_CONF_LOG.exists():
            try:
                with open(LOW_CONF_LOG, mode="r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    if "text" in reader.fieldnames:  # avoid malformed CSVs
                        existing_texts = {row["text"].strip() for row in reader if row.get("text")}
            except Exception as e:
                print(f"[WARN] Could not read existing low-confidence log: {e}")

        # Only log if text is new and confidence is below threshold
        if sanitized_text not in existing_texts and confidence < CONFIDENCE_THRESHOLD:
            try:
                file_exists = LOW_CONF_LOG.exists()
                with open(LOW_CONF_LOG, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    if not file_exists or LOW_CONF_LOG.stat().st_size == 0:
                        writer.writerow(["timestamp", "text", "predicted_class", "confidence", "model_version"])
                    writer.writerow([
                        datetime.utcnow().isoformat(),
                        sanitized_text,
                        predicted_class,
                        round(confidence, 4),
                        VERSION
                    ])
                existing_texts.add(sanitized_text)
                print(f"[LOGGED] Low-confidence sample saved: '{sanitized_text}' ({confidence:.4f})")
            except Exception as e:
                print(f"[ERROR] Failed to write to low-confidence log: {e}")

        # Trigger augmentation if threshold reached
        if len(existing_texts) >= AUGMENT_TRIGGER_THRESHOLD:
            print(f"[INFO] {len(existing_texts)} low-confidence entries found. Triggering /augment...")
            try:
                response = requests.post(AUGMENT_ENDPOINT, timeout=10)
                if response.status_code == 200:
                    print("[SUCCESS] Augmentation triggered.")
                else:
                    print(f"[ERROR] Augment API failed. Status: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"[ERROR] Failed to call /augment: {e}")

    except Exception as e:
        print(f"[LOGGING ERROR] Unexpected error in augmentation logger: {e}")


def predict_texts(texts):
    """
    Run the trained BERT model on input text(s).
    Supports single string or list of strings.
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return {"results": []}

    # Tokenize as batch
    encoding = tokenizer(
        texts, return_tensors="pt",
        padding=True, truncation=True, max_length=128
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)

    results = []
    for i, text in enumerate(texts):
        pred_class = predictions[i].item()
        conf_val = confidences[i].item()
        conf_str = f"{round(conf_val * 100, 2)}%"
        label = "Misogynistic" if pred_class == 1 else "Non-misogynistic"

        # Background logging/augmentation
        threading.Thread(
            target=_log_and_trigger_augmentation,
            args=(text, pred_class, conf_val),
            daemon=True  # ensure threads don't block shutdown
        ).start()

        results.append({
            "input": text,
            "is_misogynistic": pred_class == 1,
            "label": label,
            "confidence_score": conf_str,
            "predicted_class": pred_class,
            "model": MODEL_NAME,
            "version": VERSION
        })

    # Keep backward compatibility
    if len(results) == 1:
        return results[0]
    return {"results": results}
