# services/predictor.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from misogyny_detection_api.services.model_loader import load_model, load_tokenizer
from misogyny_detection_api.config import (
    MODEL_NAME, VERSION, CONFIDENCE_THRESHOLD
)

# Load actual model and tokenizer (from model_loader.py)
model = load_model()
tokenizer = load_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_text(text: str):
    """
    Run the trained BERT model on input text to classify misogyny.
    """
    # Tokenize and move to device
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    # Format result
    label = "Misogynistic" if prediction.item() == 1 else "Non-misogynistic"
    is_misogynistic = prediction.item() == 1
    confidence_score = round(confidence.item(), 4)

    return {
        "input": text,
        "is_misogynistic": is_misogynistic,
        "label": label,
        "confidence_score": confidence_score,
        "predicted_class": prediction.item(),
        "model": MODEL_NAME,
        "version": VERSION
    }
