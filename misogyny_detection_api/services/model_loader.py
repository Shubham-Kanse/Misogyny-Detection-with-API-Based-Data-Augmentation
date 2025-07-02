# services/model_loader.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from misogyny_detection_api.config import MODEL_PATH, MODEL_NAME, NUM_LABELS

def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return model

def load_tokenizer():
    return BertTokenizer.from_pretrained(MODEL_NAME)
