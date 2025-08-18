# api/predict.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Union, List
from misogyny_detection_api.services.predictor import predict_texts

# Define router for prediction functionality
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Request schema
class PredictRequest(BaseModel):
    text: Union[str, List[str]]  # Accept single string or list of strings

@router.post("")
def predict(request: PredictRequest):
    """
    Predict whether the input text(s) are misogynistic.
    Supports both single text and batch input.
    """
    return predict_texts(request.text)
