# api/predict.py

from fastapi import APIRouter
from pydantic import BaseModel
from misogyny_detection_api.services.predictor import predict_text

# Define router for prediction functionality
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Request schema
class PredictRequest(BaseModel):
    text: str  # Input text for prediction

@router.post("/")
def predict(request: PredictRequest):
    """
    Predict whether the input text is misogynistic.
    """
    return predict_text(request.text)