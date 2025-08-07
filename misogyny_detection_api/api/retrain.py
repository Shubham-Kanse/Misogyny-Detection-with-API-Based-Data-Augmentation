# File: api/retrain.py

from fastapi import APIRouter
from misogyny_detection_api.services.retrainer import run_retraining

router = APIRouter()

@router.post("/retrain")
async def retrain_model():
    result = run_retraining()
    if result["success"]:
        return {"status": "✅ Model retrained successfully", "log": result["output"]}
    else:
        return {"status": "❌ Retraining failed", "error": result["error"]}
