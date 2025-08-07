# File: api/retrain.py

from fastapi import APIRouter, HTTPException
from misogyny_detection_api.services.retrainer import trigger_retraining

router = APIRouter()

@router.post("/retrain")
def retrain_model():
    try:
        trigger_retraining()
        return {
            "status": "in-progress",
            "message": "Retraining started in background using latest dataset. This may take 1–2 hours."
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")
