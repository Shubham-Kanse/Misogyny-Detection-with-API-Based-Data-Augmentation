from fastapi import APIRouter, HTTPException
from misogyny_detection_api.services.augmentor import augment_inputs

router = APIRouter(prefix="/augment", tags=["Augmentation"])

@router.post("")
def augment():
    """
    Automatically triggered augmentation endpoint.
    Verifies low-confidence inputs and generates synthetic data.
    """
    try:
        num_generated = augment_inputs()
        return {
            "message": f"Augmented {num_generated} synthetic examples.",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
