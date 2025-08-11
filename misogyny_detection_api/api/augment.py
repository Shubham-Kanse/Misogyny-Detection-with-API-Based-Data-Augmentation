# File: routers/augment.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from misogyny_detection_api.services.augmentor import (
    augment_inputs,
    generate_augmented_samples_for_text
)

router = APIRouter(prefix="/augment", tags=["Augmentation"])

@router.post("")
def augment():
    """
    Existing automatic augmentation endpoint.
    Reads low-confidence log, writes files, triggers retrain, cleans up.
    """
    try:
        num_generated = augment_inputs()
        return {
            "message": f"Augmented {num_generated} synthetic examples.",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PreviewRequest(BaseModel):
    text: str = Field(..., description="A single low-confidence text to augment for demo/preview.")
    force_label: Optional[int] = Field(
        default=None,
        description="Optionally force label 0 (non) or 1 (misogynistic) to skip LLM verification."
    )

@router.post("/preview")
def augment_preview(payload: PreviewRequest):
    """
    Side-effect-free preview endpoint.
    - Accepts a single text
    - Returns up to 5 augmented samples (same techniques as batch)
    - Does NOT write to disk, trigger retrain, or clear logs
    """
    try:
        samples: List[Dict] = generate_augmented_samples_for_text(
            text=payload.text,
            force_label=payload.force_label
        )
        return {
            "status": "success",
            "count": len(samples),
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
