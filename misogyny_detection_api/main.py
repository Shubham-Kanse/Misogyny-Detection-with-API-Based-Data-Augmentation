# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from misogyny_detection_api.api import predict, augment, retrain

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Misogyny Detection API",
    description="Real-time detection, augmentation, and retraining framework for misogynistic content classification.",
    version="v0.1"
)

# Allow all origins for development (update to restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API route modules
app.include_router(predict.router)
app.include_router(augment.router)
app.include_router(retrain.router)

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Welcome to the Misogyny Detection API"}
