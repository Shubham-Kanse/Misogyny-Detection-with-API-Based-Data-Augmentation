# services/predictor.py

import csv
import logging
import queue
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
from torch.nn.utils.rnn import pad_sequence

from misogyny_detection_api.services.model_loader import load_model, load_tokenizer
from misogyny_detection_api.config import (
    MODEL_NAME,
    VERSION,
    CONFIDENCE_THRESHOLD,
    LOW_CONF_LOG,
    AUGMENT_TRIGGER_THRESHOLD,
    AUGMENT_ENDPOINT,
)

# Configure logging
logger = logging.getLogger(__name__)

# ----------------------------
# Async Background Task Handling
# ----------------------------
def _handle_background_tasks_async(text: str, predicted_class: int, confidence: float) -> None:
    """Handle background tasks asynchronously without blocking API response."""
    def background_worker():
        try:
            _enqueue_low_confidence_prediction(text, predicted_class, confidence)
            _trigger_augmentation_if_needed()
        except Exception as e:
            logger.error(f"Background task error: {e}")
    
    # Run in background thread
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()

def _handle_batch_background_tasks_async(background_tasks: List[Tuple[str, int, float]]) -> None:
    """Handle batch background tasks asynchronously."""
    def batch_background_worker():
        try:
            # Process all low-confidence logging
            for text, pred_class, confidence in background_tasks:
                _enqueue_low_confidence_prediction(text, pred_class, confidence)
            
            # Trigger augmentation check once for the entire batch
            _trigger_augmentation_if_needed()
        except Exception as e:
            logger.error(f"Batch background task error: {e}")
    
    # Run in background thread
    thread = threading.Thread(target=batch_background_worker, daemon=True)
    thread.start()

# ----------------------------
# Constants and Configuration
# ----------------------------
MAX_SEQUENCE_LENGTH = 128
MAX_BATCH_SIZE = 32
LOG_QUEUE_SIZE = 2048
DEDUP_CACHE_SIZE = 10000
HTTP_TIMEOUT = 10
CSV_FIELDNAMES = ["timestamp", "text", "predicted_class", "confidence", "model_version"]

# ----------------------------
# Global State (Singleton Pattern)
# ----------------------------
class PredictorState:
    """Thread-safe singleton for managing global predictor state."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Model components
        self.tokenizer = load_tokenizer()
        self.model = load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        
        # HTTP session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=5,
            max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Logging infrastructure
        self.log_queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)
        self.seen_texts = set()
        self.seen_texts_lock = threading.RLock()
        self.log_thread = None
        self.log_thread_started = False
        
        # Batch processing optimization
        self.batch_lock = threading.Lock()
        
        self._initialized = True

# Global state instance
_state = PredictorState()

# ----------------------------
# Utility Functions
# ----------------------------
def _ensure_log_dir() -> None:
    """Ensure log directory exists."""
    LOW_CONF_LOG.parent.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def _load_seen_texts() -> set:
    """Load existing texts from log file with caching."""
    seen_texts = set()
    if not LOW_CONF_LOG.exists():
        return seen_texts
    
    try:
        with open(LOW_CONF_LOG, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "text" not in reader.fieldnames:
                return seen_texts
                
            for row in reader:
                text = (row.get("text", "") or "").strip()
                if text:
                    seen_texts.add(text)
                    
        logger.info(f"Loaded {len(seen_texts)} seen texts from log")
    except Exception as e:
        logger.warning(f"Could not load seen texts: {e}")
    
    return seen_texts

def _initialize_seen_texts() -> None:
    """Initialize the seen texts set thread-safely."""
    with _state.seen_texts_lock:
        if not _state.seen_texts:
            _state.seen_texts.update(_load_seen_texts())

@contextmanager
def _csv_writer(filepath: Path):
    """Context manager for CSV writing with proper error handling."""
    try:
        with open(filepath, mode="a", encoding="utf-8", newline="") as f:
            # Check if file is empty to write header
            needs_header = f.tell() == 0
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(CSV_FIELDNAMES)
            yield writer
    except Exception as e:
        logger.error(f"CSV writing error: {e}")
        raise

def _log_writer_loop() -> None:
    """Background thread for writing low-confidence predictions to CSV."""
    _ensure_log_dir()
    
    while True:
        try:
            # Block until item available
            item = _state.log_queue.get()
            if item is None:  # Sentinel for shutdown
                break
                
            timestamp, text, predicted_class, confidence, version = item
            
            # Double-check deduplication at write time
            with _state.seen_texts_lock:
                if text in _state.seen_texts:
                    _state.log_queue.task_done()
                    continue
            
            # Write to CSV
            with _csv_writer(LOW_CONF_LOG) as writer:
                writer.writerow([timestamp, text, predicted_class, 
                               round(confidence, 4), version])
            
            # Update seen texts
            with _state.seen_texts_lock:
                _state.seen_texts.add(text)
                # Prevent unbounded memory growth
                if len(_state.seen_texts) > DEDUP_CACHE_SIZE:
                    # Keep most recent 80% of entries
                    keep_size = int(DEDUP_CACHE_SIZE * 0.8)
                    _state.seen_texts = set(list(_state.seen_texts)[-keep_size:])
            
            _state.log_queue.task_done()
            
        except Exception as e:
            logger.error(f"Log writer loop error: {e}")
            # Continue processing to avoid thread death

def _start_log_thread() -> None:
    """Start the logging thread if not already started."""
    if _state.log_thread_started:
        return
        
    with _state.seen_texts_lock:
        if _state.log_thread_started:
            return
            
        _initialize_seen_texts()
        _state.log_thread = threading.Thread(
            target=_log_writer_loop,
            name="low_conf_log_writer",
            daemon=True
        )
        _state.log_thread.start()
        _state.log_thread_started = True
        logger.info("Started low-confidence logging thread")

def _enqueue_low_confidence_prediction(
    text: str, 
    predicted_class: int, 
    confidence: float
) -> None:
    """Queue low-confidence prediction for logging."""
    if confidence >= CONFIDENCE_THRESHOLD:
        return
        
    sanitized_text = text.strip()
    if not sanitized_text:
        return
    
    # Check if already seen (with lock)
    with _state.seen_texts_lock:
        if sanitized_text in _state.seen_texts:
            return
    
    # Start logging thread if needed
    _start_log_thread()
    
    # Enqueue for background processing
    try:
        _state.log_queue.put_nowait((
            datetime.utcnow().isoformat(),
            sanitized_text,
            int(predicted_class),
            float(confidence),
            str(VERSION)
        ))
    except queue.Full:
        logger.warning("Low-confidence log queue is full; dropping entry")

def _trigger_augmentation_if_needed() -> None:
    """Trigger model augmentation if threshold reached."""
    with _state.seen_texts_lock:
        unique_count = len(_state.seen_texts)
    
    if unique_count < AUGMENT_TRIGGER_THRESHOLD:
        return
    
    try:
        response = _state.session.post(AUGMENT_ENDPOINT, timeout=HTTP_TIMEOUT)
        if response.status_code == 200:
            logger.info(f"Augmentation triggered (logged={unique_count})")
        else:
            logger.error(f"Augmentation failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Could not trigger augmentation: {e}")

# ----------------------------
# Optimized Inference Engine
# ----------------------------
def _prepare_batch_inputs(texts: List[str]) -> Dict[str, torch.Tensor]:
    """Prepare batch inputs with optimized tokenization."""
    # Filter and clean texts
    cleaned_texts = [text.strip() for text in texts if text and text.strip()]
    if not cleaned_texts:
        return {}
    
    # Tokenize batch
    encoded = _state.tokenizer(
        cleaned_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        add_special_tokens=True
    )
    
    # Move to device
    return {k: v.to(_state.device) for k, v in encoded.items()}

def _run_inference_batch(texts: List[str]) -> Tuple[List[int], List[float]]:
    """Run inference on a batch of texts with optimizations."""
    if not texts:
        return [], []
    
    # Prepare inputs
    inputs = _prepare_batch_inputs(texts)
    if not inputs:
        return [], []
    
    # Run inference
    with torch.no_grad():
        # Use torch.jit optimizations if available
        if hasattr(_state.model, 'forward'):
            outputs = _state.model(**inputs)
        else:
            outputs = _state.model.forward(**inputs)
        
        # Compute probabilities and predictions
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
    
    return predictions.cpu().tolist(), confidences.cpu().tolist()

def _format_prediction_result(
    text: str, 
    predicted_class: int, 
    confidence: float
) -> Dict:
    """Format a single prediction result."""
    is_misogynistic = predicted_class == 1
    label = "Misogynistic" if is_misogynistic else "Non-misogynistic"
    confidence_percentage = f"{round(confidence * 100, 2)}%"
    
    return {
        "input": text,
        "is_misogynistic": is_misogynistic,
        "label": label,
        "confidence_score": confidence_percentage,
        "predicted_class": predicted_class,
        "model": MODEL_NAME,
        "version": VERSION,
    }

# ----------------------------
# Public API Functions
# ----------------------------
def predict_text(text: str) -> Dict:
    """
    Predict misogyny for a single text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dict containing prediction results
    """
    if not isinstance(text, str) or not text.strip():
        return {"results": []}
    
    predictions, confidences = _run_inference_batch([text])
    
    if not predictions:
        return {"results": []}
    
    predicted_class = predictions[0]
    confidence = confidences[0]
    
    # Format result immediately for fast response
    result = _format_prediction_result(text, predicted_class, confidence)
    
    # Handle background tasks asynchronously (non-blocking)
    _handle_background_tasks_async(text, predicted_class, confidence)
    
    return result

def predict_texts(texts: Union[str, List[str]]) -> Dict[str, List[Dict]]:
    """
    Predict misogyny for multiple texts with batch optimization.
    
    Args:
        texts: Single text string or list of text strings
        
    Returns:
        Dict with 'results' key containing list of predictions
    """
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        return {"results": []}
    
    # Process in batches to manage memory
    all_results = []
    background_tasks = []  # Collect background tasks for async processing
    
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i:i + MAX_BATCH_SIZE]
        predictions, confidences = _run_inference_batch(batch)
        
        # Format results immediately
        for text, pred_class, confidence in zip(batch, predictions, confidences):
            result = _format_prediction_result(text, pred_class, confidence)
            all_results.append(result)
            
            # Queue background task data
            background_tasks.append((text, pred_class, confidence))
    
    # Handle all background tasks asynchronously after response is ready
    _handle_batch_background_tasks_async(background_tasks)
    
    return {"results": all_results}

# ----------------------------
# Health Check and Utilities
# ----------------------------
def get_model_info() -> Dict:
    """Get information about the loaded model."""
    return {
        "model_name": MODEL_NAME,
        "version": VERSION,
        "device": str(_state.device),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }

def get_statistics() -> Dict:
    """Get runtime statistics."""
    with _state.seen_texts_lock:
        seen_count = len(_state.seen_texts)
    
    return {
        "unique_low_confidence_predictions": seen_count,
        "augmentation_threshold": AUGMENT_TRIGGER_THRESHOLD,
        "log_queue_size": _state.log_queue.qsize() if _state.log_thread_started else 0,
        "logging_active": _state.log_thread_started,
    }

# ----------------------------
# Cleanup
# ----------------------------
def shutdown() -> None:
    """Clean shutdown of background threads."""
    if _state.log_thread_started and _state.log_thread:
        # Send shutdown sentinel
        _state.log_queue.put(None)
        _state.log_thread.join(timeout=5)
        logger.info("Predictor service shut down")