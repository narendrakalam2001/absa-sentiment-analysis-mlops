# ============================================================
# ABSA API — FastAPI Serving
# Endpoints: /predict  /predict_batch  /health  /model_info
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
import time
import os
import json

from src.model_loader          import load_latest_model
from services.prediction_service import predict_review, predict_reviews_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "ABSA — Aspect-Based Sentiment Analysis API",
    description = (
        "Aspect-level sentiment classification for e-commerce and banking reviews. "
        "SemEval 2014 Task 4 trained. Multi-class: positive / negative / neutral / conflict."
    ),
    version     = "1.0.0",
)

# ── Load model on startup ─────────────────────────────────────
try:
    model, thresholds = load_latest_model()
    logger.info("ABSA model loaded successfully | thresholds=%s", thresholds)
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model      = None
    thresholds = {"negative": 0.45}


# ============================================================
# INPUT / OUTPUT SCHEMAS
# ============================================================

class ReviewInput(BaseModel):
    text:        str         = Field(..., description="Full review sentence")
    aspect_term: str         = Field(..., description="Aspect term (e.g. 'food', 'battery')")
    domain:      Optional[str] = Field("restaurants", description="Domain: restaurants/laptops/banking")

    class Config:
        schema_extra = {
            "example": {
                "text":        "The pasta was incredible but the service was really slow.",
                "aspect_term": "service",
                "domain":      "restaurants"
            }
        }


class BatchReviewInput(BaseModel):
    reviews: List[ReviewInput] = Field(..., description="List of review inputs (max 100)")

    class Config:
        schema_extra = {
            "example": {
                "reviews": [
                    {"text": "The food was amazing!", "aspect_term": "food", "domain": "restaurants"},
                    {"text": "Battery life is terrible.", "aspect_term": "battery", "domain": "laptops"},
                ]
            }
        }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "ABSA Sentiment Analysis API is live 🚀",
        "docs":    "/docs",
        "health":  "/health",
        "version": "1.0.0",
        "model":   "SemEval 2014 Task 4 — Multi-class ABSA",
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": model is not None,
        "thresholds":   thresholds,
    }


@app.get("/model_info")
def model_info():
    registry_path = os.path.join("absa_models", "latest_model.json")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            info = json.load(f)

        # Attach model card metrics if available
        card_path = info.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            info["metrics"]       = card.get("metrics", {})
            info["per_class_f1"]  = card.get("per_class_f1", {})
            info["intended_use"]  = card.get("intended_use", {})

        return info
    return {"error": "Model registry not found"}


# ── Single prediction ─────────────────────────────────────────

@app.post("/predict")
def predict(review: ReviewInput):
    """
    Predict sentiment for a single (text, aspect_term) pair.
    Returns: predicted_sentiment, probabilities, decision, business_action.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    start = time.time()

    result = predict_review(
        model      = model,
        text       = review.text,
        aspect     = review.aspect_term,
        domain     = review.domain or "restaurants",
        thresholds = thresholds,
    )

    # ── Log prediction ────────────────────────────────────────
    _log_prediction(review, result)

    result["latency_seconds"] = round(time.time() - start, 4)
    return result


# ── Batch prediction ──────────────────────────────────────────

@app.post("/predict_batch")
def predict_batch(batch: BatchReviewInput):
    """
    Predict sentiment for a batch of reviews (up to 100).
    Returns list of predictions + batch-level analytics.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(batch.reviews) > 100:
        raise HTTPException(status_code=400, detail="Max 100 reviews per batch.")

    start = time.time()

    reviews_data = [
        {
            "text":   r.text,
            "aspect": r.aspect_term,
            "domain": r.domain or "restaurants"
        }
        for r in batch.reviews
    ]

    result = predict_reviews_batch(model, reviews_data, thresholds)

    result["latency_seconds"] = round(time.time() - start, 4)
    return result


# ============================================================
# PREDICTION LOGGER
# ============================================================

def _log_prediction(review: ReviewInput, result: dict):
    log_record = {
        "timestamp":           time.time(),
        "text":                review.text[:100],
        "aspect_term":         review.aspect_term,
        "domain":              review.domain,
        "predicted_sentiment": result.get("predicted_sentiment"),
        "decision":            result.get("decision"),
        "confidence":          result.get("confidence"),
        "rule_triggered":      result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
