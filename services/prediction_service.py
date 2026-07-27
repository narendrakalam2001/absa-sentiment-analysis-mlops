# ============================================================
# PREDICTION SERVICE — Aspect-Based Sentiment Analysis (ABSA)
# ============================================================

import pandas as pd
import numpy as np
import logging

from src.data_loader      import add_engineered_features
from src.sentiment_engine import score_review, score_batch, summarise_batch

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(text: str, aspect: str) -> pd.DataFrame:
    """
    Prepares a single (text, aspect) pair into a feature DataFrame.
    Mirrors the training pipeline feature engineering exactly.
    """
    from src.config import TEXT_COL, ASPECT_COL, LABEL_COL

    df = pd.DataFrame([{
        TEXT_COL:   text,
        ASPECT_COL: aspect,
        LABEL_COL:  "neutral",   # placeholder — not used in inference
        "domain":   "api",
        "split":    "api",
    }])

    df = add_engineered_features(df)

    # Drop columns that are not features
    drop_cols = [LABEL_COL, "label", "split", "domain"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


# ============================================================
# PREDICT — SINGLE REVIEW
# ============================================================

def predict_review(
    model,
    text:       str,
    aspect:     str,
    domain:     str  = "restaurants",
    thresholds: dict = None,
) -> dict:
    """
    Full prediction flow for one (review, aspect) pair:
      1. Feature engineering
      2. Model probability estimation
      3. Sentiment engine scoring (rules + ML)

    Returns structured output dict.
    """
    if thresholds is None:
        thresholds = {"negative": 0.45}

    neg_threshold = float(thresholds.get("negative", 0.45))

    df = prepare_features(text, aspect)

    try:
        proba = model.predict_proba(df)[0]
    except AttributeError:
        # LinearSVC / RidgeClassifier fallback
        from src.evaluation import safe_predict_proba
        proba = safe_predict_proba(model, df)[0]
    except Exception as e:
        logger.error("predict_proba failed: %s", e)
        proba = np.array([0.25, 0.25, 0.25, 0.25])

    result = score_review(
        review_text   = text,
        aspect_term   = aspect,
        proba         = proba,
        domain        = domain,
        neg_threshold = neg_threshold,
    )

    logger.info(
        "Predict  |  aspect='%s'  sentiment=%s  decision=%s  confidence=%.4f",
        aspect,
        result["predicted_sentiment"],
        result["decision"],
        result["confidence"],
    )

    return result


# ============================================================
# PREDICT — BATCH
# ============================================================

def predict_reviews_batch(
    model,
    reviews:    list,       # list of {"text": str, "aspect": str, "domain": str}
    thresholds: dict = None,
) -> dict:
    """
    Batch prediction for multiple (review, aspect) pairs.

    Args:
        model     : loaded sklearn pipeline
        reviews   : list of dicts with keys: text, aspect, domain
        thresholds: class-specific threshold overrides

    Returns:
        dict with 'predictions' (list) and 'analytics' (batch KPIs)
    """
    if thresholds is None:
        thresholds = {"negative": 0.45}

    neg_threshold = float(thresholds.get("negative", 0.45))

    # ── Build batch DataFrame ─────────────────────────────────
    rows = []
    for r in reviews:
        df_row = prepare_features(r["text"], r["aspect"])
        rows.append(df_row)

    if not rows:
        return {"predictions": [], "analytics": {}}

    X_batch = pd.concat(rows, ignore_index=True)

    try:
        probas = model.predict_proba(X_batch)
    except AttributeError:
        from src.evaluation import safe_predict_proba
        probas = safe_predict_proba(model, X_batch)
    except Exception as e:
        logger.error("Batch predict_proba failed: %s", e)
        probas = np.full((len(reviews), 4), 0.25)

    # ── Score each review ──────────────────────────────────────
    review_tuples = [(r["text"], r["aspect"]) for r in reviews]
    results       = score_batch(review_tuples, probas, domain="mixed", neg_threshold=neg_threshold)

    # Add domain info
    for i, (res, rev) in enumerate(zip(results, reviews)):
        res["domain"]      = rev.get("domain", "restaurants")
        res["aspect_term"] = rev["aspect"]
        res["text"]        = rev["text"][:100]

    analytics = summarise_batch(results)

    logger.info(
        "Batch scored  |  n=%d  escalate=%.1f%%  negative=%.1f%%",
        len(results),
        analytics.get("escalate_rate", 0) * 100,
        analytics.get("negative_rate",  0) * 100,
    )

    return {
        "predictions": results,
        "analytics":   analytics,
    }
