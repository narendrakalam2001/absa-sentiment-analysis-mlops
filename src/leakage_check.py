# ============================================================
# LEAKAGE CHECK — Aspect-Based Sentiment Analysis (ABSA)
# FIX: Reduced keyword sensitivity threshold (93% → 98%)
#      to avoid false positives on normal NLP words like
#      'excellent', 'great service' which are valid features.
# ============================================================

import numpy as np
import pandas as pd
import logging
import re

from src.config import TEXT_COL, LABEL_COL, ASPECT_COL, SENTIMENT_NAMES

logger = logging.getLogger(__name__)


def detect_leakage(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    X_test:         pd.DataFrame = None,
    y_test:         pd.Series    = None,
    threshold_corr: float        = 0.99,
) -> list:
    """
    Runs ABSA-specific leakage heuristics.
    Returns list of warning strings. Empty = no leakage.
    
    NOTE: Warnings are logged HERE — caller should NOT re-log them.
    """
    warnings_list = []

    # ── Check 1: Numeric feature near-perfect correlation ─────
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        try:
            if X_train[col].equals(y_train.astype(X_train[col].dtype)):
                msg = f"[LEAKAGE] '{col}' is identical to target → remove this feature"
                warnings_list.append(msg)
                logger.warning(msg)
                continue

            corr = abs(np.corrcoef(
                X_train[col].fillna(0),
                y_train.fillna(0)
            )[0, 1])

            if corr >= threshold_corr:
                msg = (f"[LEAKAGE] '{col}' corr={corr:.4f} with target "
                       f"(>= {threshold_corr}) → possible leakage")
                warnings_list.append(msg)
                logger.warning(msg)
        except Exception as e:
            logger.warning("Leakage check failed for '%s': %s", col, e)

    # ── Check 2: Sentiment keywords in text ───────────────────
    # Threshold raised to 0.98 — common NLP words like "excellent",
    # "great service" are valid features, NOT leakage.
    LABEL_KEYWORDS = {
        "positive": ["five star", "highly recommend"],
        "negative": ["do not recommend", "terrible service"],
    }
    KEYWORD_THRESHOLD = 0.98   # was 0.90 — too sensitive, caused false positives

    if TEXT_COL in X_train.columns:
        for sentiment, keywords in LABEL_KEYWORDS.items():
            label_val = {"positive": 0, "negative": 1}.get(sentiment)
            if label_val is None:
                continue
            for kw in keywords:
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                flagged = X_train[TEXT_COL].fillna("").apply(
                    lambda t: bool(pattern.search(t))
                )
                if flagged.any():
                    overlap = (y_train[flagged] == label_val).mean()
                    if overlap >= KEYWORD_THRESHOLD:
                        msg = (f"[LEAKAGE] keyword '{kw}' appears in {flagged.sum()} texts "
                               f"and {overlap:.0%} are labeled '{sentiment}' → "
                               "text may encode label directly")
                        warnings_list.append(msg)
                        logger.warning(msg)

    # ── Check 3: Train-test text overlap ──────────────────────
    if X_test is not None and TEXT_COL in X_train.columns and TEXT_COL in X_test.columns:
        train_texts = set(X_train[TEXT_COL].fillna("").str.lower().str.strip())
        test_texts  = set(X_test[TEXT_COL].fillna("").str.lower().str.strip())
        overlap_n   = len(train_texts & test_texts)
        overlap_pct = overlap_n / max(len(test_texts), 1)

        if overlap_pct > 0.05:
            msg = (f"[LEAKAGE] {overlap_n} test sentences ({overlap_pct:.1%}) "
                   "found verbatim in training set → data contamination")
            warnings_list.append(msg)
            logger.warning(msg)
        elif overlap_pct > 0.00:
            logger.info(
                "Minor text overlap: %d sentences (%.1f%%) in both train and test "
                "(SemEval known characteristic — same sentence, different aspects)",
                overlap_n, overlap_pct * 100
            )

    # ── Check 4: Aspect term trivially predicts label ─────────
    if ASPECT_COL in X_train.columns:
        aspect_label = pd.DataFrame({
            "aspect": X_train[ASPECT_COL].fillna("GENERAL").str.lower(),
            "label":  y_train
        })
        for aspect, grp in aspect_label.groupby("aspect"):
            if len(grp) < 5:   # skip rare aspects (< 5 samples)
                continue
            dominant_ratio = grp["label"].value_counts(normalize=True).iloc[0]
            if dominant_ratio >= 0.95:
                msg = (f"[LEAKAGE] aspect='{aspect}' has {dominant_ratio:.0%} one label "
                       "(>= 95%) — could be target leakage or over-specific term")
                warnings_list.append(msg)
                logger.warning(msg)

    if not warnings_list:
        logger.info("Leakage check passed — no obvious leakage detected")

    return warnings_list