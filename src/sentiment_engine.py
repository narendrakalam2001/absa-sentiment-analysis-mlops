# ============================================================
# SENTIMENT ENGINE — Business Logic Layer (ABSA)
# ============================================================
# Translates raw model probabilities into actionable decisions
# for e-commerce / banking / healthcare use cases.
#
# Business rules layered on top of ML probabilities:
#   ESCALATE  → high-confidence negative + specific aspect
#   REVIEW    → moderate negative or conflict
#   POSITIVE  → high-confidence positive
#   NEUTRAL   → neutral or low-confidence prediction
# ============================================================

import numpy as np
from typing import Optional

from src.config import SENTIMENT_NAMES, NUM_CLASSES


# ============================================================
# SENTIMENT BANDS
# ============================================================

SENTIMENT_BANDS = {
    "STRONG_NEGATIVE": (0.70, 1.01),   # escalate immediately
    "MODERATE_NEGATIVE":(0.45, 0.70),  # flag for review
    "UNCERTAIN":        (0.00, 0.45),  # use overall max prob
}


def get_sentiment_band(prob_negative: float) -> str:
    """Classifies negative probability into business action band."""
    for band, (lo, hi) in SENTIMENT_BANDS.items():
        if lo <= prob_negative < hi:
            return band
    return "UNCERTAIN"


# ============================================================
# SCORE REVIEW — SINGLE PREDICTION
# ============================================================

def score_review(
    review_text:    str,
    aspect_term:    str,
    proba:          np.ndarray,         # shape (n_classes,)
    domain:         str = "restaurants",
    neg_threshold:  float = 0.45,       # business-tuned
) -> dict:
    """
    Converts model probabilities into a structured business output.

    Returns:
        {
          predicted_sentiment : str  (positive/negative/neutral/conflict)
          probabilities        : dict {class: float}
          sentiment_band       : str  (STRONG_NEGATIVE / MODERATE_NEGATIVE / UNCERTAIN)
          decision             : str  (ESCALATE / REVIEW / POSITIVE / NEUTRAL)
          confidence           : float
          rule_triggered       : str or None
          business_action      : str
        }
    """
    if len(proba) < NUM_CLASSES:
        proba = np.pad(proba, (0, NUM_CLASSES - len(proba)))

    prob_positive = float(proba[0])
    prob_negative = float(proba[1])
    prob_neutral  = float(proba[2])
    prob_conflict = float(proba[3]) if len(proba) > 3 else 0.0

    predicted_cls   = int(np.argmax(proba))
    predicted_label = SENTIMENT_NAMES[predicted_cls] if predicted_cls < len(SENTIMENT_NAMES) else "unknown"
    confidence      = float(np.max(proba))
    band            = get_sentiment_band(prob_negative)

    # ── Business decision rules ───────────────────────────────
    rule_triggered = None
    decision       = None

    # Rule 1: Strong negative — escalate
    if prob_negative >= 0.70:
        decision       = "ESCALATE"
        rule_triggered = f"prob_negative={prob_negative:.3f} >= 0.70 → immediate escalation"

    # Rule 2: Moderate negative or conflict
    elif prob_negative >= neg_threshold or prob_conflict >= 0.50:
        decision       = "REVIEW"
        if prob_conflict >= 0.50:
            rule_triggered = f"prob_conflict={prob_conflict:.3f} >= 0.50 → conflicting signals"
        else:
            rule_triggered = f"prob_negative={prob_negative:.3f} >= {neg_threshold} → human review"

    # Rule 3: High-confidence positive
    elif prob_positive >= 0.60:
        decision = "POSITIVE"

    # Rule 4: Neutral / low-confidence
    else:
        decision = "NEUTRAL"

    # ── Domain-specific business action ───────────────────────
    business_action = _domain_action(decision, domain, aspect_term, review_text)

    return {
        "predicted_sentiment": predicted_label,
        "probabilities": {
            "positive": round(prob_positive, 4),
            "negative": round(prob_negative, 4),
            "neutral":  round(prob_neutral,  4),
            "conflict": round(prob_conflict, 4),
        },
        "sentiment_band":  band,
        "decision":        decision,
        "confidence":      round(confidence, 4),
        "rule_triggered":  rule_triggered,
        "business_action": business_action,
    }


# ============================================================
# BATCH SCORING
# ============================================================

def score_batch(
    reviews:       list,           # list of (review_text, aspect_term) tuples
    probas:        np.ndarray,     # shape (n_samples, n_classes)
    domain:        str = "restaurants",
    neg_threshold: float = 0.45,
) -> list:
    """
    Scores a batch of (review, aspect) pairs.
    Returns list of score_review() dicts.
    """
    results = []
    for i, (text, aspect) in enumerate(reviews):
        prob_row = probas[i] if i < len(probas) else np.array([0.25, 0.25, 0.25, 0.25])
        result   = score_review(text, aspect, prob_row, domain, neg_threshold)
        results.append(result)
    return results


# ============================================================
# SUMMARY ANALYTICS
# ============================================================

def summarise_batch(results: list) -> dict:
    """
    Computes batch-level analytics from scored reviews.
    Useful for monitoring dashboard KPIs.
    """
    if not results:
        return {}

    n = len(results)
    decisions   = [r["decision"]        for r in results]
    sentiments  = [r["predicted_sentiment"] for r in results]

    escalate_rate  = decisions.count("ESCALATE")  / n
    review_rate    = decisions.count("REVIEW")     / n
    positive_rate  = decisions.count("POSITIVE")   / n
    negative_rate  = sentiments.count("negative")  / n
    avg_confidence = float(np.mean([r["confidence"] for r in results]))

    return {
        "total_reviews":    n,
        "escalate_rate":    round(escalate_rate,  4),
        "review_rate":      round(review_rate,    4),
        "positive_rate":    round(positive_rate,  4),
        "negative_rate":    round(negative_rate,  4),
        "avg_confidence":   round(avg_confidence, 4),
        "alert": escalate_rate > 0.20 or negative_rate > 0.35,
    }


# ============================================================
# DOMAIN ACTION HELPER
# ============================================================

def _domain_action(
    decision:    str,
    domain:      str,
    aspect_term: str,
    review_text: str,
) -> str:
    """
    Maps decision → domain-specific business action string.
    """
    aspect = aspect_term.lower() if aspect_term else "general"

    if domain in ("restaurants", "restaurant"):
        actions = {
            "ESCALATE": f"🔴 Alert kitchen/management — negative '{aspect}' feedback. Reply within 24h.",
            "REVIEW":   f"🟡 Flag for supervisor — '{aspect}' mentioned negatively.",
            "POSITIVE": f"✅ Highlight in weekly digest — customers love '{aspect}'.",
            "NEUTRAL":  f"📝 Logged — '{aspect}' review neutral.",
        }
    elif domain in ("laptops", "laptop", "tech"):
        actions = {
            "ESCALATE": f"🔴 Escalate to product team — critical issue with '{aspect}'.",
            "REVIEW":   f"🟡 Route to support team — '{aspect}' complaints increasing.",
            "POSITIVE": f"✅ Share with marketing — positive '{aspect}' reviews.",
            "NEUTRAL":  f"📝 Logged — '{aspect}' feedback neutral.",
        }
    elif domain in ("banking", "bank", "finance"):
        actions = {
            "ESCALATE": f"🔴 URGENT: Customer dissatisfied with '{aspect}' — assign RM immediately.",
            "REVIEW":   f"🟡 Route to grievance team — '{aspect}' concern flagged.",
            "POSITIVE": f"✅ NPS positive signal — '{aspect}' rated well.",
            "NEUTRAL":  f"📝 Logged — '{aspect}' feedback neutral.",
        }
    else:
        actions = {
            "ESCALATE": f"🔴 High-priority negative feedback on '{aspect}'.",
            "REVIEW":   f"🟡 Review required for '{aspect}' feedback.",
            "POSITIVE": f"✅ Positive feedback on '{aspect}'.",
            "NEUTRAL":  f"📝 Neutral feedback on '{aspect}'.",
        }

    return actions.get(decision, "📝 No action required.")
