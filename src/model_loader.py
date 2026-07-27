# ============================================================
# MODEL LOADER + CHAMPION-CHALLENGER — ABSA System
# ============================================================
#
# CHALLENGER MODEL SYSTEM (identical pattern to Credit Risk):
#   Champion  = current production model (latest_model.json)
#   Challenger = newly trained model (passed in from pipeline)
#
#   Promotion: Challenger promoted ONLY if ALL 3 gates pass:
#     Gate 1: macro-F1 improvement >= MIN_F1_IMPROVEMENT
#     Gate 2: macro ROC-AUC       >= MIN_ROCAUC_THRESHOLD
#     Gate 3: train-val F1 gap    <= MAX_GENERALIZATION_GAP
#
#   Full comparison saved to absa_models/challenger_log.json
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import (
    MODEL_DIR,
    MIN_F1_IMPROVEMENT,
    MIN_ROCAUC_THRESHOLD,
    MAX_GENERALIZATION_GAP,
)

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads absa_models/latest_model.json → loads .joblib + metadata.
    Returns: (model_pipeline, threshold_dict)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "Run scripts/train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    # threshold_dict: {"negative": 0.42} for business-tuned negative detection
    threshold_dict = registry.get("thresholds", {"negative": 0.5})

    logger.info(
        "Champion model loaded: %s  |  thresholds=%s",
        model_path, threshold_dict
    )

    return model, threshold_dict


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    """
    Reads current champion model card to get its performance metrics.
    Returns empty dict if no champion exists.
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # model_card_path stored directly (no brittle string parsing)
    card_path = registry.get("model_card_path", "")

    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})

    return {
        "model_name": card.get("model_name", "unknown"),
        "macro_f1":   float(metrics.get("macro_f1",   0)),
        "roc_auc":    float(metrics.get("roc_auc",    0)),
        "gap":        float(metrics.get("train_val_gap", 0)),
        "thresholds": card.get("thresholds", {"negative": 0.5}),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC (3-gate)
# ============================================================

def run_challenger_comparison(
    challenger_name:       str,
    challenger_macro_f1:   float,
    challenger_roc_auc:    float,
    challenger_gap:        float,
    challenger_model_path: str,
    challenger_thresholds: dict,
    challenger_card_path:  str = "",
) -> dict:
    """
    Compares challenger vs current champion using 3 promotion gates.

    Args:
        challenger_name       : model name e.g. 'LightGBM'
        challenger_macro_f1   : challenger test macro-F1
        challenger_roc_auc    : challenger macro OvR ROC-AUC
        challenger_gap        : challenger train-val macro-F1 gap
        challenger_model_path : path to challenger .joblib
        challenger_thresholds : dict of class-specific thresholds
        challenger_card_path  : path to challenger model card JSON

    Returns:
        result dict with decision: 'PROMOTED' or 'REJECTED'
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    # ── No champion → auto-promote first model ────────────────
    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_thresholds, challenger_card_path)
        result = {
            "decision":           "PROMOTED",
            "reason":             "No existing champion — first model auto-promoted",
            "challenger_name":    challenger_name,
            "challenger_macro_f1":round(challenger_macro_f1, 4),
            "challenger_roc_auc": round(challenger_roc_auc,  4),
            "champion_name":      None,
            "champion_macro_f1":  None,
            "evaluated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_f1   = champion.get("macro_f1", 0.0)
    champion_roc  = champion.get("roc_auc",  0.0)
    champion_name = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-20s  F1=%.4f  ROC=%.4f",
                champion_name,   champion_f1,          champion_roc)
    logger.info("  Challenger: %-20s  F1=%.4f  ROC=%.4f",
                challenger_name, challenger_macro_f1, challenger_roc_auc)
    logger.info("=" * 55)

    # ── Promotion gates ───────────────────────────────────────
    gate1_f1_improvement = (challenger_macro_f1 - champion_f1) >= MIN_F1_IMPROVEMENT
    gate2_roc_auc        = challenger_roc_auc >= MIN_ROCAUC_THRESHOLD
    gate3_gap            = challenger_gap      <= MAX_GENERALIZATION_GAP

    gates_passed = gate1_f1_improvement and gate2_roc_auc and gate3_gap

    if gates_passed:
        decision = "PROMOTED"
        reason   = (
            f"Challenger beats champion: "
            f"macro-F1 {champion_f1:.4f} → {challenger_macro_f1:.4f} "
            f"(+{challenger_macro_f1 - champion_f1:.4f})"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_thresholds, challenger_card_path)

    else:
        decision = "REJECTED"
        failed   = []
        if not gate1_f1_improvement:
            failed.append(
                f"macro-F1 improvement {challenger_macro_f1 - champion_f1:+.4f} "
                f"< {MIN_F1_IMPROVEMENT}"
            )
        if not gate2_roc_auc:
            failed.append(f"ROC-AUC {challenger_roc_auc:.4f} < {MIN_ROCAUC_THRESHOLD}")
        if not gate3_gap:
            failed.append(f"train-val gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":              decision,
        "reason":                reason,
        "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":       challenger_name,
        "challenger_macro_f1":   round(challenger_macro_f1, 4),
        "challenger_roc_auc":    round(challenger_roc_auc,  4),
        "challenger_gap":        round(challenger_gap,       4),
        "champion_name":         champion_name,
        "champion_macro_f1":     round(champion_f1,          4),
        "champion_roc_auc":      round(champion_roc,         4),
        "gates": {
            "f1_improvement_passed": gate1_f1_improvement,
            "roc_auc_passed":        gate2_roc_auc,
            "gap_passed":            gate3_gap,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, thresholds: dict, model_card_path: str = ""):
    """
    Updates latest_model.json with new champion.
    Stores model_card_path directly — no brittle filename parsing.
    """
    registry = {
        "model_name":      os.path.basename(model_path),
        "thresholds":      thresholds,
        "model_card_path": model_card_path or "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _save_challenger_log(result: dict):
    """Appends challenger comparison result to challenger_log.json."""
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved → %s", CHALLENGER_LOG)
