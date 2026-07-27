# ============================================================
# METRICS — Aspect-Based Sentiment Analysis (ABSA)
# FIX: compute_roc_auc handles missing classes gracefully
#      compute_per_class_f1 uses only actual labels present
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    f1_score, classification_report
)

from src.config import SENTIMENT_NAMES, NUM_CLASSES, COST_MATRIX

logger = logging.getLogger(__name__)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class F1 — only for classes ACTUALLY present in y_true."""
    actual_labels = sorted(np.unique(y_true).tolist())
    f1s = f1_score(
        y_true, y_pred,
        labels    = actual_labels,
        average   = None,
        zero_division = 0
    )
    result = {}
    for i, lbl in enumerate(actual_labels):
        name = SENTIMENT_NAMES[lbl] if lbl < len(SENTIMENT_NAMES) else f"class_{lbl}"
        result[name] = round(float(f1s[i]), 4)
    return result


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Macro OvR ROC-AUC — handles case where y_prob columns != actual classes.
    Returns 0.0 instead of crashing when classes are missing.
    """
    try:
        actual_labels = sorted(np.unique(y_true).tolist())
        n_actual      = len(actual_labels)
        n_prob_cols   = y_prob.shape[1]

        # y_prob must have same number of columns as actual classes
        if n_prob_cols != n_actual:
            logger.warning(
                "ROC-AUC skipped: y_prob has %d columns but %d actual classes. "
                "This happens when conflict class is absent from test data.",
                n_prob_cols, n_actual
            )
            return float("nan")

        auc = roc_auc_score(
            y_true, y_prob,
            multi_class = "ovr",
            average     = "macro",
            labels      = actual_labels
        )
        return float(auc)

    except Exception as e:
        logger.warning("ROC-AUC computation failed: %s", e)
        return float("nan")


def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index (edge-based bin computation).
    PSI < 0.10 → stable
    PSI 0.10–0.20 → moderate drift
    PSI > 0.20 → critical drift
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.percentile(expected, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

    except Exception as e:
        logger.warning("PSI failed: %s", e)
        return float("nan")


def label_psi(y_expected: np.ndarray, y_actual: np.ndarray) -> float:
    """PSI on class label distributions."""
    all_labels  = sorted(set(np.unique(y_expected)) | set(np.unique(y_actual)))
    n_classes   = len(all_labels)

    exp_counts = np.array([np.sum(y_expected == l) for l in all_labels], dtype=float)
    act_counts = np.array([np.sum(y_actual   == l) for l in all_labels], dtype=float)

    exp_pct = np.where(exp_counts == 0, 1e-6, exp_counts / (exp_counts.sum() + 1e-9))
    act_pct = np.where(act_counts == 0, 1e-6, act_counts / (act_counts.sum() + 1e-9))

    return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))


def ks_statistic(y_true: np.ndarray, y_prob_positive: np.ndarray) -> float:
    """KS statistic for positive class (class 0) vs rest."""
    try:
        y_bin = (y_true == 0).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return 0.0
        fpr, tpr, _ = roc_curve(y_bin, y_prob_positive)
        return float(np.max(tpr - fpr))
    except Exception as e:
        logger.warning("KS statistic failed: %s", e)
        return 0.0


def cost_sensitive_evaluation(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    domain:      str   = "e-commerce",
    review_cost: float = 50.0
) -> dict:
    """Business cost of ABSA misclassification (Rs.-based)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_cost = 0.0
    n_misclass = 0
    cost_matrix = np.array(COST_MATRIX)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            n_misclass += 1
            try:
                cost = cost_matrix[int(true_label)][int(pred_label)]
            except IndexError:
                cost = review_cost
            total_cost += cost

    per_sample_cost = total_cost / max(len(y_true), 1)
    neg_as_pos      = int(np.sum((y_true == 1) & (y_pred == 0)))
    pos_as_neg      = int(np.sum((y_true == 0) & (y_pred == 1)))

    result = {
        "n_misclassified":        n_misclass,
        "total_cost_inr":         round(total_cost, 2),
        "per_sample_cost_inr":    round(per_sample_cost, 4),
        "negative_missed":        neg_as_pos,
        "positive_flagged_wrong": pos_as_neg,
        "accuracy":               round(float(np.mean(y_true == y_pred)), 4),
    }

    logger.info(
        "Cost eval | misclass=%d  total_cost=Rs.%.2f  neg_missed=%d",
        n_misclass, total_cost, neg_as_pos
    )
    return result


def tune_negative_threshold(
    y_true:        np.ndarray,
    y_prob_neg:    np.ndarray,
    target_recall: float = 0.85
) -> float:
    """Tunes threshold for negative class detection (recall >= target)."""
    try:
        y_bin = (y_true == 1).astype(int)
        if y_bin.sum() == 0:
            return 0.5
        precision, recall, thresholds = precision_recall_curve(y_bin, y_prob_neg)
        idxs = np.where(recall >= target_recall)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(precision[idxs])]
            thr    = thresholds[chosen] if chosen < len(thresholds) else 0.5
        else:
            thr = 0.5
        logger.info("Negative threshold: %.4f  |  recall>=%.2f", thr, target_recall)
        return float(thr)
    except Exception as e:
        logger.warning("Threshold tuning failed: %s", e)
        return 0.5