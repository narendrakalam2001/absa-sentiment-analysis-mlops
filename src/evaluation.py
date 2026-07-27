# ============================================================
# EVALUATION — Aspect-Based Sentiment Analysis (ABSA)
# ============================================================

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config  import (
    SENTIMENT_NAMES, NUM_CLASSES, MODEL_DIR, RANDOM_STATE
)
from src.metrics import compute_macro_f1, compute_roc_auc, cost_sensitive_evaluation

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT PROBA
# ============================================================

def safe_predict_proba(pipe, X) -> np.ndarray:
    """
    Returns (n_samples, n_classes) probability array.
    Falls back to decision function for models without predict_proba.
    """
    clf = pipe[-1]  # last step

    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)

    # LinearSVC / RidgeClassifier → use decision function
    if hasattr(clf, "decision_function"):
        df_scores = pipe.decision_function(X)
        if df_scores.ndim == 1:
            # Binary case — shouldn't happen for 4-class, but safe
            df_scores = np.column_stack([-df_scores, df_scores])
        # Softmax normalization
        e = np.exp(df_scores - df_scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # Last resort: one-hot of predicted class
    y_pred = pipe.predict(X)
    proba  = np.zeros((len(y_pred), NUM_CLASSES))
    for i, p in enumerate(y_pred):
        proba[i, int(p)] = 1.0
    return proba


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    pipelines: dict,
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
) -> pd.DataFrame:
    """
    Evaluates all trained pipelines on test set.
    Returns summary DataFrame sorted by macro-F1 descending.
    """
    records = []

    for name, pipe in pipelines.items():
        try:
            y_pred     = pipe.predict(X_test)
            y_prob     = safe_predict_proba(pipe, X_test)
            y_pred_tr  = pipe.predict(X_train)

            macro_f1   = compute_macro_f1(y_test, y_pred)
            weighted_f1= float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            accuracy   = float(accuracy_score(y_test, y_pred))
            roc_auc    = compute_roc_auc(y_test.values, y_prob)
            train_f1   = compute_macro_f1(y_train, y_pred_tr)
            gap        = float(abs(train_f1 - macro_f1))

            records.append({
                "model":       name,
                "macro_f1":    round(macro_f1,    4),
                "weighted_f1": round(weighted_f1, 4),
                "accuracy":    round(accuracy,    4),
                "roc_auc":     round(roc_auc,     4),
                "train_f1":    round(train_f1,    4),
                "gap":         round(gap,          4),
            })

            logger.info(
                "%-20s  macro_f1=%.4f  weighted_f1=%.4f  roc_auc=%.4f  gap=%.4f",
                name, macro_f1, weighted_f1, roc_auc, gap
            )

        except Exception as e:
            logger.error("Evaluation failed for %s: %s", name, e)

    df = pd.DataFrame(records).sort_values("macro_f1", ascending=False)
    return df.reset_index(drop=True)


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    summary:   pd.DataFrame,
    pipelines: dict,
    linear_pipes: dict,
    tree_pipes:   dict,
) -> tuple:
    """
    Selects the best model by macro-F1.
    Ties broken by weighted-F1 then ROC-AUC.
    Returns (name, pipeline).
    """
    if summary.empty:
        raise ValueError("No models evaluated successfully")

    best_row  = summary.iloc[0]
    best_name = best_row["model"]
    best_pipe = pipelines.get(best_name)

    logger.info(
        "Best model: %s  macro_f1=%.4f  roc_auc=%.4f",
        best_name, best_row["macro_f1"], best_row["roc_auc"]
    )
    return best_name, best_pipe


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_with_holdout(pipe, X_cal, y_cal):
    """
    Isotonic-regression probability calibration on holdout set.
    Skips models that don't support calibration (e.g. LinearSVC with cv=prefit).
    """
    try:
        cal_pipe = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        cal_pipe.fit(X_cal, y_cal)
        logger.info("Probability calibration applied (isotonic)")
        return cal_pipe
    except Exception as e:
        logger.warning("Calibration skipped: %s", e)
        return None


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(
    pipe,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n:   int = 20
) -> pd.DataFrame | None:
    """
    Extracts feature importances from tree-based classifiers.
    For linear models, uses |coefficient| as importance.
    """
    try:
        clf = pipe[-1]
        pre = pipe[0]   # preprocessor

        # Get feature names
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = None

        importances = None

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_

        elif hasattr(clf, "coef_"):
            coef = clf.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        if importances is None:
            return None

        if feat_names is None or len(feat_names) != len(importances):
            feat_names = [f"feature_{i}" for i in range(len(importances))]

        fi = pd.DataFrame({
            "feature":    feat_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)

        logger.info("Feature importance computed — top feature: %s", fi.iloc[0]["feature"])
        return fi

    except Exception as e:
        logger.warning("Feature importance failed: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(
    pipe,
    X_train: pd.DataFrame,
    X_sample: pd.DataFrame,
    max_samples: int = 200
) -> dict | None:
    """
    Computes SHAP summary for tree-based or linear models.
    Returns dict with top SHAP features per class.
    """
    try:
        import shap

        clf = pipe[-1]
        pre = pipe[0]

        X_tr = pre.transform(X_train[:max_samples])
        X_sp = pre.transform(X_sample)

        from scipy.sparse import issparse
        if issparse(X_tr): X_tr = X_tr.toarray()
        if issparse(X_sp): X_sp = X_sp.toarray()

        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(X_tr.shape[1])]

        if hasattr(clf, "feature_importances_"):
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sp)
        else:
            explainer   = shap.LinearExplainer(clf, X_tr)
            shap_values = explainer.shap_values(X_sp)

        # Top features per class
        shap_top = {}
        if isinstance(shap_values, list):
            for class_idx, sv in enumerate(shap_values):
                mean_abs = np.abs(sv).mean(axis=0)
                top_idx  = mean_abs.argsort()[::-1][:10]
                shap_top[SENTIMENT_NAMES[class_idx]] = {
                    str(feat_names[i]): float(mean_abs[i]) for i in top_idx
                }
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx  = mean_abs.argsort()[::-1][:10]
            shap_top["overall"] = {
                str(feat_names[i]): float(mean_abs[i]) for i in top_idx
            }

        logger.info("SHAP computed successfully")
        return {"shap_top": shap_top}

    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        return None


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=SENTIMENT_NAMES, yticklabels=SENTIMENT_NAMES
    )
    plt.title(f"{model_name} — Confusion Matrix (ABSA)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info("Confusion matrix saved: %s", save_path)


# ============================================================
# SAVE MODEL + CARD
# ============================================================

def save_model_and_card(
    model_name: str,
    pipe,
    model_card: dict
) -> str:
    """
    Saves pipeline (.joblib) and returns the model path.
    Also saves model card JSON.
    Filename format: absa_model_{model_name}_v{version}.joblib
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Auto-increment version
    version = 1
    while os.path.exists(
        os.path.join(MODEL_DIR, f"absa_model_{model_name}_v{version}.joblib")
    ):
        version += 1

    model_filename = f"absa_model_{model_name}_v{version}.joblib"
    model_path     = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(pipe, model_path)
    logger.info("Model saved: %s", model_path)

    return model_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:   str,
    model_name: str,
    pipe,
    model_card: dict,
    X_train_sample: pd.DataFrame = None,
):
    """
    Logs training run to MLflow.
    Gracefully skips if MLflow unavailable.
    """
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("ABSA-SemEval-2014")

        with mlflow.start_run(run_name=run_name):
            metrics = model_card.get("metrics", {})

            mlflow.log_metric("macro_f1",    metrics.get("macro_f1",    0))
            mlflow.log_metric("weighted_f1", metrics.get("weighted_f1", 0))
            mlflow.log_metric("accuracy",    metrics.get("accuracy",    0))
            mlflow.log_metric("roc_auc",     metrics.get("roc_auc",     0))

            mlflow.log_param("model_name",   model_name)
            mlflow.log_param("dataset",      "SemEval 2014 Task 4")
            mlflow.log_param("domain",       model_card.get("domain", "restaurants+laptops"))

            mlflow.sklearn.log_model(pipe, artifact_path="absa_model")

            logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.warning("MLflow logging skipped: %s", e)
