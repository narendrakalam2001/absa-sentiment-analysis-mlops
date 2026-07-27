# ============================================================
# TRAINING PIPELINE — Aspect-Based Sentiment Analysis (ABSA)
# FIXES:
#   1. conflict class missing → dynamic NUM_CLASSES from data
#   2. classification_report crash → labels= from actual y values
#   3. ROC-AUC nan → handle missing classes gracefully
#   4. Duplicate leakage warnings removed
#   5. LinearSVC convergence → handled in model_tuning.py
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection   import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics           import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config       import (
    RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR, LABEL_COL,
    SENTIMENT_NAMES, SENTIMENT_MAP, NUM_CLASSES
)
from src.data_loader  import (
    load_semeval_dataset, load_all_domains,
    validate_input_data, add_engineered_features, detect_feature_types
)
from src.preprocessing  import build_preprocessors
from src.model_tuning   import linear_models, tree_models, tune_models, train_mlp_pipeline
from src.evaluation     import (
    evaluate_models, select_best_model, calibrate_with_holdout,
    compute_feature_importance, compute_shap,
    save_model_and_card, mlflow_log_run, safe_predict_proba,
    plot_confusion_matrix
)
from src.leakage_check  import detect_leakage
from src.model_card     import build_model_card, save_model_card
from src.model_loader   import run_challenger_comparison
from src.metrics        import (
    compute_macro_f1, compute_roc_auc,
    psi, label_psi, ks_statistic,
    cost_sensitive_evaluation, tune_negative_threshold
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def run_training(domain: str = "restaurants"):

    start_time = time.time()

    # ── STEP 1: LOAD DATA ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading SemEval 2014 Task 4 | domain=%s", domain)
    logger.info("=" * 60)

    if domain == "both":
        df = load_all_domains()
    else:
        df = load_semeval_dataset(domain=domain, split="all")

    df = validate_input_data(df)

    # ── Detect actual classes present in data ─────────────────
    actual_labels    = sorted(df["label"].unique().tolist())
    actual_sentiments = [SENTIMENT_NAMES[i] for i in actual_labels
                         if i < len(SENTIMENT_NAMES)]
    n_actual_classes = len(actual_labels)

    logger.info(
        "Dataset  |  shape=%s  |  classes=%s  |  actual_labels=%s",
        df.shape,
        df[LABEL_COL].value_counts().to_dict(),
        actual_labels
    )

    if n_actual_classes < 2:
        raise ValueError(f"Need at least 2 classes, got: {actual_labels}")

    # ── STEP 2: FEATURE ENGINEERING ───────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Feature Engineering")
    logger.info("=" * 60)
    df = add_engineered_features(df)

    # ── STEP 3: FEATURE TYPE DETECTION ────────────────────────
    logger.info("STEP 3 — Feature Type Detection")
    text_cols, num_cols, bin_cols = detect_feature_types(df)

    X = df.drop(columns=[LABEL_COL, "label", "split", "domain"], errors="ignore")
    y = df["label"]

    # ── STEP 4: TRAIN / VAL / TEST SPLIT ──────────────────────
    logger.info("STEP 4 — Train / Val / Test Split")

    if "split" in df.columns and df["split"].nunique() > 1:
        train_mask   = df["split"] == "train"
        X_train_full = X[train_mask].reset_index(drop=True)
        y_train_full = y[train_mask].reset_index(drop=True)
        X_test       = X[~train_mask].reset_index(drop=True)
        y_test       = y[~train_mask].reset_index(drop=True)
        logger.info("Using official SemEval train/test split")
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full,
        random_state=RANDOM_STATE
    )
    logger.info("Split  |  train_fit=%d  val=%d  test=%d",
                len(X_train_fit), len(X_val), len(X_test))

    # ── STEP 5: LEAKAGE DETECTION ─────────────────────────────
    logger.info("STEP 5 — Leakage Detection")
    leak_warnings = detect_leakage(X_train_fit, y_train_fit, X_test, y_test)
    if not leak_warnings:
        logger.info("No obvious leakage detected")

    # ── STEP 6: BUILD PREPROCESSORS ───────────────────────────
    logger.info("STEP 6 — Building Preprocessors")
    preprocessor_main, preprocessor_context, feature_order = build_preprocessors(
        text_cols, num_cols, bin_cols, X_train_fit
    )

    # ── STEP 7: LINEAR MODELS ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7 — Tuning Linear Models ...")
    logger.info("=" * 60)
    linear_pipelines = tune_models(
        linear_models, preprocessor_main,
        X_train_fit, y_train_fit, use_smote=True,
    )

    # ── STEP 8: TREE MODELS ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8 — Tuning Tree Models ...")
    logger.info("=" * 60)
    tree_pipelines = tune_models(
        tree_models, preprocessor_context,
        X_train_fit, y_train_fit, use_smote=True,
    )

    # ── STEP 9: NEURAL NETWORK ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9 — Training Neural Network (MLP) ...")
    logger.info("=" * 60)
    mlp_pipe = train_mlp_pipeline(
        X_train_fit, y_train_fit, preprocessor_main, use_smote=True
    )

    all_pipelines = {**linear_pipelines, **tree_pipelines, "NeuralNet": mlp_pipe}

    # ── STEP 10: EVALUATE ALL MODELS ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 10 — Evaluating All Models ...")
    logger.info("=" * 60)
    summary = evaluate_models(all_pipelines, X_train_fit, y_train_fit, X_test, y_test)

    print("\n" + "=" * 60)
    print("ALL MODELS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # ── STEP 11: SELECT BEST MODEL ────────────────────────────
    logger.info("STEP 11 — Selecting Best Model")
    selected_name, selected_pipe = select_best_model(
        summary, all_pipelines, linear_pipelines, tree_pipelines
    )

    # ── STEP 12: DETAILED EVALUATION ──────────────────────────
    logger.info("STEP 12 — Detailed Evaluation of Best Model")

    y_prob_sel = safe_predict_proba(selected_pipe, X_test)
    y_pred_sel = selected_pipe.predict(X_test)

    # ── Tune negative threshold ────────────────────────────────
    # Use only columns that actually exist in probability output
    n_prob_cols = y_prob_sel.shape[1]
    neg_label   = SENTIMENT_MAP.get("negative", 1)

    if neg_label < n_prob_cols:
        neg_threshold = tune_negative_threshold(
            y_test.values, y_prob_sel[:, neg_label], target_recall=0.85
        )
    else:
        neg_threshold = 0.45
        logger.warning("negative class not in probability output — using default threshold 0.45")

    thresholds_dict = {"negative": round(neg_threshold, 4)}

    # ── Classification report — use ONLY actual labels ────────
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {selected_name}")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred_sel,
        labels       = actual_labels,
        target_names = actual_sentiments,
        zero_division= 0
    ))

    # ── Confusion matrix ──────────────────────────────────────
    os.makedirs(os.path.join("docs", "plots"), exist_ok=True)

    cm = confusion_matrix(y_test, y_pred_sel, labels=actual_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=actual_sentiments,
        yticklabels=actual_sentiments
    )
    plt.title(f"{selected_name} — Confusion Matrix (ABSA)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join("docs", "plots", "confusion_matrix.png"), dpi=120)
    plt.close()
    logger.info("Confusion matrix saved")

    # ── ROC curves (only if >2 classes and prob available) ────
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics       import RocCurveDisplay

        y_bin = label_binarize(y_test, classes=actual_labels)
        fig, axes = plt.subplots(1, n_actual_classes, figsize=(5*n_actual_classes, 4))
        if n_actual_classes == 1:
            axes = [axes]

        for i, (ax, cls_name) in enumerate(zip(axes, actual_sentiments)):
            if y_bin[:, i].sum() == 0 or i >= n_prob_cols:
                ax.text(0.5, 0.5, f"No samples\n{cls_name}", ha="center", va="center")
                continue
            RocCurveDisplay.from_predictions(
                y_bin[:, i], y_prob_sel[:, i],
                name=cls_name, ax=ax
            )
            ax.set_title(f"ROC — {cls_name}")

        plt.suptitle(f"{selected_name} — OvR ROC Curves (ABSA)")
        plt.tight_layout()
        plt.savefig(os.path.join("docs", "plots", "roc_curves.png"), dpi=120)
        plt.close()
    except Exception as e:
        logger.warning("ROC curve plotting skipped: %s", e)

    # ── STEP 13: COST-SENSITIVE EVALUATION ────────────────────
    logger.info("STEP 13 — Cost-Sensitive Business Evaluation")
    cost_result = cost_sensitive_evaluation(y_test.values, y_pred_sel, domain=domain)
    print("\nCOST EVALUATION")
    for k, v in cost_result.items():
        print(f"  {k}: {v}")

    # ── STEP 14: CALIBRATION ──────────────────────────────────
    logger.info("STEP 14 — Probability Calibration (isotonic)")
    cal_pipe     = calibrate_with_holdout(selected_pipe, X_val, y_val)
    y_prob_final = None

    if cal_pipe is not None:
        try:
            y_prob_final = cal_pipe.predict_proba(X_test)
            y_pred_final = cal_pipe.predict(X_test)
            logger.info("Post-calibration macro-F1: %.4f",
                        compute_macro_f1(y_test, y_pred_final))
        except Exception as e:
            logger.warning("Calibrated prediction failed: %s", e)
            cal_pipe = None

    final_pipe  = cal_pipe if cal_pipe is not None else selected_pipe
    final_probs = y_prob_final if y_prob_final is not None else y_prob_sel
    final_preds = final_pipe.predict(X_test)

    # ── STEP 15: REPEATED CV ──────────────────────────────────
    logger.info("STEP 15 — Repeated CV Stability Check")
    try:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
        rep_scores = cross_val_score(
            selected_pipe, X_train_fit, y_train_fit,
            scoring="f1_macro", cv=rskf, n_jobs=1,
        )
        logger.info("Repeated CV  |  mean_macro_f1=%.4f  std=%.4f",
                    rep_scores.mean(), rep_scores.std())
    except Exception as e:
        logger.warning("Repeated CV failed: %s", e)
        rep_scores = np.array([0.0])

    # ── STEP 16: FEATURE IMPORTANCE ───────────────────────────
    logger.info("STEP 16 — Feature Importance")
    fi = compute_feature_importance(selected_pipe, X_train_fit, y_train_fit)
    if fi is not None:
        print("\nTOP FEATURE IMPORTANCES")
        print(fi.head(10).to_string())

    # ── STEP 17: SHAP ─────────────────────────────────────────
    logger.info("STEP 17 — SHAP Explainability")
    shap_result = compute_shap(selected_pipe, X_train_fit, X_test.head(100))

    # ── STEP 18: PSI DRIFT ────────────────────────────────────
    logger.info("STEP 18 — PSI Feature Drift (train vs test)")
    psi_scores = {}
    for col in num_cols:
        if col in X_train_fit.columns and col in X_test.columns:
            psi_scores[col] = psi(X_train_fit[col].values, X_test[col].values)

    psi_scores["_label_distribution"] = label_psi(y_train_fit.values, y_test.values)

    psi_df = (
        pd.Series(psi_scores)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "drift_score"})
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    psi_df.to_csv(os.path.join(MODEL_DIR, "feature_drift_report.csv"), index=False)
    print("\nTOP PSI SCORES (train vs test)")
    print(psi_df.head(10).to_string())

    # ── STEP 19: MONITOR SCORES ───────────────────────────────
    logger.info("STEP 19 — Saving Monitor Scores")
    label_inv = {v: k for k, v in SENTIMENT_MAP.items()}

    # Build probability columns dynamically
    monitor_data = {
        "predicted": [label_inv.get(p, f"class_{p}") for p in final_preds],
        "true_label":[label_inv.get(t, f"class_{t}") for t in y_test.values],
        "correct":   (final_preds == y_test.values).astype(int),
    }
    for i, lbl in enumerate(actual_labels):
        sent_name = label_inv.get(lbl, f"class_{lbl}")
        if i < final_probs.shape[1]:
            monitor_data[f"prob_{sent_name}"] = final_probs[:, i]

    monitor_df = pd.DataFrame(monitor_data)
    monitor_df.to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)
    logger.info("Monitor scores saved — %d records", len(monitor_df))

    # ── STEP 20: FINAL METRICS ────────────────────────────────
    logger.info("STEP 20 — Computing Final Metrics")

    from src.metrics import compute_per_class_f1, compute_weighted_f1

    macro_f1    = compute_macro_f1(y_test, final_preds)
    weighted_f1 = compute_weighted_f1(y_test, final_preds)
    accuracy    = float(accuracy_score(y_test, final_preds))
    per_cls_f1  = compute_per_class_f1(y_test, final_preds)
    gap         = float(abs(
        compute_macro_f1(y_train_fit, selected_pipe.predict(X_train_fit)) - macro_f1
    ))

    # ROC-AUC — only if prob columns match actual classes
    try:
        roc_auc = compute_roc_auc(y_test.values, final_probs)
        if np.isnan(roc_auc):
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0

    # KS statistic
    try:
        neg_col_idx = actual_labels.index(neg_label) if neg_label in actual_labels else -1
        ks = ks_statistic(y_test.values, final_probs[:, neg_col_idx]) \
             if neg_col_idx >= 0 else 0.0
    except Exception:
        ks = 0.0

    metrics_dict = {
        "macro_f1":      round(macro_f1,    4),
        "weighted_f1":   round(weighted_f1, 4),
        "accuracy":      round(accuracy,    4),
        "roc_auc":       round(roc_auc,     4),
        "ks_negative":   round(ks,          4),
        "train_val_gap": round(gap,          4),
        "cv_mean_f1":    round(float(rep_scores.mean()), 4),
        "cv_std_f1":     round(float(rep_scores.std()),  4),
        "n_classes_actual": n_actual_classes,
    }

    print(f"\n{'='*60}")
    print("FINAL METRICS")
    print(f"{'='*60}")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v}")

    # ── STEP 21: MODEL CARD ───────────────────────────────────
    logger.info("STEP 21 — Building Model Card")

    class_dist = {
        SENTIMENT_NAMES[i]: int((y_test == i).sum())
        for i in actual_labels
    }

    model_card = build_model_card(
        selected_name      = selected_name,
        train_fit_size     = int(len(X_train_fit)),
        val_size           = int(len(X_val)),
        test_size          = int(len(X_test)),
        domain             = domain,
        class_distribution = class_dist,
        metrics            = metrics_dict,
        thresholds         = thresholds_dict,
        cost_result        = cost_result,
        feature_order      = feature_order,
        fi_dict            = fi.head(20).to_dict() if fi is not None else None,
        shap_dict          = shap_result.get("shap_top", {}) if shap_result else None,
        per_class_f1       = per_cls_f1,
    )

    card_path = save_model_card(model_card, MODEL_DIR, selected_name)

    # ── STEP 22: SAVE MODEL ───────────────────────────────────
    logger.info("STEP 22 — Saving Model Artifact")
    model_path = save_model_and_card(selected_name, final_pipe, model_card)
    summary.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    # ── STEP 23: CHAMPION-CHALLENGER ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 23 — Champion vs Challenger (3-gate)")
    logger.info("=" * 60)

    challenger_result = run_challenger_comparison(
        challenger_name       = selected_name,
        challenger_macro_f1   = macro_f1,
        challenger_roc_auc    = roc_auc,
        challenger_gap        = gap,
        challenger_model_path = model_path,
        challenger_thresholds = thresholds_dict,
        challenger_card_path  = card_path,
    )

    print("\n" + "=" * 60)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"Reason: {challenger_result['reason']}")
    print("=" * 60)

    # ── STEP 24: MLFLOW ───────────────────────────────────────
    logger.info("STEP 24 — MLflow Experiment Tracking")
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{selected_name}"
    mlflow_log_run(run_name, selected_name, final_pipe, model_card)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best model: {selected_name}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Challenger: {challenger_result['decision']}")
    print(f"Time      : {elapsed:.1f}s")
    print("=" * 60)

    return selected_name, final_pipe, model_card


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="restaurants",
                        choices=["restaurants", "laptops", "both"])
    args = parser.parse_args()
    run_training(domain=args.domain)