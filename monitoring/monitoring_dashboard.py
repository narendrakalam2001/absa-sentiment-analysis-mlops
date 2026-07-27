# ============================================================
# MONITORING DASHBOARD — Aspect-Based Sentiment Analysis (ABSA)
# Streamlit — 5 sections:
#   1. Real-Time Alerts
#   2. Champion vs Challenger History
#   3. KPIs + Charts
#   4. PSI Drift Monitoring
#   5. Recent Predictions
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json
import os

st.set_page_config(
    page_title = "ABSA Sentiment Dashboard",
    page_icon  = "💬",
    layout     = "wide",
)

st.title("💬 ABSA Sentiment Analysis — Monitoring Dashboard")
st.caption("SemEval 2014 Task 4 | E-Commerce · Banking · Healthcare | Champion-Challenger System")

# ── API URL ────────────────────────────────────────────────────
API_URL = os.getenv(
    "ABSA_API_URL", "http://localhost:8000"
) + "/predict"

# ── PSI thresholds ─────────────────────────────────────────────
PSI_MODERATE         = 0.10
PSI_HIGH             = 0.20
NEGATIVE_RATE_ALERT  = 0.35
ESCALATE_RATE_ALERT  = 0.20

# ── Path resolution — Streamlit Cloud safe ─────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "absa_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "absa_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "absa_models", "challenger_log.json")
REGISTRY_PATH   = os.path.join(BASE_DIR, "absa_models", "latest_model.json")
EXP_PATH        = os.path.join(BASE_DIR, "absa_models", "model_experiment_results.csv")

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
    "conflict": "#f39c12",
}


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔮 Live Aspect Sentiment Prediction")

review_text  = st.sidebar.text_area(
    "Review Text",
    value="The pasta was absolutely delicious but the service was very slow.",
    height=100
)
aspect_term  = st.sidebar.text_input("Aspect Term", value="service")
domain       = st.sidebar.selectbox(
    "Domain",
    ["restaurants", "laptops", "banking"],
    index=0
)

if st.sidebar.button("🚀 Predict Sentiment"):
    payload = {
        "text":        review_text,
        "aspect_term": aspect_term,
        "domain":      domain,
    }
    with st.sidebar:
        with st.spinner("Calling API... (first request may take 30–60s on Render cold start)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                if response.status_code == 200:
                    result    = response.json()
                    sentiment = result.get("predicted_sentiment", "—")
                    decision  = result.get("decision", "—")
                    conf      = result.get("confidence", 0)

                    color_map = {
                        "ESCALATE": "red", "REVIEW": "orange",
                        "POSITIVE": "green", "NEUTRAL": "gray"
                    }
                    color = color_map.get(decision, "gray")

                    st.success("✅ Prediction received!")
                    st.markdown(f"**Sentiment:** `{sentiment}`")
                    st.markdown(f"**Confidence:** `{conf:.4f}`")
                    st.markdown(
                        f"<h3 style='color:{color}'>Decision: {decision}</h3>",
                        unsafe_allow_html=True
                    )

                    probs = result.get("probabilities", {})
                    if probs:
                        prob_df = pd.DataFrame(
                            list(probs.items()),
                            columns=["Sentiment", "Probability"]
                        )
                        st.bar_chart(prob_df.set_index("Sentiment"))

                    if result.get("rule_triggered"):
                        st.warning(f"⚡ Rule: {result['rule_triggered']}")
                    if result.get("business_action"):
                        st.info(result["business_action"])

                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("⏳ Request timed out (90s). Render waking up — retry in 30s.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Section 1 — Real-Time Monitoring Alerts")

alerts_found = False

# ── Alert: negative rate ──────────────────────────────────────
if os.path.exists(MONITOR_PATH):
    df_mon = pd.read_csv(MONITOR_PATH)

    if "predicted" in df_mon.columns:
        neg_rate = (df_mon["predicted"] == "negative").mean()
        if neg_rate > NEGATIVE_RATE_ALERT:
            st.error(
                f"🔴 HIGH NEGATIVE RATE: {neg_rate:.1%} of reviews negative "
                f"(threshold {NEGATIVE_RATE_ALERT:.0%}). Review product/service quality."
            )
            alerts_found = True

    if "correct" in df_mon.columns:
        accuracy = df_mon["correct"].mean()
        if accuracy < 0.70:
            st.warning(
                f"🟡 LOW ACCURACY on monitor set: {accuracy:.1%} (expected ≥ 70%). "
                "Consider retraining."
            )
            alerts_found = True

# ── Alert: PSI drift ──────────────────────────────────────────
if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns and len(df_psi) > 0:
        max_psi     = df_psi["drift_score"].max()
        top_feature = df_psi.iloc[0].get("feature", "unknown")
        if max_psi >= PSI_HIGH:
            st.error(
                f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}'. "
                "Immediate retraining recommended."
            )
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(
                f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Monitor closely."
            )
            alerts_found = True

# ── Alert: prediction log volume ─────────────────────────────
if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    if len(df_log) > 0:
        # Conflict rate
        conf_rate = (df_log.get("predicted_sentiment", pd.Series([])) == "conflict").mean() \
            if "predicted_sentiment" in df_log.columns else 0
        if conf_rate > 0.15:
            st.warning(
                f"🟡 HIGH CONFLICT RATE: {conf_rate:.1%} of predictions flagged as 'conflict'. "
                "Model may be uncertain — review data quality."
            )
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no monitoring alerts triggered.")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Section 2 — Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)

    if challenger_log:
        latest        = challenger_log[-1]
        dec_color     = "green" if latest["decision"] == "PROMOTED" else "red"
        icon          = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Challenger",   latest.get("challenger_name", "—"))
        col2.metric("Challenger Macro-F1", latest.get("challenger_macro_f1", "—"))
        col3.metric("Champion Macro-F1",   latest.get("champion_macro_f1",   "—") or "First Run")
        col4.metric("ROC-AUC",             latest.get("challenger_roc_auc",  "—"))

        st.markdown(
            f"<h4 style='color:{dec_color}'>"
            f"{icon} Decision: {latest['decision']} — {latest.get('reason', '')}"
            f"</h4>",
            unsafe_allow_html=True
        )

        # ── 3-Gate visual ──────────────────────────────────────
        if latest.get("gates"):
            g = latest["gates"]
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric(
                "Gate 1: Macro-F1 Improvement",
                "✅ PASS" if g.get("f1_improvement_passed") else "❌ FAIL"
            )
            gc2.metric(
                "Gate 2: ROC-AUC ≥ 0.80",
                "✅ PASS" if g.get("roc_auc_passed") else "❌ FAIL"
            )
            gc3.metric(
                "Gate 3: Train-Val Gap ≤ 10%",
                "✅ PASS" if g.get("gap_passed") else "❌ FAIL"
            )

        if len(challenger_log) > 1:
            with st.expander("📋 View Full Challenger History"):
                hist_df = pd.DataFrame(challenger_log)
                keep    = [c for c in [
                    "evaluated_at", "challenger_name", "challenger_macro_f1",
                    "challenger_roc_auc", "champion_name", "champion_macro_f1",
                    "decision", "reason"
                ] if c in hist_df.columns]
                st.dataframe(hist_df[keep], use_container_width=True)
    else:
        st.info("No challenger runs found yet.")
else:
    st.info("No challenger log found. Run training pipeline to generate challenger history.")


# ============================================================
# SECTION 3 — KPIs + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Section 3 — Model Performance KPIs & Charts")

# ── KPI metrics from model card ───────────────────────────────
if os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)

    card_path = reg.get("model_card_path", "")
    if card_path and os.path.exists(card_path):
        with open(card_path) as f:
            card = json.load(f)

        metrics = card.get("metrics", {})
        pcf1    = card.get("per_class_f1", {})

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Macro F1",      f"{metrics.get('macro_f1',    '—')}")
        k2.metric("Weighted F1",   f"{metrics.get('weighted_f1', '—')}")
        k3.metric("Accuracy",      f"{metrics.get('accuracy',    '—')}")
        k4.metric("ROC-AUC (OvR)", f"{metrics.get('roc_auc',    '—')}")
        k5.metric("CV Mean F1",    f"{metrics.get('cv_mean_f1', '—')}")

        # Per-class F1 bar chart
        if pcf1:
            fig, ax = plt.subplots(figsize=(8, 3))
            classes = list(pcf1.keys())
            values  = list(pcf1.values())
            colors  = [SENTIMENT_COLORS.get(c, "#3498db") for c in classes]
            ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("F1 Score")
            ax.set_title("Per-Class F1 Score (Test Set)")
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
            ax.axhline(0.70, color="red", linestyle="--", alpha=0.4, label="0.70 threshold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── Experiment comparison table ────────────────────────────────
if os.path.exists(EXP_PATH):
    df_exp = pd.read_csv(EXP_PATH)
    with st.expander("📋 All Models — Experiment Results"):
        st.dataframe(
            df_exp.style.highlight_max(
                subset=["macro_f1", "weighted_f1", "roc_auc"],
                color="#d4efdf"
            ).highlight_min(
                subset=["gap"],
                color="#d4efdf"
            ),
            use_container_width=True
        )

# ── Monitor scores distribution ────────────────────────────────
if os.path.exists(MONITOR_PATH):
    df_mon = pd.read_csv(MONITOR_PATH)

    col_a, col_b = st.columns(2)

    with col_a:
        if "predicted" in df_mon.columns:
            vc = df_mon["predicted"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_list = [SENTIMENT_COLORS.get(c, "#3498db") for c in vc.index]
            ax.bar(vc.index, vc.values, color=colors_list)
            ax.set_title("Predicted Sentiment Distribution")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_b:
        if "prob_negative" in df_mon.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df_mon["prob_negative"], bins=30, color="#e74c3c", alpha=0.7, edgecolor="white")
            ax.axvline(0.45, color="orange", linestyle="--", label="neg_threshold=0.45")
            ax.axvline(0.70, color="red",    linestyle="--", label="escalate=0.70")
            ax.set_title("Negative Probability Distribution")
            ax.set_xlabel("P(negative)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ============================================================
# SECTION 4 — PSI DRIFT MONITORING
# ============================================================

st.markdown("---")
st.subheader("📉 Section 4 — PSI Drift Monitoring (Train vs Test)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)

    if "drift_score" in df_psi.columns and "feature" in df_psi.columns:
        df_psi = df_psi.sort_values("drift_score", ascending=False)

        # Color-code by drift level
        def _psi_color(v):
            if v >= PSI_HIGH:     return "background-color: #f5b7b1"
            if v >= PSI_MODERATE: return "background-color: #fdebd0"
            return "background-color: #d4efdf"

        col_psi1, col_psi2 = st.columns([1, 1])

        with col_psi1:
            st.dataframe(
                df_psi.style.applymap(_psi_color, subset=["drift_score"]),
                use_container_width=True
            )

        with col_psi2:
            fig, ax = plt.subplots(figsize=(6, 4))
            top10 = df_psi.head(10)
            bar_colors = [
                "#e74c3c" if v >= PSI_HIGH else
                "#f39c12" if v >= PSI_MODERATE else
                "#2ecc71"
                for v in top10["drift_score"]
            ]
            ax.barh(top10["feature"][::-1], top10["drift_score"][::-1], color=bar_colors[::-1])
            ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
            ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label=f"Critical ({PSI_HIGH})")
            ax.set_xlabel("PSI Score")
            ax.set_title("Feature Drift — PSI (Top 10)")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Legend
        st.markdown(
            "🟢 PSI < 0.10 → Stable &nbsp;&nbsp; "
            "🟡 0.10–0.20 → Monitor &nbsp;&nbsp; "
            "🔴 > 0.20 → Retrain",
            unsafe_allow_html=True
        )
    else:
        st.info("PSI report found but missing expected columns (feature, drift_score).")
else:
    st.info("No PSI drift report found. Run training pipeline to generate.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("🧾 Section 5 — Recent API Predictions")

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)

    if len(df_log) > 0:
        df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], unit="s", errors="coerce")
        df_log = df_log.sort_values("timestamp", ascending=False)

        # Summary KPIs
        lk1, lk2, lk3, lk4 = st.columns(4)
        lk1.metric("Total Predictions",   len(df_log))
        if "predicted_sentiment" in df_log.columns:
            lk2.metric("Negative Rate",   f"{(df_log['predicted_sentiment']=='negative').mean():.1%}")
            lk3.metric("Escalate Rate",   f"{(df_log.get('decision', pd.Series([])) == 'ESCALATE').mean():.1%}"
                       if 'decision' in df_log.columns else "—")
            lk4.metric("Avg Confidence",  f"{df_log.get('confidence', pd.Series([0])).mean():.3f}"
                       if 'confidence' in df_log.columns else "—")

        # Recent 20 predictions
        display_cols = [c for c in [
            "timestamp", "aspect_term", "predicted_sentiment",
            "decision", "confidence", "domain", "rule_triggered"
        ] if c in df_log.columns]

        st.dataframe(df_log[display_cols].head(20), use_container_width=True)

        # Trend: sentiment over last 100 predictions
        if "predicted_sentiment" in df_log.columns and len(df_log) >= 10:
            last100 = df_log.head(100)
            sent_trend = last100["predicted_sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            colors_t = [SENTIMENT_COLORS.get(c, "#3498db") for c in sent_trend.index]
            ax.pie(
                sent_trend.values,
                labels     = sent_trend.index,
                colors     = colors_t,
                autopct    = "%1.1f%%",
                startangle = 90,
            )
            ax.set_title("Last 100 Predictions — Sentiment Mix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No predictions logged yet. Use the sidebar to make live predictions.")
else:
    st.info("No prediction log found at logs/prediction_logs.csv.")

st.markdown("---")
st.caption(
    "ABSA Monitoring Dashboard · SemEval 2014 Task 4 · "
    "Built by Narendra Kalam (MSc CS — NASSCOM Gold Medalist)"
)
