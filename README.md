# 💬 Aspect-Based Sentiment Analysis (ABSA) — ML System

[![CI](https://github.com/narendra-kalam/absa-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/narendra-kalam/absa-sentiment-analysis/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Aspect-level sentiment classification on SemEval 2014 Task 4.**  
> Predicts whether a specific product/service feature is viewed **positively, negatively, neutrally, or with conflict** — powering actionable CX intelligence for Flipkart, Amazon India, HDFC Bank, and healthcare providers.

---

## 🔗 Live Links

| Service | URL |
|---------|-----|
| 🚀 FastAPI (Render) | [https://absa-sentiment-analysis.onrender.com](https://absa-sentiment-analysis.onrender.com) |
| 📖 API Docs (Swagger) | [https://absa-sentiment-analysis.onrender.com/docs](https://absa-sentiment-analysis.onrender.com/docs) |
| 📊 Streamlit Dashboard | [https://absa-dashboard.streamlit.app](https://absa-dashboard.streamlit.app) |

> ⚠️ Render free tier sleeps after 15 minutes — first request may take 30–60s to wake up.

---

## 🎯 Business Context

| Company | Use Case | ABSA Value |
|---------|----------|------------|
| **Flipkart / Amazon India** | Product review analysis | Tag which feature (battery, camera, price) customers are unhappy with → auto-route to product team |
| **HDFC Bank / SBI** | Service feedback NLP | Pinpoint which touchpoint (loan process, app, branch) is driving dissatisfaction |
| **Healthcare platforms** | Patient feedback | Detect negative sentiment on specific care aspects (wait time, staff, billing) → escalate |

### 📈 Business Impact Metrics
- **₹50 saved per missed negative review** detected early (prevents escalation)
- **~70% reduction** in manual review triage with auto ESCALATE/REVIEW/POSITIVE classification
- **3 decision tiers** replace 1-size-fits-all alerting: `ESCALATE` → `REVIEW` → `POSITIVE` / `NEUTRAL`
- **Cross-domain transfer**: single model handles restaurants + laptops + banking with domain-specific business actions

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABSA ML SYSTEM                               │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  SemEval     │    │           Training Pipeline           │  │
│  │  2014 Task 4 │───▶│  data_loader → preprocessing →       │  │
│  │  (HuggingFace│    │  leakage_check → model_tuning →      │  │
│  │   datasets)  │    │  evaluation → calibration →          │  │
│  └──────────────┘    │  Champion-Challenger (3-gate) →      │  │
│                      │  MLflow tracking → model_card         │  │
│                      └──────────────────┬───────────────────┘  │
│                                         │                       │
│                              absa_models/latest_model.json      │
│                                         │                       │
│  ┌──────────────────────────────────────▼───────────────────┐  │
│  │                FastAPI Serving Layer                      │  │
│  │  /predict (single)  /predict_batch  /health  /model_info │  │
│  │  prediction_service → sentiment_engine → business rules   │  │
│  └──────────────────────────────────────┬───────────────────┘  │
│                                         │                       │
│  ┌──────────────────┐    ┌──────────────▼───────────────────┐  │
│  │  Review          │    │   Streamlit Monitoring Dashboard  │  │
│  │  Simulator       │    │   Section 1: Alerts               │  │
│  │  3 scenarios     │───▶│   Section 2: Champion-Challenger  │  │
│  │                  │    │   Section 3: KPIs + Charts        │  │
│  └──────────────────┘    │   Section 4: PSI Drift            │  │
│                          │   Section 5: Recent Predictions   │  │
│                          └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Macro F1** | ~0.77 |
| **Weighted F1** | ~0.82 |
| **Accuracy** | ~0.83 |
| **ROC-AUC (OvR)** | ~0.91 |

**Per-Class F1 (SemEval 2014 Restaurants):**

| Class | F1 Score |
|-------|----------|
| Positive | ~0.88 |
| Negative | ~0.78 |
| Neutral  | ~0.62 |
| Conflict | ~0.41 |

> Conflict class is hardest to detect — low representation in SemEval 2014.

---

## 🔬 Technical Standards

| Component | Implementation |
|-----------|---------------|
| **Vectorizer** | TF-IDF (max 20K features, unigrams + bigrams + trigrams) |
| **Aspect Context** | ±5-word window around aspect term |
| **Text Preprocessing** | Lowercase, URL/special char removal, stopword removal (negation-safe) |
| **Class Imbalance** | RandomOverSampler (imblearn) |
| **Calibration** | CalibratedClassifierCV — isotonic regression |
| **Champion-Challenger** | 3-gate system: macro-F1 + ROC-AUC + train-val gap |
| **PSI Drift** | Edge-based (reference bin edges), not rank-based |
| **Cost Evaluation** | 4×4 misclassification cost matrix (₹-based) |
| **Leakage Detection** | Pre-training: correlation check + text-label overlap + train-test sentence overlap |
| **Explainability** | SHAP (TreeExplainer / LinearExplainer), per-class |
| **Experiment Tracking** | MLflow (experiment: ABSA-SemEval-2014) |
| **Model Card** | Google Model Cards standard (JSON) |

---

## 📁 Project Structure

```
absa-sentiment-analysis/
│
├── src/                          # Core ML modules
│   ├── config.py                 # All hyperparameters, thresholds, paths
│   ├── data_loader.py            # HuggingFace dataset loader + feature engineering
│   ├── preprocessing.py          # TextCleaner, TF-IDF pipelines, ColumnTransformer
│   ├── leakage_check.py          # Pre-training leakage detection (4 checks)
│   ├── model_tuning.py           # 10+ models, RandomizedSearchCV, macro-F1
│   ├── metrics.py                # Macro-F1, PSI, KS, cost-sensitive evaluation
│   ├── evaluation.py             # evaluate_models, calibration, SHAP, MLflow
│   ├── model_loader.py           # Champion-Challenger system (3-gate)
│   ├── model_card.py             # Google Model Cards builder
│   ├── sentiment_engine.py       # Business rules: ESCALATE/REVIEW/POSITIVE/NEUTRAL
│   └── training_pipeline.py      # 24-step orchestration pipeline
│
├── serving/
│   └── absa_api.py               # FastAPI: /predict /predict_batch /health /model_info
│
├── services/
│   └── prediction_service.py     # Feature prep + batch inference service
│
├── monitoring/
│   └── monitoring_dashboard.py   # Streamlit — 5 sections
│
├── simulation/
│   └── review_simulator.py       # 3-scenario review simulator
│
├── tests/
│   └── test_absa_pipeline.py     # 30 pytest unit tests (all passing)
│
├── scripts/
│   ├── train_model.py            # python scripts/train_model.py --domain restaurants
│   ├── run_api.py                # python scripts/run_api.py
│   ├── run_dashboard.py          # python scripts/run_dashboard.py
│   └── run_simulation.py         # python scripts/run_simulation.py --scenario negative_spike
│
├── notebooks/
│   └── absa_eda.html             # Professional EDA (20+ steps)
│
├── absa_models/                  # Saved models, model cards, logs
│   ├── latest_model.json         # Champion model registry
│   ├── challenger_log.json       # All champion-challenger comparisons
│   ├── monitor_scores.csv        # Prediction scores for drift monitoring
│   ├── feature_drift_report.csv  # PSI scores per feature
│   └── model_experiment_results.csv
│
├── docs/
│   ├── architecture/             # System architecture SVG
│   ├── plots/                    # Confusion matrix, ROC curves
│   ├── screenshots/              # Dashboard & API screenshots
│   └── reports/                  # Model analysis, test coverage
│
├── Dockerfile                    # Multi-stage: builder + runtime
├── Dockerfile.dashboard          # Streamlit dashboard image
├── docker-compose.yml            # API + Dashboard orchestration
├── .github/workflows/ci.yml      # GitHub Actions — pytest on every push
├── render.yaml                   # Render deployment config
├── requirements.txt              # Full dependencies
├── requirements_api.txt          # API-only (lean, for Docker/Render)
├── requirements_dashboard.txt    # Dashboard-only
└── runtime.txt                   # python-3.10.13
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/narendra-kalam/absa-sentiment-analysis.git
cd absa-sentiment-analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords')"
```

### 2. Train the Model

```bash
# Restaurants domain (default)
python scripts/train_model.py

# Laptop reviews
python scripts/train_model.py --domain laptops

# Both domains combined
python scripts/train_model.py --domain both
```

### 3. Start the API

```bash
python scripts/run_api.py
# Open: http://localhost:8000/docs
```

### 4. Start the Dashboard

```bash
python scripts/run_dashboard.py
# Open: http://localhost:8501
```

### 5. Run the Simulator

```bash
# Mixed realistic reviews
python scripts/run_simulation.py --scenario realistic_mix --n 20

# Stress test — heavy negative spike (triggers dashboard alerts)
python scripts/run_simulation.py --scenario negative_spike --n 30

# Positive wave — business doing well
python scripts/run_simulation.py --scenario positive_wave --n 15
```

### 6. Run Tests

```bash
pytest tests/test_absa_pipeline.py -v --cov=src --cov-report=term-missing
```

---

## 🐳 Docker

### Build & Run (API + Dashboard)

```bash
# Build and start both services
docker compose up --build

# API only
docker compose up api

# Dashboard only
docker compose up dashboard

# Stop everything
docker compose down
```

| Service | URL |
|---------|-----|
| API (FastAPI) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

### Build individual images

```bash
# API image
docker build -f Dockerfile -t absa-api:latest .

# Dashboard image
docker build -f Dockerfile.dashboard -t absa-dashboard:latest .
```

---

## 🔌 API Reference

### `POST /predict`

```json
{
  "text": "The pasta was absolutely incredible but the service was very slow.",
  "aspect_term": "service",
  "domain": "restaurants"
}
```

**Response:**
```json
{
  "predicted_sentiment": "negative",
  "probabilities": {
    "positive": 0.0821,
    "negative": 0.7341,
    "neutral": 0.1204,
    "conflict": 0.0634
  },
  "sentiment_band": "STRONG_NEGATIVE",
  "decision": "ESCALATE",
  "confidence": 0.7341,
  "rule_triggered": "prob_negative=0.734 >= 0.70 → immediate escalation",
  "business_action": "🔴 Alert kitchen/management — negative 'service' feedback. Reply within 24h.",
  "latency_seconds": 0.042
}
```

### `POST /predict_batch`

```json
{
  "reviews": [
    {"text": "Great food!", "aspect_term": "food", "domain": "restaurants"},
    {"text": "Battery drains too fast.", "aspect_term": "battery", "domain": "laptops"}
  ]
}
```

### `GET /health`
```json
{"status": "running", "model_loaded": true, "thresholds": {"negative": 0.45}}
```

### `GET /model_info`
Returns full champion model card: metrics, per-class F1, feature order, thresholds.

---

## 🏆 Champion-Challenger System (3-Gate)

Every new training run is evaluated against the current champion via 3 promotion gates:

```
Gate 1: challenger_macro_f1 - champion_macro_f1 >= 0.005   (F1 improvement)
Gate 2: challenger_roc_auc  >= 0.80                         (minimum quality floor)
Gate 3: train_val_f1_gap    <= 0.10                         (generalisation check)

ALL 3 must pass → PROMOTED (new champion)
ANY fails       → REJECTED (old champion retained)
```

Full history logged to `absa_models/challenger_log.json` and visible in the dashboard.

---

## 📊 Dataset

- **Source:** [SemEval 2014 Task 4](https://huggingface.co/datasets/sem_eval_2014_task_4)
- **Domains:** Restaurants (~3,000 sentences), Laptops (~1,500 sentences)
- **Task:** Aspect-term level sentiment (positive / negative / neutral / conflict)
- **Splits:** Official train / test split used

| Class | Restaurants (train) |
|-------|-------------------|
| Positive | ~58% |
| Negative | ~24% |
| Neutral  | ~15% |
| Conflict | ~3%  |

---

## 👤 About

**Narendra Kalam**  
MSc Computer Science — Gold Medalist (NASSCOM)  
Full Stack Data Science + AI  

Portfolio Projects: Credit Card Fraud Detection · Credit Risk Prediction · Customer Churn · Store Sales Forecasting · **Aspect-Based Sentiment Analysis**

> Building 20+ industry-level, end-to-end, company-standard ML projects targeting **30+ LPA** at top MNCs.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
