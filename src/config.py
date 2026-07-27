# ============================================================
# CONFIGURATION — Aspect-Based Sentiment Analysis (ABSA) System
# ============================================================

import os

# ── Reproducibility ───────────────────────────────────────────
RANDOM_STATE  = 42
N_JOBS        = -1

# ── Cross-validation ──────────────────────────────────────────
CV_FOLDS            = 5
RANDOM_SEARCH_ITERS = 20

# ── Dataset (SemEval 2014 Task 4) ─────────────────────────────
DATASET_NAME      = "Chow05/SemEval-2014-Task-4"
# Note: This HuggingFace version has only 'default' config —
# restaurants domain only. Laptops data unavailable here.
DOMAINS           = ["restaurants"]   # laptops not available in Chow05

# ── Target label ─────────────────────────────────────────────
LABEL_COL    = "polarity"
TEXT_COL     = "text"
ASPECT_COL   = "aspect_term"
SENTIMENT_MAP = {
    "positive": 0,
    "negative": 1,
    "neutral":  2,
    "conflict": 3,
}
SENTIMENT_NAMES = ["positive", "negative", "neutral", "conflict"]
NUM_CLASSES     = 4

# ── Text preprocessing ────────────────────────────────────────
MAX_FEATURES    = 20_000   # TF-IDF / BoW vocab size
MAX_SEQ_LEN     = 128      # for BERT-style tokenizer (if used)
NGRAM_RANGE     = (1, 3)   # unigrams, bigrams, trigrams

# ── TF-IDF vectorizer ─────────────────────────────────────────
TFIDF_MAX_FEATURES = 20_000
TFIDF_SUBLINEAR    = True

# ── Aspect context window ─────────────────────────────────────
CONTEXT_WINDOW = 5         # words each side of aspect term

# ── PSI drift thresholds ──────────────────────────────────────
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Score monitoring alert ────────────────────────────────────
SCORE_MEAN_ALERT = 0.35

# ── Challenger promotion gates (3-gate system) ────────────────
MIN_F1_IMPROVEMENT     = 0.005    # challenger must beat champion by >= 0.5% macro-F1
MIN_ROCAUC_THRESHOLD   = 0.80     # challenger macro-OvR ROC-AUC >= 0.80
MAX_GENERALIZATION_GAP = 0.10     # train-val macro-F1 gap <= 10%

# ── Business cost matrix (₹ / misclassification) ─────────────
# Rows = True label, Cols = Predicted label  [pos, neg, neu, con]
COST_MATRIX = [
    [0,   30,  10,  15],   # true=positive
    [50,   0,  25,  20],   # true=negative   (FN on negative is costliest)
    [10,  20,   0,  10],   # true=neutral
    [15,  25,  15,   0],   # true=conflict
]

# ── Paths ──────────────────────────────────────────────────────
MODEL_DIR   = "absa_models"
METRICS_LOG = "absa_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)