# ============================================================
# PYTEST UNIT TESTS — Aspect-Based Sentiment Analysis (ABSA)
# FIX: test_preprocessor_transform_shape uses large_sample_df
#      (30 rows) instead of sample_df (5 rows).
#      min_df=2 in TfidfVectorizer needs terms in >= 2 docs.
#      5 rows = each term appears only once = ValueError.
# Run: pytest tests/test_absa_pipeline.py -v
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing   import TextCleaner, build_preprocessors, safe_k
from src.data_loader     import (
    validate_input_data, add_engineered_features, detect_feature_types,
    _extract_context, _aspect_position
)
from src.leakage_check   import detect_leakage
from src.metrics         import (
    compute_macro_f1, compute_weighted_f1, psi,
    label_psi, cost_sensitive_evaluation, tune_negative_threshold
)
from src.sentiment_engine import (
    score_review, get_sentiment_band, summarise_batch
)
from src.config          import (
    SENTIMENT_MAP, SENTIMENT_NAMES, NUM_CLASSES,
    PSI_MODERATE, PSI_HIGH,
    MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP,
    COST_MATRIX
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df():
    """5-row fixture for basic tests (not for preprocessor fit)."""
    return pd.DataFrame({
        "text": [
            "The pasta was delicious and well presented.",
            "Service was very slow and staff was rude.",
            "The ambiance is quite standard, nothing special.",
            "Food quality is excellent but prices are high.",
            "Battery life is terrible, only lasts 2 hours.",
        ],
        "aspect_term": ["pasta", "service", "ambiance", "food quality", "battery life"],
        "polarity":    ["positive", "negative", "neutral", "conflict", "negative"],
        "domain":      ["restaurants"] * 5,
        "label":       [0, 1, 2, 3, 1],
        "split":       ["train"] * 5,
    })


@pytest.fixture
def large_sample_df():
    """
    30-row fixture for preprocessor tests.
    TfidfVectorizer(min_df=2) requires each term to appear
    in at least 2 documents — 5-row fixture fails this.
    """
    texts = [
        "The food was great and delicious.",
        "The service was slow and bad.",
        "The food quality is excellent here.",
        "Service staff was very rude and slow.",
        "Great ambiance and good food quality.",
        "The price is too high for this food.",
        "I love the pasta and the service was fast.",
        "The staff was friendly and food was great.",
        "Terrible service and the food was cold.",
        "The ambiance was nice but food was bland.",
        "Food was fresh and the service was good.",
        "The place has great food and bad service.",
        "Excellent food quality at a great price.",
        "The service was slow but food was delicious.",
        "Great pasta and excellent ambiance here.",
        "The food was cold and service was rude.",
        "Good price but terrible food quality.",
        "The staff was helpful and food was great.",
        "Bad service and the food was overpriced.",
        "The ambiance is great and food is fresh.",
        "Service was fast and food was delicious.",
        "The pasta was bland and service was slow.",
        "Excellent service and great food quality.",
        "The price is good and food is fresh.",
        "The staff was rude and food was terrible.",
        "Great ambiance but bad food quality.",
        "The service was excellent and price fair.",
        "Food was great but ambiance was noisy.",
        "The pasta was delicious and price good.",
        "Slow service but excellent food quality.",
    ]
    aspects  = [
        "food","service","food","service","ambiance","price","pasta","staff",
        "service","ambiance","food","food","food","service","pasta","food",
        "price","staff","service","ambiance","service","pasta","service",
        "price","staff","ambiance","service","ambiance","pasta","service"
    ]
    polarity = [
        "positive","negative","positive","negative","positive","negative",
        "positive","positive","negative","neutral","positive","positive",
        "positive","negative","positive","negative","negative","positive",
        "negative","positive","positive","negative","positive","positive",
        "negative","negative","positive","neutral","positive","negative"
    ]
    label = [SENTIMENT_MAP[p] for p in polarity]

    return pd.DataFrame({
        "text":        texts,
        "aspect_term": aspects,
        "polarity":    polarity,
        "domain":      ["restaurants"] * 30,
        "label":       label,
        "split":       ["train"] * 30,
    })


@pytest.fixture
def sample_proba():
    np.random.seed(42)
    return np.random.dirichlet([2, 1, 1, 0.5], size=20)


# ============================================================
# TEXT CLEANER TESTS (5)
# ============================================================

class TestTextCleaner:

    def test_lowercase_output(self):
        tc = TextCleaner(remove_stopwords=False)
        tc.fit(["dummy"])
        result = tc.transform(["Hello World THIS IS A TEST"])
        assert result[0] == result[0].lower()

    def test_url_removed(self):
        tc = TextCleaner(remove_stopwords=False)
        tc.fit(["dummy"])
        result = tc.transform(["Visit https://example.com for info"])
        assert "http" not in result[0]
        assert "example" not in result[0]

    def test_negation_preserved(self):
        tc = TextCleaner(remove_stopwords=True)
        tc.fit(["dummy"])
        result = tc.transform(["The food was not good at all"])
        assert "not" in result[0], "Negation 'not' must be preserved"

    def test_empty_string_handled(self):
        tc = TextCleaner(remove_stopwords=True)
        tc.fit(["dummy"])
        result = tc.transform([""])
        assert result[0] == ""

    def test_non_string_handled(self):
        tc = TextCleaner(remove_stopwords=False)
        tc.fit(["dummy"])
        result = tc.transform([None])
        assert isinstance(result[0], str)


# ============================================================
# PREPROCESSING TESTS (4)
# ============================================================

class TestPreprocessing:

    def test_build_preprocessors_returns_tuple(self, large_sample_df):
        df = add_engineered_features(large_sample_df)
        text_cols, num_cols, bin_cols = detect_feature_types(df)
        result = build_preprocessors(text_cols, num_cols, bin_cols, df)
        assert len(result) == 3

    def test_feature_order_not_empty(self, large_sample_df):
        df = add_engineered_features(large_sample_df)
        text_cols, num_cols, bin_cols = detect_feature_types(df)
        _, _, feature_order = build_preprocessors(text_cols, num_cols, bin_cols, df)
        assert len(feature_order) > 0

    def test_preprocessor_transform_shape(self, large_sample_df):
        """
        FIX: Uses large_sample_df (30 rows).
        min_df=2 needs terms appearing in >= 2 docs.
        5-row sample_df caused: ValueError: After pruning, no terms remain.
        """
        df = add_engineered_features(large_sample_df)
        text_cols, num_cols, bin_cols = detect_feature_types(df)
        pre_main, _, _ = build_preprocessors(text_cols, num_cols, bin_cols, df)
        drop = ["polarity", "label", "split", "domain", "text", "aspect_term"]
        X = df.drop(columns=[c for c in drop if c in df.columns])
        pre_main.fit(X)
        out = pre_main.transform(X)
        assert out.shape[0] == len(df)
        assert out.shape[1] > 0

    def test_safe_k_does_not_exceed_features(self, large_sample_df):
        df = add_engineered_features(large_sample_df)
        text_cols, num_cols, bin_cols = detect_feature_types(df)
        pre_main, _, _ = build_preprocessors(text_cols, num_cols, bin_cols, df)
        drop = ["polarity", "label", "split", "domain", "text", "aspect_term"]
        X = df.drop(columns=[c for c in drop if c in df.columns])
        k = safe_k(9999, pre_main, X)
        assert k <= 9999


# ============================================================
# DATA LOADER TESTS (6)
# ============================================================

class TestDataLoader:

    def test_validate_drops_unknown_polarity(self):
        df = pd.DataFrame({
            "text":        ["good", "bad", "unknown feeling"],
            "aspect_term": ["food", "service", "price"],
            "polarity":    ["positive", "negative", "GARBAGE_LABEL"],
            "domain":      ["restaurants"] * 3,
        })
        result = validate_input_data(df)
        assert "GARBAGE_LABEL" not in result["polarity"].values

    def test_validate_adds_label_column(self, sample_df):
        result = validate_input_data(sample_df.copy())
        assert "label" in result.columns
        assert result["label"].dtype in [np.int64, np.int32, int]

    def test_add_engineered_features_columns(self, sample_df):
        result = add_engineered_features(sample_df.copy())
        for col in ["combined_text", "aspect_in_context", "text_len",
                    "has_negation", "lexicon_score"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_extract_context_window(self):
        text   = "the food was absolutely delicious and fresh"
        aspect = "food"
        ctx    = _extract_context(text, aspect, window=2)
        assert "food" in ctx
        assert len(ctx.split()) <= 5

    def test_aspect_position_valid_range(self):
        pos = _aspect_position("The battery life is great", "battery")
        assert 0.0 <= pos <= 1.0

    def test_aspect_position_not_found(self):
        pos = _aspect_position("The food was great", "battery")
        assert pos == -1.0


# ============================================================
# LEAKAGE CHECK TESTS (3)
# ============================================================

class TestLeakageCheck:

    def test_no_leakage_on_clean_data(self, sample_df):
        df = add_engineered_features(sample_df.copy())
        drop = ["polarity", "split", "domain", "text", "aspect_term"]
        X = df.drop(columns=[c for c in drop if c in df.columns])
        y = df["label"]
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)

    def test_identical_feature_flagged(self):
        X = pd.DataFrame({"innocent": [1,2,3,4,5], "leak_col": [0,1,1,0,1]})
        y = pd.Series([0,1,1,0,1])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) >= 1

    def test_returns_list_type(self, sample_df):
        df = add_engineered_features(sample_df)
        X  = df[["text_len", "word_count"]].fillna(0)
        y  = df["label"]
        result = detect_leakage(X, y)
        assert isinstance(result, list)


# ============================================================
# METRICS TESTS (7)
# ============================================================

class TestMetrics:

    def test_macro_f1_perfect(self):
        y = np.array([0, 1, 2, 3, 0, 1])
        assert compute_macro_f1(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_macro_f1_range(self, sample_proba):
        y_true = np.array([0, 1, 2, 3] * 5)
        y_pred = np.argmax(sample_proba, axis=1)
        score  = compute_macro_f1(y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_psi_stable_distributions(self):
        x = np.random.normal(0, 1, 1000)
        assert psi(x, x) == pytest.approx(0.0, abs=0.01)

    def test_psi_shifted_distribution(self):
        x_ref = np.random.normal(0, 1, 1000)
        x_new = np.random.normal(5, 1, 1000)
        assert psi(x_ref, x_new) > PSI_MODERATE

    def test_label_psi_type(self):
        y_exp = np.array([0, 1, 2, 3] * 25)
        y_act = np.array([0, 1, 2, 3] * 25)
        result = label_psi(y_exp, y_act)
        assert isinstance(result, float)

    def test_cost_evaluation_keys(self):
        y_true = np.array([0, 1, 1, 0, 2])
        y_pred = np.array([0, 0, 1, 1, 2])
        result = cost_sensitive_evaluation(y_true, y_pred)
        for key in ["n_misclassified", "total_cost_inr", "per_sample_cost_inr"]:
            assert key in result

    def test_tune_negative_threshold_range(self):
        np.random.seed(0)
        y_true = np.array([0, 1, 1, 0, 2, 1, 0, 3, 1, 0])
        y_prob = np.random.uniform(0, 1, 10)
        thr    = tune_negative_threshold(y_true, y_prob, target_recall=0.80)
        assert 0.0 < thr < 1.0


# ============================================================
# SENTIMENT ENGINE TESTS (6)
# ============================================================

class TestSentimentEngine:

    def test_score_review_keys(self):
        proba  = np.array([0.1, 0.7, 0.1, 0.1])
        result = score_review("The service was terrible.", "service", proba)
        for key in ["predicted_sentiment", "probabilities", "decision",
                    "confidence", "sentiment_band", "business_action"]:
            assert key in result, f"Missing key: {key}"

    def test_high_negative_prob_escalates(self):
        proba  = np.array([0.05, 0.85, 0.05, 0.05])
        result = score_review("Absolutely terrible food.", "food", proba)
        assert result["decision"] == "ESCALATE"

    def test_high_positive_prob_decision(self):
        proba  = np.array([0.75, 0.10, 0.10, 0.05])
        result = score_review("The food was amazing.", "food", proba)
        assert result["decision"] == "POSITIVE"

    def test_confidence_is_max_prob(self):
        proba  = np.array([0.20, 0.60, 0.15, 0.05])
        result = score_review("Test review.", "service", proba)
        assert result["confidence"] == pytest.approx(0.60, abs=1e-4)

    def test_get_sentiment_band_strong_negative(self):
        assert get_sentiment_band(0.75) == "STRONG_NEGATIVE"

    def test_summarise_batch_keys(self):
        batch_results = [
            score_review("Great food",   "food",    np.array([0.8, 0.1, 0.1, 0.0])),
            score_review("Bad service",  "service", np.array([0.1, 0.7, 0.1, 0.1])),
            score_review("OK price",     "price",   np.array([0.2, 0.2, 0.5, 0.1])),
        ]
        summary = summarise_batch(batch_results)
        for key in ["total_reviews", "escalate_rate", "negative_rate", "avg_confidence"]:
            assert key in summary


# ============================================================
# CONFIG TESTS (4)
# ============================================================

class TestConfig:

    def test_sentiment_map_has_four_classes(self):
        assert len(SENTIMENT_MAP) == NUM_CLASSES

    def test_psi_thresholds_ordering(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_cost_matrix_shape(self):
        cm = COST_MATRIX
        assert len(cm) == NUM_CLASSES
        for row in cm:
            assert len(row) == NUM_CLASSES

    def test_challenger_gates_positive(self):
        assert MIN_F1_IMPROVEMENT    > 0
        assert MIN_ROCAUC_THRESHOLD  > 0
        assert MAX_GENERALIZATION_GAP > 0


# ============================================================
# MODEL LOADER TESTS (3)
# ============================================================

class TestModelLoader:

    def test_load_missing_model_raises(self, tmp_path, monkeypatch):
        from src import model_loader
        monkeypatch.setattr(model_loader, "MODEL_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            model_loader.load_latest_model()

    def test_challenger_log_appends(self, tmp_path, monkeypatch):
        from src import model_loader
        monkeypatch.setattr(model_loader, "MODEL_DIR",      str(tmp_path))
        monkeypatch.setattr(model_loader, "CHALLENGER_LOG", str(tmp_path / "challenger_log.json"))

        result = {"decision": "PROMOTED", "challenger_name": "TestModel", "reason": "test"}
        model_loader._save_challenger_log(result)
        model_loader._save_challenger_log(result)

        import json
        with open(tmp_path / "challenger_log.json") as f:
            log = json.load(f)
        assert len(log) == 2

    def test_update_registry_creates_file(self, tmp_path, monkeypatch):
        from src import model_loader
        monkeypatch.setattr(model_loader, "MODEL_DIR", str(tmp_path))

        model_loader._update_registry(
            model_path      = str(tmp_path / "absa_model_Test_v1.joblib"),
            thresholds      = {"negative": 0.45},
            model_card_path = str(tmp_path / "model_card_Test_v1.json"),
        )

        import json
        with open(tmp_path / "latest_model.json") as f:
            reg = json.load(f)
        assert "model_name"      in reg
        assert "thresholds"      in reg
        assert "model_card_path" in reg