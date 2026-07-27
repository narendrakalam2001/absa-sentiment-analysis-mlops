# ============================================================
# DATA LOADER — Aspect-Based Sentiment Analysis (ABSA)
# Dataset : Chow05/SemEval-2014-Task-4 (HuggingFace)
# Same approach as EDA notebook — direct load_dataset()
# ============================================================

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.config import (
    DATASET_NAME, LABEL_COL, TEXT_COL, ASPECT_COL,
    SENTIMENT_MAP, SENTIMENT_NAMES, CONTEXT_WINDOW, RANDOM_STATE
)

logger = logging.getLogger(__name__)

# ── Same POLARITY_MAP as EDA notebook ─────────────────────────
POLARITY_MAP = {
    "-1": None,  -1: None,
    "0" : "negative",  0: "negative",
    "1" : "neutral",   1: "neutral",
    "2" : "positive",  2: "positive",
    "3" : "conflict",  3: "conflict",
}


# ============================================================
# NER PARSER — exact same as EDA notebook
# ============================================================

def _parse_semeval_ner(item: dict, split_name: str, domain: str = "restaurants") -> list:
    """
    Converts NER token-level format to flat (sentence, aspect, polarity) rows.
    Tags: 0=O, 1=B-ASP (begin aspect), 2=I-ASP (inside aspect)
    Polarities: -1=non-aspect, 0=negative, 1=neutral, 2=positive, 3=conflict
    """
    tokens     = item["Tokens"]
    tags       = item["Tags"]
    polarities = item["Polarities"]
    sentence   = " ".join(tokens)
    rows       = []
    i          = 0

    while i < len(tags):
        tag = int(tags[i]) if not isinstance(tags[i], int) else tags[i]

        if tag == 1:  # B-ASP — beginning of aspect span
            aspect_tokens = [tokens[i]]
            polarity_val  = polarities[i]

            j = i + 1
            while j < len(tags):
                t = int(tags[j]) if not isinstance(tags[j], int) else tags[j]
                if t == 2:
                    aspect_tokens.append(tokens[j])
                    j += 1
                else:
                    break

            aspect_term = " ".join(aspect_tokens)
            pol_key     = int(polarity_val) if not isinstance(polarity_val, int) else polarity_val
            polarity    = POLARITY_MAP.get(pol_key)

            if polarity is not None:
                rows.append({
                    TEXT_COL:   sentence,
                    ASPECT_COL: aspect_term,
                    LABEL_COL:  polarity,
                    "domain":   domain,
                    "split":    split_name,
                })
            i = j
        else:
            i += 1

    return rows


# ============================================================
# LOAD DATASET FROM HUGGINGFACE
# ============================================================

def load_semeval_dataset(
    domain: str = "restaurants",
    split:  str = "all",
) -> pd.DataFrame:
    """
    Loads Chow05/SemEval-2014-Task-4 from HuggingFace.

    NOTE: Only 'default' config exists — restaurants domain only.
          Laptops data unavailable in this HuggingFace version.

    Args:
        domain : 'restaurants' only (laptops unavailable)
        split  : 'train' | 'test' | 'all'
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Run: pip install datasets")

    if domain == "laptops":
        raise ValueError(
            "Chow05/SemEval-2014-Task-4 mein sirf restaurants domain hai. "
            "Laptops unavailable. Use --domain restaurants."
        )

    logger.info("Loading %s  split=%s", DATASET_NAME, split)

    # ── Direct load — no config arg (only 'default' exists) ───
    ds = hf_load(DATASET_NAME)
    logger.info("Available splits: %s", list(ds.keys()))

    frames = []

    if split in ("train", "all") and "train" in ds:
        rows = []
        for item in ds["train"]:
            rows.extend(_parse_semeval_ner(item, "train", "restaurants"))
        train_df = pd.DataFrame(rows)
        frames.append(train_df)
        logger.info("Train loaded  |  rows=%d", len(train_df))

    if split in ("test", "all"):
        test_key = "test" if "test" in ds else ("validation" if "validation" in ds else None)
        if test_key:
            rows = []
            for item in ds[test_key]:
                rows.extend(_parse_semeval_ner(item, "test", "restaurants"))
            test_df = pd.DataFrame(rows)
            frames.append(test_df)
            logger.info("Test loaded   |  rows=%d", len(test_df))
        else:
            logger.warning("No test/validation split found. Available: %s", list(ds.keys()))

    if not frames:
        raise ValueError(f"No data loaded. split='{split}'. Available: {list(ds.keys())}")

    df = pd.concat(frames, ignore_index=True)
    logger.info("Combined shape: %s", df.shape)
    return df


def load_all_domains() -> pd.DataFrame:
    """
    Loads all available domains.
    NOTE: Only restaurants available in Chow05/SemEval-2014-Task-4.
    """
    logger.info("load_all_domains() — loading restaurants (only available domain)")
    return load_semeval_dataset(domain="restaurants", split="all")


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans raw ABSA DataFrame.
    - Drops nulls
    - Filters to known sentiment classes
    - Adds numeric label column
    """
    logger.info("Validating  |  initial shape=%s", df.shape)

    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].str.strip().str.len() > 0]

    known  = set(SENTIMENT_MAP.keys())
    before = len(df)
    df     = df[df[LABEL_COL].isin(known)].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unknown polarity", dropped)

    df["label"] = df[LABEL_COL].map(SENTIMENT_MAP)

    # Conflict guard — very few samples can crash StratifiedKFold
    conflict_n = (df["label"] == 3).sum()
    if 0 < conflict_n < 2:
        logger.warning("Conflict class only %d sample(s) — dropping to avoid CV crash", conflict_n)
        df = df[df["label"] != 3].copy()

    logger.info("Class distribution:\n%s", df[LABEL_COL].value_counts().to_string())
    logger.info("Final shape: %s", df.shape)
    return df.reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds aspect-aware features:
      combined_text, aspect_in_context, text_len, word_count,
      aspect_position, has_negation, has_exclamation,
      has_question, lexicon_score
    """
    df = df.copy()

    # 1. combined_text — primary NLP feature
    df["combined_text"] = (
        df[ASPECT_COL].fillna("GENERAL").astype(str)
        + " [SEP] "
        + df[TEXT_COL].fillna("").astype(str)
    )

    # 2. aspect_in_context — local ±5 word window
    df["aspect_in_context"] = df.apply(
        lambda r: _extract_context(r[TEXT_COL], r[ASPECT_COL], CONTEXT_WINDOW), axis=1
    )

    # 3. text_len
    df["text_len"] = df[TEXT_COL].fillna("").str.len()

    # 4. word_count
    df["word_count"] = df[TEXT_COL].fillna("").str.split().str.len()

    # 5. aspect_position (0-1)
    df["aspect_position"] = df.apply(
        lambda r: _aspect_position(r[TEXT_COL], r[ASPECT_COL]), axis=1
    )

    # 6. has_negation
    NEGATIONS = {"not", "no", "never", "nobody", "nothing", "neither",
                 "nor", "none", "n't", "without", "hardly", "barely", "scarcely"}
    df["has_negation"] = df["aspect_in_context"].apply(
        lambda t: int(any(neg in str(t).lower().split() for neg in NEGATIONS))
    )

    # 7. has_exclamation / question
    df["has_exclamation"] = df[TEXT_COL].fillna("").str.contains("!").astype(int)
    df["has_question"]    = df[TEXT_COL].fillna("").str.contains(r"\?").astype(int)

    # 8. lexicon_score
    POS_WORDS = {"great", "excellent", "good", "love", "amazing", "perfect",
                 "best", "wonderful", "fantastic", "fresh", "delicious",
                 "friendly", "fast", "clean", "recommend", "superb",
                 "outstanding", "brilliant", "nice", "awesome", "enjoyed"}
    NEG_WORDS = {"bad", "terrible", "awful", "horrible", "poor", "slow",
                 "rude", "worst", "disappointing", "overpriced", "dirty",
                 "cold", "late", "wrong", "broken", "mediocre", "bland",
                 "stale", "unfriendly", "noisy", "disgusting"}

    def _lex(text):
        words = set(str(text).lower().split())
        return len(words & POS_WORDS) - len(words & NEG_WORDS)

    df["lexicon_score"] = df["aspect_in_context"].apply(_lex)

    logger.info("Feature engineering done  |  shape=%s", df.shape)
    return df


# ============================================================
# HELPERS
# ============================================================

def _extract_context(text: str, aspect: str, window: int = 5) -> str:
    if not isinstance(text, str) or not isinstance(aspect, str):
        return str(text) if isinstance(text, str) else ""
    words  = text.lower().split()
    awords = aspect.lower().split()
    for i in range(len(words) - len(awords) + 1):
        if words[i: i + len(awords)] == awords:
            s = max(0, i - window)
            e = min(len(words), i + len(awords) + window)
            return " ".join(words[s:e])
    return text


def _aspect_position(text: str, aspect: str) -> float:
    if not isinstance(text, str) or not isinstance(aspect, str):
        return -1.0
    idx = text.lower().find(aspect.lower())
    return round(idx / max(len(text), 1), 4) if idx >= 0 else -1.0


def detect_feature_types(
    df: pd.DataFrame,
    text_cols: List[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Returns (text_cols, num_cols, bin_cols)."""
    if text_cols is None:
        text_cols = ["combined_text", "aspect_in_context"]

    skip = {"label", LABEL_COL, "split", "domain", TEXT_COL, ASPECT_COL} | set(text_cols)
    num_cols, bin_cols = [], []

    for col in df.columns:
        if col in skip or df[col].dtype == object:
            continue
        if df[col].nunique() <= 2 and set(df[col].dropna().unique()).issubset({0, 1}):
            bin_cols.append(col)
        else:
            num_cols.append(col)

    logger.info("Text=%s | Num=%s | Bin=%s", text_cols, num_cols, bin_cols)
    return text_cols, num_cols, bin_cols