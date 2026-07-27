# ============================================================
# MODEL TUNING — Aspect-Based Sentiment Analysis (ABSA)
# FIX: LinearSVC max_iter=10000 to fix ConvergenceWarning
# ============================================================

import numpy as np
import logging
from typing import Dict, Tuple

from sklearn.base            import BaseEstimator
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline        import Pipeline

from sklearn.linear_model  import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm           import LinearSVC
from sklearn.ensemble      import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree          import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from imblearn.pipeline      import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler
    _HAS_IMBL = True
except ImportError:
    _HAS_IMBL = False
    ImbPipeline = Pipeline

from src.config import RANDOM_STATE, N_JOBS, CV_FOLDS

logger = logging.getLogger(__name__)

SEARCH_ITERS = 10


# ============================================================
# LINEAR MODELS
# ============================================================

linear_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(
            random_state=RANDOM_STATE, max_iter=3000,
            multi_class="multinomial", solver="lbfgs"
        ),
        {
            "classifier__C":            [0.1, 1.0, 5.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "LinearSVC": (
        LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=10000,   # ← FIX: was 5000, still not converging
            dual=True
        ),
        {
            "classifier__C":            [0.1, 1.0, 5.0],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "SGD": (
        SGDClassifier(
            loss="modified_huber", random_state=RANDOM_STATE,
            max_iter=2000, tol=1e-3, n_jobs=N_JOBS
        ),
        {
            "classifier__alpha":        [1e-4, 1e-3, 1e-2],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "RidgeClassifier": (
        RidgeClassifier(class_weight="balanced"),
        {
            "classifier__alpha": [0.1, 1.0, 5.0, 10.0],
        }
    ),
}

# ============================================================
# TREE MODELS
# NOTE: GradientBoosting removed (too slow)
#       ComplementNB removed (fails on negative lexicon_score)
# ============================================================

tree_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "RandomForest": (
        RandomForestClassifier(
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        {
            "classifier__n_estimators":     [100, 200],
            "classifier__max_depth":        [10, 20, None],
            "classifier__min_samples_leaf": [1, 2],
        }
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [10, 20, None],
        }
    ),

    "DecisionTree": (
        DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        {
            "classifier__max_depth":        [5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    ),
}

if _HAS_LGB:
    tree_models["LightGBM"] = (
        LGBMClassifier(
            random_state=RANDOM_STATE, verbose=-1,
            class_weight="balanced", n_jobs=N_JOBS,
        ),
        {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__num_leaves":    [31, 63],
            "classifier__max_depth":     [-1, 10],
        }
    )

if _HAS_XGB:
    tree_models["XGBoost"] = (
        XGBClassifier(
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            use_label_encoder=False, n_jobs=N_JOBS,
        ),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [3, 5],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:       Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor: ColumnTransformer,
    X_train:      "pd.DataFrame",
    y_train:      "pd.Series",
    use_smote:    bool = True,
) -> Dict[str, Pipeline]:

    final_pipelines = {}

    for name, (clf, param_dist) in models.items():
        logger.info("Tuning: %s", name)

        n_iter = min(SEARCH_ITERS, max(1, _count_combinations(param_dist)))

        if _HAS_IMBL and use_smote:
            pipe = ImbPipeline([
                ("preprocessor", preprocessor),
                ("oversample",   RandomOverSampler(random_state=RANDOM_STATE)),
                ("classifier",   clf),
            ])
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier",   clf),
            ])

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "f1_macro",
            cv                  = StratifiedKFold(CV_FOLDS, shuffle=True,
                                                  random_state=RANDOM_STATE),
            n_jobs              = 1,
            random_state        = RANDOM_STATE,
            verbose             = 0,
            error_score         = 0.0,
        )

        try:
            search.fit(X_train, y_train)
            logger.info("%s best_macro_f1=%.4f  params=%s",
                        name, search.best_score_, search.best_params_)
            final_pipelines[name] = search.best_estimator_
        except Exception as e:
            logger.error("Tuning failed for %s: %s", name, e)

    return final_pipelines


def _count_combinations(param_dist: dict) -> int:
    prod = 1
    for v in param_dist.values():
        try:    prod *= len(v)
        except: prod *= SEARCH_ITERS
    return prod


# ============================================================
# MLP NEURAL NETWORK
# ============================================================

def train_mlp_pipeline(
    X_train:      "pd.DataFrame",
    y_train:      "pd.Series",
    preprocessor: ColumnTransformer,
    use_smote:    bool = True,
) -> Pipeline:
    logger.info("Training MLP Neural Network ...")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        solver="adam", alpha=0.001, max_iter=50,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, random_state=RANDOM_STATE,
    )

    if _HAS_IMBL and use_smote:
        pipe = ImbPipeline([
            ("preprocessor", preprocessor),
            ("oversample",   RandomOverSampler(random_state=RANDOM_STATE)),
            ("classifier",   mlp),
        ])
    else:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   mlp),
        ])

    pipe.fit(X_train, y_train)
    logger.info("MLP training complete")
    return pipe