"""Model definitions, imbalance strategies and the majority-voting ensemble.

The imbalance problem has several standard answers and this project compares
them rather than picking one by habit:

* ``class_weight`` / ``scale_pos_weight``  -- reweight the loss, keep all rows.
* ``SMOTE``                                 -- synthesise minority rows (over).
* ``RandomUnderSampler``                    -- drop majority rows (under).
* untouched                                 -- the honest baseline to beat.

Each model family is paired with the strategy that suits it, because a linear
model and a bagged forest do not respond to resampling in the same way.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from .config import SEED


def imbalance_ratio(y) -> float:
    """How many legitimate rows there are per fraud row."""
    y = np.asarray(y)
    positives = max(int((y == 1).sum()), 1)
    return float((y == 0).sum()) / positives


# ── imbalance strategies ────────────────────────────────────────────────────

def smote(random_state: int = SEED, sampling_strategy: float = 0.20):
    return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5,
                 random_state=random_state)


def undersample(random_state: int = SEED, sampling_strategy: float = 0.20):
    """Reduce the majority class to `sampling_strategy` times the minority size."""
    return RandomUnderSampler(sampling_strategy=sampling_strategy,
                              random_state=random_state)


# ── base estimators ─────────────────────────────────────────────────────────

def logistic_regression(imbalance: float, seed: int = SEED, max_iter: int = 300):
    return LogisticRegression(
        solver="saga", max_iter=max_iter, tol=1e-3, C=1.0,
        class_weight="balanced", random_state=seed)


def random_forest(imbalance: float, seed: int = SEED, n_estimators: int = 300,
                  **kwargs):
    """Bagged trees, typically paired with under-sampling so each tree sees a
    balanced bootstrap.

    Measured caveat on this dataset: balancing 1.04M rows to 50/50 keeps only
    ~11.9k rows (98.9% discarded), and the resulting forest scored PR-AUC 0.7500
    against XGBoost's 0.9097. The `sampling_strategy` below is a tunable, not a
    fixed rule, precisely because the right value depends on the row count."""
    params = dict(
        n_estimators=n_estimators,
        max_depth=kwargs.pop("max_depth", 20),
        min_samples_leaf=kwargs.pop("min_samples_leaf", 1),
        max_features=kwargs.pop("max_features", "sqrt"),
        n_jobs=-1,
        random_state=seed,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def xgboost(imbalance: float, seed: int = SEED, **kwargs):
    params = dict(
        n_estimators=kwargs.pop("n_estimators", 600),
        learning_rate=kwargs.pop("learning_rate", 0.05),
        max_depth=kwargs.pop("max_depth", 8),
        min_child_weight=kwargs.pop("min_child_weight", 1),
        subsample=kwargs.pop("subsample", 1.0),
        colsample_bytree=kwargs.pop("colsample_bytree", 1.0),
        gamma=kwargs.pop("gamma", 0.0),
        reg_alpha=kwargs.pop("reg_alpha", 0.0),
        reg_lambda=kwargs.pop("reg_lambda", 1.0),
        scale_pos_weight=imbalance,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=kwargs.pop("early_stopping_rounds", 50),
        random_state=seed,
        n_jobs=-1,
    )
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def lightgbm(imbalance: float, seed: int = SEED, **kwargs):
    params = dict(
        n_estimators=kwargs.pop("n_estimators", 600),
        learning_rate=kwargs.pop("learning_rate", 0.05),
        num_leaves=kwargs.pop("num_leaves", 64),
        max_depth=kwargs.pop("max_depth", -1),
        min_child_samples=kwargs.pop("min_child_samples", 20),
        subsample=kwargs.pop("subsample", 1.0),
        colsample_bytree=kwargs.pop("colsample_bytree", 1.0),
        reg_alpha=kwargs.pop("reg_alpha", 0.0),
        reg_lambda=kwargs.pop("reg_lambda", 0.0),
        scale_pos_weight=imbalance,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


# ── the five candidates compared in Stage 6 ─────────────────────────────────

def build_models(imbalance: float, seed: int = SEED) -> dict[str, Pipeline]:
    """Baseline (untuned) version of every candidate, each with its imbalance fix.

    The numeric block is already standardised by the fitted `Preprocessor`, so
    no model here re-scales. The SMOTE/undersample steps sit inside the pipeline
    so resampling only ever touches training folds during cross-validation.
    """
    return {
        "Logistic Regression + SMOTE": ImbPipeline([
            ("resample", smote(seed)),
            ("model", logistic_regression(imbalance, seed)),
        ]),
        "Random Forest + Undersampling": ImbPipeline([
            ("resample", undersample(seed)),
            ("model", random_forest(imbalance, seed)),
        ]),
        "XGBoost + Class Weight": Pipeline([("model", xgboost(imbalance, seed))]),
        "LightGBM + Class Weight": Pipeline([("model", lightgbm(imbalance, seed))]),
    }


def build_voting(members: dict[str, Pipeline], voting: str = "soft",
                 weights: dict[str, float] | None = None) -> VotingClassifier:
    """Majority vote across the tree models.

    ``hard`` averages the 0/1 class labels, which throws away all confidence
    information -- with a 0.58% fraud rate the interesting signal lives entirely
    in the probability, so ``soft`` (mean of probabilities) is expected to win.
    Both are measured rather than assumed.
    """
    estimators = [(name.replace(" ", "_"), model) for name, model in members.items()]
    if weights:
        weights = [weights[name] for name in members]
    # n_jobs=None: every member already uses all cores, and nesting two levels of
    # parallelism just oversubscribes the CPU.
    return VotingClassifier(estimators=estimators, voting=voting,
                            weights=weights, n_jobs=None, flatten_transform=True)
