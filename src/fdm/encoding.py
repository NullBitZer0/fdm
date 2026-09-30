"""Frequency and target encoding for the high-cardinality categorical columns.

Both encoders exist to solve the same problem: 693 one-hot merchant columns and 51
state columns are wide, sparse, and split on few examples, so a tree has to work
hard to learn that "this merchant" is risky. Collapsing each category to one dense
number gives every split an immediately usable signal.

The two are used for different reasons:

**Frequency encoding** answers "how common is this category?" and uses no label at
all, so it cannot leak. It is a genuinely safe feature.

**Target encoding** answers "how risky is this category?", which is a much stronger
signal, but it is computed *from the label* and is therefore the most dangerous
feature in the project. Computed naively on the training set it leaks badly: the
encoder memorises each category's own label, and a model scores near-perfectly on
data it will never see again. Three defences are used here:

1. **Out-of-fold construction.** Each training row is encoded using statistics from
   folds that exclude that row (see `TargetEncoder.fit_transform`). This is the
   standard defence and is not optional.
2. **Smoothing.** The estimate is pulled toward the global fraud rate in proportion
   to how little evidence a category has, so a merchant seen once does not get a
   0.0 or 1.0 rate. Smoothing strength is fitted by cross-validation, not guessed.
3. **Fitted on training rows only**, then frozen. The mapping is persisted inside
   the `Preprocessor` artifact, so validation and test rows — and live API
   requests — are encoded with training-period statistics and never with their own
   label. This is what makes it legal to use at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

UNSEEN = "Unknown"


class FrequencyEncoder:
    """Maps each category value to its share of the training rows.

    Uses no target information, so it is safe by construction: no out-of-fold
    machinery is needed because nothing in the mapping depends on `y`.
    """

    def __init__(self, columns: list[str], normalize: bool = True):
        self.columns = list(columns)
        self.normalize = normalize
        self.maps_: dict[str, dict[str, float]] = {}
        self.fallback_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, y=None) -> "FrequencyEncoder":
        for column in self.columns:
            counts = frame[column].astype(str).value_counts(normalize=self.normalize)
            self.maps_[column] = counts.to_dict()
            self.fallback_[column] = float(counts.iloc[0]) if len(counts) else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in self.columns:
            counts = self.maps_[column]
            values = frame[column].astype(str).map(counts)
            # An unseen category gets the global frequency, which is the honest
            # answer: we know nothing about it, so treat it as ordinary.
            values = values.fillna(self.fallback_[column])
            columns.append(values.to_numpy(dtype="float32"))
        return np.column_stack(columns)

    def feature_names(self) -> list[str]:
        return [f"{column}_freq" for column in self.columns]

    @property
    def n_features_(self) -> int:
        return len(self.columns)


class TargetEncoder:
    """Maps each category value to a smoothed fraud rate, fitted out-of-fold.

    The smoothing follows the standard form:

        encoded(v) = (sum_fraud(v) + prior * m) / (count(v) + m)

    where `prior` is the global fraud rate and `m` the smoothing strength. A
    category with many rows lands near its own rate; a category with two rows lands
    near the global rate, because two rows are not evidence of anything.
    """

    def __init__(self, columns: list[str], smoothing: float = 20.0, n_splits: int = 5,
                 random_state: int = 42):
        self.columns = list(columns)
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.random_state = random_state
        self.prior_ = 0.0
        # value -> smoothed rate, per column. This is what gets persisted.
        self.maps_: dict[str, dict[str, float]] = {}

    # ── internals ──

    def _rates(self, column: str, values: pd.Series, y: np.ndarray) -> dict[str, float]:
        frame = pd.DataFrame({"v": values.astype(str).to_numpy(), "y": y})
        grouped = frame.groupby("v")["y"].agg(["sum", "count"])
        rates = ((grouped["sum"] + self.prior_ * self.smoothing) /
                 (grouped["count"] + self.smoothing))
        return {str(key): float(value) for key, value in rates.items()}

    # ── public API ──

    def fit(self, frame: pd.DataFrame, y) -> "TargetEncoder":
        """Fit the frozen mapping used for validation, test and live requests."""
        y = np.asarray(y, dtype="float64")
        self.prior_ = float(y.mean())
        for column in self.columns:
            self.maps_[column] = self._rates(column, frame[column], y)
        return self

    def fit_transform(self, frame: pd.DataFrame, y) -> np.ndarray:
        """Out-of-fold encoding for training rows.

        This is the method that must be used on training data. Each row's value is
        computed from folds that do not contain it, so no row contributes to its own
        encoded feature and the model cannot cheat.
        """
        y = np.asarray(y, dtype="float64")
        self.prior_ = float(y.mean())
        n = len(y)
        n_splits = max(2, min(self.n_splits, n // 2))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        out = np.zeros((n, len(self.columns)), dtype="float32")
        for train_idx, apply_idx in splitter.split(np.arange(n)):
            fold_y = y[train_idx]
            # prior from the fitting folds only, for the same reason
            saved_prior = self.prior_
            self.prior_ = float(fold_y.mean())
            for index, column in enumerate(self.columns):
                rates = self._rates(column, frame[column].iloc[train_idx], fold_y)
                column_values = frame[column].iloc[apply_idx].astype(str)
                out[apply_idx, index] = (
                    column_values.map(rates).fillna(saved_prior).to_numpy(dtype="float32"))
            self.prior_ = saved_prior

        # Fit the final mapping on all of training for later use.
        self.fit(frame, y)
        return out

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode with the frozen training mapping. Never touches the target."""
        columns = []
        for column in self.columns:
            rates = self.maps_[column]
            values = frame[column].astype(str).map(rates)
            values = values.fillna(self.prior_)
            columns.append(values.to_numpy(dtype="float32"))
        return np.column_stack(columns)

    def feature_names(self) -> list[str]:
        return [f"{column}_target" for column in self.columns]

    @property
    def n_features_(self) -> int:
        return len(self.columns)


def choose_smoothing(frame: pd.DataFrame, y, column: str, candidates=(1, 5, 10, 20, 50, 100),
                     n_splits: int = 5, random_state: int = 42) -> tuple[float, float]:
    """Pick the smoothing strength by out-of-fold error on one column.

    Scored as how well the encoding predicts the label, not on the final model, so
    it is cheap and independent of which algorithm will consume the feature. The
    best candidate is the one whose encoded values are most informative without
    overfitting the small categories.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype="float64")
    n = len(y)
    splitter = KFold(n_splits=max(2, min(n_splits, n // 2)), shuffle=True,
                     random_state=random_state)
    prior = float(y.mean())
    results: list[tuple[float, float]] = []

    for m in candidates:
        encoded = np.zeros(n, dtype="float64")
        for train_idx, apply_idx in splitter.split(np.arange(n)):
            fold_y = y[train_idx]
            sub = pd.DataFrame({"v": frame[column].astype(str).to_numpy()[train_idx],
                                "y": fold_y})
            grouped = sub.groupby("v")["y"].agg(["sum", "count"])
            rates = ((grouped["sum"] + prior * m) / (grouped["count"] + m)).to_dict()
            encoded[apply_idx] = (frame[column].astype(str).iloc[apply_idx]
                                 .map(rates).fillna(prior).to_numpy())
        try:
            score = float(roc_auc_score(y, encoded))
        except ValueError:
            score = 0.5
        results.append((m, score))

    best_m, best_score = max(results, key=lambda item: item[1])
    return float(best_m), round(best_score, 4)
