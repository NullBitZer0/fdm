"""Encoding and the persisted preprocessing artifact.

`Preprocessor` is the single source of truth for "how a row becomes a matrix".
It is fitted once on training data only and then saved, so the API never has to
refit anything and cannot accidentally fit on data it should not see.

Two encoding strategies are available, because one-hot is genuinely bad for
`merchant` (693 levels) and `state` (51 levels) — a tree needs several splits to
isolate a rare merchant, and 760 dummy columns on 1M rows is a lot of width for
very little signal:

  * ``count``  — one-hot only (the original 772-feature setup)
  * ``count+freq``     — add frequency encoding (label-free, cannot leak)
  * ``count+target``   — add out-of-fold target encoding (strong, but see below)
  * ``target_only``    — replace one-hot with frequency + target

Target encoding is built from the label, so it is fitted with strict train/test
separation: training rows are encoded out-of-fold, and validation, test and live
API rows are encoded with a mapping frozen from the training period. A row is
never encoded using its own label. `tests/test_contract.py` asserts this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC
from .encoding import FrequencyEncoder, TargetEncoder

# Frequency encoding is useful for every categorical column. Target encoding is
# aimed at the high-cardinality ones: category and gender have so few levels that
# one-hot already represents them perfectly, and encoding them adds no information.
TARGET_COLUMNS = ["merchant", "state", "category"]
FREQ_COLUMNS = list(CATEGORICAL)

SCHEMES = ("count", "count+freq", "count+target", "target_only")


@dataclass
class Preprocessor:
    scaler: StandardScaler
    encoder: OneHotEncoder
    feature_names: list[str]
    scheme: str = "count"
    frequency: FrequencyEncoder | None = None
    target: TargetEncoder | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame, y=None, scheme: str = "count",
            target_columns: list[str] | None = None,
            target_smoothing: float = 20.0) -> "Preprocessor":
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")

        columns = target_columns or TARGET_COLUMNS
        scaler = StandardScaler()
        scaler.fit(frame[NUMERIC])
        names = list(NUMERIC)

        # Target encoding needs y; without it we fall back rather than fail, so a
        # caller that only wants a count-based preprocessor stays simple.
        frequency = target = None

        if "freq" in scheme:
            frequency = FrequencyEncoder(FREQ_COLUMNS).fit(frame)
            names += frequency.feature_names()

        if "target" in scheme:
            if y is None:
                raise ValueError(f"scheme {scheme!r} needs y to fit target encoding")
            target = TargetEncoder(columns, smoothing=target_smoothing).fit(frame, y)
            names += target.feature_names()

        if scheme != "target_only":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                    dtype="float32")
            encoder.fit(frame[CATEGORICAL].astype(str))
            names += [str(n) for n in encoder.get_feature_names_out(CATEGORICAL)]
        else:
            encoder = None

        return cls(scaler=scaler, encoder=encoder, feature_names=names,
                   scheme=scheme, frequency=frequency, target=target,
                   metadata={"target_columns": columns,
                             "target_smoothing": target_smoothing,
                             "target_prior": getattr(target, "prior_", None)})

    def fit_transform(self, frame: pd.DataFrame, y=None) -> sparse.csr_matrix:
        """Encode TRAINING rows. Target encoding goes strictly out-of-fold here."""
        blocks = [sparse.csr_matrix(self.scaler.transform(frame[NUMERIC]).astype("float32"))]

        if self.frequency is not None:
            blocks.append(sparse.csr_matrix(self.frequency.transform(frame)))

        if self.target is not None:
            if y is None:
                raise ValueError("out-of-fold target encoding needs y")
            # The one call that is allowed to see y on training data.
            out_of_fold = self.target.fit_transform(frame, y)
            blocks.append(sparse.csr_matrix(out_of_fold))
            # The frozen mapping for later data is fitted on all of training.
            self.target.fit(frame, y)

        if self.encoder is not None:
            blocks.append(self.encoder.transform(frame[CATEGORICAL].astype(str)))

        return sparse.hstack(blocks, format="csr")

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        """Encode validation / test / live rows. Never looks at any label."""
        blocks = [sparse.csr_matrix(self.scaler.transform(frame[NUMERIC]).astype("float32"))]

        if self.frequency is not None:
            blocks.append(sparse.csr_matrix(self.frequency.transform(frame)))

        if self.target is not None:
            blocks.append(sparse.csr_matrix(self.target.transform(frame)))

        if self.encoder is not None:
            blocks.append(self.encoder.transform(frame[CATEGORICAL].astype(str)))

        return sparse.hstack(blocks, format="csr")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)


def matrix_summary(matrix) -> str:
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    return f"{matrix.shape[0]:,} rows x {matrix.shape[1]:,} features, {density:.3%} dense"
