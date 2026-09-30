"""Shared data assembly used by every script."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SAMPLE_FRAC, SEED, VALID_FRAC
from .features import add_features, chronological_split, clean, load_frames
from .preprocess import Preprocessor, matrix_summary


@dataclass
class Dataset:
    X_train: object
    y_train: object
    X_valid: object
    y_valid: object
    X_test: object
    y_test: object
    preprocessor: Preprocessor
    report: pd.DataFrame

    @property
    def imbalance(self) -> float:
        positives = max(int(self.y_train.sum()), 1)
        return float((len(self.y_train) - self.y_train.sum()) / positives)


def build_dataset(sample_frac: float = SAMPLE_FRAC, valid_frac: float = VALID_FRAC,
                  seed: int = SEED, verbose: bool = True) -> Dataset:
    """Clean -> split on time -> engineer features -> fit preprocessing on train only."""
    raw_train, raw_test = load_frames(sample_frac, seed, verbose)

    train_clean = clean(raw_train, "train", verbose)
    test_clean = clean(raw_test, "test", verbose)

    train_part, valid_part = chronological_split(train_clean, valid_frac)
    train_part = add_features(train_part)
    valid_part = add_features(valid_part)
    test_part = add_features(test_clean)

    preprocessor = Preprocessor.fit(train_part)
    X_train = preprocessor.transform(train_part)
    X_valid = preprocessor.transform(valid_part)
    X_test = preprocessor.transform(test_part)

    y_train = train_part["is_fraud"].to_numpy()
    y_valid = valid_part["is_fraud"].to_numpy()
    y_test = test_part["is_fraud"].to_numpy()

    report = pd.DataFrame([
        {"split": "train", "rows": len(train_part),
         "fraud_rate": round(float(train_part["is_fraud"].mean()), 6),
         "starts": str(train_part["trans_date_trans_time"].min()),
         "ends": str(train_part["trans_date_trans_time"].max())},
        {"split": "validation", "rows": len(valid_part),
         "fraud_rate": round(float(valid_part["is_fraud"].mean()), 6),
         "starts": str(valid_part["trans_date_trans_time"].min()),
         "ends": str(valid_part["trans_date_trans_time"].max())},
        {"split": "test", "rows": len(test_part),
         "fraud_rate": round(float(test_part["is_fraud"].mean()), 6),
         "starts": str(test_part["trans_date_trans_time"].min()),
         "ends": str(test_part["trans_date_trans_time"].max())},
    ])

    if verbose:
        print()
        print(report.to_string(index=False))
        print()
        print("X_train  ", matrix_summary(X_train))
        print("X_valid  ", matrix_summary(X_valid))
        print("X_test   ", matrix_summary(X_test))
        print("features ", f"{len(preprocessor.feature_names)} "
                          f"({12} numeric + "
                          f"{len(preprocessor.feature_names) - 12} one-hot)")

    return Dataset(X_train, y_train, X_valid, y_valid, X_test, y_test,
                   preprocessor, report)
