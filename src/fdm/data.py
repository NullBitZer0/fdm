"""Shared data assembly used by every script."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import NUMERIC, SAMPLE_FRAC, SEED, VALID_FRAC
from .features import (BEHAVIOURAL, add_behavioural_features, add_features,
                      chronological_split, clean, load_frames)
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
                  seed: int = SEED, verbose: bool = True,
                  behavioural: bool = False) -> Dataset:
    """Clean -> split on time -> engineer features -> fit preprocessing on train only.

    With `behavioural=True`, the per-card history features are added. They are
    computed on train and test CONCATENATED IN TIME ORDER and only then split, which
    is what makes them correct: a test-period transaction must see the training
    period's history, exactly as it would in production. Each row still uses only
    strictly-earlier rows, so nothing leaks (see add_behavioural_features).
    """
    raw_train, raw_test = load_frames(sample_frac, seed, verbose)

    train_clean = clean(raw_train, "train", verbose)
    test_clean = clean(raw_test, "test", verbose)

    if behavioural:
        boundary = train_clean["trans_date_trans_time"].max()
        combined = pd.concat([train_clean, test_clean], ignore_index=True)
        combined = add_behavioural_features(combined)
        train_part, valid_part = chronological_split(
            combined[combined["trans_date_trans_time"] <= boundary], valid_frac)
        test_part = combined[combined["trans_date_trans_time"] > boundary].reset_index(drop=True)
        train_part = add_features(train_part)
        valid_part = add_features(valid_part)
        test_part = add_features(test_part)
    else:
        train_part, valid_part = chronological_split(train_clean, valid_frac)
        train_part = add_features(train_part)
        valid_part = add_features(valid_part)
        test_part = add_features(test_clean)

    y_train = train_part["is_fraud"].to_numpy()
    y_valid = valid_part["is_fraud"].to_numpy()
    y_test = test_part["is_fraud"].to_numpy()

    numeric_columns = NUMERIC + (BEHAVIOURAL + ["card_is_new"] if behavioural else [])
    preprocessor = Preprocessor.fit(train_part, y_train,
                                    numeric_columns=numeric_columns)
    X_train = preprocessor.transform(train_part)
    X_valid = preprocessor.transform(valid_part)
    X_test = preprocessor.transform(test_part)

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
        numeric_count = len(preprocessor.numeric_columns)
        print("features ", f"{len(preprocessor.feature_names)} "
                          f"({numeric_count} numeric + "
                          f"{len(preprocessor.feature_names) - numeric_count} one-hot)")

    return Dataset(X_train, y_train, X_valid, y_valid, X_test, y_test,
                   preprocessor, report)
