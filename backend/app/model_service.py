"""Model loading, feature construction and prediction.

The single most important rule in this module: the API must reproduce the
training preprocessing exactly. It reuses `fdm.features.add_features` and the
fitted `Preprocessor` rather than reimplementing either, so there is no way for
the two to drift apart silently.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fdm.config import CATEGORICAL, NUMERIC  # noqa: E402
from fdm.features import add_features, haversine_km  # noqa: E402
from fdm.preprocess import Preprocessor  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "final_model.joblib"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}


class ModelBundle:
    """Everything needed to serve a prediction, loaded once at startup."""

    def __init__(self, model, preprocessor: Preprocessor, threshold: float,
                 name: str, metadata: dict[str, Any] | None = None):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = float(threshold)
        self.name = name
        self.metadata = metadata or {}

    # ── feature construction ──

    def build_frame(self, records: list[dict]) -> pd.DataFrame:
        """Turn raw API payloads into the exact frame the model was trained on."""
        frame = pd.DataFrame(records)
        frame["trans_date_trans_time"] = pd.to_datetime(frame["trans_date_trans_time"])
        frame["dob"] = pd.to_datetime(frame["dob"])
        # cc_num is an identifier: the model never sees it, but the schema does.
        frame["city_pop"] = frame["city_pop"].astype(float)
        return add_features(frame)

    # ── prediction ──

    def predict(self, records: list[dict]) -> np.ndarray:
        frame = self.build_frame(records)
        matrix = self.preprocessor.transform(frame)
        return self.model.predict_proba(matrix)[:, 1]

    def top_features(self, limit: int = 15) -> list[dict]:
        names = self.preprocessor.feature_names
        model = self.model
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_).ravel()
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_)).ravel()
        else:
            inner = getattr(model, "named_steps", {}).get("model")
            if inner is None:
                return []
            if hasattr(inner, "feature_importances_"):
                values = np.asarray(inner.feature_importances_).ravel()
            else:
                values = np.abs(np.asarray(inner.coef_)).ravel()

        order = np.argsort(values)[::-1][:limit]
        total = float(values.sum()) or 1.0
        return [{"feature": names[i], "importance": round(float(values[i] / total), 5)}
                for i in order]


@lru_cache(maxsize=1)
def load_bundle() -> ModelBundle:
    """Load and cache the serving artifacts. Raises if they are missing."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing artifacts in {ARTIFACT_DIR}. Run "
            "`python scripts/tune_models.py` to train and save them.")

    payload = joblib.load(MODEL_PATH)
    preprocessor = Preprocessor.load(PREPROCESSOR_PATH)

    # Saved either as a bare dict (current) or as the model itself (older runs).
    if isinstance(payload, dict):
        model = payload["model"]
        threshold = payload.get("threshold", 0.5)
        name = payload.get("model_name", "unknown")
    else:
        model, threshold, name = payload, 0.5, type(payload).__name__

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())

    return ModelBundle(model=model, preprocessor=preprocessor,
                       threshold=threshold, name=name, metadata=metadata)


def risk_band(probability: float, threshold: float) -> str:
    """Coarse display band, anchored on the real decision threshold."""
    if probability >= threshold:
        return "high"
    if probability >= threshold * 0.4:
        return "medium"
    return "low"


def explain(frame: pd.DataFrame, probabilities: np.ndarray) -> list[list[str]]:
    """Cheap, honest per-transaction reasons.

    These are the EDA findings restated for one row, not a model
    interpretation: each reason names an input that pushed the score up. They are
    labelled as "why this looks unusual", not "why the model decided".
    """
    reasons: list[list[str]] = []
    for (_, row), probability in zip(frame.iterrows(), probabilities):
        notes: list[str] = []
        if row["is_night"] == 1:
            notes.append("Night-time transaction (22:00-04:00): fraud rate is 18x daytime")
        if row["amt"] > 500:
            notes.append(f"Unusually high amount (${row['amt']:,.2f}); fraud median is $396")
        if row["distance_km"] > 500:
            notes.append(f"Cardholder is {row['distance_km']:,.0f} km from the merchant")
        if row["amt_is_round"] == 1 and row["amt"] >= 100:
            notes.append("Round dollar amount, a pattern fraudsters favour")
        if row["age"] < 25:
            notes.append("Cardholder under 25")
        if row["category"] in {"shopping_net", "misc_net", "grocery_pos"}:
            notes.append(f"Category '{row['category']}' has one of the highest fraud rates")
        if not notes:
            notes.append("No individual factor stands out; score is driven by the "
                         "combination of fields")
        if probability < 0.2:
            notes = ["Nothing about this transaction looks unusual"] + notes[:1]
        reasons.append(notes)
    return reasons
