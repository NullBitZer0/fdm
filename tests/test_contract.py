"""Tests for the training-to-serving contract.

The riskiest failure in this project is silent: the API builds features slightly
differently from training, the model still returns a number, and nobody notices
until the scores are quietly wrong. These tests pin the contract.

Run:  python -m pytest tests/ -v      (or  python tests/test_contract.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from fdm.config import CATEGORICAL, NUMERIC  # noqa: E402
from fdm.features import add_features, clean, haversine_km  # noqa: E402
from fdm.preprocess import Preprocessor  # noqa: E402

RAW = {
    "trans_date_trans_time": "2020-06-21 23:44:07",
    "cc_num": 371449635398431,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "gas_transport",
    "amt": 1024.55,
    "first": "Jennifer",
    "last": "Banks",
    "gender": "F",
    "street": "561 Perry Cove",
    "city": "Moravian Falls",
    "state": "NC",
    "zip": 28654,
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 3495,
    "job": "Psychologist, counselling",
    "dob": "1988-03-09",
    "trans_num": "0b242abb623afc578575680df30655b9",
    "unix_time": 1325376018,
    "merch_lat": 36.0113,
    "merch_long": -82.0483,
    "is_fraud": 1,
}


def _frame(rows: int = 1) -> pd.DataFrame:
    """Raw rows exactly as they appear in the CSV, still un-parsed."""
    base = pd.DataFrame([RAW])
    return pd.concat([base] * rows, ignore_index=True)


def _distinct_raw(rows: int) -> pd.DataFrame:
    """`rows` raw records that differ in every categorical field.

    Needed because clean() drops exact duplicates: repeating one record verbatim
    would collapse to a single row and quietly weaken every downstream assertion.
    """
    base = RAW | {"is_fraud": 0}
    records = []
    for index in range(rows):
        record = dict(base)
        record["category"] = f"category_{index}"
        record["state"] = f"S{index % 50:02d}"
        record["merchant"] = f"merchant_{index}"
        record["gender"] = "F" if index % 2 == 0 else "M"
        record["trans_num"] = f"{index:032x}"
        record["unix_time"] = 1_325_376_018 + index
        record["trans_date_trans_time"] = f"2020-06-21 12:{index % 60:02d}:{index % 60:02d}"
        records.append(record)
    return pd.DataFrame(records)


def _prepared(rows: int = 1, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The real training path: clean (which parses the dates) then engineer."""
    raw = _distinct_raw(rows) if frame is None else frame
    return add_features(clean(raw, "test", verbose=False))


def test_haversine_matches_known_distance():
    """One degree of latitude is ~111.2 km anywhere on the globe."""
    distance = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert abs(distance - 111.19) < 0.5, distance
    assert haversine_km(36.0788, -81.1781, 36.0788, -81.1781) == 0.0


def test_add_features_creates_every_model_column():
    frame = _prepared(frame=_frame(1))
    for column in NUMERIC:
        assert column in frame.columns, f"missing {column}"
    # The derived columns must be numerically sane, not merely present.
    row = frame.iloc[0]
    assert row["hour"] == 23
    assert row["is_night"] == 1
    assert row["is_weekend"] == 1          # 2020-06-21 was a Sunday (dayofweek 6)
    assert abs(row["age"] - 32.3) < 0.5
    assert abs(row["amt_log"] - np.log1p(1024.55)) < 1e-9
    assert row["amt_is_round"] == 0
    assert abs(row["city_pop_log"] - np.log1p(3495)) < 1e-9
    assert 0 < row["distance_km"] < 200


def test_is_night_boundary_is_exact():
    """22:00 and 03:00 are night; 21:00 and 04:00 are not."""
    hours = [21, 22, 23, 0, 3, 4, 12]
    frame = _frame(len(hours))
    frame["trans_date_trans_time"] = [f"2020-06-21 {h:02d}:30:00" for h in hours]
    flags = _prepared(frame=frame)["is_night"].tolist()
    assert flags == [0, 1, 1, 1, 1, 0, 0], flags


def test_preprocessor_roundtrip_and_unknown_categories():
    # 40 rows differing in every categorical field, so the expected one-hot width
    # is a real assertion rather than a tautology.
    train = _prepared(40)
    assert len(train) == 40, len(train)
    preprocessor = Preprocessor.fit(train)
    matrix = preprocessor.transform(train)

    assert matrix.shape == (40, len(preprocessor.feature_names))
    assert matrix.shape[1] == len(NUMERIC) + 40 + 2 + 40 + 40
    # Sparse output must stay sparse: 772 columns at 1M rows is only viable if so.
    assert matrix.nnz < matrix.shape[0] * matrix.shape[1]

    # An unseen merchant and state must not raise; handle_unknown="ignore".
    unseen = train.head(1).copy()
    unseen["merchant"] = "fraud_This Merchant Did Not Exist"
    unseen["state"] = "ZZ"
    assert preprocessor.transform(unseen).shape[1] == matrix.shape[1]


def CATEGORICAL_LEVELS(frame: pd.DataFrame) -> int:
    return int(sum(frame[column].nunique() for column in CATEGORICAL))


def test_feature_order_is_stable():
    """Feature order is the contract. A silent reorder would corrupt every score."""
    train = _prepared(10)
    preprocessor = Preprocessor.fit(train)
    assert preprocessor.feature_names[:len(NUMERIC)] == NUMERIC
    for column in CATEGORICAL:
        assert any(name.startswith(f"{column}_") for name in preprocessor.feature_names)


def test_api_and_training_produce_identical_features():
    """The core Stage 9 guarantee: one request == one training row."""
    from app.model_service import ModelBundle
    from app.schemas import Transaction

    class _Stub:
        def predict_proba(self, matrix):
            return np.column_stack([1 - matrix[:, 0], matrix[:, 0]])

    train = _prepared(50)
    bundle = ModelBundle(model=_Stub(), preprocessor=Preprocessor.fit(train),
                         threshold=0.5, name="stub")

    payload = Transaction(**{k: v for k, v in RAW.items() if k != "is_fraud"})
    from_api = bundle.build_frame([payload.model_dump()])
    from_training = _prepared(frame=_frame(1))

    for column in NUMERIC:
        assert np.isclose(from_api[column].iloc[0], from_training[column].iloc[0]), column


def test_clean_drops_impossible_rows():
    frame = _frame(3)
    frame.loc[1, "amt"] = -5.0
    frame.loc[2, "lat"] = 999.0
    cleaned = clean(frame, "test", verbose=False)
    assert len(cleaned) == 1, len(cleaned)


def test_schema_rejects_bad_input():
    """The API must refuse bad input rather than scoring it."""
    from pydantic import ValidationError
    from app.schemas import Transaction

    good = {k: v for k, v in RAW.items() if k != "is_fraud"}
    Transaction(**good)  # the valid case must pass

    for patch in ({"amt": -1.0}, {"gender": "X"}, {"lat": 999.0},
                  {"state": "North Carolina"}, {"dob": "2099-01-01"},
                  {"trans_date_trans_time": "2099-01-01 00:00:00"}):
        payload = {**good, **patch}
        try:
            Transaction(**payload)
        except ValidationError:
            continue
        raise AssertionError(f"schema accepted invalid input: {patch}")


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
