"""Paths, constants and settings shared by every script and by the API."""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

TRAIN_PATH = DATA_DIR / "fraudTrain.csv"
TEST_PATH = DATA_DIR / "fraudTest.csv"

# 1.0 = full data. Lower it for a fast demo run, e.g. FDM_SAMPLE_FRAC=0.1
SAMPLE_FRAC = float(os.environ.get("FDM_SAMPLE_FRAC", "1.0"))

# Validation is the last VALID_FRAC of the *training* file, cut on time.
VALID_FRAC = 0.20

# Column groups. These are the contract between training and serving: the API
# builds exactly these columns, in exactly this order, from raw request fields.
NUMERIC = [
    "amt",
    "amt_log",
    "amt_is_round",
    "age",
    "distance_km",
    "city_pop_log",
    "hour",
    "day",
    "month",
    "dayofweek",
    "is_weekend",
    "is_night",
]
CATEGORICAL = ["category", "gender", "state", "merchant"]

# Raw fields the frontend/API must supply, mapped to the columns the model needs.
REQUIRED_INPUT_FIELDS = {
    "trans_date_trans_time": "Transaction date and time (YYYY-MM-DD HH:MM:SS)",
    "amt": "Transaction amount in dollars",
    "cc_num": "Card number",
    "merchant": "Merchant name",
    "category": "Merchant category",
    "gender": "Cardholder gender",
    "state": "Cardholder state",
    "lat": "Cardholder latitude",
    "long": "Cardholder longitude",
    "city_pop": "Cardholder city population",
    "dob": "Cardholder date of birth (YYYY-MM-DD)",
    "merch_lat": "Merchant latitude",
    "merch_long": "Merchant longitude",
}

for _directory in (ARTIFACT_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
