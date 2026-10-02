"""Cleaning and feature engineering.

Everything here is row-level and stateless on purpose. The same functions run at
training time and at serving time, so a live request goes through exactly the same
transformations as a training row. No column is ever computed by aggregating the
target across rows, because that would leak and would not exist online.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR

EARTH_RADIUS_KM = 6371.0
NIGHT_HOURS = [22, 23, 0, 1, 2, 3]

RAW_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first",
    "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop",
    "job", "dob", "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud",
]

_GAP_FILL_COLUMNS = ["merchant", "category", "first", "last", "street", "city",
                     "state", "job", "gender"]


def clean(df: pd.DataFrame, name: str = "frame", verbose: bool = True) -> pd.DataFrame:
    """Remove bad rows and fill gaps. Reports what it changed."""
    df = df.copy()
    start = len(df)

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    bad_dates = int(df["trans_date_trans_time"].isna().sum())
    df = df[df["trans_date_trans_time"].notna()]

    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()

    bad_amount = int((df["amt"] <= 0).sum())
    df = df[df["amt"] > 0]

    valid_geo = (df["lat"].between(-90, 90) & df["long"].between(-180, 180) &
                 df["merch_lat"].between(-90, 90) & df["merch_long"].between(-180, 180))
    bad_geo = int((~valid_geo).sum())
    df = df[valid_geo]

    gaps = int(df.isna().sum().sum())
    for column in df.select_dtypes(include="number").columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())
    for column in _GAP_FILL_COLUMNS:
        if column in df.columns and df[column].isna().any():
            df[column] = df[column].fillna("Unknown")

    if verbose:
        print(f"{name}: {start:,} -> {len(df):,} rows | bad dates {bad_dates} "
              f"| duplicates {duplicates} | bad amount {bad_amount} "
              f"| bad coordinates {bad_geo} | gaps filled {gaps}")
    return df.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, vectorised."""
    p1, o1 = np.radians(lat1), np.radians(lon1)
    p2, o2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((p2 - p1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) * np.sin((o2 - o1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived columns. Identical formula for train, validation, test and API."""
    df = df.copy()
    time = df["trans_date_trans_time"]

    df["hour"] = time.dt.hour
    df["dayofweek"] = time.dt.dayofweek
    df["day"] = time.dt.day
    df["month"] = time.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)

    df["age"] = (time - df["dob"]).dt.days / 365.25

    df["amt_log"] = np.log1p(df["amt"])
    df["amt_is_round"] = (df["amt"] == df["amt"].round(0)).astype(int)
    df["distance_km"] = haversine_km(df["lat"], df["long"],
                                     df["merch_lat"], df["merch_long"])
    df["city_pop_log"] = np.log1p(df["city_pop"])
    return df


BEHAVIOURAL = ["card_txn_1h", "card_txn_24h", "card_amt_ratio", "card_amt_dev",
               "card_hours_since_last", "card_dist_from_home_km", "card_category_seen"]

_HOUR_NS = 3600 * 10**9
_DAY_NS = 24 * _HOUR_NS


def _trailing_counts(times_ns: np.ndarray, positions: np.ndarray, window_ns: int) -> np.ndarray:
    """How many of `positions` fall in the `window_ns` strictly BEFORE each one.

    Computed with searchsorted rather than pandas rolling, for two reasons. First,
    pandas groupby.rolling returns a (card, timestamp) MultiIndex that cannot be
    aligned back onto the frame's positional index. Second, searchsorted makes the
    "strictly before" rule explicit instead of leaving it to `closed=` semantics.
    """
    ordered = times_ns[positions]                 # positions are already time-sorted
    left = np.searchsorted(ordered, ordered - window_ns, side="left")
    return np.arange(len(ordered)) - left         # excludes the current row


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-card behavioural features computed from STRICTLY PAST transactions.

    Everything else in this project is row-level: amount, hour, distance, merchant.
    The strongest fraud signals in practice are behavioural -- a card used three
    times in an hour, an amount far above that card's own habit, a purchase nowhere
    near where the card normally transacts -- and none of them exist yet.

    Leakage control is the whole difficulty. Every feature below uses only rows that
    occur BEFORE the row being scored:

      * the frame is sorted by transaction time first
      * the two trailing windows exclude the current row by construction
      * rolling statistics are shifted by one row within each card
      * cumcount() reports the occurrence index, so "seen before" never counts itself

    A validation or test row therefore sees the history a real model would have had
    at that moment, which is exactly the history a live request would need too.

    The cost is a serving contract change: these features cannot be computed from a
    single transaction, because they need that card's recent history.
    """
    df = df.sort_values("trans_date_trans_time", kind="mergesort").copy()
    times = df["trans_date_trans_time"]
    times_ns = times.astype("int64").to_numpy()
    n = len(df)

    txn_1h = np.zeros(n, dtype="float32")
    txn_24h = np.zeros(n, dtype="float32")
    for positions in df.groupby("cc_num", sort=False).indices.values():
        positions = np.sort(positions)
        txn_1h[positions] = _trailing_counts(times_ns, positions, _HOUR_NS)
        txn_24h[positions] = _trailing_counts(times_ns, positions, _DAY_NS)
    df["card_txn_1h"] = txn_1h
    df["card_txn_24h"] = txn_24h

    # How unusual this amount is FOR THIS CARD rather than for the population.
    # transform() keeps the result aligned to the frame, which groupby.rolling does not.
    grouped = df.groupby("cc_num", sort=False)["amt"]
    median = grouped.transform(lambda s: s.rolling(50, min_periods=10).median().shift(1))
    mean = grouped.transform(lambda s: s.rolling(50, min_periods=10).mean().shift(1))
    std = grouped.transform(lambda s: s.rolling(50, min_periods=10).std().shift(1))
    df["card_amt_ratio"] = (df["amt"] / median.replace(0, np.nan)).to_numpy()
    df["card_amt_dev"] = ((df["amt"] - mean).abs() / std.replace(0, np.nan)).to_numpy()

    # dormancy: a card quiet for months and then used is worth noticing
    df["card_hours_since_last"] = (times.diff().dt.total_seconds() / 3600.0).to_numpy()

    # "home" location: the median of this card's past transactions. Distance from it
    # compares against where THIS CARD behaves, which is a sharper signal than the
    # distance from the cardholder's registered city.
    home_lat = df.groupby("cc_num", sort=False)["merch_lat"].transform(
        lambda s: s.rolling(50, min_periods=10).median().shift(1))
    home_long = df.groupby("cc_num", sort=False)["merch_long"].transform(
        lambda s: s.rolling(50, min_periods=10).median().shift(1))
    df["card_dist_from_home_km"] = haversine_km(home_lat, home_long,
                                                df["merch_lat"], df["merch_long"])

    # has this card ever used this category before? cumcount is 0 on first sight
    df["card_category_seen"] = (df.groupby(["cc_num", "category"], sort=False)
                                .cumcount() > 0).astype(int).to_numpy()

    # No history yet: neutral values, plus a flag so the model can tell "a quiet
    # card" apart from "a card we have never seen".
    df["card_is_new"] = (txn_24h == 0).astype(int)
    for column in BEHAVIOURAL:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.reset_index(drop=True)


def load_frames(sample_frac: float = 1.0, seed: int = 42, verbose: bool = True):
    """Load both CSVs, optionally subsample while keeping time order."""
    train = pd.read_csv(DATA_DIR / "fraudTrain.csv", index_col=0)
    test = pd.read_csv(DATA_DIR / "fraudTest.csv", index_col=0)

    if sample_frac < 1.0:
        train = (train.sample(frac=sample_frac, random_state=seed)
                      .sort_values("trans_date_trans_time").reset_index(drop=True))
        test = (test.sample(frac=sample_frac, random_state=seed)
                     .sort_values("trans_date_trans_time").reset_index(drop=True))

    if verbose:
        print(f"train {len(train):,} rows  fraud {train['is_fraud'].mean():.4%}")
        print(f"test  {len(test):,} rows  fraud {test['is_fraud'].mean():.4%}")
    return train, test


def chronological_split(train_clean: pd.DataFrame, valid_frac: float = 0.20):
    """Cut the *end* of the training data as validation. Never shuffles.

    The data is in time order and card/merchant behaviour drifts, so a random
    split would let the model memorise identities across the boundary.
    """
    frame = train_clean.sort_values("trans_date_trans_time").reset_index(drop=True)
    cut = frame["trans_date_trans_time"].quantile(1.0 - valid_frac)
    train_part = frame[frame["trans_date_trans_time"] < cut]
    valid_part = frame[frame["trans_date_trans_time"] >= cut]
    return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)
