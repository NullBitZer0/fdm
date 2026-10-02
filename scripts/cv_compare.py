"""Five-fold time-series cross-validation: does tuning actually help?

This closes the gap left by the rest of the project. Almost every decision here —
four models, five resampling strategies, three encodings, ten feature-set sizes —
was made on ONE chronological validation block. That invites a fair objection: some
of the reported margin is selection luck rather than a real difference. Cross-
validation over several time periods gives a variance estimate and re-tests the one
comparison that matters most, tuned versus untuned.

Why TimeSeriesSplit and not StratifiedKFold
--------------------------------------------
KFold shuffles, and this dataset is in time order. We measured what that costs in
scripts/feature_study.py: the same rows and the same model score 0.0329 PR-AUC
higher under a shuffled split than under a chronological one, because card and
merchant identities recur on both sides of a random boundary. Running StratifiedKFold
would reintroduce exactly the inflation this project is trying to avoid, and would
make both arms look better than either really is.

TimeSeriesSplit keeps the property that matters: every fold trains only on the past
and is scored on the future. It also mirrors deployment, where a model scores
transactions it has not seen.

Leakage control
---------------
The scaler, the one-hot encoder and the frequency encoder are refitted inside each
fold, on that fold's training rows only. Fitting them once on all the data and then
cross-validating would leak the validation periods' category distributions into
training, which inflates scores in exactly the way the shuffle test warns about.

Run:  python scripts/cv_compare.py [--sample-frac 0.3] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xgboost as xgb

from fdm.config import DATA_DIR, REPORT_DIR, SAMPLE_FRAC, SEED
from fdm.config import NUMERIC
from fdm.features import (BEHAVIOURAL, add_behavioural_features,
                           add_features, chronological_split, clean, load_frames)
from fdm.preprocess import Preprocessor

CONFIGS = {
    "untuned (Stage 6)": "untuned_baseline",
    "tuned (random search)": "random_search_winner",
}


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def load(sample_frac: float, behavioural: bool = False):
    """Load, clean and engineer. The split is cut later, by TimeSeriesSplit."""
    raw_train, _ = load_frames(sample_frac, SEED, verbose=False)
    train_clean = clean(raw_train, "train", verbose=False)
    frame = train_clean.sort_values("trans_date_trans_time").reset_index(drop=True)
    if behavioural:
        frame = add_behavioural_features(frame)
    return add_features(frame)


def fit_one(params, frame_train, frame_valid, imbalance, behavioural: bool = False):
    """Fit preprocessing on this fold's training rows only, then the model.

    Both arms of the --behavioural comparison share one frame; they differ only in
    which numeric columns the Preprocessor is allowed to use. Selecting the columns
    here rather than rebuilding the frame is what guarantees the arms are identical
    in every other respect, so the difference measured is the feature set alone.
    """
    y_train = frame_train["is_fraud"].to_numpy()
    y_valid = frame_valid["is_fraud"].to_numpy()

    numeric_columns = list(NUMERIC) + (
        BEHAVIOURAL + ["card_is_new"] if behavioural else [])
    preprocessor = Preprocessor.fit(frame_train, y_train, scheme="count+freq",
                                    numeric_columns=numeric_columns)
    X_train = preprocessor.fit_transform(frame_train, y_train)
    X_valid = preprocessor.transform(frame_valid)

    # No eval_set and no early stopping. Early stopping needs a validation set, and
    # the only one available is the fold we are trying to score honestly, so every
    # fit runs its full n_estimators. That costs time and is the correct trade.
    model = xgb.XGBClassifier(**params, scale_pos_weight=imbalance,
                              tree_method="hist", eval_metric="aucpr",
                              random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train, verbose=False)
    proba = model.predict_proba(X_valid)[:, 1]
    return (average_precision_score(y_valid, proba),
            roc_auc_score(y_valid, proba),
            float(y_valid.mean()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--behavioural", action="store_true",
                        help="compare row-level features against row-level + "
                             "per-card history features")
    args = parser.parse_args()

    if args.behavioural:
        CONFIGS.clear()
        CONFIGS["row-level only"] = None
        CONFIGS["+ behavioural history"] = True

    banner("5-fold time-series cross-validation: tuned versus untuned")
    configs = json.loads((REPORT_DIR / "tuning_configs.json").read_text())

    frame = load(args.sample_frac, behavioural=args.behavioural)
    y_all = frame["is_fraud"].to_numpy()
    imbalance = float((y_all == 0).sum() / max(int(y_all.sum()), 1))
    print(f"{len(frame):,} rows, {frame['trans_date_trans_time'].min()} to "
          f"{frame['trans_date_trans_time'].max()}")
    print(f"overall fraud rate {y_all.mean():.4%}, imbalance {imbalance:.1f}:1")
    print(f"folds {args.folds} (TimeSeriesSplit: every fold trains only on the past)")

    splits = TimeSeriesSplit(n_splits=args.folds).split(frame)
    folds = list(splits)

    print()
    print("  fold  train rows  valid rows  train period              valid period")
    for number, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"  {number:>4}  {len(train_idx):>10,}  {len(valid_idx):>10,}   "
              f"{str(frame['trans_date_trans_time'].iloc[train_idx].min())[:16]}  "
              f"{str(frame['trans_date_trans_time'].iloc[valid_idx].min())[:16]}")

    results: dict[str, list] = {label: [] for label in CONFIGS}
    for label, key in CONFIGS.items():
        # the behavioural arm reuses the untuned parameters; only the
        # feature set differs, which is the whole point of the comparison
        params = configs[key] if isinstance(key, str) else configs["untuned_baseline"]
        banner(f"{label}")
        for number, (train_idx, valid_idx) in enumerate(folds, start=1):
            start = time.time()
            pr_auc, roc_auc, fraud_rate = fit_one(
                params, frame.iloc[train_idx], frame.iloc[valid_idx], imbalance,
                behavioural=bool(key))
            results[label].append({"fold": number, "pr_auc": round(pr_auc, 4),
                                   "roc_auc": round(roc_auc, 4),
                                   "fraud_rate": round(fraud_rate, 4),
                                   "seconds": round(time.time() - start, 1)})
            print(f"  fold {number}: PR-AUC {pr_auc:.4f}  ROC-AUC {roc_auc:.4f}  "
                  f"(fraud {fraud_rate:.3%}, {time.time() - start:.0f}s)")

    banner("summary")
    rows = []
    for label, folds_out in results.items():
        pr = np.array([fold["pr_auc"] for fold in folds_out])
        roc = np.array([fold["roc_auc"] for fold in folds_out])
        rows.append({"config": label, "folds": len(pr),
                     "mean_pr_auc": round(float(pr.mean()), 4),
                     "std_pr_auc": round(float(pr.std(ddof=1)), 4),
                     "min_pr_auc": round(float(pr.min()), 4),
                     "max_pr_auc": round(float(pr.max()), 4),
                     "mean_roc_auc": round(float(roc.mean()), 4)})
    summary = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)
    print(summary.to_string(index=False))

    first, second = list(CONFIGS)[0], list(CONFIGS)[1]
    untuned = np.array([f["pr_auc"] for f in results[first]])
    tuned = np.array([f["pr_auc"] for f in results[second]])
    difference = tuned - untuned
    print()
    print(f"  {second} minus {first}, per fold: "
          f"{', '.join(f'{d:+.4f}' for d in difference)}")
    print(f"  mean difference {difference.mean():+.4f}, "
          f"std {difference.std(ddof=1):.4f}")
    print(f"  {second} wins {int((difference > 0).sum())} of {len(difference)} folds")

    # The paired difference is the meaningful quantity: same folds, so period
    # effects cancel. Thresholding the MEAN alone is not enough -- a mean of +0.002
    # with a spread of 0.0055 is one standard error from zero, and calling that a
    # win is how a search "improvement" turns out to be noise. Compare the mean to
    # its own standard error instead.
    n = len(difference)
    standard_error = float(difference.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    t_stat = float(difference.mean() / standard_error) if standard_error else 0.0
    significant = abs(t_stat) >= 2.776          # two-sided, p<0.05, df=4
    subject = "the behavioural features" if args.behavioural else "tuning"
    verdict = (f"{subject} helps" if difference.mean() > 0 and significant
               else f"{subject} does not measurably help")
    print()
    print(f"  mean difference {difference.mean():+.4f}  "
          f"standard error {standard_error:.4f}  t = {t_stat:.2f} (df={n - 1})")
    print(f"  |t| >= 2.776 would be needed for significance at p<0.05; "
          f"this is {abs(t_stat):.2f}")
    print()
    print(f"  VERDICT: {verdict}.")
    if not significant:
        print(f"  The paired difference is within one standard error of zero, so the two")
        print(f"  arms are indistinguishable on this data. The hand-set, row-level model is")
        print(f"  what serves, and nothing is adopted on a sub-significant gain.")

    tag = "behavioural" if args.behavioural else "tuned"
    summary.to_csv(REPORT_DIR / f"cv_{tag}_summary.csv", index=False)
    pd.DataFrame([dict(config=label, **fold)
                  for label, folds_out in results.items() for fold in folds_out]
                 ).to_csv(REPORT_DIR / ("cv_per_fold_behavioural.csv" if args.behavioural else "cv_per_fold.csv"), index=False)
    (REPORT_DIR / ("cv_behavioural_summary.json" if args.behavioural else "cv_summary.json")).write_text(json.dumps({
        "sample_frac": args.sample_frac, "folds": args.folds,
        "n_rows": int(len(frame)), "imbalance": round(imbalance, 1),
        "mean_pr_auc": {row["config"]: row["mean_pr_auc"] for row in rows},
        "std_pr_auc": {row["config"]: row["std_pr_auc"] for row in rows},
        "mean_difference": round(float(difference.mean()), 4),
        "std_difference": round(float(difference.std(ddof=1)), 4),
        "folds_won_by_tuned": int((difference > 0).sum()),
        "verdict": verdict}, indent=2))
    print()
    print(f"wrote cv_tuned_vs_untuned.csv, cv_per_fold.csv and cv_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
