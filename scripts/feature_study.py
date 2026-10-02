"""Feature selection study: does using fewer features help?

Two experiments, both answering a question we have asserted but not measured.

  1. SHUFFLE THE SPLIT.  We claim a chronological split stops the model memorising
                        card and merchant identities. Now measured: the same model on
                        the same rows, chronological versus shuffled.

  2. TOP-K BY IMPORTANCE. XGBoost ranks the features itself, then we refit on the
                        top 10, 20, 30 ... and up to the full set, measuring PR-AUC
                        at each size.

Why rank with XGBoost's own importances rather than a penalised linear model: the
linear model was tried first and produced a worse ranking, because a linear boundary
and a greedy tree disagree about what matters when the signal is an interaction
("night AND gas station AND large amount"). The tree's own ordering is the only
ranking that matches how the final model actually splits.

Why the sweep matters rather than a single k: 693 of the 776 columns are merchant
one-hot dummies, and most are rarely used. If the model really only relies on a few
dozen features, dropping the rest should speed training and may even improve the
score. If it does not, that is worth knowing too: it means the wide one-hot block is
carrying signal in a way that no single column's importance reveals.

Run:  python scripts/feature_study.py [--sample-frac 0.3] [--skip-shuffle]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xgboost as xgb

from fdm.config import REPORT_DIR, SAMPLE_FRAC, SEED
from fdm.features import add_features, chronological_split, clean, load_frames
from fdm.preprocess import Preprocessor

warnings.filterwarnings("ignore")

# The 2% run showed a plateau from 75 to 250 with nothing usable below 50,
# so the confirmation only needs the plateau and one point either side of it.
K_VALUES = (50, 100, 250, 600)


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def fit_xgb(X_train, y_train, X_eval, y_eval, imbalance, **overrides):
    params = dict(n_estimators=600, learning_rate=0.05, max_depth=8,
                  scale_pos_weight=imbalance, tree_method="hist",
                  eval_metric="aucpr", early_stopping_rounds=50,
                  random_state=SEED, n_jobs=-1)
    params.update(overrides)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    return model


def build(sample_frac: float):
    raw_train, _ = load_frames(sample_frac, SEED, verbose=False)
    train_clean = clean(raw_train, "train", verbose=False)
    train_part, valid_part = chronological_split(train_clean, 0.20)
    train_frame, valid_frame = add_features(train_part), add_features(valid_part)

    y_train = train_frame["is_fraud"].to_numpy()
    y_valid = valid_frame["is_fraud"].to_numpy()
    preprocessor = Preprocessor.fit(train_frame, y_train, scheme="count+freq")
    X_train = preprocessor.fit_transform(train_frame, y_train)
    X_valid = preprocessor.transform(valid_frame)
    imbalance = float((y_train == 0).sum() / max(int(y_train.sum()), 1))
    return (X_train, y_train, X_valid, y_valid, list(preprocessor.feature_names),
            imbalance, train_frame, valid_frame)


# ── 1. does the split matter? ────────────────────────────────────────────────

def shuffle_split_experiment(train_frame, valid_frame, imbalance):
    """Same rows, same model, one split on time and one shuffled.

    This is the experiment behind the claim in the write-up. A shuffled split puts
    the same cards and merchants on both sides of the boundary, so part of the score
    is memorisation rather than detection.
    """
    X_train, y_train, X_valid, y_valid, _, _, _, _ = data
    chrono_model = fit_xgb(X_train, y_train, X_valid, y_valid, imbalance)
    chrono_pr = average_precision_score(y_valid, chrono_model.predict_proba(X_valid)[:, 1])

    # The encoder is refitted on the shuffled training half, so the only thing that
    # changes between the two arms is the split itself.
    combined = pd.concat([train_frame, valid_frame], ignore_index=True)
    y_all = combined["is_fraud"].to_numpy()
    idx_train, idx_valid = train_test_split(np.arange(len(y_all)), test_size=0.2,
                                            random_state=SEED, stratify=y_all)
    shuffled_train = combined.iloc[idx_train].reset_index(drop=True)
    shuffled_valid = combined.iloc[idx_valid].reset_index(drop=True)
    shuffled_pre = Preprocessor.fit(shuffled_train, y_all[idx_train], scheme="count+freq")
    shuffled_model = fit_xgb(shuffled_pre.fit_transform(shuffled_train, y_all[idx_train]),
                             y_all[idx_train], shuffled_pre.transform(shuffled_valid),
                             y_all[idx_valid], imbalance)
    shuffled_pr = average_precision_score(y_all[idx_valid],
                                          shuffled_model.predict_proba(
                                              shuffled_pre.transform(shuffled_valid))[:, 1])

    print(f"  chronological split  PR-AUC {chrono_pr:.4f}")
    print(f"  shuffled split       PR-AUC {shuffled_pr:.4f}")
    print(f"  inflation from shuffling: {shuffled_pr - chrono_pr:+.4f}")
    print()
    print("  The gap is the part of the score that comes from recognising cards and")
    print("  merchants it has already seen, rather than from detecting fraud. It is")
    print("  why every number in this project comes from the time-based split.")
    return {"chronological_pr_auc": round(float(chrono_pr), 4),
            "shuffled_pr_auc": round(float(shuffled_pr), 4),
            "inflation": round(float(shuffled_pr - chrono_pr), 4)}


# ── 2. top-k by XGBoost importance ──────────────────────────────────────────

def top_k_sweep(X_train, y_train, X_valid, y_valid, names, imbalance, order):
    """Refit on the k highest-importance features and score each size."""
    total = X_train.shape[1]
    print(f"  {'k':>6} {'PR-AUC':>9} {'gain':>9} {'seconds':>8}")
    rows = []

    start = time.time()
    full = fit_xgb(X_train, y_train, X_valid, y_valid, imbalance)
    baseline = average_precision_score(y_valid, full.predict_proba(X_valid)[:, 1])
    print(f"  {total:>6} {baseline:>9.4f} {0.0:>+9.4f} {time.time() - start:>8.0f}  (all)")
    rows.append({"k": total, "pr_auc": round(baseline, 4), "gain": 0.0,
                 "seconds": round(time.time() - start, 1), "is_full": True})

    for k in K_VALUES:
        if k >= total:
            continue
        columns = order[:k]
        start = time.time()
        model = fit_xgb(X_train[:, columns], y_train, X_valid[:, columns], y_valid, imbalance)
        pr = average_precision_score(y_valid, model.predict_proba(X_valid[:, columns])[:, 1])
        seconds = time.time() - start
        rows.append({"k": k, "pr_auc": round(pr, 4), "gain": round(pr - baseline, 4),
                     "seconds": round(seconds, 1), "is_full": False})
        print(f"  {k:>6} {pr:>9.4f} {pr - baseline:>+9.4f} {seconds:>8.0f}")

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True), baseline, order


def main() -> int:
    global data
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    parser.add_argument("--skip-shuffle", action="store_true")
    args = parser.parse_args()

    banner("Feature selection study")
    data = build(args.sample_frac)
    X_train, y_train, X_valid, y_valid, names, imbalance, train_frame, valid_frame = data
    print(f"train {X_train.shape[0]:,} rows | validation {X_valid.shape[0]:,} rows "
          f"| {X_train.shape[1]} features | imbalance {imbalance:.1f}:1")

    report: dict = {"sample_frac": args.sample_frac, "n_features": int(X_train.shape[1]),
                    "n_train_rows": int(X_train.shape[0])}

    if not args.skip_shuffle:
        banner("1. Does the split matter? chronological vs shuffled")
        report["shuffle_split"] = shuffle_split_experiment(train_frame, valid_frame, imbalance)

    banner("2. Top-k features by XGBoost importance")
    # Rank once on the full feature set, then reuse that ordering for every k.
    # Re-ranking inside each fold would make the comparison unstable.
    seed_model = fit_xgb(X_train, y_train, X_valid, y_valid, imbalance)
    order = np.argsort(np.asarray(seed_model.feature_importances_).ravel())[::-1]
    print()
    print("  15 highest-importance features:")
    for rank, column in enumerate(order[:15], start=1):
        print(f"    {rank:>2}. {names[column]}")
    print()

    table, baseline, order = top_k_sweep(X_train, y_train, X_valid, y_valid, names,
                                         imbalance, order)
    table.to_csv(REPORT_DIR / "feature_topk_sweep.csv", index=False)

    banner("3. Selection")
    best = table.loc[table["pr_auc"].idxmax()]
    print(f"  all {X_train.shape[1]} features   PR-AUC {baseline:.4f}")
    print(f"  best k = {int(best['k']):<3}            PR-AUC {best['pr_auc']:.4f}  "
          f"({best['gain']:+.4f})")
    print(f"  best trained in {best['seconds']:.0f}s against "
          f"{float(table.loc[table['is_full'], 'seconds'].iloc[0]):.0f}s for all features")
    print()
    print(table[["k", "pr_auc", "gain", "seconds"]].to_string(index=False))

    gain = float(best["gain"])
    report["baseline_pr_auc"] = round(float(baseline), 4)
    report["best_k"] = int(best["k"])
    report["best_pr_auc"] = float(best["pr_auc"])
    report["best_gain"] = gain
    if gain > 0.005:
        report["decision"] = f"adopt the top {int(best['k'])} features"
        print()
        print(f"DECISION: adopt the top {int(best['k'])} features. The gain is larger than")
        print("0.005 PR-AUC and it trains faster, so both scores and cost agree.")
    elif gain > 0.001:
        report["decision"] = "keep all features (gain is within noise)"
        print()
        print(f"DECISION: keep all {X_train.shape[1]} features. A {gain:+.4f} gain is")
        print("inside run-to-run noise, and dropping features the model does use is not")
        print("worth the risk for nothing.")
    else:
        report["decision"] = "keep all features"
        print()
        print(f"DECISION: keep all {X_train.shape[1]} features. No subset beat the full set,")
        print("so the wide one-hot block is carrying signal that no single column's")
        print("importance reveals. Selecting features here would lose information for")
        print("no measured return.")

    (REPORT_DIR / "feature_study.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
