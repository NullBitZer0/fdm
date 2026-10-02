"""Stage 7 - hyperparameter tuning with a time-aware random search.

Replaces the Optuna TPE run, which took 10 hours on 30% of the data. Random search
over a documented space is a recognised, appropriate strategy, and at this cost it
is the better trade: the goal is to find a configuration better than the hand-set
baseline, not to prove one search algorithm beats another.

What is deliberately NOT here:

  * No Bayesian sampler. TPE took 35,494s for 20 trials and pruned none of them, so
    the pruning argument that motivated it did not hold in practice.
  * No cross-validation inside the search. See the note on validation below.

Validation: every trial is scored on the same chronological validation block used
throughout Stage 6 (the last 20% of the training file, split on time). `KFold` is
not used because it shuffles, which on time-ordered data lets a model be scored on
January 2019 transactions after training on June 2020.

How many trials: enough to cover the space coarsely. One fit on 10% of the data is
roughly 40s, so 12 trials in both directions costs ~8 minutes rather than half a day.

Run:  python scripts/tune_random.py [--trials 12] [--sample-frac 0.1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xgboost as xgb

from fdm.config import ARTIFACT_DIR, REPORT_DIR, SAMPLE_FRAC, SEED
from fdm.data import build_dataset
from fdm.metrics import evaluate
from fdm.models import imbalance_ratio

# Hand-set baseline from Stage 6, for the tuned-versus-baseline comparison.
BASELINE = {
    "n_estimators": 600, "learning_rate": 0.05, "max_depth": 8,
    "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
    "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
}

# Log-scaled where the useful range is multiplicative: learning rate spans two
# orders of magnitude, and uniform sampling wastes most trials in the top half.
SPACE = {
    "n_estimators": ([200, 300, 400, 600, 800, 1000], False),
    "learning_rate": (list(np.geomspace(0.01, 0.3, 12)), False),
    "max_depth": (list(range(3, 13)), False),
    "min_child_weight": (list(np.geomspace(0.5, 50, 8)), False),
    "subsample": (list(np.linspace(0.6, 1.0, 5)), False),
    "colsample_bytree": (list(np.linspace(0.4, 1.0, 7)), False),
    "gamma": (list(np.geomspace(1e-8, 5.0, 6)), False),
    "reg_alpha": (list(np.geomspace(1e-8, 5.0, 6)), False),
    "reg_lambda": (list(np.geomspace(1e-3, 10.0, 6)), False),
}


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def fit(params, X_train, y_train, X_valid, y_valid, imbalance):
    model = xgb.XGBClassifier(**params, scale_pos_weight=imbalance,
                              tree_method="hist", eval_metric="aucpr",
                              early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    banner("Stage 7 - random search hyperparameter tuning")
    dataset = build_dataset(sample_frac=args.sample_frac)
    imbalance = imbalance_ratio(dataset.y_train)
    X_train, y_train = dataset.X_train, dataset.y_train
    X_valid, y_valid = dataset.X_valid, dataset.y_valid
    print(f"\n{len(y_train):,} training rows, {len(y_valid):,} validation rows, "
          f"{X_train.shape[1]} features, imbalance {imbalance:.1f}:1")
    print("scored on the same chronological validation block as Stage 6 "
          "(no shuffling, no KFold)")

    # Baseline first, on identical data, so the comparison is like for like.
    start = time.time()
    baseline_model = fit(BASELINE, X_train, y_train, X_valid, y_valid, imbalance)
    baseline_pr = evaluate("baseline (Stage 6 settings)", y_valid,
                           baseline_model.predict_proba(X_valid)[:, 1],
                           seconds=time.time() - start)
    print(f"\nbaseline PR-AUC {baseline_pr['pr_auc']:.4f}  "
          f"({baseline_pr['seconds']:.0f}s)")

    banner(f"random search, {args.trials} trials")
    candidates = list(ParameterSampler({k: v[0] for k, v in SPACE.items()},
                                       n_iter=args.trials, random_state=SEED))
    trials, best_pr, best_params, best_model = [], -1.0, None, None
    for number, params in enumerate(candidates, start=1):
        start = time.time()
        model = fit(params, X_train, y_train, X_valid, y_valid, imbalance)
        row = evaluate(f"trial {number}", y_valid, model.predict_proba(X_valid)[:, 1],
                       seconds=time.time() - start)
        trials.append(row)
        if row["pr_auc"] > best_pr:
            best_pr, best_params, best_model = row["pr_auc"], params, model
        print(f"  trial {number:>3}/{args.trials}  PR-AUC {row['pr_auc']:.4f}  "
              f"({row['seconds']:.0f}s)  best so far {best_pr:.4f}")

    banner("tuned versus baseline")
    tuned_row = evaluate("tuned (random search)", y_valid,
                         best_model.predict_proba(X_valid)[:, 1])
    comparison = pd.DataFrame([
        {"model": "baseline (Stage 6 settings)", "pr_auc": baseline_pr["pr_auc"],
         "roc_auc": baseline_pr["roc_auc"], "f1": baseline_pr["f1"],
         "seconds": baseline_pr["seconds"]},
        {"model": "tuned (random search)", "pr_auc": tuned_row["pr_auc"],
         "roc_auc": tuned_row["roc_auc"], "f1": tuned_row["f1"],
         "seconds": tuned_row["seconds"]},
    ])
    comparison["gain"] = (comparison["pr_auc"] - baseline_pr["pr_auc"]).round(4)
    print(comparison[["model", "pr_auc", "roc_auc", "f1", "gain", "seconds"]]
          .to_string(index=False))
    print()
    print("best parameters:")
    for key in sorted(best_params):
        print(f"  {key:<18} {best_params[key]:.6g}")

    comparison.to_csv(REPORT_DIR / "stage7_tuned_vs_baseline.csv", index=False)
    pd.DataFrame(trials)[["model", "pr_auc", "f1", "seconds"]].to_csv(
        REPORT_DIR / "stage7_trials.csv", index=False)

    threshold = tuned_row["threshold"]
    metadata = {
        "model_name": "XGBoost + Class Weight (tuned)",
        "params": best_params,
        "threshold": float(threshold),
        "validation_pr_auc": float(tuned_row["pr_auc"]),
        "validation_roc_auc": float(tuned_row["roc_auc"]),
        "validation_f1": float(tuned_row["f1"]),
        "baseline_pr_auc": float(baseline_pr["pr_auc"]),
        "gain": round(float(tuned_row["pr_auc"] - baseline_pr["pr_auc"]), 4),
        "strategy": f"random search, {args.trials} trials, time-based validation",
        "sample_frac": float(args.sample_frac),
        "n_train_rows": int(len(y_train)),
        "n_valid_rows": int(len(y_valid)),
        "n_features": int(X_train.shape[1]),
        "validation_fraud_rate": float(y_valid.mean()),
        "trained_on": (f"{args.sample_frac:.0%} chronological sample of the "
                       f"training set" if args.sample_frac < 1.0 else "full dataset"),
    }

    if not args.no_test:
        banner("final check on the untouched test set")
        start = time.time()
        prob_test = best_model.predict_proba(dataset.X_test)[:, 1]
        test_row = evaluate("tuned (test)", dataset.y_test, prob_test,
                            threshold=threshold, seconds=time.time() - start)
        print(pd.DataFrame([test_row])[["model", "pr_auc", "roc_auc", "threshold",
                                        "precision", "recall", "f1", "tp", "fp", "fn"]]
              .to_string(index=False))
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            metadata[f"test_{key}"] = float(test_row[key])
        metadata["test_fraud_rate"] = float(dataset.y_test.mean())
        metadata["test_rows"] = int(len(dataset.y_test))

    banner("artifacts")
    dataset.preprocessor.save(ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump({"model": best_model, "threshold": threshold,
                 "model_name": metadata["model_name"], "params": best_params,
                 "encoding_scheme": dataset.preprocessor.scheme,
                 "n_features": int(X_train.shape[1]),
                 "validation_pr_auc": float(tuned_row["pr_auc"])},
                ARTIFACT_DIR / "final_model.joblib")
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote final_model.joblib ({metadata['model_name']}), preprocessor.joblib "
          f"and model_metadata.json")

    verdict = ("tuning helped" if metadata["gain"] > 0.002
               else "tuning did not beat the hand-set baseline")
    print()
    print(f"VERDICT: {verdict} (gain {metadata['gain']:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
