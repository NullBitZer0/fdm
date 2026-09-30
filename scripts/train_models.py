"""Stage 6 - develop and compare multiple models.

Produces three things:
  1. reports/stage6_model_comparison.csv   four models, validation PR-AUC
  2. reports/stage6_undersampling.csv      under- vs over-sampling, same model
  3. reports/stage6_ensemble.csv           majority vote, hard vs soft

Run:  python scripts/train_models.py [--sample-frac 0.1] [--skip-ensemble]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdm import (SEED, build_dataset, build_models, evaluate,
                 lift_over_random, smote, undersample, xgboost)
from fdm.config import ARTIFACT_DIR, REPORT_DIR, SAMPLE_FRAC
from fdm.models import imbalance_ratio
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid):
    """Fit, handing the validation set to the boosters that support early stopping.

    XGBoost and LightGBM are built with early_stopping_rounds set, and both raise
    if it is set without an eval_set. imblearn pipelines forward **fit_params to
    the final step, so the same call works for the resampled and plain variants.
    """
    final = model[-1] if hasattr(model, "steps") else model
    kind = type(final).__name__

    if kind == "XGBClassifier":
        model.fit(X_train, y_train, model__eval_set=[(X_valid, y_valid)],
                  model__verbose=False)
    elif kind == "LGBMClassifier":
        # LightGBM silences its own logging via the verbose constructor arg, and
        # early stopping is requested with a callback rather than a fit kwarg.
        import lightgbm as lgb
        model.fit(X_train, y_train, model__eval_set=[(X_valid, y_valid)],
                  model__callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model


def compare_models(X_train, y_train, X_valid, y_valid, imbalance, seed=SEED):
    """Four algorithms, each with the imbalance strategy that suits it."""
    candidates = build_models(imbalance, seed)
    rows, probabilities = [], {}

    for name, model in candidates.items():
        start = time.time()
        fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid)
        seconds = time.time() - start

        proba = model.predict_proba(X_valid)[:, 1]
        row = evaluate(name, y_valid, proba, seconds=seconds)
        rows.append(row)
        probabilities[name] = proba
        print(f"{name:<32} PR-AUC {row['pr_auc']:.4f}  ROC-AUC {row['roc_auc']:.4f}  "
              f"F1 {row['f1']:.4f}  ({seconds:.0f}s)")

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True), \
        probabilities, candidates


def resampled_length(sampler, y_train) -> int:
    """How many rows the learner will see after this sampler runs.

    Resampled from y alone: running fit_resample on the 772-column matrix just to
    read one integer would cost as much as the fit itself on the full dataset.
    """
    if sampler is None:
        return len(y_train)
    y = np.asarray(y_train)
    n_minority = int((y == 1).sum())
    n_majority = int((y == 0).sum())
    ratio = sampler.sampling_strategy
    if sampler.__class__.__name__ == "SMOTE":
        # ratio is the desired minority *share* of the output
        target_minority = round(ratio * n_minority / (1 - ratio))
        return n_majority + target_minority
    # RandomUnderSampler: ratio is majority rows per minority row
    target_majority = min(int(np.ceil(ratio * n_minority)), n_majority)
    return target_majority + n_minority


def undersampling_study(X_train, y_train, X_valid, y_valid, imbalance, seed=SEED):
    """Does removing majority rows beat synthesising them?

    Same algorithm (gradient boosting) and same ratio (20% minority share) in
    every row, so the only variable is the resampling direction.
    """
    variants = {
        "no resampling + class weight": xgboost(imbalance, seed),
        "over-sampling (SMOTE 20%)": xgboost(imbalance, seed),
        "under-sampling (to 20%)": xgboost(imbalance, seed),
        "under-sampling (to 50%)": xgboost(imbalance, seed),
        "under-sampling (to 100%)": xgboost(imbalance, seed),
    }
    samplers = {
        "no resampling + class weight": None,
        "over-sampling (SMOTE 20%)": smote(seed, 0.20),
        "under-sampling (to 20%)": undersample(seed, 0.20),
        "under-sampling (to 50%)": undersample(seed, 0.50),
        "under-sampling (to 100%)": undersample(seed, 1.00),
    }

    rows = []
    for name, estimator in variants.items():
        sampler = samplers[name]
        model = (Pipeline([("model", estimator)]) if sampler is None
                 else ImbPipeline([("resample", sampler), ("model", estimator)]))

        start = time.time()
        fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid)
        seconds = time.time() - start

        proba = model.predict_proba(X_valid)[:, 1]
        row = evaluate(f"XGBoost {name}", y_valid, proba, seconds=seconds)
        row["strategy"] = name
        row["train_rows_seen"] = int(resampled_length(sampler, y_train))
        rows.append(row)
        print(f"{name:<32} PR-AUC {row['pr_auc']:.4f}  F1 {row['f1']:.4f}  "
              f"rows seen {row['train_rows_seen']:>9,}  ({seconds:.0f}s)")

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def ensemble_study(probabilities, y_valid):
    """Majority voting across the three tree models, scored on validation.

    The members are already fitted with their own resampling, so we combine the
    stored validation probabilities rather than calling `VotingClassifier.fit`,
    which would refit all three (another ~20 minutes on full data) to recompute
    the same numbers. `fdm.models.build_voting` remains the canonical estimator
    and is what the unit tests exercise; this is the same arithmetic, evaluated
    without the refit.
    """
    import numpy as np

    members = [name for name in probabilities if name != "Logistic Regression + SMOTE"]
    member_probs = {name: probabilities[name] for name in members}
    # Each member votes at the threshold that maximised its own validation F1.
    thresholds = {name: evaluate(name, y_valid, probabilities[name])["threshold"]
                  for name in members}

    rows = []

    # hard voting: majority of the 0/1 decisions
    votes = sum((member_probs[name] >= thresholds[name]).astype(int)
                for name in members)
    hard = (votes >= 2).astype(float)
    row = evaluate("Majority vote (hard, 2 of 3)", y_valid, hard)
    row["voting"] = "hard"
    rows.append(row)

    # soft voting: unweighted mean of the probabilities
    soft = np.mean([member_probs[name] for name in members], axis=0)
    row = evaluate("Majority vote (soft, mean probability)", y_valid, soft)
    row["voting"] = "soft"
    rows.append(row)

    # weighted soft vote: each member counts in proportion to its own PR-AUC
    scores = {name: evaluate(name, y_valid, probabilities[name])["pr_auc"]
              for name in members}
    total = sum(scores.values())
    weights = {name: scores[name] / total for name in members}
    weighted = sum(weights[name] * member_probs[name] for name in members)
    row = evaluate("Majority vote (soft, PR-AUC weighted)", y_valid, weighted)
    row["voting"] = "soft_weighted"
    rows.append(row)

    for entry in rows:
        print(f"{entry['model']:<40} PR-AUC {entry['pr_auc']:.4f}  F1 {entry['f1']:.4f}")
    print("member weights:", {name.split()[0]: round(weight, 3)
                              for name, weight in weights.items()})

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-undersampling", action="store_true")
    args = parser.parse_args()

    banner("Stage 6 - model development and comparison")
    dataset = build_dataset(sample_frac=args.sample_frac)
    imbalance = imbalance_ratio(dataset.y_train)
    print(f"\nimbalance: {imbalance:.1f} legitimate rows per fraud row")

    banner("1. Four algorithms on validation PR-AUC")
    table, probabilities, candidates = compare_models(
        dataset.X_train, dataset.y_train, dataset.X_valid, dataset.y_valid, imbalance)
    print()
    print(table[["model", "pr_auc", "roc_auc", "precision", "recall", "f1",
                 "seconds"]].to_string(index=False))
    print(f"\nbaseline (random guessing) = validation fraud rate "
          f"{dataset.y_valid.mean():.4f}")
    table.to_csv(REPORT_DIR / "stage6_model_comparison.csv", index=False)

    under_table = None
    if not args.skip_undersampling:
        banner("2. Under-sampling vs over-sampling vs class weights")
        under_table = undersampling_study(dataset.X_train, dataset.y_train,
                                          dataset.X_valid, dataset.y_valid, imbalance)
        print()
        print(under_table[["strategy", "pr_auc", "precision", "recall", "f1",
                           "train_rows_seen", "seconds"]].to_string(index=False))
        under_table.to_csv(REPORT_DIR / "stage6_undersampling.csv", index=False)

    ens_table = None
    if not args.skip_ensemble:
        banner("3. Majority voting ensemble")
        ens_table = ensemble_study(probabilities, dataset.y_valid)
        print()
        print(ens_table[["model", "pr_auc", "precision", "recall", "f1"]].to_string(index=False))
        ens_table.to_csv(REPORT_DIR / "stage6_ensemble.csv", index=False)

    best = table.loc[0, "model"]
    print()
    print(f"winner on validation: {best} (PR-AUC {table.loc[0, 'pr_auc']:.4f}, "
          f"{lift_over_random(dataset.y_valid, probabilities[best]):.0f}x random)")

    banner("saving fitted baseline models for the API")
    dataset.preprocessor.save(ARTIFACT_DIR / "preprocessor.joblib")
    for name, model in candidates.items():
        joblib.dump(model, ARTIFACT_DIR / f"baseline_{name.split()[0].lower()}.joblib")
    print(f"wrote preprocessor.joblib and {len(candidates)} baseline models "
          f"to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
