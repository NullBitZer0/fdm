"""Do per-card behavioural features improve the model?

Everything else in this project is row-level. This tests whether adding features
derived from a card's own transaction HISTORY helps, now that they are built
leak-free: each row is scored using only strictly-earlier rows of the same card.

The features (see fdm.features.add_behavioural_features):
  card_txn_1h / card_txn_24h   how busy the card has been
  card_amt_ratio / card_amt_dev how far this amount is from the card's own habit
  card_hours_since_last        dormancy before this transaction
  card_dist_from_home_km       distance from the card's median past location
  card_category_seen           has this card used this category before
  card_is_new                  a card with no history at all

Run:  python scripts/behavioural_test.py [--sample-frac 0.3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xgboost as xgb

from fdm.config import REPORT_DIR, SAMPLE_FRAC, SEED
from fdm.data import build_dataset
from fdm.metrics import evaluate
from fdm.models import imbalance_ratio

BASE = dict(n_estimators=600, learning_rate=0.05, max_depth=8, tree_method="hist",
            eval_metric="aucpr", early_stopping_rounds=50, random_state=SEED, n_jobs=-1)


def fit(dataset):
    imbalance = imbalance_ratio(dataset.y_train)
    model = xgb.XGBClassifier(**BASE, scale_pos_weight=imbalance)
    model.fit(dataset.X_train, dataset.y_train,
              eval_set=[(dataset.X_valid, dataset.y_valid)], verbose=False)
    proba = model.predict_proba(dataset.X_valid)[:, 1]
    return evaluate("model", dataset.y_valid, proba), model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    args = parser.parse_args()

    print("=" * 78)
    print("Behavioural features: do a card's own history help?")
    print("=" * 78)

    report = {"sample_frac": args.sample_frac}
    results = {}
    datasets = {}

    for label, flag in (("row-level only (current model)", False),
                        ("+ behavioural history", True)):
        print(f"\nbuilding: {label}", flush=True)
        start = time.time()
        dataset = build_dataset(sample_frac=args.sample_frac, behavioural=flag, verbose=False)
        datasets[label] = dataset
        row, model = fit(dataset)
        row["seconds"] = round(time.time() - start, 1)
        row["features"] = int(dataset.X_train.shape[1])
        row["train_rows"] = int(dataset.X_train.shape[0])
        results[label] = row
        print(f"  {dataset.X_train.shape[1]} features | "
              f"validation PR-AUC {row['pr_auc']:.4f} | {row['seconds']:.0f}s")

    print()
    print("=" * 78)
    print("comparison")
    print("=" * 78)
    table = pd.DataFrame([
        {"featureset": label,
         "n_features": row["features"],
         "pr_auc": row["pr_auc"], "roc_auc": row["roc_auc"],
         "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
         "seconds": row["seconds"]}
        for label, row in results.items()])
    base_label = "row-level only (current model)"
    table["gain"] = (table["pr_auc"] - results[base_label]["pr_auc"]).round(4)
    print(table.to_string(index=False))

    base = datasets[base_label]
    enhanced = datasets["+ behavioural history"]
    base_model = fit(base)[1]
    enhanced_model = fit(enhanced)[1]

    importance = pd.Series(enhanced_model.feature_importances_.ravel(),
                           index=enhanced.preprocessor.feature_names)
    behavioural_names = [n for n in importance.index if n.startswith("card_")]
    print()
    print(f"behavioural features occupy {sum(importance[n] for n in behavioural_names):.1%} "
          f"of total importance across {len(behavioural_names)} columns")
    print()
    print("behavioural feature importances:")
    for name in sorted(behavioural_names, key=lambda n: -importance[n]):
        print(f"  {name:<28} {importance[name]:.4f}")
    print()
    print("top 10 features overall:")
    for name in importance.sort_values(ascending=False).head(10).index:
        tag = "  <- behavioural" if name.startswith("card_") else ""
        print(f"  {name:<32} {importance[name]:.4f}{tag}")

    gain = float(table.loc[1, "gain"])
    table.to_csv(REPORT_DIR / "behavioural_features.csv", index=False)
    importance.sort_values(ascending=False).to_csv(
        REPORT_DIR / "behavioural_importances.csv")
    report.update({
        "baseline_pr_auc": results[base_label]["pr_auc"],
        "behavioural_pr_auc": results["+ behavioural history"]["pr_auc"],
        "gain": round(gain, 4),
        "n_features_added": int(table.loc[1, "n_features"] - table.loc[0, "n_features"]),
        "behavioural_share_of_importance": round(
            float(sum(importance[n] for n in behavioural_names)), 4),
        "decision": ("adopt" if gain > 0.01 else
                     "not adopted - gain is inside the 0.021 noise band"),
    })
    (REPORT_DIR / "behavioural_study.json").write_text(json.dumps(report, indent=2))

    print()
    if gain > 0.01:
        print(f"DECISION: adopt. +{gain:.4f} clears the 0.021 noise band.")
        print("This needs a serving change: the API can no longer score a single")
        print("self-contained transaction, it needs that card's recent history.")
    else:
        print(f"DECISION: not adopted. +{gain:.4f} is inside the 0.021 noise band, so the")
        print("extra serving complexity would buy nothing measurable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
