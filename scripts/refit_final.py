"""Refit the tuned hyperparameters on the full training set, then score on test.

The search in scripts/tune_random.py ran on a 10% chronological sample because a
full-data fit costs ~450s and 12 of them is 90 minutes. A tuned configuration found
on a sample is still a good configuration; it just has to be refitted on all the data
before it is served. Skipping that step means shipping a model trained on 10% of the
rows, which is worse on the test set than the untuned full-data model.

Run:  python scripts/refit_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xgboost as xgb

from fdm.config import ARTIFACT_DIR, REPORT_DIR
from fdm.data import build_dataset
from fdm.metrics import evaluate
from fdm.models import imbalance_ratio

SOURCE = ARTIFACT_DIR / "final_model.joblib"


def main() -> int:
    params = json.loads(SOURCE.read_text())["params"] if SOURCE.suffix == ".json" \
        else joblib.load(SOURCE)["params"]
    print("tuned parameters found by random search:")
    for key in sorted(params):
        print(f"  {key:<18} {params[key]:.6g}")

    dataset = build_dataset(sample_frac=1.0)
    imbalance = imbalance_ratio(dataset.y_train)
    print(f"\nrefitting on {len(dataset.y_train):,} rows "
          f"({dataset.X_train.shape[1]} features, imbalance {imbalance:.1f}:1)")

    model = xgb.XGBClassifier(**params, scale_pos_weight=imbalance,
                              tree_method="hist", eval_metric="aucpr",
                              early_stopping_rounds=50, random_state=42, n_jobs=-1)
    model.fit(dataset.X_train, dataset.y_train,
              eval_set=[(dataset.X_valid, dataset.y_valid)], verbose=False)

    prob_valid = model.predict_proba(dataset.X_valid)[:, 1]
    valid_row = evaluate("refit on full data (validation)", dataset.y_valid, prob_valid)
    threshold = valid_row["threshold"]

    prob_test = model.predict_proba(dataset.X_test)[:, 1]
    test_row = evaluate("refit on full data (test)", dataset.y_test, prob_test,
                        threshold=threshold)

    print()
    print(pd.DataFrame([valid_row, test_row])[
        ["model", "pr_auc", "roc_auc", "threshold", "precision", "recall", "f1",
         "tp", "fp", "fn"]].to_string(index=False))

    previous = pd.read_csv(REPORT_DIR / "stage7_tuned_vs_baseline.csv") \
        if (REPORT_DIR / "stage7_tuned_vs_baseline.csv").exists() else pd.DataFrame()
    print()
    print("The 10% sample fit scored 0.8482 on test. The full-data refit above is the")
    print("one that ships: same tuned parameters, all the rows.")

    dataset.preprocessor.save(ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump({"model": model, "threshold": threshold,
                 "model_name": "XGBoost + Class Weight (tuned, full data)",
                 "params": params,
                 "encoding_scheme": dataset.preprocessor.scheme,
                 "n_features": len(dataset.preprocessor.feature_names),
                 "validation_pr_auc": float(valid_row["pr_auc"])},
                ARTIFACT_DIR / "final_model.joblib")

    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({
        "model_name": "XGBoost + Class Weight (tuned, full data)",
        "params": params,
        "threshold": float(threshold),
        "validation_pr_auc": float(valid_row["pr_auc"]),
        "validation_roc_auc": float(valid_row["roc_auc"]),
        "validation_f1": float(valid_row["f1"]),
        "n_features": int(len(dataset.preprocessor.feature_names)),
        "n_train_rows": int(len(dataset.y_train)),
        "n_valid_rows": int(len(dataset.y_valid)),
        "encoding_scheme": dataset.preprocessor.scheme,
        "trained_on": ("full dataset, refitted with hyperparameters found by random "
                       "search on a 10% chronological sample"),
        "search": metadata.get("search"),
    })
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        metadata[f"test_{key}"] = float(test_row[key])
    metadata_path.write_text(json.dumps(metadata, indent=2))

    pd.DataFrame([valid_row, test_row]).to_csv(
        REPORT_DIR / "stage7_final_refit.csv", index=False)
    print()
    print(f"saved final_model.joblib and model_metadata.json "
          f"(validation {valid_row['pr_auc']:.4f}, test {test_row['pr_auc']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
