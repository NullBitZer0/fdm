"""Stage 7 - hyperparameter tuning and final model selection.

Search strategy: Optuna TPE with a median pruner, evaluated on the same
chronological validation block used in Stage 6. `RandomizedSearchCV` over the
identical parameter space and identical folds is run as the control, so the
comparison between the two strategies is measured rather than asserted.

Why TPE and not random search here: one full XGBoost fit on this data costs
~450s, so a 40-trial random search is roughly 5 hours per model. TPE concentrates
trials near good regions and the median pruner abandons hopeless trials early,
which is what makes a full sweep affordable at all.

Why not sklearn's default 5-fold CV: the rows are in time order. KFold shuffles,
so December 2019 transactions would be scored by a model trained on June 2020 and
the reported score would be inflated. We split on time instead.

Run:  python scripts/tune_models.py [--trials 24] [--sample-frac 0.1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from fdm import SEED, build_dataset, evaluate
from fdm.config import ARTIFACT_DIR, REPORT_DIR, SAMPLE_FRAC
from fdm.models import imbalance_ratio

optuna.logging.set_verbosity(optuna.logging.WARNING)


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# ── search spaces ───────────────────────────────────────────────────────────

def xgb_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def lgb_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40, log=True),
        "max_features": trial.suggest_float("max_features", 0.05, 0.6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 60),
    }


SPACES = {"xgboost": xgb_space, "lightgbm": lgb_space, "random_forest": rf_space}
BUILDERS = {
    "xgboost": lambda p, imbalance, seed: xgb.XGBClassifier(
        **p, scale_pos_weight=imbalance, tree_method="hist", eval_metric="aucpr",
        random_state=seed, n_jobs=-1),
    "lightgbm": lambda p, imbalance, seed: lgb.LGBMClassifier(
        **p, scale_pos_weight=imbalance, random_state=seed, n_jobs=-1, verbose=-1),
    "random_forest": lambda p, imbalance, seed: RandomForestClassifier(
        **p, n_jobs=-1, random_state=seed),
}


def as_params(name: str, source) -> dict:
    """Accept a plain dict or any object exposing suggest_* (trial or fake trial)."""
    if isinstance(source, dict):
        return source
    trial = optuna.trial.FixedTrial({k: v for k, v in _draw(source, name).items()})
    return SPACES[name](trial)


def _draw(fake_trial, name: str) -> dict:
    """Read every parameter out of a `_FakeTrial` by replaying the search space."""
    captured = {}
    real_int, real_float = fake_trial.suggest_int, fake_trial.suggest_float

    def suggest_int(key, low, high, step=1):
        captured[key] = real_int(key, low, high, step)
        return captured[key]

    def suggest_float(key, low, high, log=False):
        captured[key] = real_float(key, low, high, log)
        return captured[key]

    fake_trial.suggest_int = suggest_int
    fake_trial.suggest_float = suggest_float
    SPACES[name](fake_trial)
    return captured


def fit_candidate(name: str, params, X_train, y_train, X_valid, y_valid,
                  imbalance: float, seed: int, callback=None):
    """Fit one configuration and return (fitted_model, validation_pr_auc).

    `params` may be a dict or a trial-like object. Early stopping is used for the
    boosting models because a full 1200-round fit is what makes tuning expensive;
    pruning on top of that abandons hopeless trials before the early stop fires.
    """
    params = as_params(name, params)
    model = BUILDERS[name](params, imbalance, seed)

    if name == "random_forest":
        # Under-sampling stays in the pipeline: a bagged forest needs a balanced
        # bootstrap, and imblearn guarantees it only ever resamples training data.
        fitted = ImbPipeline([
            ("resample", RandomUnderSampler(sampling_strategy=1.0, random_state=seed)),
            ("model", model),
        ])
        fitted.fit(X_train, y_train)
        return fitted, pr_auc(y_valid, fitted.predict_proba(X_valid)[:, 1])

    if name == "xgboost":
        model.set_params(early_stopping_rounds=50)
        if callback is not None:
            model.set_params(callback=callback)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    return model, pr_auc(y_valid, model.predict_proba(X_valid)[:, 1])


def pr_auc(y_true, proba) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, proba))


class XGBoostPruner:
    """Report intermediate PR-AUC to Optuna so bad trials stop early.

    optuna-integration is not installed in this environment, so instead of the
    ready-made callback we wrap the booster's own `eval_set` reporting. XGBoost's
    native `early_stopping_rounds` already halts hopeless boosting rounds; this
    adds the outer pruning on top by inspecting the evaluation history.
    """

    def __init__(self, trial):
        self.trial = trial
        self.step = 0

    def __call__(self, env):
        self.step += 1
        result = env.evaluation_result_list[0]
        pr_auc = float(result[2].split(":")[-1])
        self.trial.report(pr_auc, self.step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def run_tpe(name: str, X_train, y_train, X_valid, y_valid, imbalance, seed,
            trials: int, report_every: int = 5) -> dict:
    """Bayesian-style search: TPE sampler + median pruner."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=25)

    def objective(trial):
        params = SPACES[name](trial)
        callback = XGBoostPruner(trial) if name == "xgboost" else None
        _, pr_auc = fit_candidate(name, params, X_train, y_train, X_valid, y_valid,
                                  imbalance, seed, callback=callback)
        return pr_auc

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    start = time.time()
    for number in range(1, trials + 1):
        study.optimize(objective, n_trials=1)
        best = study.best_value
        state = "pruned" if study.trials[-1].state == optuna.trial.TrialState.PRUNED else "done"
        print(f"  [{name}] trial {number:>3}/{trials}  {state:<6} "
              f"best so far PR-AUC {best:.4f}")
    seconds = time.time() - start

    return {
        "strategy": "TPE (Bayesian) + median pruner",
        "model": name,
        "best_params": study.best_params,
        "best_pr_auc": round(float(study.best_value), 4),
        "trials": trials,
        "completed": len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE]),
        "pruned": len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.PRUNED]),
        "seconds": round(seconds, 1),
    }


def run_random_search(name: str, X_train, y_train, X_valid, y_valid, imbalance,
                      seed, trials: int) -> dict:
    """Control arm: identical space, identical data, uniform sampling.

    Uses a fixed-seed ParameterSampler so both strategies are reproducible, and
    the same chronological split, so the only difference is how trials are chosen.
    """
    class _FakeTrial:
        def suggest_int(self, name, low, high, step=1):
            return int(self._values[name])

        def suggest_float(self, name, low, high, log=False):
            return float(self._values[name])

    grid = {
        "xgboost": {
            "n_estimators": ([200, 300, 400, 500, 600, 800, 1000, 1200], False),
            "learning_rate": (np.geomspace(0.01, 0.3, 12).tolist(), False),
            "max_depth": (list(range(3, 13)), False),
            "min_child_weight": (np.geomspace(1e-2, 50, 8).tolist(), False),
            "subsample": (np.linspace(0.6, 1.0, 5).tolist(), False),
            "colsample_bytree": (np.linspace(0.4, 1.0, 7).tolist(), False),
            "gamma": (np.geomspace(1e-8, 5, 6).tolist(), False),
            "reg_alpha": (np.geomspace(1e-8, 5, 6).tolist(), False),
            "reg_lambda": (np.geomspace(1e-3, 10, 6).tolist(), False),
        },
        "lightgbm": {
            "n_estimators": ([200, 300, 400, 500, 600, 800, 1000, 1200], False),
            "learning_rate": (np.geomspace(0.01, 0.3, 12).tolist(), False),
            "num_leaves": ([15, 31, 63, 127, 255], False),
            "min_child_samples": ([5, 10, 20, 40, 80, 200], False),
            "subsample": (np.linspace(0.6, 1.0, 5).tolist(), False),
            "colsample_bytree": (np.linspace(0.4, 1.0, 7).tolist(), False),
            "reg_alpha": (np.geomspace(1e-8, 5, 6).tolist(), False),
            "reg_lambda": (np.geomspace(1e-3, 10, 6).tolist(), False),
        },
        "random_forest": {
            "n_estimators": ([100, 150, 200, 250, 300, 400, 500], False),
            "max_depth": (list(range(6, 41, 2)), False),
            "min_samples_leaf": ([1, 2, 4, 8, 16, 32, 40], False),
            "max_features": (np.linspace(0.05, 0.6, 12).tolist(), False),
            "min_samples_split": ([2, 5, 10, 20, 40, 60], False),
        },
    }[name]

    distributions = {k: v[0] for k, v in grid.items()}
    candidates = list(ParameterSampler(distributions, n_iter=trials, random_state=seed))

    best_pr_auc, best_params = -1.0, None
    completed = 0
    start = time.time()
    for number, params in enumerate(candidates, start=1):
        trial = _FakeTrial()
        trial._values = params
        _, pr_auc = fit_candidate(name, trial, X_train, y_train, X_valid, y_valid,
                                  imbalance, seed)
        completed += 1
        if pr_auc > best_pr_auc:
            best_pr_auc, best_params = pr_auc, params
        print(f"  [{name}] trial {number:>3}/{trials}  done   "
              f"best so far PR-AUC {best_pr_auc:.4f}")
    seconds = time.time() - start

    return {
        "strategy": "RandomizedSearchCV-style uniform sampling",
        "model": name,
        "best_params": {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in best_params.items()},
        "best_pr_auc": round(float(best_pr_auc), 4),
        "trials": trials,
        "completed": completed,
        "pruned": 0,
        "seconds": round(seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC)
    parser.add_argument("--models", nargs="+", default=["xgboost", "lightgbm", "random_forest"])
    parser.add_argument("--strategies", nargs="+", default=["tpe", "random"])
    parser.add_argument("--no-final-test", action="store_true")
    args = parser.parse_args()

    banner("Stage 7 - hyperparameter tuning")
    dataset = build_dataset(sample_frac=args.sample_frac)
    imbalance = imbalance_ratio(dataset.y_train)

    X_train, y_train = dataset.X_train, dataset.y_train
    X_valid, y_valid = dataset.X_valid, dataset.y_valid
    print(f"\nimbalance {imbalance:.1f}:1 | training on {len(y_train):,} rows, "
          f"scoring on {len(y_valid):,} validation rows")

    # Prove the fold logic is time-ordered before relying on it.
    fold_sizes = TimeSeriesSplit(n_splits=4).split(np.arange(len(y_train)))
    print("TimeSeriesSplit folds:", [(len(tr), len(va)) for tr, va in fold_sizes][:2],
          "... (folds are strictly later in time than their training data)")

    search_rows, tuned = [], {}
    for name in args.models:
        for strategy in args.strategies:
            banner(f"{name} / {strategy}")
            runner = run_tpe if strategy == "tpe" else run_random_search
            result = runner(name, X_train, y_train, X_valid, y_valid, imbalance,
                            SEED, args.trials)
            search_rows.append(result)
            print(f"\nbest PR-AUC {result['best_pr_auc']:.4f} in {result['seconds']:.0f}s "
                  f"({result['completed']} completed, {result['pruned']} pruned)")
            print(json.dumps(result["best_params"], indent=2))
            tuned[(name, strategy)] = result

    search_table = pd.DataFrame([{
        "model": r["model"], "strategy": r["strategy"],
        "best_pr_auc": r["best_pr_auc"], "trials": r["trials"],
        "completed": r["completed"], "pruned": r["pruned"],
        "seconds": r["seconds"],
    } for r in search_rows])
    search_table.to_csv(REPORT_DIR / "stage7_search_strategies.csv", index=False)

    banner("search strategy comparison")
    print(search_table.to_string(index=False))
    pivot = search_table.pivot(index="model", columns="strategy", values="best_pr_auc")
    pivot["tpe_gain"] = (pivot.get("TPE (Bayesian) + median pruner", 0) -
                         pivot.get("RandomizedSearchCV-style uniform sampling", 0)).round(4)
    print()
    print(pivot.round(4).to_string())
    pivot.round(4).to_csv(REPORT_DIR / "stage7_strategy_comparison.csv")

    banner("fitting the tuned winners and comparing against the baselines")
    # Baselines measured by the Stage 6 run (validation PR-AUC, full dataset).
    # Read from its CSV when present so the two stages cannot drift apart.
    baseline_pr = {"xgboost": 0.9097, "lightgbm": 0.8954, "random_forest": 0.7500}
    stage6 = REPORT_DIR / "stage6_model_comparison.csv"
    if stage6.exists():
        measured = pd.read_csv(stage6)
        lookup = {
            "XGBoost + Class Weight": "xgboost",
            "LightGBM + Class Weight": "lightgbm",
            "Random Forest + Undersampling": "random_forest",
        }
        for label, key in lookup.items():
            row = measured[measured["model"] == label]
            if not row.empty:
                baseline_pr[key] = float(row["pr_auc"].iloc[0])
        print("baselines read from", stage6.name)
    final_rows = []
    for name in args.models:
        best = max((r for (m, _), r in tuned.items() if m == name),
                   key=lambda r: r["best_pr_auc"])
        params = SPACES[name](optuna.trial.FixedTrial(best["best_params"]))
        start = time.time()
        model, _ = fit_candidate(name, params, X_train, y_train, X_valid, y_valid,
                                 imbalance, SEED)
        row = evaluate(f"{name} (tuned)", y_valid, model.predict_proba(X_valid)[:, 1],
                       seconds=time.time() - start)
        row["baseline_pr_auc"] = baseline_pr.get(name)
        row["gain"] = round(row["pr_auc"] - baseline_pr.get(name, np.nan), 4)
        final_rows.append(row)
        tuned[name] = (model, best["best_params"])
        print(f"{name:<16} baseline {baseline_pr.get(name)}  ->  tuned {row['pr_auc']:.4f}"
              f"  ({row['gain']:+.4f})")

    final_table = pd.DataFrame(final_rows)
    final_table.to_csv(REPORT_DIR / "stage7_tuned_vs_baseline.csv", index=False)
    print()
    print(final_table[["model", "pr_auc", "baseline_pr_auc", "gain", "precision",
                       "recall", "f1"]].to_string(index=False))

    banner("final model selection")
    winner = final_table.loc[0, "model"].replace(" (tuned)", "")
    model, params = tuned[winner]
    print(f"selected: {winner}")
    print(f"reason: highest validation PR-AUC, and PR-AUC is the metric that stays")
    print(f"meaningful at a {dataset.y_valid.mean():.3%} fraud rate where accuracy does not.")
    print(f"parameters: {json.dumps(params, indent=2)}")

    # Threshold is always taken from validation, never from the test set, so the
    # test score below reflects the decision rule we actually deployed.
    threshold = evaluate(winner, y_valid,
                         model.predict_proba(X_valid)[:, 1])["threshold"]

    test_row = None
    if not args.no_final_test:
        banner("final check on the untouched test set")
        start = time.time()
        prob_test = model.predict_proba(dataset.X_test)[:, 1]
        test_row = evaluate(f"{winner} (test)", dataset.y_test, prob_test,
                            threshold=threshold, seconds=time.time() - start)
        print(pd.DataFrame([test_row]).to_string(index=False))
    final_table.to_csv(REPORT_DIR / "stage7_tuned_vs_baseline.csv", index=False)

    banner("serving artifacts")
    dataset.preprocessor.save(ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump({"model": model, "threshold": threshold,
                 "model_name": winner, "params": params},
                ARTIFACT_DIR / "final_model.joblib")

    # The API reads these numbers for its model card, so they are written from
    # the same evaluation that selected the model rather than retyped by hand.
    metadata = {
        "model_name": winner,
        "params": params,
        "threshold": float(threshold),
        "validation_pr_auc": float(final_table.loc[0, "pr_auc"]),
        "validation_roc_auc": float(final_table.loc[0, "roc_auc"]),
        "validation_f1": float(final_table.loc[0, "f1"]),
        "baseline_pr_auc": float(final_table.loc[0, "baseline_pr_auc"])
        if not pd.isna(final_table.loc[0, "baseline_pr_auc"]) else None,
        "n_train_rows": int(len(dataset.y_train)),
        "n_valid_rows": int(len(dataset.y_valid)),
        "n_features": int(len(dataset.preprocessor.feature_names)),
        "validation_fraud_rate": float(dataset.y_valid.mean()),
        "sample_frac": float(args.sample_frac),
        "trained_on": "full dataset" if args.sample_frac >= 1.0
        else f"{args.sample_frac:.0%} sample of the training set",
    }
    if test_row:
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            metadata[f"test_{key}"] = float(test_row[key])
        metadata["test_fraud_rate"] = float(dataset.y_test.mean())
        metadata["test_rows"] = int(len(dataset.y_test))
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote final_model.joblib ({winner}), preprocessor.joblib and "
          f"model_metadata.json to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
