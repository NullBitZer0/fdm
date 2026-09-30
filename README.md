# Credit Card Fraud Detection

Sparkov fraud dataset (Kaggle `kartik2112/fraud-detection`), 1,296,675 training and
555,719 test transactions. Covers Stage 6 (model development), Stage 7 (tuning and
final selection), Stage 9 (backend) and Stage 10 (frontend) of the assignment.

## Headline results

Validation (last 20% of training data, split on time, never shuffled):

| Model | PR-AUC | Test PR-AUC |
|---|---|---|
| **XGBoost + class weight** | **0.9097** | **0.8593** |
| LightGBM + class weight | 0.8954 | |
| Random Forest + under-sampling | 0.7500 | |
| Logistic Regression + SMOTE | 0.2780 | |

Final model: XGBoost with `scale_pos_weight=172.8`, at the validation-derived
threshold 0.9704 it reaches precision 0.8688 / recall 0.7814 / F1 0.8228 on 2,145
real fraud cases — 223x better than random guessing.

## Layout

```
fraud_detection_viva.ipynb   the full analysis, EDA through final test score
src/fdm/                     the library both the scripts and the API import
  config.py                  paths, column groups, the train/serve contract
  features.py                clean(), add_features(), chronological_split()
  preprocess.py              Preprocessor: scaler + encoder, saved to disk
  metrics.py                 PR-AUC, best-F1 threshold, confusion counts
  models.py                  model builders, imbalance strategies, voting
  data.py                    build_dataset(): the whole pipeline in one call
scripts/train_models.py      Stage 6
scripts/tune_models.py       Stage 7
backend/app/                 Stage 9 (FastAPI)
frontend/                    Stage 10 (React + TypeScript + Vite)
tests/test_contract.py       8 tests pinning the train/serve contract
artifacts/                   fitted model + preprocessor + metadata
reports/                     result tables as CSV
```

## The three findings worth defending

**1. Class weighting beat both resampling directions.** Holding XGBoost fixed and
changing only the resampling strategy: no resampling 0.9097, under-sample to 50%
0.8950, under-sample to 20% 0.8939, SMOTE 0.8689, under-sample to balanced 0.8672.
Reweighting the loss keeps all 1,037,340 rows. Resampling either invents rows
(SMOTE) or discards them.

**2. Under-sampling is the wrong partner for a Random Forest at this ratio.** The
standard recipe scored 0.7500, because balancing to 50/50 kept 11,936 of 1,037,340
rows — 98.9% of the data deleted, almost all of it legitimate transactions. The
technique isn't wrong in general; it's wrong at this ratio on 1M rows.

**3. Majority voting lost to the best single model** (0.8999 weighted soft vs
0.9097). Soft beat hard decisively (0.8989 vs 0.7411), confirming that
confidence carries the signal at a 0.58% fraud rate. But two of the three members
are weaker models making correlated errors, so averaging them in pulled the score
down. The ensemble is implemented and measured but not selected.

## Running it

```bash
pip install -r requirements.txt

# Stage 6 — four models, the resampling study, and the ensemble  (~45 min full data)
python scripts/train_models.py

# Stage 7 — TPE tuning vs random search, then the final test score  (long)
python scripts/tune_models.py --trials 24

# Stage 9 — API on :8002
bash backend/start_dev.sh          # interactive docs at /docs

# Stage 10 — frontend on :5173, proxying to the API
cd frontend && npm install && npm run dev
```

Both scripts take `--sample-frac`. A fraction gives a runnable demo in a few
minutes; the default is `1.0`, the full dataset, which is what the numbers above
come from. The notebook is already executed on full data, so it is the
authoritative record of Stage 6 — the scripts exist to produce the saved artifacts
and the Stage 7 tables.

```bash
python tests/test_contract.py      # no pytest needed
```

## Design decisions

**Why PR-AUC and not accuracy.** Always predicting "legit" scores 99.42% accuracy,
so accuracy measures nothing here. PR-AUC is threshold-free, focuses on the fraud
class, and has a known baseline (the fraud rate), which is what makes it
comparable across the 0.593% validation and 0.386% test periods.

**Why the split is on time.** The data is chronologically ordered, and the Kaggle
train/test boundary is itself a time boundary. A random split would put the same
card and merchant on both sides and let the model memorise identities. The same
reasoning rules out `KFold` for the Stage 7 search — `TimeSeriesSplit` is used
instead.

**Why the API reuses the training code.** `backend/app/model_service.py` imports
`add_features` and `Preprocessor` from `src/fdm` rather than reimplementing them,
so a live request cannot drift from a training row. `tests/test_contract.py`
asserts exactly that by comparing the two paths feature by feature.

**Why CatBoost is gone.** It was fed one-hot encoded sparse input with no
`cat_features` argument, so its categorical handling — the reason to pick it — was
never exercised. It was also 158s slower than XGBoost for a worse score. Random
Forest replaced it to reach four algorithms with genuinely different inductive
biases.
