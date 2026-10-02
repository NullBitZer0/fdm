# Credit Card Fraud Detection

Sparkov fraud dataset (Kaggle `kartik2112/fraud-detection`), 1,296,675 training and
555,719 test transactions. Covers Stage 6 (model development), Stage 7 (tuning and
final selection), Stage 9 (backend) and Stage 10 (frontend) of the assignment.

## Headline results

Validation (last 20% of training data, split on time, never shuffled):

| Model | PR-AUC | Test PR-AUC |
|---|---|---|
| **XGBoost + class weight** | **0.9150** | **0.8669** |
| LightGBM + class weight | 0.8942 | |
| Random Forest + under-sampling | 0.7742 | |
| Logistic Regression + SMOTE | 0.2781 | |

Final model: XGBoost with `scale_pos_weight=172.8` on one-hot + frequency encoding.
At the validation-derived threshold 0.9697 it reaches precision 0.8682 / recall
0.7800 / F1 0.8217 on 2,145 real fraud cases — 225x better than random guessing.

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
tests/test_contract.py       10 tests: train/serve contract + leakage guards
scripts/record_results.py    extracts the notebook's printed tables into reports/
scripts/cv_compare.py       5-fold time-series CV, tuned vs untuned
scripts/behavioural_test.py  per-card history features, with/without
scripts/render_frontend.py  drives the UI in a real browser, screenshots it
artifacts/                   fitted model + preprocessor + metadata (776-feature, count+freq)
reports/                     result tables as CSV + screenshots/
```

## Seven findings worth defending

**1. Class weighting beat both resampling directions.** Holding XGBoost fixed and
changing only the resampling strategy: no resampling 0.9150, under-sample to 20%
0.8924, under-sample to 50% 0.8781, SMOTE 0.8715, under-sample to balanced 0.8624.
Reweighting the loss keeps all 1,037,340 rows. Resampling either invents rows
(SMOTE) or discards them.

**2. Under-sampling is the wrong partner for a Random Forest at this ratio.** The
standard recipe scored 0.7742, because balancing to 50/50 kept 11,936 of 1,037,340
rows — 98.8% of the data deleted, almost all of it legitimate transactions. The
technique isn't wrong in general; it's wrong at this ratio on 1M rows.

**3. Majority voting lost to the best single model** (0.8980 weighted soft vs
0.9150). Soft beat hard decisively (0.8972 vs 0.7477), confirming that confidence
carries the signal at a 0.58% fraud rate. But two of the three members are weaker
models making correlated errors, so averaging them in pulled the score down. The
ensemble is implemented and measured but not selected. Notably its hard variant had
the best F1 in the notebook (0.8625) while ranking worst on PR-AUC by 0.15.

**4. Frequency encoding helped (+0.0053); target encoding hurt (−0.0071).**
Frequency encoding answers "how common is this merchant" — a different question
from one-hot's "which merchant" — using no label, so it cannot leak. It cost 4
columns against 760 existing ones and `category_freq` now ranks 6th in the feature
importances. Target encoding answers "how risky", which is the stronger signal, but
it is built from the label: the naive in-fold version scored 0.7565 AUC against the
label on training rows versus 0.7327 out-of-fold. Built correctly it still lost,
because the trees can already reconstruct merchant identity from the one-hot
dummies. It is implemented and tested but not enabled.

**5. Small-sample gains do not survive more data — seen three times.**
Tree regularisation +0.0282 at 3% became +0.0031 at 30%. Feature selection +0.0048
at 2% became +0.0028 at 30%. Hyperparameter tuning +0.0066 at 10% became −0.0106
when refitted on 100%. Nothing here was adopted on a gain measured on a subsample,
which is why three of the four results above are negative.

**The shuffle test.** The claim that a chronological split stops the model
memorising identities is now measured, not asserted: identical rows, identical
model, one split on time and one shuffled. The shuffled split scores +0.0069 higher
at 3% of the data but +0.0329 higher at 30% — the inflation *grows* with data,
because more rows means more identities recurring across the boundary.

**6. Most of the gaps in this project are inside the noise.** Five-fold
TimeSeriesSplit gives a standard deviation of **0.0233** PR-AUC across time periods.
Against that, only two comparisons are comfortably real: XGBoost over Random Forest
(0.1408) and the ensemble losing to its best member. XGBoost over LightGBM (0.0208),
the resampling result (0.0226) and the encoding gain (0.0053) are all inside the band
and should be read as "no measurable difference". Nothing in the final model choice
depends on those margins — which is the point of measuring them.

**Cross-validation says tuning and not-tuning are indistinguishable.** 5-fold
`TimeSeriesSplit` at both 30% and 100% of the data:

| Run | Untuned | Tuned | Difference | t (df=4) |
|---|---|---|---|---|
| 30% | 0.8819 ± 0.0233 | 0.8816 ± 0.0211 | −0.0003 | −0.06 |
| 100% | 0.8827 ± 0.0211 | 0.8851 ± 0.0213 | +0.0024 | +0.98 |

Both within one standard error of zero, and the sign flips when you add data — the
signature of noise, not effect. Three evaluations (full-data refit −0.0106, 30% CV
−0.0003, 100% CV +0.0024) gave three answers, all inside the noise band. The single
holdout's 0.9150 is ~1.4 standard deviations above the CV mean, so the honest
estimate for an arbitrary future period is **~0.883**.

**7. Behavioural features were built and rejected.** Every feature above is
row-level, so we added per-card history: transactions in the last hour, amount
against that card's own median, distance from the card's historical centre. Built
leak-free (sorted by time, each row scored only on its strictly-earlier rows of the
same card) and verified with three leak checks. They carry real univariate signal —
and two AUCs below 0.5, showing fraud happens on *quiet* cards and *closer to home*
than normal, the opposite of the usual intuition. But +0.0128 on one holdout became
+0.0039 across five folds (t=0.97), and adopting them would have forced the API to
stop accepting a single self-contained transaction. Rejected; kept in the code
behind `build_dataset(behavioural=True)`.

**Four improvement attempts, all built, all measured, all rejected.** Target
encoding, feature selection, hyperparameter tuning, behavioural features. Each looked
real on one split and did not survive proper measurement. The error is dominated by
period and label noise, not by model configuration.

## Running it

```bash
pip install -r requirements.txt

# Stage 6 — four models, the resampling study, and the ensemble  (~45 min full data)
python scripts/train_models.py

# Stage 7 — random search, then refit the winner on the full dataset
python scripts/tune_random.py --trials 12 --sample-frac 0.1   # ~25 min
python scripts/refit_final.py                                 # ~7 min

# Feature selection and the shuffle test
python scripts/feature_study.py --sample-frac 0.3             # ~20 min

# 5-fold time-series CV: tuned vs untuned, plus the variance estimate
python scripts/cv_compare.py --sample-frac 0.3 --folds 5      # ~8 min
python scripts/cv_compare.py --sample-frac 1.0 --folds 5      # ~25 min

# Behavioural (per-card history) features, with and without
python scripts/behavioural_test.py --sample-frac 0.3           # ~20 min

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
python tests/test_contract.py      # no pytest needed, 10/10
python scripts/record_results.py   # notebook tables -> reports/*.csv
python scripts/render_frontend.py  # headless browser check + screenshots
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

**Why target encoding is not enabled, even though it is implemented.** It is the
only feature here computed from the label. The naive version leaks: fitted and
transformed on the same rows, the in-fold encoding scored 0.7565 AUC against the
label on training data versus 0.7327 out-of-fold. The out-of-fold construction
closes that, and two tests in `tests/test_contract.py` pin it. It still lost on
score (0.9079 vs 0.9150 for frequency alone), because the trees can already
reconstruct merchant identity from the one-hot dummies. It also collapses to a
global constant for any merchant unseen in training, where one-hot at least reads
as "unknown". Taking leakage risk for a negative gain is the wrong trade, so
`count+freq` is the default and the target encoder is kept, exercised and tested
but off.

**Why CatBoost is gone.** It was fed one-hot encoded sparse input with no
`cat_features` argument, so its categorical handling — the reason to pick it — was
never exercised. It was also 158s slower than XGBoost for a worse score. Random
Forest replaced it to reach four algorithms with genuinely different inductive
biases.
