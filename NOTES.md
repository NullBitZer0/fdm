# Progress notes — resume checklist

Superseded by "State as of 1:20 on 1 Oct 2026" below, which is the current
situation. The early sections are kept for the record.

## Running right now

| Process | Status |
|---|---|
| Full notebook execution (`nbconvert`) | **running**, ~40 min in, cell 23 of 25 |
| FastAPI on port 8002 | **running and verified**, `bash backend/start_dev.sh` |

The notebook is on the full 1.29M-row dataset, so it takes ~45 min. It writes
back into `fraud_detection_viva.ipynb` with `--inplace`, so it is only safe to
kill after it finishes. Check progress with:

```bash
python -c "import json; nb=json.load(open('fraud_detection_viva.ipynb')); \
  ex=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']; \
  print(sum(1 for e in ex if e), '/', len(ex))"

## State as of 1:20 on 1 Oct 2026

### Running

- `scripts/train_models.py` on full data, regenerating the Stage 6 CSVs and the
  serving artifacts. Started after fixing three real bugs the smoke test missed:
  boosters raising on `early_stopping_rounds` without an `eval_set`, LightGBM
  rejecting a `verbose` fit kwarg, and a `Pipeline.transform` call that does not
  exist on a resampling pipeline.

### Frontend — dependency blocker RESOLVED

`npm install` completed once the network came back, with vite 8.1.0 and
`@vitejs/plugin-react` 6.0.3 pinned in `package.json`. All three checks pass:

- `npx tsc -b --noEmit` — clean (two real type errors fixed: a bad import path and
  a `datetime-local` value outside the declared union)
- `npx vite build` — 19 modules, 205 kB / 64.7 kB gzipped
- `npx vite` on :5173, proxying to the API

### Verified against the live API

| Check | Result |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true,"model_name":"xgboost"}` |
| `GET /api/model` | 771 features, threshold 0.9634 |
| `GET /api/categories` | 14 categories, 2 genders, 50 states, 693 merchants |
| `POST /api/predict` night, $1024, gas | 0.9950 → `fraud`, high risk |
| `POST /api/predict` daytime, $12, dining | 0.0013 → `legitimate`, low risk |

Rejected with 422 and a field-specific message: negative amount, gender `X`,
latitude 999, three-letter state, future DOB, future transaction time, missing
fields.

### Tests

`python tests/test_contract.py` — 8/8. These pin the train/serve contract, the one
thing that fails silently. `test_api_and_training_produce_identical_features`
compares the API's feature path against the training path column by column.

### Notebook: executed on full data

Ran clean, all 25 code cells, ~50 min. Authoritative Stage 6 results are in
`README.md`.

## Two results that contradicted the predictions

The narrative was written before the run, and two cells disagreed with what
happened. Both are now corrected:

1. **Under-sampling badly hurt the Random Forest** (0.7500, where the standard
   recipe implies competitive). Balancing to 50/50 kept 11,936 of 1,037,340 rows —
   98.9% of the data deleted. Corrected in cells 41, 46 and 51.
2. **Majority voting lost to XGBoost alone** (0.8999 vs 0.9097). Soft beat hard
   decisively (0.8989 vs 0.7411), confirming that confidence carries the signal at
   a 0.58% fraud rate, but two of three members were weaker models with correlated
   errors. Recorded in a new cell 52. The ensemble is implemented and measured,
   not selected.

## Remaining

1. Let `train_models.py` finish, then `python scripts/tune_models.py --trials 24`
   for the real Stage 7 tables. The current `reports/stage7_*.csv` are from a 2%
   smoke test and must not be quoted.
2. `tune_models.py` writes `artifacts/model_metadata.json`, which the API's
   `/api/model` card reads. It shows 0 for `validation_pr_auc` until then, so the
   "About the model" tab needs a real run to populate.
