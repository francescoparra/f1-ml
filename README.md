# Formula 1 Qualifying Prediction

An end-to-end machine learning pipeline that predicts **Formula 1 qualifying positions** from [FastF1](https://docs.fastf1.dev/) data, **before** the qualifying session starts.

Pipeline: `data ingestion → leak-free feature engineering → training → prediction → walk-forward validation → CSV artifacts`

---

## Current capabilities

- Downloads and caches historical sessions (Qualifying, FP1, FP2, FP3) and condenses every weekend into a small per-driver CSV (`data/processed/weekends/`), so reruns are fast.
- Predicts a **past** qualifying (and compares it with the real result) or an **upcoming** one (predictions only), using the free practice of the same weekend.
- Uses only information available before qualifying: no feature is taken from the session being predicted.
- Validates the model with a **walk-forward backtest** over several previous weekends and compares it with two naive baselines.
- Handles sprint weekends (missing FP2/FP3 are treated as missing values) and renamed teams (e.g. AlphaTauri → RB → Racing Bulls).

### Features

| Feature | Description |
|---|---|
| `fp1_gap_pct`, `fp2_gap_pct`, `fp3_gap_pct` | Best lap gap (%) to the fastest lap of each free practice session |
| `fp_best_gap_pct`, `fp_best_rank` | Same, using each driver's best lap across all enabled FP sessions |
| `team_fp_gap_pct` | Best FP gap (%) of the team's two cars (car pace) |
| `fp_gap_vs_teammate` | FP gap (%) minus the teammate's: isolates the driver from the car |
| `driver_prev_qual_mean` | Mean qualifying position of the driver in **earlier** weekends |
| `driver_last3_qual_mean` | Mean position in the driver's last 3 earlier qualifyings |
| `team_form_ewm` | Exponentially weighted team qualifying position over earlier weekends (half-life in `features.form_halflife`) |
| `is_sprint_weekend` | 1 if the weekend has a sprint |

All form features are shifted by one weekend before aggregating, so a label can never leak into its own features (this is covered by tests).

---

## Tech stack

- **Python 3.12+** and **Poetry**
- **FastF1** – timing and results data
- **pandas / numpy** – data processing
- **scikit-learn / SciPy** – metrics
- **XGBoost** – gradient boosted trees (handles missing values natively)
- **PyYAML**, **tqdm**, **pytest**

---

## Project structure

```
.
├── run.py                       # Pipeline entry point
├── config.example.yaml          # Configuration template (copy to config.yaml)
├── src/
│   ├── data/
│   │   ├── fetch_sessions.py    # FastF1 download, retries, per-weekend CSV cache
│   │   └── build_features.py    # Leak-free features, train/target split
│   └── models/
│       ├── train_model.py       # Model factory + fit
│       ├── predict_model.py     # Predictions + unique 1..N ranking
│       ├── evaluate.py          # mae, spearman, top3_accuracy
│       └── validate.py          # Walk-forward validation + baselines
├── tests/                       # Unit tests (synthetic data, no network)
├── data/raw/fastf1_cache/       # FastF1 cache (git-ignored)
├── data/processed/weekends/     # Condensed weekends (git-ignored)
└── outputs/                     # Generated CSVs (git-ignored)
```

---

## How to run

### 1. Install dependencies

```bash
poetry install
```

### 2. Configure the experiment

```bash
cp config.example.yaml config.yaml
```

| Section | Key | Meaning |
|---|---|---|
| `target_session` | `season`, `round` | The qualifying to predict (past or upcoming) |
| `history` | `seasons` | Seasons used for training. The target season is always added, using only the rounds **before** the target |
| `features` | `use_fp1/2/3` | Enable or disable each free practice session |
| `features` | `form_halflife` | Half-life (weekends) of the team form. Lower = reacts faster to upgrades |
| `model` | `type`, `params` | Model type (`xgboost`) and its hyperparameters |
| `validation` | `enabled`, `n_rounds`, `min_train_weekends` | Walk-forward backtest settings |
| `evaluation` | `metrics` | Any of `mae`, `spearman`, `top3_accuracy` |
| `output` | `*_file` | Where the CSVs are written (folders are created automatically) |

### 3. Run the pipeline

```bash
poetry run python run.py
poetry run python run.py --config path/to/config.yaml
poetry run python run.py --refresh      # ignore the processed cache
poetry run python run.py --season 2026 --round 17   # override the target of the config
```

The first run downloads every weekend from FastF1 (roughly 10–40 s per weekend, so a few seasons can take a while); later runs read the local cache and finish in seconds.

### Predicting an upcoming qualifying

The target qualifying has no results yet, so only predictions are produced (no metrics). The prediction improves as the weekend goes on, and you can **rerun the same command after each session** (the pipeline picks up whatever has been run):

| When | What the model uses |
|---|---|
| Before FP1 (nothing has run) | Form only. The line-up of the latest weekend is reused. The model is trained **without** FP features, since they do not exist yet |
| After FP1 (or FP2/FP3) | Form + the FP sessions that have run. Missing sessions are dropped from the features |
| After the weekend | Everything runs as a normal past target and metrics appear |

Features that are missing for the whole target weekend are removed and the model (and the backtest) is trained without them, so it never relies on a signal it will not get at prediction time.

### 4. Run the tests

```bash
poetry run pytest
```

---

## Outputs

**`outputs/predictions.csv`** – one row per driver, sorted by predicted grid order:

| Column | Description |
|---|---|
| `driver`, `team` | Driver abbreviation and team |
| `predicted_position` | Raw regression output |
| `predicted_rank` | Unique 1..N grid order from the raw output |
| `actual_position` | Real qualifying position (empty if not run yet) |

**`outputs/metrics.csv`** – metrics of the target weekend (only if it has been run).

**`outputs/backtest.csv`** and **`outputs/backtest_summary.csv`** – walk-forward validation: per weekend and averaged, for `model`, `baseline_fp_rank` (order by best FP lap) and `baseline_driver_form` (last 3 qualifying positions).

| Metric | Description |
|---|---|
| `mae` | Mean absolute error in grid positions (lower is better) |
| `spearman` | Rank correlation between predicted and actual order (higher is better) |
| `top3_accuracy` | Share of the real top 3 found in the predicted top 3 (0–1) |

---

## How validation works

For each of the last `n_rounds` weekends before the target, the model is trained **only on earlier weekends** and scored on that weekend. The mean across weekends is the number to look at: a single weekend is too noisy to judge a model. A model that does not beat the baselines is not adding value.

---

## Known limitations

- **Pre-FP predictions are essentially recent form:** in the backtest the model is on par with the driver-form baseline until FP data exists.
- **New teams/drivers** (e.g. a new constructor) have no history, so they are predicted from missing values.
- **No wet/dry or track information:** weather, circuit type and tyre compounds are not modelled yet.
- **FP is a noisy signal:** teams run different programmes (race simulations, tyre tests) and track conditions change between sessions.
- **Sprint weekends:** the sprint (qualifying) session result is not used as a feature yet, even though it is available before qualifying.
- **Regulation changes:** form carried over from previous seasons loses value when the cars change a lot (the half-life partially covers this).
- **Few samples:** ~20 drivers × ~24 weekends per season, so hyperparameter differences smaller than ~0.05 MAE are within noise.

---

## Roadmap

- Use sprint session results and weather as features
- Circuit-specific history of each driver and team
- Ranking objectives (XGBoost ranker / LightGBM) and model comparison
- Experiment tracking with MLflow
