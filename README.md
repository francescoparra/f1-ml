# Formula 1 Qualifying Prediction

An end-to-end machine learning pipeline that predicts **Formula 1 qualifying positions** from [FastF1](https://docs.fastf1.dev/) data, **before** the qualifying session starts.

Pipeline: `data ingestion → leak-free feature engineering → training → prediction → walk-forward validation → CSV artifacts`

---

## Current capabilities

- Downloads and caches historical sessions (Qualifying, FP1-FP3, sprint sessions) and condenses every weekend into a small per-driver CSV (`data/processed/weekends/`), so reruns are fast.
- Predicts a **past** qualifying (and compares it with the real result) or an **upcoming** one (predictions only), using everything that has already run in the same weekend.
- Uses only information available before qualifying: no feature is taken from the session being predicted, and sprint sessions held *after* qualifying (2021-2023 formats) are ignored.
- **Recency-weighted form:** the last 12 races matter the most, the 12 before them add data with less weight. The windows ignore season boundaries, so early-season rounds still know the end of the previous season (momentum, upgrades).
- **Track record:** each driver's results at the same circuit over all the seasons in the history (e.g. Leclerc in Monaco, Verstappen in Austria).
- **Sprint weekends:** sprint qualifying/shootout pace and the sprint result are used when they precede qualifying.
- **New drivers and teams:** a rookie leans on his team's form, a brand-new team on its drivers' form.
- Validates the model with a **walk-forward backtest** over several previous weekends and compares it with naive baselines.
- Handles renamed teams (AlphaTauri → RB → Racing Bulls, Sauber → Audi) and renamed circuits.

### Features

All of them are computed from information available before qualifying starts.

**Free practice (same weekend)**

| Feature | Description |
|---|---|
| `fp1_gap_pct`, `fp2_gap_pct`, `fp3_gap_pct` | Best lap gap (%) to the fastest lap of each session |
| `fp_best_gap_pct`, `fp_best_rank` | Same, using each driver's best lap across all enabled FP sessions |
| `fp_soft_gap_pct`, `fp_soft_rank` | Same, using only laps on the **soft** compound (the closest thing to a qualifying push lap) |
| `team_fp_gap_pct` | Best FP gap (%) of the team's cars (car pace) |
| `fp_gap_vs_teammate` | FP gap (%) minus the teammate's: isolates the driver from the car |

**Sprint (same weekend, only sessions that precede qualifying)**

| Feature | Description |
|---|---|
| `sq_gap_pct`, `sq_rank` | Gap (%) to the fastest lap of sprint qualifying / shootout, and rank by that lap |
| `sprint_position`, `sprint_gap_pct` | Sprint finishing position and best-lap gap (%) |
| `is_sprint_weekend` | 1 if the weekend has a sprint |

**Form (earlier weekends, any season)**

| Feature | Description |
|---|---|
| `driver_last3`, `team_last3` | Mean qualifying position in the last 3 races: streak / momentum |
| `driver_recent`, `team_recent` | Mean position in the last `recent_window` (12) races |
| `driver_older`, `team_older` | Mean position in the `older_window` (12) races before those |
| `driver_form`, `team_form` | Weighted mean of both blocks (older races count `older_weight` = 0.4 as much) |
| `driver_n_prior`, `team_n_prior` | Number of races the form above is based on (0 for a rookie or a new team) |
| `form_blend` | Expected position mixing driver and team form, each trusted in proportion to its history |

Comparing `team_recent` with `team_older` shows whether the car improved (upgrades) compared to earlier in the season or last year.

**Track record (all earlier seasons)**

| Feature | Description |
|---|---|
| `driver_circuit_visits` | Earlier qualifyings of the driver at this circuit |
| `driver_circuit_mean` | Mean position of the driver at this circuit |
| `driver_circuit_delta` | How much better (negative) or worse than his own form the driver usually is at this circuit, shrunk towards 0 when there are few visits |

All form and track features only see earlier weekends, so a label can never leak into its own features (this is covered by tests).

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
| `history` | `seasons` | Seasons used for training. The target season is always added, using only the rounds **before** the target. Older seasons feed the track record |
| `features` | `use_fp1/2/3` | Enable or disable each free practice session |
| `features` | `use_sprint` | Use sprint sessions that precede qualifying |
| `features` | `recent_window`, `older_window`, `older_weight` | Form windows in races (12 + 12) and the weight of the older block (0.4) |
| `features` | `circuit_shrinkage` | How strongly the track record is pulled towards 0 when there are few visits |
| `model` | `type`, `params` | Model type (`xgboost`) and its hyperparameters |
| `validation` | `enabled`, `n_rounds`, `min_train_weekends`, `same_kind_only` | Walk-forward backtest settings. `same_kind_only` scores only weekends of the same kind (sprint / normal) as the target |
| `evaluation` | `metrics` | Any of `mae`, `spearman`, `top3_accuracy` |
| `output` | `*_file` | Where the CSVs are written (folders are created automatically) |

### 3. Run the pipeline

```bash
poetry run python run.py
poetry run python run.py --config path/to/config.yaml
poetry run python run.py --refresh      # ignore the processed cache
poetry run python run.py --season 2026 --round 17   # override the target of the config
```

The first run downloads every weekend from FastF1 (the default config covers 2019-2026, around 170 weekends, so expect tens of minutes). The API throttles heavy use: failed or empty downloads are retried with a growing delay and are never cached half-finished, so if a weekend is reported as missing just rerun the command. Later runs read the local cache and only download the target weekend (about 20 s).

### Predicting an upcoming qualifying

The target qualifying has no results yet, so only predictions are produced (no metrics). The prediction improves as the weekend goes on, and you can **rerun the same command after each session** (the pipeline picks up whatever has been run):

| When | What the model uses |
|---|---|
| Before FP1 (nothing has run) | Form and track record only. The line-up of the latest weekend is reused. The model is trained **without** the FP and sprint features, since they do not exist yet |
| After FP1 (or FP2/FP3) | Form, track record + the FP sessions that have run (including the soft-tyre runs) |
| After sprint qualifying / the sprint (sprint weekends) | All of the above + the sprint features. Sprint sessions are used only if they start before qualifying |
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

**`outputs/backtest.csv`** and **`outputs/backtest_summary.csv`** – walk-forward validation: per weekend and averaged, for `model` and three naive baselines: `baseline_fp_rank` (order by best FP lap), `baseline_driver_form` (last 3 qualifying positions) and `baseline_form_blend` (the hand-made driver/team form blend, no learning).

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

- **No weather or circuit-type information:** the driver's track record is used, but wet/dry conditions and circuit characteristics (high downforce, street circuit, ...) are not modelled yet.
- **FP is a noisy signal:** teams run different programmes and track conditions change between sessions. It is still used (with extra weight on soft-tyre runs); it should become more valuable once race predictions are added.
- **Regulation changes cannot be predicted:** when the cars change a lot, form from the previous year loses value. The only data that could help is pre-season testing, which is not used.
- **Few samples:** ~20 drivers × ~24 weekends per season, and only ~6 sprint weekends per season, so differences smaller than ~0.05 MAE between model variants are within noise.

---

## Roadmap

- Weather (wet/dry) and circuit-type features
- Race prediction, where FP long runs matter more
- Pre-season testing as a signal after regulation changes
- Ranking objectives (XGBoost ranker / LightGBM) and model comparison
- Experiment tracking with MLflow
