# Formula 1 Qualifying Prediction (ML Baseline)

A **baseline end-to-end machine learning pipeline** that predicts **Formula 1 qualifying positions** from historical data provided by the [FastF1](https://docs.fastf1.dev/) library.

The goal is not a complex model, but a **clean, reproducible pipeline** that can be extended step by step:

`data ingestion → feature engineering → training → prediction → evaluation → CSV artifacts`

---

## Current capabilities

- Downloads and caches historical session data (Qualifying, FP1, FP2, FP3) with FastF1
- Builds one row per driver with these features:
  | Feature | Description |
  |---|---|
  | `best_qual_time` | Driver's best qualifying lap time (seconds) |
  | `driver_avg_qual_pos` | Driver's mean qualifying position in the training seasons |
  | `constructor_strength` | Recency-weighted mean qualifying position of the team, using only seasons *before* the target season (lower = stronger) |
  | `fp1_gap`, `fp2_gap`, `fp3_gap` | Gap (s) to the fastest lap of each free practice session |
- Trains an XGBoost regressor to predict the qualifying position
- Predicts the qualifying session of a target race weekend
- Evaluates it with `mae`, `spearman` and `top3_accuracy`
- Saves predictions and metrics as CSV files

---

## Tech stack

- **Python 3.12+** and **Poetry** (dependency/environment management)
- **FastF1** – timing and results data
- **pandas / numpy** – data processing
- **scikit-learn / SciPy** – metrics
- **XGBoost** – gradient boosted trees
- **PyYAML** – configuration
- **pytest** – tests

---

## Project structure

```
.
├── run.py                    # Pipeline entry point
├── config.example.yaml       # Configuration template (copy to config.yaml)
├── src/
│   ├── data/
│   │   ├── fetch_sessions.py   # FastF1 download + local cache
│   │   └── build_features.py   # Feature engineering, constructor strength
│   └── models/
│       ├── train_model.py      # Model factory + fit
│       ├── predict_model.py    # Model → predictions DataFrame
│       └── evaluate.py         # mae, spearman, top3_accuracy
├── tests/                    # Unit tests
├── data/raw/fastf1_cache/    # FastF1 cache (git-ignored)
└── outputs/                  # Generated predictions and metrics (git-ignored)
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

Edit `config.yaml` to choose:

| Section | Key | Meaning |
|---|---|---|
| `target_session` | `season`, `round` | The qualifying session to predict |
| `history` | `seasons` | Seasons used for training. Include seasons **earlier** than the target one, otherwise `constructor_strength` falls back to the global mean |
| `features` | `use_fp1/2/3` | Enable or disable each free practice feature |
| `model` | `type`, `params` | Model type (`xgboost`) and its hyperparameters |
| `evaluation` | `metrics` | Any of `mae`, `spearman`, `top3_accuracy` (unknown names raise an error) |
| `output` | `predictions_file`, `metrics_file` | Where the CSVs are written (folders are created automatically) |

### 3. Run the pipeline

```bash
poetry run python run.py
# or with a custom config
poetry run python run.py --config path/to/config.yaml
```

The first run downloads data from FastF1 and can take several minutes; later runs reuse the local cache.

The pipeline will:

1. Fetch historical and target-session data (cached locally)
2. Build training and test features
3. Train the model
4. Predict the target qualifying
5. Evaluate and save the outputs

### 4. Run the tests

```bash
poetry run pytest
```

---

## Outputs

**`outputs/predictions.csv`** – one row per driver, sorted by predicted position:

| Column | Description |
|---|---|
| `driver` | Driver abbreviation (e.g. `VER`) |
| `predicted_position` | Raw regression output |
| `predicted_rank` | Predicted position converted to a 1..N ranking |
| `actual_position` | Real qualifying position |

**`outputs/metrics.csv`** – one row with the metrics selected in the config:

| Metric | Description |
|---|---|
| `mae` | Mean absolute error between predicted and actual position (lower is better) |
| `spearman` | Rank correlation between predicted and actual order (higher is better) |
| `top3_accuracy` | Share of the real top 3 drivers found in the predicted top 3 (0–1) |

---

## Known limitations

This is a baseline MVP, so keep these caveats in mind when reading the metrics:

- **Target leakage:** `best_qual_time` comes from the same session being predicted, which makes the task much easier than a real pre-session forecast.
- **Free practice features at test time:** the target session's FP gaps are not fetched yet; they are filled with the training mean.
- **Driver average:** `driver_avg_qual_pos` is computed on the training set, including each row's own label.
- **Single target session:** the evaluation uses one race weekend, so metrics are noisy.

---

## Roadmap

- Fetch real FP data for the target weekend and remove `best_qual_time` as a feature
- Time-based validation across several rounds/seasons
- Additional features (track, weather, tyre compounds, recent form)
- Compare more models (LightGBM, ranking objectives) and track experiments with MLflow
