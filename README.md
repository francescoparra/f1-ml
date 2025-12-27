# Formula 1 Qualifying Prediction (ML Baseline)

This project implements a **baseline end-to-end machine learning pipeline** to predict **Formula 1 qualifying positions** using historical data from the **FastF1** library.

The focus of the project is not on complex models, but on building a **clean, reproducible ML pipeline**:
- structured data ingestion
- feature engineering
- model training
- evaluation
- artifact generation (predictions and metrics)

The project is designed to be extended incrementally with more advanced models and features.

---

## What the program does (current capabilities)

At its current stage, the program:

- Downloads and caches historical Formula 1 session data using FastF1
- Builds tabular features for each driver based on:
  - best qualifying lap time
  - free practice session performance (FP1, FP2, FP3 gaps to fastest lap)
- Trains a regression model to predict qualifying finishing positions
- Predicts qualifying positions for a target race weekend
- Evaluates predictions using standard regression and ranking metrics
- Saves predictions and evaluation metrics to CSV files

This represents a **baseline MVP** for qualifying prediction using tabular machine learning.

---

## Tech Stack

- **Python 3.12+**
- **Poetry** – dependency and environment management
- **FastF1** – Formula 1 timing and results data
- **pandas / numpy** – data processing
- **scikit-learn** – evaluation and baselines
- **XGBoost** – gradient boosted tree model
- **PyYAML** – configuration management

---

## How to Run

### 1. Install dependencies

```bash
poetry install
```

### 2. Configure the experiment

Rename `config.example.yaml`

```bash
cp config.example.yaml config.yaml
```

Edit `config/config.yml` to select:

- Historical seasons
- Target season and race
- Model parameters
- Output paths

### 3. Run the pipeline

```bash
python run.py
```

The program will:

1. Fetch historical data (cached locally)
2. Build training and test features
3. Train the model
4. Generate predictions
5. Save outputs to the outputs/ directory
