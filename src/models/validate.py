import numpy as np
import pandas as pd

from src.models.evaluate import evaluate
from src.models.predict_model import predict
from src.models.train_model import train_model


def baseline_predictions(test, use_fp=True):
    """
    Naive predictors the model must beat to be worth anything.

    - `baseline_fp_rank`: order of the drivers by their best free practice
      lap (falls back to the driver's recent form when FP data is missing).
      Skipped when `use_fp` is False (prediction made before any FP).
    - `baseline_driver_form`: mean of the driver's last 3 qualifying positions.
    - `baseline_form_blend`: the hand-made driver/team form blend, with no
      learning. The model has to beat it to justify itself.
    """
    fallback = (len(test) + 1) / 2
    form = test['driver_last3'].fillna(fallback)

    baselines = {}
    if use_fp:
        baselines['baseline_fp_rank'] = test['fp_best_rank'].fillna(form)
    baselines['baseline_driver_form'] = form
    baselines['baseline_form_blend'] = test['form_blend'].fillna(form)
    return baselines


def walk_forward_validation(
    table,
    feature_cols,
    model_config,
    metrics,
    n_rounds=10,
    min_train_weekends=5,
    before_idx=None,
    eval_sprint=None,
):
    """
    Time-based (walk-forward) validation over several race weekends.

    For each evaluated weekend the model is trained ONLY on earlier weekends
    and then scored on that weekend. Nothing from the future is ever used, so
    the average reflects how the model would have performed live. Naive
    baselines are scored on the same weekends for comparison.

    Parameters
    ----------
    table : pd.DataFrame
        Output of `build_feature_table`.
    feature_cols : list[str]
    model_config : dict
        Same structure as the `model` section of the config.
    metrics : list[str]
        Metric names understood by `evaluate`.
    n_rounds : int
        How many of the most recent weekends to evaluate.
    min_train_weekends : int
        Minimum number of earlier weekends needed to evaluate a weekend.
    before_idx : int | None
        Only weekends with `weekend_idx` lower than this are used
        (pass the target weekend index to keep the target out of validation).
    eval_sprint : int | None
        If 0 or 1, only weekends whose `is_sprint_weekend` equals it are
        scored (training still uses every earlier weekend). Used to validate on
        weekends of the same kind as the target.

    Returns
    -------
    per_weekend : pd.DataFrame
        One row per (weekend, method) with the metrics.
    summary : pd.DataFrame
        Mean of each metric per method across the evaluated weekends.
    """
    labeled = table[table['qual_position'].notna()]
    if before_idx is not None:
        labeled = labeled[labeled['weekend_idx'] < before_idx]

    weekend_ids = sorted(labeled['weekend_idx'].unique())
    candidates = weekend_ids[min_train_weekends:]
    if eval_sprint is not None:
        kind = labeled.groupby('weekend_idx')['is_sprint_weekend'].first()
        candidates = [w for w in candidates if kind[w] == eval_sprint]
    eval_ids = candidates[-n_rounds:]

    rows = []
    for weekend_id in eval_ids:
        train = labeled[labeled['weekend_idx'] < weekend_id]
        test = labeled[labeled['weekend_idx'] == weekend_id]
        y_true = test['qual_position'].to_numpy()

        model = train_model(train[feature_cols], train['qual_position'], model_config)
        predictions = {'model': predict(model, test[feature_cols])['predicted_position']}
        for name, values in baseline_predictions(test, 'fp_best_rank' in feature_cols).items():
            predictions[name] = values

        info = test.iloc[0]
        for method, y_pred in predictions.items():
            scores = evaluate(np.asarray(y_pred, dtype=float), y_true, metrics)
            rows.append({
                'season': info['season'],
                'round': info['round'],
                'event_name': info['event_name'],
                'method': method,
                'n_drivers': len(test),
                'n_train_weekends': train['weekend_idx'].nunique(),
                **scores,
            })

    per_weekend = pd.DataFrame(rows)
    if per_weekend.empty:
        return per_weekend, pd.DataFrame()

    metric_cols = [m for m in metrics if m in per_weekend.columns]
    summary = (
        per_weekend.groupby('method', sort=False)[metric_cols]
        .mean()
        .reset_index()
    )
    summary.insert(1, 'n_weekends', len(eval_ids))

    return per_weekend, summary
