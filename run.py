import argparse
import os

import pandas as pd
import yaml

from src.data.build_features import (
    available_feature_columns,
    build_feature_table,
    get_feature_columns,
    split_train_target,
)
from src.data.fetch_sessions import fetch_weekends
from src.models.evaluate import evaluate
from src.models.predict_model import predict
from src.models.train_model import train_model
from src.models.validate import walk_forward_validation

MIN_TRAIN_ROWS = 40


def load_config(path):
    """Load the experiment configuration from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file '{path}' not found. "
            "Create it with: cp config.example.yaml config.yaml"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def save_csv(df, path):
    """Write a DataFrame to CSV, creating the parent folder if needed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)


def main(config_path='config.yaml', refresh=False, season=None, round_num=None):
    config = load_config(config_path)
    if season is not None:
        config['target_session']['season'] = season
    if round_num is not None:
        config['target_session']['round'] = round_num
    target = config['target_session']
    features_config = config['features']
    metric_names = config['evaluation']['metrics']
    validation_config = config.get('validation', {})
    output = config['output']

    # Fetch every weekend up to (and including) the target one
    weekends = fetch_weekends(config['history']['seasons'], target, refresh=refresh)

    # Build leak-free features and split by time
    table = build_feature_table(
        weekends, features_config, features_config.get('form_halflife', 6.0)
    )
    train, target_rows = split_train_target(table, target['season'], target['round'])
    feature_cols = available_feature_columns(target_rows, get_feature_columns(features_config))
    dropped = set(get_feature_columns(features_config)) - set(feature_cols)
    if dropped:
        print(f"Not available for the target yet, model trained without: {sorted(dropped)}")

    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {len(train)} training rows found (need at least {MIN_TRAIN_ROWS}). "
            "Add more seasons to `history.seasons`."
        )
    print(f"Training on {train['weekend_idx'].nunique()} weekends ({len(train)} rows).")

    # Time-based validation over several previous weekends
    if validation_config.get('enabled', True):
        per_weekend, summary = walk_forward_validation(
            table,
            feature_cols,
            config['model'],
            metric_names,
            n_rounds=validation_config.get('n_rounds', 10),
            min_train_weekends=validation_config.get('min_train_weekends', 5),
            before_idx=target_rows['weekend_idx'].iloc[0],
        )
        if per_weekend.empty:
            print("Not enough history to run the walk-forward validation.")
        else:
            save_csv(per_weekend, output.get('backtest_file', 'outputs/backtest.csv'))
            save_csv(summary, output.get('backtest_summary_file', 'outputs/backtest_summary.csv'))
            print(f"\nWalk-forward validation ({int(summary['n_weekends'].iloc[0])} weekends):")
            print(summary.round(3).to_string(index=False))

    # Train on everything before the target weekend and predict it
    model = train_model(train[feature_cols], train['qual_position'], config['model'])
    predictions = predict(model, target_rows[feature_cols])

    predictions_df = pd.concat(
        [
            target_rows[['driver', 'team']],
            predictions,
            target_rows['qual_position'].rename('actual_position'),
        ],
        axis=1,
    ).sort_values('predicted_rank')
    save_csv(predictions_df, output['predictions_file'])

    # Metrics only exist if the target qualifying has already been run
    if target_rows['qual_position'].notna().all():
        metrics = evaluate(
            predictions_df['predicted_position'].to_numpy(),
            predictions_df['actual_position'].to_numpy(),
            metric_names,
        )
        save_csv(pd.DataFrame([metrics]), output['metrics_file'])
        print("\nTarget weekend metrics:", {k: round(float(v), 3) for k, v in metrics.items()})
    else:
        print("\nTarget qualifying has not been run yet: predictions only, no metrics.")

    print(f"Done! Predictions saved in '{output['predictions_file']}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the F1 qualifying prediction pipeline.")
    parser.add_argument(
        '--config', default='config.yaml', help="Path to the YAML config (default: config.yaml)"
    )
    parser.add_argument('--season', type=int, help="Override target_session.season")
    parser.add_argument('--round', type=int, dest='round_num', help="Override target_session.round")
    parser.add_argument(
        '--refresh', action='store_true',
        help="Ignore the processed-data cache and download every weekend again",
    )
    args = parser.parse_args()
    main(args.config, args.refresh, args.season, args.round_num)
