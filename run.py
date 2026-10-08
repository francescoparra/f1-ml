import argparse
import os

import pandas as pd
import yaml

from src.data.build_features import build_features
from src.data.fetch_sessions import fetch_target_session, fetch_historical_sessions
from src.models.evaluate import evaluate
from src.models.predict_model import predict
from src.models.train_model import train_model


def load_config(path):
    """Load the experiment configuration from a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file '{path}' not found. "
            "Create it with: cp config.example.yaml config.yaml"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def save_outputs(predictions_df, metrics, output_config):
    """Write predictions and metrics to the configured CSV paths."""
    for path in (output_config['predictions_file'], output_config['metrics_file']):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    predictions_df.to_csv(output_config['predictions_file'], index=False)
    pd.DataFrame([metrics]).to_csv(output_config['metrics_file'], index=False)


def main(config_path='config.yaml'):
    config = load_config(config_path)

    # Fetch the data
    target_session = fetch_target_session(config['target_session'])
    historical_data = fetch_historical_sessions(
        config['history']['seasons'], config['features']
    )

    # Build train and test data
    X_train, y_train, X_test, y_test, test_drivers = build_features(
        historical_data, target_session
    )

    # Train the model and predict the target session
    model = train_model(X_train, y_train, config['model'])
    predictions_df = predict(model, X_test)

    # Evaluate the prediction
    metrics = evaluate(
        predictions_df['predicted_position'].to_numpy(),
        y_test,
        config['evaluation']['metrics'],
    )

    # Save everything into CSV files
    predictions_df.insert(0, 'driver', test_drivers)
    predictions_df['predicted_rank'] = (
        predictions_df['predicted_position'].rank(method='first').astype(int)
    )
    predictions_df['actual_position'] = y_test
    predictions_df = predictions_df.sort_values('predicted_position')

    save_outputs(predictions_df, metrics, config['output'])

    print("Metrics:", {k: round(float(v), 3) for k, v in metrics.items()})
    print(
        "Done! Predictions saved in "
        f"'{config['output']['predictions_file']}', metrics in "
        f"'{config['output']['metrics_file']}'."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the F1 qualifying prediction pipeline.")
    parser.add_argument(
        '--config', default='config.yaml', help="Path to the YAML config (default: config.yaml)"
    )
    main(parser.parse_args().config)
