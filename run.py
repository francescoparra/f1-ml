import yaml
from src.data.fetch_sessions import fetch_target_session, fetch_historical_sessions
from src.data.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate import evaluate
import pandas as pd

""" Load configs """
with open("config.yaml") as f:
    config = yaml.safe_load(f)

""" Fetch the data """
target_session = fetch_target_session(config['target_session'])
historical_data = fetch_historical_sessions(config['history']['seasons'], config['features'])

""" Build train and test data """
X_train, y_train, X_test, y_test, test_drivers = build_features(
    historical_data,
    target_session
)

""" Train the model """
model = train_model(X_train, y_train, config['model'])

""" Do the actual prediction """
y_pred = model.predict(X_test)

""" Evaluate the prediction """
metrics = evaluate(y_pred, y_test, config['evaluation']['metrics'])

""" Save everything into a csv """
predictions_df = pd.DataFrame({
    "driver": test_drivers,
    "predicted_position": y_pred,
    "actual_position": y_test
})

predictions_df = predictions_df.sort_values("predicted_position")

predictions_df.to_csv(config['output']['predictions_file'], index=False)
pd.DataFrame([metrics]).to_csv(
    config['output']['metrics_file'],
    index=False
)


print("Done! Predictions and metrics saved in 'outputs/' folder.")
