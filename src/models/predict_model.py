import pandas as pd


def predict(model, X_test):
    """
    Generate qualifying position predictions from a trained model.

    This function applies a fitted machine learning model to the test feature
    matrix and returns the predictions in a structured tabular format, ready
    for evaluation or persistence to disk.

    Parameters
    ----------
    model : object
        Trained model implementing a `predict(X)` method
        (e.g. XGBoost, LightGBM, scikit-learn estimator).

    X_test : pd.DataFrame
        Feature matrix for the target qualifying session.
        One row corresponds to one driver.

    Returns
    -------
    pd.DataFrame
    """
    preds = model.predict(X_test)
    return pd.DataFrame({'predicted_position': preds})
