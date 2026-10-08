import pandas as pd


def predict(model, X_test):
    """
    Generate qualifying position predictions from a trained model.

    Parameters
    ----------
    model : object
        Trained model implementing a `predict(X)` method
        (e.g. XGBoost, LightGBM, scikit-learn estimator).

    X_test : pd.DataFrame
        Feature matrix of ONE qualifying session, one row per driver.

    Returns
    -------
    pd.DataFrame
        Indexed like `X_test`, with:
        - `predicted_position`: raw regression output
        - `predicted_rank`: 1..N grid order obtained by sorting the raw output
          (ties broken by row order), so every driver gets a unique position.
    """
    preds = model.predict(X_test)
    result = pd.DataFrame({'predicted_position': preds}, index=X_test.index)
    result['predicted_rank'] = result['predicted_position'].rank(method='first').astype(int)
    return result
