from xgboost import XGBRegressor


def train_model(X_train, y_train, model_config):
    """
    Train a regression model for qualifying position prediction.

    This function instantiates and fits a machine learning model based on the
    configuration provided. Currently, supports gradient-boosted decision trees
    via XGBoost, but is structured to allow easy extension to other models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
        One row corresponds to one driver–session observation.

    y_train : np.ndarray or pd.Series
        Target values representing qualifying positions.

    model_config : dict
        Model configuration dictionary

    Returns
    -------
    model : object
        Fitted regression model implementing a `predict(X)` method.
    """
    model_type = model_config['type']
    params = model_config['params']

    if model_type == 'xgboost':
        model = XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model
