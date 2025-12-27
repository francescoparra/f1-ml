from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


def evaluate(y_pred, y_true, metrics):
    """
    Evaluate regression predictions.

    Args:
        y_pred (np.ndarray): predicted values
        y_true (np.ndarray): true values
        metrics (list[str]): list of metric names

    Returns:
        dict: metric_name -> value
    """
    results = {}

    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)

    if 'spearman' in metrics:
        corr, _ = spearmanr(y_true, y_pred)
        results['spearman'] = corr

    return results
