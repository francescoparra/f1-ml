import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

SUPPORTED_METRICS = ('mae', 'spearman', 'top3_accuracy')


def top_k_accuracy(y_pred, y_true, k=3):
    """
    Fraction of the real top-k drivers that are also in the predicted top-k.

    Lower values mean a better position, so the top-k are the k smallest values.

    Parameters
    ----------
    y_pred : array-like
        Predicted qualifying positions (continuous values are fine).
    y_true : array-like
        Actual qualifying positions.
    k : int
        Size of the top group, capped to the number of drivers.

    Returns
    -------
    float
        Value between 0.0 and 1.0.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    k = min(k, len(y_true))

    pred_top = set(np.argsort(y_pred, kind='stable')[:k])
    true_top = set(np.argsort(y_true, kind='stable')[:k])

    return len(pred_top & true_top) / k


def evaluate(y_pred, y_true, metrics):
    """
    Evaluate qualifying position predictions.

    Args:
        y_pred (np.ndarray): predicted values
        y_true (np.ndarray): true values
        metrics (list[str]): metric names, any of `SUPPORTED_METRICS`

    Returns:
        dict: metric_name -> value

    Raises:
        ValueError: if a requested metric is not supported.
    """
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. Supported: {list(SUPPORTED_METRICS)}"
        )

    results = {}

    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)

    if 'spearman' in metrics:
        corr, _ = spearmanr(y_true, y_pred)
        results['spearman'] = corr

    if 'top3_accuracy' in metrics:
        results['top3_accuracy'] = top_k_accuracy(y_pred, y_true, k=3)

    return results
