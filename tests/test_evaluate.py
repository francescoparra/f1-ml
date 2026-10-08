import numpy as np
import pytest

from src.models.evaluate import evaluate, top_k_accuracy


def test_perfect_prediction():
    y = np.array([1, 2, 3, 4, 5])
    res = evaluate(y.astype(float), y, ['mae', 'spearman', 'top3_accuracy'])
    assert res['mae'] == 0
    assert res['spearman'] == pytest.approx(1.0)
    assert res['top3_accuracy'] == 1.0


def test_top3_partial_overlap():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 4.5, 3.0, 5.0])  # predicted top 3: idx 0, 1, 3
    assert top_k_accuracy(y_pred, y_true, k=3) == pytest.approx(2 / 3)


def test_top_k_is_capped_to_number_of_drivers():
    assert top_k_accuracy([2.0, 1.0], [1, 2], k=3) == 1.0


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match='Unknown metric'):
        evaluate([1, 2], [1, 2], ['rmse'])
