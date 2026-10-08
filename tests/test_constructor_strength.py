import pandas as pd
import pytest

from src.data.build_features import compute_constructor_strength


class FakeSession:
    def __init__(self, rows):
        self.results = pd.DataFrame(rows, columns=['TeamName', 'Position'])


def make_history():
    return [
        {'season': 2023, 'session': FakeSession([('A', 1), ('A', 2), ('B', 9), ('B', 10)])},
        {'season': 2024, 'session': FakeSession([('A', 3), ('A', 4), ('B', 7), ('B', 8)])},
    ]


def test_recency_weighting():
    strength, _ = compute_constructor_strength(make_history(), target_year=2025)
    # 2024 -> weight 1.0 (mean 3.5), 2023 -> weight 0.8 (mean 1.5)
    assert strength['A'] == pytest.approx((1.0 * 3.5 + 0.8 * 1.5) / 1.8)


def test_target_season_and_later_are_ignored():
    strength, _ = compute_constructor_strength(make_history(), target_year=2024)
    assert strength['A'] == pytest.approx(1.5)


def test_no_usable_history_falls_back_to_default():
    strength, global_mean = compute_constructor_strength(make_history(), target_year=2023)
    assert strength == {}
    assert global_mean == 10.0


def test_global_mean_is_over_all_positions():
    _, global_mean = compute_constructor_strength(make_history(), target_year=2025)
    assert global_mean == pytest.approx(5.5)
