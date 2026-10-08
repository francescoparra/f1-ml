import numpy as np
import pandas as pd
import pytest


def make_weekends(n_weekends=12, seed=0, last_unlabeled=False):
    """
    Synthetic per-driver weekend table: 6 drivers in 3 teams, with a stable
    pecking order plus noise so FP pace is informative about qualifying.
    """
    rng = np.random.default_rng(seed)
    drivers = [('AAA', 'Alpha'), ('BBB', 'Alpha'), ('CCC', 'Beta'),
               ('DDD', 'Beta'), ('EEE', 'Gamma'), ('FFF', 'Gamma')]
    rows = []
    for w in range(n_weekends):
        pace = np.arange(6) + rng.normal(0, 0.8, 6)
        order = pace.argsort().argsort() + 1
        for i, (driver, team) in enumerate(drivers):
            row = {
                'season': 2024 + w // 6,
                'round': w % 6 + 1,
                'event_name': f'GP {w}',
                'event_format': 'sprint_qualifying' if w % 5 == 4 else 'conventional',
                'driver': driver,
                'team': team,
                'qual_position': float(order[i]),
            }
            for k in (1, 2, 3):
                missing = row['event_format'] != 'conventional' and k > 1
                row[f'fp{k}_best'] = np.nan if missing else 90 + pace[i] * 0.1 + rng.normal(0, 0.05)
                row[f'fp{k}_laps'] = np.nan if missing else 15
            rows.append(row)
    df = pd.DataFrame(rows)
    if last_unlabeled:
        last = (df['season'] == df['season'].max()) & (df['round'] == df['round'].max())
        df.loc[last, 'qual_position'] = np.nan
    return df


@pytest.fixture
def weekends():
    return make_weekends()
