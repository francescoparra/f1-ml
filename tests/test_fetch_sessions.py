import numpy as np
import pandas as pd

from src.data import fetch_sessions
from src.data.fetch_sessions import WEEKEND_COLUMNS, build_weekend_table


class FakeSession:
    def __init__(self, results=None, laps=None):
        if results is not None:
            self.results = pd.DataFrame(results, columns=['Abbreviation', 'TeamName', 'Position'])
        if laps is not None:
            self.laps = pd.DataFrame(
                [(d, pd.Timedelta(seconds=t) if t is not None else pd.NaT) for d, t in laps],
                columns=['Driver', 'LapTime'],
            )


def test_weekend_table_collects_positions_and_best_laps():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)])
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)],
        laps=[('VER', 91.5), ('VER', 90.2), ('VER', None), ('LEC', 90.9)],
    )

    table = build_weekend_table(2024, 3, 'Test GP', 'conventional', q, {'FP1': fp1, 'FP2': None, 'FP3': None})

    assert list(table.columns) == WEEKEND_COLUMNS
    ver = table.set_index('driver').loc['VER']
    assert ver['qual_position'] == 1
    assert ver['fp1_best'] == 90.2
    assert ver['fp1_laps'] == 2
    assert np.isnan(ver['fp2_best'])


def test_upcoming_weekend_without_qualifying_uses_fp_drivers_and_has_nan_labels():
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', np.nan), ('LEC', 'Ferrari', np.nan)],
        laps=[('VER', 90.0), ('LEC', 90.5)],
    )

    table = build_weekend_table(2026, 1, 'Future GP', 'conventional', None, {'FP1': fp1})

    assert sorted(table['driver']) == ['LEC', 'VER']
    assert table['qual_position'].isna().all()
    assert table['fp1_best'].notna().all()


def test_fp_only_drivers_are_dropped_when_qualifying_exists():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0)])
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', 1.0), ('ROO', 'Ferrari', 2.0)],
        laps=[('VER', 90.0), ('ROO', 89.0)],
    )

    table = build_weekend_table(2024, 3, 'Test GP', 'conventional', q, {'FP1': fp1})

    assert table['driver'].tolist() == ['VER']


def test_completed_weekends_are_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sessions, 'PROCESSED_DIR', str(tmp_path))
    calls = []

    def fake_load(season, round_num, name):
        calls.append(name)
        if name == 'Q':
            return FakeSession(results=[('VER', 'Red Bull Racing', 1.0)])
        return None

    monkeypatch.setattr(fetch_sessions, 'load_session', fake_load)

    first = fetch_sessions.fetch_weekend(2024, 1, 'GP', 'conventional')
    n_calls = len(calls)
    second = fetch_sessions.fetch_weekend(2024, 1, 'GP', 'conventional')

    assert len(calls) == n_calls
    assert first['driver'].tolist() == second['driver'].tolist() == ['VER']
    assert (tmp_path / '2024_01.csv').exists()


def test_weekends_without_qualifying_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sessions, 'PROCESSED_DIR', str(tmp_path))
    fp1 = FakeSession(results=[('VER', 'Red Bull Racing', np.nan)], laps=[('VER', 90.0)])
    monkeypatch.setattr(
        fetch_sessions, 'load_session', lambda s, r, n: fp1 if n == 'FP1' else None
    )

    table = fetch_sessions.fetch_weekend(2026, 1, 'GP', 'conventional')

    assert len(table) == 1
    assert list(tmp_path.iterdir()) == []


def test_lineup_placeholder_reuses_latest_weekend_with_nan_data(weekends):
    from src.data.fetch_sessions import lineup_from_previous

    placeholder = lineup_from_previous(weekends, 2030, 1, 'Future GP', 'sprint_qualifying')

    latest = weekends[(weekends['season'] == 2025) & (weekends['round'] == 6)]
    assert list(placeholder.columns) == WEEKEND_COLUMNS
    assert sorted(placeholder['driver']) == sorted(latest['driver'])
    assert (placeholder['season'] == 2030).all() and (placeholder['event_format'] == 'sprint_qualifying').all()
    value_cols = [c for c in WEEKEND_COLUMNS if c.startswith(('fp', 'qual'))]
    assert placeholder[value_cols].isna().all().all()
