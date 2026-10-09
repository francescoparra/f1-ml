import numpy as np
import pandas as pd

from src.data import fetch_sessions
from src.data.fetch_sessions import WEEKEND_COLUMNS, build_weekend_table

Q_START = pd.Timestamp('2024-04-20 07:00')


class FakeSession:
    def __init__(self, results=None, laps=None, date=None):
        self.date = date
        if results is not None:
            self.results = pd.DataFrame(results, columns=['Abbreviation', 'TeamName', 'Position'])
        if laps is not None:
            self.laps = pd.DataFrame(
                [(row[0], pd.Timedelta(seconds=row[1]) if row[1] is not None else pd.NaT,
                  row[2] if len(row) > 2 else 'MEDIUM') for row in laps],
                columns=['Driver', 'LapTime', 'Compound'],
            )


def table_for(q=None, fps=None, sprints=None, event_format='conventional'):
    return build_weekend_table(
        2024, 3, 'Test GP', event_format, 'Marina Bay', q, fps or {}, sprints
    )


def test_weekend_table_collects_positions_best_laps_and_circuit():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)])
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)],
        laps=[('VER', 91.5), ('VER', 90.2), ('VER', None), ('LEC', 90.9)],
    )

    table = table_for(q, {'FP1': fp1, 'FP2': None, 'FP3': None})

    assert list(table.columns) == WEEKEND_COLUMNS
    ver = table.set_index('driver').loc['VER']
    assert ver['qual_position'] == 1
    assert ver['circuit'] == 'Marina Bay'
    assert ver['fp1_best'] == 90.2
    assert ver['fp1_laps'] == 2
    assert np.isnan(ver['fp2_best'])


def test_soft_best_only_counts_soft_laps():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)])
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)],
        laps=[('VER', 90.0, 'HARD'), ('VER', 91.0, 'SOFT'), ('VER', 90.5, 'SOFT'),
              ('LEC', 90.2, 'MEDIUM')],
    )

    table = table_for(q, {'FP1': fp1}).set_index('driver')

    assert table.loc['VER', 'fp1_best'] == 90.0
    assert table.loc['VER', 'fp1_soft_best'] == 90.5
    assert np.isnan(table.loc['LEC', 'fp1_soft_best'])


def test_upcoming_weekend_without_qualifying_uses_fp_drivers_and_has_nan_labels():
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', np.nan), ('LEC', 'Ferrari', np.nan)],
        laps=[('VER', 90.0), ('LEC', 90.5)],
    )

    table = table_for(None, {'FP1': fp1})

    assert sorted(table['driver']) == ['LEC', 'VER']
    assert table['qual_position'].isna().all()
    assert table['fp1_best'].notna().all()


def test_fp_only_drivers_are_dropped_when_qualifying_exists():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0)])
    fp1 = FakeSession(
        results=[('VER', 'Red Bull Racing', 1.0), ('ROO', 'Ferrari', 2.0)],
        laps=[('VER', 90.0), ('ROO', 89.0)],
    )

    assert table_for(q, {'FP1': fp1})['driver'].tolist() == ['VER']


def test_sprint_sessions_before_qualifying_are_used():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0), ('LEC', 'Ferrari', 2.0)], date=Q_START)
    sq = FakeSession(
        results=[('VER', 'Red Bull Racing', np.nan), ('LEC', 'Ferrari', np.nan)],
        laps=[('VER', 95.6), ('LEC', 95.8)], date=Q_START - pd.Timedelta(days=1),
    )
    sprint = FakeSession(
        results=[('VER', 'Red Bull Racing', 2.0), ('LEC', 'Ferrari', 1.0)],
        laps=[('VER', 100.4), ('LEC', 100.1)], date=Q_START - pd.Timedelta(hours=3),
    )

    table = table_for(q, sprints={'SQ': sq, 'SS': None, 'S': sprint}).set_index('driver')

    assert table.loc['VER', 'sq_best'] == 95.6
    assert table.loc['VER', 'sprint_position'] == 2
    assert table.loc['LEC', 'sprint_best'] == 100.1


def test_sprint_sessions_after_qualifying_are_ignored():
    # 2021-2023 formats: the sprint came after qualifying, using it would leak the result
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0)], date=Q_START)
    shootout = FakeSession(
        results=[('VER', 'Red Bull Racing', np.nan)], laps=[('VER', 95.0)],
        date=Q_START + pd.Timedelta(hours=2),
    )
    sprint = FakeSession(
        results=[('VER', 'Red Bull Racing', 1.0)], laps=[('VER', 100.0)],
        date=Q_START + pd.Timedelta(hours=6),
    )

    table = table_for(q, sprints={'SS': shootout, 'S': sprint})

    assert table[['sq_best', 'sprint_position', 'sprint_best']].isna().all().all()


def test_sprint_session_without_a_known_date_is_ignored():
    q = FakeSession(results=[('VER', 'Red Bull Racing', 1.0)], date=Q_START)
    sprint = FakeSession(results=[('VER', 'Red Bull Racing', 1.0)], laps=[('VER', 100.0)], date=None)

    assert table_for(q, sprints={'S': sprint})['sprint_position'].isna().all()


def test_completed_weekends_are_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sessions, 'PROCESSED_DIR', str(tmp_path))
    calls = []

    def fake_load(season, round_num, name, before=None):
        calls.append(name)
        if name == 'Q':
            return FakeSession(results=[('VER', 'Red Bull Racing', 1.0)], date=Q_START)
        return None

    monkeypatch.setattr(fetch_sessions, 'load_session', fake_load)

    first = fetch_sessions.fetch_weekend(2024, 1, 'GP', 'conventional', 'Monaco')
    n_calls = len(calls)
    second = fetch_sessions.fetch_weekend(2024, 1, 'GP', 'conventional', 'Monaco')

    assert len(calls) == n_calls
    assert first['driver'].tolist() == second['driver'].tolist() == ['VER']
    assert (tmp_path / '2024_01.csv').exists()


def test_sprint_sessions_are_only_requested_on_sprint_weekends(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sessions, 'PROCESSED_DIR', str(tmp_path))
    requested = []

    def fake_load(season, round_num, name, before=None):
        requested.append((name, before))
        if name == 'Q':
            return FakeSession(results=[('VER', 'Red Bull Racing', 1.0)], date=Q_START)
        return None

    monkeypatch.setattr(fetch_sessions, 'load_session', fake_load)

    fetch_sessions.fetch_weekend(2024, 1, 'GP', 'conventional', 'Monaco')
    assert not any(name in ('SQ', 'SS', 'S') for name, _ in requested)

    requested.clear()
    fetch_sessions.fetch_weekend(2024, 2, 'GP', 'sprint_qualifying', 'Shanghai')
    sprint_requests = [(n, b) for n, b in requested if n in ('SQ', 'SS', 'S')]
    assert [n for n, _ in sprint_requests] == ['SQ', 'SS', 'S']
    assert all(before == Q_START for _, before in sprint_requests)


def test_stale_cache_files_missing_columns_are_downloaded_again(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sessions, 'PROCESSED_DIR', str(tmp_path))
    pd.DataFrame({'season': [2024], 'round': [1], 'driver': ['OLD']}).to_csv(
        tmp_path / '2024_01.csv', index=False
    )
    monkeypatch.setattr(
        fetch_sessions, 'load_session',
        lambda s, r, n, before=None: FakeSession(
            results=[('VER', 'Red Bull Racing', 1.0)], date=Q_START
        ) if n == 'Q' else None,
    )

    table = fetch_sessions.fetch_weekend(2024, 1, 'GP', 'conventional', 'Monaco')

    assert table['driver'].tolist() == ['VER']
    assert set(WEEKEND_COLUMNS) <= set(pd.read_csv(tmp_path / '2024_01.csv').columns)


def test_weekends_without_qualifying_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sessions, 'PROCESSED_DIR', str(tmp_path))
    fp1 = FakeSession(results=[('VER', 'Red Bull Racing', np.nan)], laps=[('VER', 90.0)])
    monkeypatch.setattr(
        fetch_sessions, 'load_session',
        lambda s, r, n, before=None: fp1 if n == 'FP1' else None,
    )

    table = fetch_sessions.fetch_weekend(2026, 1, 'GP', 'conventional', 'Monaco')

    assert len(table) == 1
    assert list(tmp_path.iterdir()) == []


def test_lineup_placeholder_reuses_latest_weekend_with_nan_data():
    from tests.conftest import make_weekends

    weekends = make_weekends()
    placeholder = fetch_sessions.lineup_from_previous(
        weekends, 2030, 1, 'Future GP', 'sprint_qualifying', 'Marina Bay'
    )

    latest = weekends[(weekends['season'] == 2025) & (weekends['round'] == 6)]
    assert list(placeholder.columns) == WEEKEND_COLUMNS
    assert sorted(placeholder['driver']) == sorted(latest['driver'])
    assert (placeholder['season'] == 2030).all()
    assert (placeholder['event_format'] == 'sprint_qualifying').all()
    assert (placeholder['circuit'] == 'Marina Bay').all()
    value_cols = [c for c in WEEKEND_COLUMNS if c.startswith(('fp', 'qual', 'sq', 'sprint'))]
    assert placeholder[value_cols].isna().all().all()


def test_schedule_failure_is_retried_then_raised_not_skipped(monkeypatch):
    import pytest

    attempts = []

    def failing(year, include_testing=False):
        attempts.append(year)
        raise RuntimeError('network down')

    monkeypatch.setattr(fetch_sessions.fastf1, 'get_event_schedule', failing)
    monkeypatch.setattr(fetch_sessions.time, 'sleep', lambda s: None)

    with pytest.raises(fetch_sessions.SessionLoadError, match='2020'):
        fetch_sessions.load_schedule(2020)
    assert len(attempts) == 3


def test_schedule_recovers_after_a_transient_failure(monkeypatch):
    calls = []

    def flaky(year, include_testing=False):
        calls.append(year)
        if len(calls) < 2:
            raise RuntimeError('rate limited')
        return 'schedule'

    monkeypatch.setattr(fetch_sessions.fastf1, 'get_event_schedule', flaky)
    monkeypatch.setattr(fetch_sessions.time, 'sleep', lambda s: None)

    assert fetch_sessions.load_schedule(2021) == 'schedule'


class LoadableSession(FakeSession):
    """FakeSession whose load() fills in the data only from the n-th attempt on."""

    def __init__(self, ok_from, results, date, counter):
        super().__init__(date=date)
        self._ok_from, self._data, self._counter = ok_from, results, counter

    def load(self, **kwargs):
        self._counter.append(1)
        if len(self._counter) >= self._ok_from:
            self.results = pd.DataFrame(self._data, columns=['Abbreviation', 'TeamName', 'Position'])
        else:
            self.results = pd.DataFrame(columns=['Abbreviation', 'TeamName', 'Position'])


def patch_sessions(monkeypatch, ok_from, date):
    counter = []
    monkeypatch.setattr(
        fetch_sessions.fastf1, 'get_session',
        lambda *a: LoadableSession(ok_from, [('VER', 'Red Bull Racing', 1.0)], date, counter),
    )
    monkeypatch.setattr(fetch_sessions.time, 'sleep', lambda s: None)
    return counter


def test_throttled_empty_session_is_retried_until_it_has_data(monkeypatch):
    counter = patch_sessions(monkeypatch, ok_from=3, date=pd.Timestamp('2020-01-01'))

    session = fetch_sessions.load_session(2020, 1, 'Q')

    assert len(counter) == 3
    assert len(session.results) == 1


def test_qualifying_that_stays_empty_raises(monkeypatch):
    import pytest

    patch_sessions(monkeypatch, ok_from=99, date=pd.Timestamp('2020-01-01'))

    with pytest.raises(fetch_sessions.SessionLoadError, match='no data'):
        fetch_sessions.load_session(2020, 1, 'Q')


def test_practice_that_stays_empty_is_accepted_after_retries(monkeypatch):
    counter = patch_sessions(monkeypatch, ok_from=99, date=pd.Timestamp('2020-01-01'))

    session = fetch_sessions.load_session(2020, 1, 'FP2')

    assert session is not None and len(counter) == 4   # e.g. cancelled by rain


def test_future_sessions_are_not_retried(monkeypatch):
    counter = patch_sessions(monkeypatch, ok_from=99, date=pd.Timestamp.now() + pd.Timedelta(days=2))

    session = fetch_sessions.load_session(2030, 1, 'Q')

    assert session is not None and len(counter) == 1


def test_missing_session_type_returns_none(monkeypatch):
    def no_session(*a):
        raise ValueError('Session type does not exist')

    monkeypatch.setattr(fetch_sessions.fastf1, 'get_session', no_session)

    assert fetch_sessions.load_session(2024, 5, 'FP2') is None


def test_session_that_just_finished_is_not_retried(monkeypatch):
    # timing data is published with a delay: empty right after the session is normal
    counter = patch_sessions(monkeypatch, ok_from=99, date=pd.Timestamp.now(tz='UTC').tz_localize(None) - pd.Timedelta(hours=1))

    session = fetch_sessions.load_session(2030, 1, 'FP1')

    assert session is not None and len(counter) == 1
