import os
import time

import fastf1
import numpy as np
import pandas as pd
from fastf1.core import DataNotLoadedError
from tqdm import tqdm

FASTF1_CACHE_DIR = 'data/raw/fastf1_cache'
PROCESSED_DIR = 'data/processed/weekends'
FP_SESSIONS = ('FP1', 'FP2', 'FP3')

BACKOFF_SECONDS = 5  # first wait after a throttled/failed request, then it grows
SPRINT_SESSIONS = ('SQ', 'SS', 'S')  # sprint qualifying, 2023 sprint shootout, sprint

WEEKEND_COLUMNS = (
    ['season', 'round', 'event_name', 'event_format', 'circuit', 'driver', 'team', 'qual_position']
    + ['sq_best', 'sprint_position', 'sprint_best']
    + [
        f'{fp.lower()}_{stat}'
        for fp in FP_SESSIONS
        for stat in ('best', 'laps', 'soft_best')
    ]
)


def enable_cache():
    """Enable the FastF1 on-disk cache (created if missing)."""
    os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


class SessionLoadError(Exception):
    """A session exists but could not be loaded (network error, rate limit, no data yet)."""


def _has_data(session, name):
    """Whether a loaded session actually holds results (qualifying) or laps/results (others)."""
    try:
        results = len(session.results) > 0
    except Exception:
        results = False
    if name == 'Q':
        return results
    try:
        laps = len(session.laps) > 0
    except Exception:
        laps = False
    return laps or results


def load_session(season, round_num, name, retries=4, before=None):
    """
    Load one session with laps only (no telemetry/weather/messages, much faster).

    Returns None when the session does not exist for the event (e.g. FP2 on a
    sprint weekend), or, if `before` is given, when it does not start before
    that timestamp (used to ignore sessions held after qualifying).

    FastF1 does not raise when the API throttles a request: it logs a warning
    and returns an empty session. A session that took place (more than a few
    hours ago) but came back empty is therefore retried with a growing delay. If it is still empty
    the session is accepted as it is (e.g. a free practice cancelled by rain),
    except for qualifying, which is raised as `SessionLoadError` because a
    weekend without results is useless. Sessions that are running, just finished or not started
    yet (the target weekend) are returned as they are, without retrying.
    """
    error = None
    for attempt in range(retries):
        try:
            session = fastf1.get_session(season, round_num, name)
        except ValueError:
            return None

        if before is not None and not _starts_before(session, before):
            return None

        try:
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as e:
            error = e
        else:
            if _has_data(session, name) or not _is_settled(session):
                return session
            error = None
            last_empty = session

        if attempt < retries - 1:
            time.sleep(BACKOFF_SECONDS * 2 ** attempt)

    if error is not None:
        raise SessionLoadError(f"{name}: {error}")
    if name == 'Q':
        raise SessionLoadError("Q: no data returned")
    return last_empty


def load_schedule(year, retries=3):
    """
    Load the event schedule of a season, retrying transient failures.

    A season that cannot be loaded raises `SessionLoadError` instead of being
    skipped: silently training on fewer seasons would change the results
    without any warning.
    """
    error = None
    for attempt in range(retries):
        try:
            return fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            error = e
            time.sleep(BACKOFF_SECONDS * 3 ** attempt)
    raise SessionLoadError(f"Could not load the {year} schedule: {error}")


def _results(session):
    """Abbreviation / TeamName / Position of a session, empty if unavailable."""
    columns = ['Abbreviation', 'TeamName', 'Position']
    if session is None:
        return pd.DataFrame(columns=columns)
    try:
        results = session.results
    except Exception:
        return pd.DataFrame(columns=columns)
    if results is None or len(results) == 0:
        return pd.DataFrame(columns=columns)
    return results[columns].copy()


def _is_settled(session, grace_hours=6):
    """
    True if the session started more than `grace_hours` ago.

    Timing data takes a while to be published after a session ends, so a
    session that just finished (or is running) is not expected to have data yet
    and must not be retried as if the API had failed.
    """
    start = getattr(session, 'date', None)
    if start is None or pd.isna(start):
        return False
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)  # FastF1 dates are UTC
    return pd.Timestamp(start) + pd.Timedelta(hours=grace_hours) < now


def _starts_before(session, moment):
    """True only if the session has a known start strictly before `moment`."""
    start = getattr(session, 'date', None)
    if start is None or moment is None or pd.isna(start) or pd.isna(moment):
        return False
    return pd.Timestamp(start) < pd.Timestamp(moment)


def _best_laps(session):
    """
    Per-driver best lap time (seconds), number of timed laps and best lap on
    the SOFT compound (the tyre used for a qualifying push lap).
    """
    empty = pd.DataFrame(columns=['best', 'laps', 'soft_best'])
    if session is None:
        return empty
    try:
        laps = session.laps
    except (DataNotLoadedError, AttributeError):
        return empty
    if laps is None or len(laps) == 0:
        return empty

    seconds = laps['LapTime'].dt.total_seconds()
    grouped = seconds.groupby(laps['Driver'])
    result = pd.DataFrame({'best': grouped.min(), 'laps': grouped.count()})

    if 'Compound' in laps.columns:
        soft = seconds[laps['Compound'].astype(str).str.upper() == 'SOFT']
        result['soft_best'] = soft.groupby(laps.loc[soft.index, 'Driver']).min()
    else:
        result['soft_best'] = np.nan

    return result


def build_weekend_table(
    season, round_num, event_name, event_format, circuit, q_session, fp_sessions, sprint_sessions=None
):
    """
    Collapse the sessions of one race weekend into one row per driver.

    Only compact numbers are kept (qualifying position, best lap and lap count
    of each free practice, best lap on soft tyres, sprint qualifying lap times
    and sprint result), so the
    result can be cached as a small CSV and the feature stage does not depend
    on FastF1 objects.

    Parameters
    ----------
    season, round_num : int
    event_name, event_format, circuit : str
        `circuit` is the FastF1 event location (e.g. 'Marina Bay').
    q_session : fastf1.core.Session | None
        Qualifying. When missing or not run yet, `qual_position` is NaN
        (this is the case when predicting an upcoming weekend).
    fp_sessions : dict[str, fastf1.core.Session | None]
        Keys 'FP1', 'FP2', 'FP3'. Missing sessions produce NaN columns.
    sprint_sessions : dict[str, fastf1.core.Session | None] | None
        Keys 'SQ', 'SS', 'S'. Only sessions that start BEFORE qualifying are
        used: in the 2021-2023 formats the sprint was held after qualifying and
        using it would leak the future.

    Returns
    -------
    pd.DataFrame
        Columns listed in `WEEKEND_COLUMNS`.
    """
    q_start = getattr(q_session, 'date', None)
    sprint_sessions = sprint_sessions or {}
    usable_sprints = {
        key: session for key, session in sprint_sessions.items()
        if session is not None and _starts_before(session, q_start)
    }
    sprint_qualifying = usable_sprints.get('SQ') or usable_sprints.get('SS')
    sprint_race = usable_sprints.get('S')

    q_results = _results(q_session)
    fp_results = [_results(fp_sessions.get(fp)) for fp in FP_SESSIONS]

    frames = [f for f in [q_results] + fp_results if not f.empty]
    people = pd.concat(frames, ignore_index=True) if frames else q_results
    people = people.dropna(subset=['Abbreviation', 'TeamName'])
    people = people.drop_duplicates('Abbreviation')  # qualifying team name wins

    if q_results['Abbreviation'].notna().any():
        people = people[people['Abbreviation'].isin(q_results['Abbreviation'])]

    table = pd.DataFrame({
        'season': season,
        'round': round_num,
        'event_name': event_name,
        'event_format': event_format,
        'circuit': circuit,
        'driver': people['Abbreviation'].to_numpy(),
        'team': people['TeamName'].to_numpy(),
    })

    positions = q_results.dropna(subset=['Abbreviation']).set_index('Abbreviation')['Position']
    table['qual_position'] = pd.to_numeric(table['driver'].map(positions), errors='coerce')

    # FastF1 has no classification for sprint qualifying: only its lap times are kept
    table['sq_best'] = table['driver'].map(_best_laps(sprint_qualifying)['best'])

    sprint_results = _results(sprint_race).dropna(subset=['Abbreviation'])
    sprint_positions = sprint_results.set_index('Abbreviation')['Position']
    table['sprint_position'] = pd.to_numeric(table['driver'].map(sprint_positions), errors='coerce')
    table['sprint_best'] = table['driver'].map(_best_laps(sprint_race)['best'])

    for fp in FP_SESSIONS:
        laps = _best_laps(fp_sessions.get(fp))
        table[f'{fp.lower()}_best'] = table['driver'].map(laps['best'])
        table[f'{fp.lower()}_laps'] = table['driver'].map(laps['laps'])
        table[f'{fp.lower()}_soft_best'] = table['driver'].map(laps['soft_best'])

    return table[WEEKEND_COLUMNS]


def _cache_path(season, round_num):
    return os.path.join(PROCESSED_DIR, f'{season}_{int(round_num):02d}.csv')


def fetch_weekend(season, round_num, event_name, event_format, circuit, refresh=False):
    """
    Fetch one race weekend as a per-driver table, using the processed cache.

    A weekend is written to `data/processed/weekends/` only when qualifying
    results are available AND every existing session loaded without errors,
    so a weekend in progress, or one hit by a rate limit, is never frozen
    half-downloaded. Cache files written by an older version (missing
    columns) are downloaded again.
    """
    path = _cache_path(season, round_num)
    if os.path.exists(path) and not refresh:
        cached = pd.read_csv(path)
        if set(WEEKEND_COLUMNS) <= set(cached.columns):
            return cached[WEEKEND_COLUMNS]

    failures = []

    def load(name, before=None):
        try:
            return load_session(season, round_num, name, before=before)
        except SessionLoadError as e:
            failures.append(str(e))
            return None

    q_session = load('Q')
    fp_sessions = {fp: load(fp) for fp in FP_SESSIONS}

    sprint_sessions = {}
    if event_format != 'conventional' and q_session is not None:
        for name in SPRINT_SESSIONS:
            sprint_sessions[name] = load(name, before=q_session.date)

    table = build_weekend_table(
        season, round_num, event_name, event_format, circuit,
        q_session, fp_sessions, sprint_sessions,
    )

    if failures:
        print(f"{season} round {round_num} ({event_name}) incomplete, not cached: {failures}")
    elif table['qual_position'].notna().any():
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        table.to_csv(path, index=False)

    return table


def lineup_from_previous(weekends, season, round_num, event_name, event_format, circuit):
    """
    Placeholder rows for a weekend with no session data at all yet.

    Before FP1 there is nothing to read the entry list from, so the drivers and
    teams of the most recent weekend are reused. Qualifying and FP values are
    NaN, so only form-based features can be computed for these rows.
    """
    last = weekends.sort_values(['season', 'round'])[['season', 'round']].iloc[-1]
    previous = weekends[(weekends['season'] == last['season']) & (weekends['round'] == last['round'])]

    table = previous[['driver', 'team']].copy()
    table['season'] = season
    table['round'] = round_num
    table['event_name'] = event_name
    table['event_format'] = event_format
    table['circuit'] = circuit
    for column in WEEKEND_COLUMNS:
        if column not in table:
            table[column] = np.nan
    return table[WEEKEND_COLUMNS]


def fetch_weekends(seasons, target_config, refresh=False):
    """
    Fetch every race weekend needed to train and predict the target session.

    The target season is always included. Weekends after the target round, and
    events that have not happened yet, are never fetched, so the pipeline
    cannot see the future.

    Parameters
    ----------
    seasons : list[int]
        Historical seasons used for training.
    target_config : dict
        Keys 'season' and 'round' of the qualifying to predict.
    refresh : bool
        Ignore the processed cache and download everything again.

    Returns
    -------
    pd.DataFrame
        One row per driver per weekend (see `build_weekend_table`).

    Raises
    ------
    ValueError
        If no weekend could be loaded or the target weekend is not in the schedule.
    """
    enable_cache()

    target_season = int(target_config['season'])
    target_round = int(target_config['round'])
    all_seasons = sorted({int(s) for s in seasons if int(s) <= target_season} | {target_season})

    tables = []
    skipped = []
    target_event = None
    now = pd.Timestamp.now()

    # Schedules are small and are loaded up front: asking for them in the middle
    # of a long download is when the API throttles the most.
    schedules = {year: load_schedule(year) for year in all_seasons}

    for year in all_seasons:
        schedule = schedules[year]

        events = []
        for _, event in schedule.iterrows():
            round_num = int(event['RoundNumber'])
            is_target = year == target_season and round_num == target_round
            if year == target_season and round_num > target_round:
                continue
            if is_target:
                target_event = (event['EventName'], event['EventFormat'], event['Location'])
            if not is_target and pd.Timestamp(event['EventDate']) > now:
                continue
            events.append((round_num, event['EventName'], event['EventFormat'], event['Location']))

        for round_num, name, event_format, location in tqdm(
            events, desc=f"Season {year}", unit="weekend"
        ):
            table = fetch_weekend(year, round_num, name, event_format, location, refresh=refresh)
            if table.empty and (year, round_num) == (target_season, target_round):
                continue  # no data yet: the line-up is rebuilt below
            is_target_event = (year, round_num) == (target_season, target_round)
            if table.empty or (table['qual_position'].isna().all() and not is_target_event):
                skipped.append(f"{year} R{round_num} {name}")
                continue
            tables.append(table)

    if skipped:
        print(
            f"WARNING: {len(skipped)} past weekend(s) have no qualifying data and are missing "
            f"from the training set (cancelled, or the API failed: rerun to retry): {skipped}"
        )

    if not tables:
        raise ValueError("No sessions were found")

    weekends = pd.concat(tables, ignore_index=True)

    is_target = (weekends['season'] == target_season) & (weekends['round'] == target_round)
    if not is_target.any() and target_event is not None:
        print(
            f"No session data for {target_season} round {target_round} yet: "
            "using the latest known line-up. The prediction will rely on form only "
            "until free practice has run."
        )
        placeholder = lineup_from_previous(weekends, target_season, target_round, *target_event)
        weekends = pd.concat([weekends, placeholder], ignore_index=True)
        is_target = (weekends['season'] == target_season) & (weekends['round'] == target_round)

    if not is_target.any():
        raise ValueError(
            f"Target weekend {target_season} round {target_round} could not be loaded"
        )

    return weekends
