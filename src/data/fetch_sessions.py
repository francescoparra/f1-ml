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

WEEKEND_COLUMNS = (
    ['season', 'round', 'event_name', 'event_format', 'driver', 'team', 'qual_position']
    + [f'{fp.lower()}_{stat}' for fp in FP_SESSIONS for stat in ('best', 'laps')]
)


def enable_cache():
    """Enable the FastF1 on-disk cache (created if missing)."""
    os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


class SessionLoadError(Exception):
    """A session exists but could not be loaded (network error, rate limit, no data yet)."""


def load_session(season, round_num, name, retries=3):
    """
    Load one session with laps only (no telemetry/weather/messages, much faster).

    Returns None when the session does not exist for the event (e.g. FP2 on a
    sprint weekend). Transient failures such as API rate limits are retried
    with a growing delay, then raised as `SessionLoadError`, so a failed
    download is never confused with a session that does not exist.
    """
    try:
        session = fastf1.get_session(season, round_num, name)
    except ValueError:
        return None

    error = None
    for attempt in range(retries):
        try:
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            return session
        except Exception as e:
            error = e
            time.sleep(2 ** attempt)

    raise SessionLoadError(f"{name}: {error}")


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


def _best_laps(session):
    """Per-driver best lap time (seconds) and number of timed laps."""
    if session is None:
        return pd.DataFrame(columns=['best', 'laps'])
    try:
        laps = session.laps
    except (DataNotLoadedError, AttributeError):
        return pd.DataFrame(columns=['best', 'laps'])
    if laps is None or len(laps) == 0:
        return pd.DataFrame(columns=['best', 'laps'])

    seconds = laps['LapTime'].dt.total_seconds()
    grouped = seconds.groupby(laps['Driver'])
    return pd.DataFrame({'best': grouped.min(), 'laps': grouped.count()})


def build_weekend_table(season, round_num, event_name, event_format, q_session, fp_sessions):
    """
    Collapse the sessions of one race weekend into one row per driver.

    Only compact numbers are kept (qualifying position, best lap and lap count
    of each free practice), so the result can be cached as a small CSV and the
    feature stage does not depend on FastF1 objects.

    Parameters
    ----------
    season, round_num : int
    event_name, event_format : str
    q_session : fastf1.core.Session | None
        Qualifying. When missing or not run yet, `qual_position` is NaN
        (this is the case when predicting an upcoming weekend).
    fp_sessions : dict[str, fastf1.core.Session | None]
        Keys 'FP1', 'FP2', 'FP3'. Missing sessions produce NaN columns.

    Returns
    -------
    pd.DataFrame
        Columns listed in `WEEKEND_COLUMNS`.
    """
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
        'driver': people['Abbreviation'].to_numpy(),
        'team': people['TeamName'].to_numpy(),
    })

    positions = q_results.dropna(subset=['Abbreviation']).set_index('Abbreviation')['Position']
    table['qual_position'] = pd.to_numeric(table['driver'].map(positions), errors='coerce')

    for fp in FP_SESSIONS:
        laps = _best_laps(fp_sessions.get(fp))
        table[f'{fp.lower()}_best'] = table['driver'].map(laps['best'])
        table[f'{fp.lower()}_laps'] = table['driver'].map(laps['laps'])

    return table[WEEKEND_COLUMNS]


def _cache_path(season, round_num):
    return os.path.join(PROCESSED_DIR, f'{season}_{int(round_num):02d}.csv')


def fetch_weekend(season, round_num, event_name, event_format, refresh=False):
    """
    Fetch one race weekend as a per-driver table, using the processed cache.

    A weekend is written to `data/processed/weekends/` only when qualifying
    results are available AND every existing session loaded without errors,
    so a weekend in progress, or one hit by a rate limit, is never frozen
    half-downloaded.
    """
    path = _cache_path(season, round_num)
    if os.path.exists(path) and not refresh:
        cached = pd.read_csv(path)
        if set(WEEKEND_COLUMNS) <= set(cached.columns):
            return cached[WEEKEND_COLUMNS]

    failures = []

    def load(name):
        try:
            return load_session(season, round_num, name)
        except SessionLoadError as e:
            failures.append(str(e))
            return None

    q_session = load('Q')
    fp_sessions = {fp: load(fp) for fp in FP_SESSIONS}
    table = build_weekend_table(
        season, round_num, event_name, event_format, q_session, fp_sessions
    )

    if failures:
        print(f"{season} round {round_num} ({event_name}) incomplete, not cached: {failures}")
    elif table['qual_position'].notna().any():
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        table.to_csv(path, index=False)

    return table


def lineup_from_previous(weekends, season, round_num, event_name, event_format):
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
    target_event = None
    now = pd.Timestamp.now()

    for year in all_seasons:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"Skipping season {year}: failed to load schedule ({e})")
            continue

        events = []
        for _, event in schedule.iterrows():
            round_num = int(event['RoundNumber'])
            is_target = year == target_season and round_num == target_round
            if year == target_season and round_num > target_round:
                continue
            if is_target:
                target_event = (event['EventName'], event['EventFormat'])
            if not is_target and pd.Timestamp(event['EventDate']) > now:
                continue
            events.append((round_num, event['EventName'], event['EventFormat']))

        for round_num, name, event_format in tqdm(events, desc=f"Season {year}", unit="weekend"):
            table = fetch_weekend(year, round_num, name, event_format, refresh=refresh)
            if table.empty and (year, round_num) == (target_season, target_round):
                continue  # no data yet: the line-up is rebuilt below
            if table.empty:
                print(f"Skipping {year} round {round_num} ({name}): no data available")
                continue
            if table['qual_position'].isna().all() and not (
                year == target_season and round_num == target_round
            ):
                print(f"Skipping {year} round {round_num} ({name}): no qualifying results")
                continue
            tables.append(table)

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
