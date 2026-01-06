import pandas as pd
import numpy as np
from fastf1.core import DataNotLoadedError


def build_features(historical_sessions, target_session):
    """
    Build tabular features for baseline qualifying prediction.

    One row = one driver
    Target = qualifying position

    Parameters
    ----------
    historical_sessions : dict

    target_session : dict

    Returns
    -------
    X_train : pd.DataFrame
    y_train : np.ndarray
    X_test : pd.DataFrame
    y_test : np.ndarray
    """
    y_train, df_train = build_training_data(historical_sessions)

    constructor_strength, global_constructor_mean = compute_constructor_strength(
        historical_sessions, target_session.event.year
    )

    driver_avg_qual = (
        df_train
        .groupby('driver')['qual_position']
        .mean()
    )

    global_driver_mean = driver_avg_qual.mean()

    df_train['driver_avg_qual_pos'] = df_train['driver'].map(
        lambda d: driver_avg_qual.get(d, global_driver_mean)
    )

    for col in ['fp1_gap', 'fp2_gap', 'fp3_gap']:
        df_train[col] = df_train[col].fillna(df_train[col].mean())

    df_train['constructor_strength'] = df_train['constructor'].map(
        lambda c: constructor_strength.get(c, global_constructor_mean)
    )

    feature_cols = [
        'best_qual_time',
        'driver_avg_qual_pos',
        'constructor_strength',
        'fp1_gap',
        'fp2_gap',
        'fp3_gap',
    ]

    X_train = df_train[feature_cols]

    test_rows = []
    test_labels = []
    test_driver_ids = []

    q = target_session
    results = q.results[['Abbreviation', 'Position']].dropna()

    for driver in q.laps['Driver'].unique():
        if driver not in results['Abbreviation'].values:
            continue

        laps = q.laps.pick_drivers([driver])
        if laps.empty:
            continue

        best_lap = laps['LapTime'].min().total_seconds()
        constructor = laps['Team'].iloc[0]

        row = {
            'best_qual_time': best_lap,
            'driver_avg_qual_pos': driver_avg_qual.get(driver, global_driver_mean),
            'constructor_strength': constructor_strength.get(constructor, global_constructor_mean),
            # Phase 1: FP gaps unavailable → fallback to training means
            'fp1_gap': X_train['fp1_gap'].mean(),
            'fp2_gap': X_train['fp2_gap'].mean(),
            'fp3_gap': X_train['fp3_gap'].mean(),
        }

        qual_position = int(
            results.loc[results['Abbreviation'] == driver, 'Position'].iloc[0]
        )

        test_rows.append(row)
        test_labels.append(qual_position)
        test_driver_ids.append(driver)

    X_test = pd.DataFrame(test_rows)
    y_test = np.array(test_labels)

    return X_train, y_train, X_test, y_test, test_driver_ids


def build_training_data(historical_sessions):
    """
    Build training data for baseline qualifying prediction.

    Parameters
    ----------
    historical_sessions : dict

    Returns
    ----------
    X_train : pd.DataFrame
    y_train : np.ndarray
    """
    train_rows = []
    train_labels = []

    for s in historical_sessions:
        q = safe_load_laps(s['session'])
        if not hasattr(q, 'laps') or q.laps is None:
            q.load(laps=True, telemetry=False)

        fp1 = safe_load_laps(s.get("fp1"))
        fp2 = safe_load_laps(s.get("fp2"))
        fp3 = safe_load_laps(s.get("fp3"))

        for fp in (fp1, fp2, fp3):
            if fp is not None and (not hasattr(fp, 'laps') or fp.laps is None):
                fp.load(laps=True, telemetry=False)

        results = q.results[['Abbreviation', 'Position']].dropna()

        for driver in q.laps['Driver'].unique():
            if driver not in results['Abbreviation'].values:
                continue

            laps = q.laps.pick_drivers([driver])
            if laps.empty:
                continue

            best_lap = laps['LapTime'].min().total_seconds()
            constructor = laps['Team'].iloc[0]

            if pd.isna(constructor):
                continue

            qual_position = int(
                results.loc[results['Abbreviation'] == driver, 'Position'].iloc[0]
            )

            if not np.isfinite(qual_position):
                continue

            row = {
                'driver': driver,
                'constructor': constructor,
                'best_qual_time': best_lap,
            }

            # FP gap features
            for fp_name, fp_session in zip(["fp1", "fp2", "fp3"], [fp1, fp2, fp3]):
                gap_col = f"{fp_name}_gap"

                if fp_session is None:
                    row[gap_col] = np.nan
                    continue

                try:
                    fp_laps = fp_session.laps
                except DataNotLoadedError:
                    row[gap_col] = np.nan
                    continue

                driver_laps = fp_laps.pick_drivers([driver])
                if driver_laps.empty:
                    row[gap_col] = np.nan
                    continue

                driver_time = driver_laps["LapTime"].min().total_seconds()
                fastest = fp_laps["LapTime"].min().total_seconds()
                row[gap_col] = driver_time - fastest

            train_rows.append(row)
            train_labels.append(qual_position)

    df_train = pd.DataFrame(train_rows)
    y_train = np.array(train_labels)
    df_train['qual_position'] = y_train

    return y_train, df_train


def safe_load_laps(sess):
    """
    Safely load lap data for a FastF1 session.

    This helper ensures that lap data is loaded before access and prevents
    `DataNotLoadedError` when working with optional sessions (e.g. FP1/FP2/FP3).
    If the session is None or lap data cannot be accessed, the function returns None.

    Parameters
    ----------
    sess : fastf1.core.Session | None

    Returns
    ----------
    sess : fastf1.core.Session | None
    """
    if sess is None:
        return None
    sess.load(laps=True, telemetry=False)
    return sess if hasattr(sess, "laps") else None


def compute_constructor_strength(historical_sessions, target_year):
    """
    Compute constructor strength using recency-weighted historical qualifying positions.
    Weights:
        current=1.0, -1yr=0.8, -2yr=0.6, -3yr=0.4, -4yr+=0.2

    Parameters
    ----------
    historical_sessions : list[dict]

    target_year : int

    Returns
    ----------
    constructor_strength : dict[str, float]
        Mapping from constructor (team) name to its recency-weighted mean
        qualifying position. Lower values indicate stronger constructors.

    global_mean : float
        Global mean qualifying position across all constructors and seasons.
        Used as a fallback value for constructors with insufficient history.
    """
    from collections import defaultdict

    DECAY_WEIGHTS = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
    DEFAULT_WEIGHT = 0.2  # 4+ years ago

    constructor_season_positions = defaultdict(lambda: defaultdict(list))

    for session in historical_sessions:
        year = session['season']
        if year >= target_year:
            continue
        results = session['session'].results

        for _, entry in results.iterrows():
            constructor = entry['TeamName']
            qual_pos = entry['Position']

            if pd.notna(qual_pos):
                constructor_season_positions[constructor][year].append(int(qual_pos))

    constructor_strength = {}
    all_positions = []

    for constructor, seasons in constructor_season_positions.items():
        weighted_sum = 0.0
        weight_total = 0.0

        for year, positions in seasons.items():
            years_ago = target_year - year - 1
            weight = DECAY_WEIGHTS.get(years_ago, DEFAULT_WEIGHT)
            mean_pos = sum(positions) / len(positions)
            weighted_sum += weight * mean_pos
            weight_total += weight
            all_positions.extend(positions)

        if weight_total > 0:
            constructor_strength[constructor] = weighted_sum / weight_total

    global_mean = sum(all_positions) / len(all_positions) if all_positions else 10.0

    return constructor_strength, global_mean

