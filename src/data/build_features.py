import numpy as np
import pandas as pd

FP_NUMBERS = (1, 2, 3)

DEFAULT_RECENT_WINDOW = 12     # most recent races: the strongest signal
DEFAULT_OLDER_WINDOW = 12      # the 12 races before those: extra data, less weight
DEFAULT_OLDER_WEIGHT = 0.4
DEFAULT_CIRCUIT_SHRINKAGE = 2.0

# FastF1 reports the same franchise under different names across seasons.
TEAM_ALIASES = {
    'Audi': 'Sauber',
    'Scuderia Toro Rosso': 'RB',
    'Alfa Romeo': 'Sauber',
    'Alfa Romeo Racing': 'Sauber',
    'Kick Sauber': 'Sauber',
    'AlphaTauri': 'RB',
    'Scuderia AlphaTauri': 'RB',
    'Racing Bulls': 'RB',
    'Visa Cash App RB': 'RB',
    'Toro Rosso': 'RB',
    'Renault': 'Alpine',
    'Alpine F1 Team': 'Alpine',
    'Racing Point': 'Aston Martin',
    'Force India': 'Aston Martin',
    'Haas F1 Team': 'Haas',
    'Red Bull': 'Red Bull Racing',
}


# The same circuit appears under different FastF1 locations across seasons.
CIRCUIT_ALIASES = {
    'Monte Carlo': 'Monaco',
    'Miami Gardens': 'Miami',
    'Yas Marina': 'Yas Island',
    'Singapore': 'Marina Bay',     # 2019 location of the Singapore GP
}


def normalize_team(name):
    """Map a FastF1 team name to a stable franchise name."""
    return TEAM_ALIASES.get(name, name)


def normalize_circuit(name):
    """Map a FastF1 event location to a stable circuit name."""
    return CIRCUIT_ALIASES.get(name, name)


FORM_PREFIXES = ('driver', 'team')


def get_feature_columns(features_config):
    """
    Names of the model features for the given feature configuration.

    Every feature is available BEFORE the target qualifying starts: sessions
    of the same weekend that precede qualifying (free practice, sprint) and
    form/track history computed from earlier weekends.
    """
    use_fp = [features_config.get(f'use_fp{k}', True) for k in FP_NUMBERS]

    columns = [f'fp{k}_gap_pct' for k, use in zip(FP_NUMBERS, use_fp) if use]
    if any(use_fp):
        columns += [
            'fp_best_gap_pct',
            'fp_best_rank',
            'fp_soft_gap_pct',
            'fp_soft_rank',
            'team_fp_gap_pct',
            'fp_gap_vs_teammate',
        ]
    if features_config.get('use_sprint', True):
        columns += ['sq_gap_pct', 'sq_rank', 'sprint_position', 'sprint_gap_pct']

    for prefix in FORM_PREFIXES:
        columns += [
            f'{prefix}_last3',
            f'{prefix}_recent',
            f'{prefix}_older',
            f'{prefix}_form',
            f'{prefix}_n_prior',
        ]
    columns += [
        'form_blend',
        'driver_circuit_visits',
        'driver_circuit_mean',
        'driver_circuit_delta',
        'is_sprint_weekend',
    ]
    return columns


def available_feature_columns(target_rows, feature_cols):
    """
    Features the target weekend can actually provide.

    A feature that is NaN for every target driver (e.g. all FP features
    before FP1, or sprint features on a normal weekend) is dropped, and the
    model is trained without it. Training a model on features that are never
    present at prediction time would make it rely on a signal it will not get.
    """
    return [c for c in feature_cols if target_rows[c].notna().any()]


def _gap_pct(values, group):
    """Gap (%) of each value to the best (lowest) value of its group."""
    best = values.groupby(group).transform('min')
    return (values - best) / best * 100


def add_session_features(df, features_config):
    """
    Add pace features from sessions of the same weekend that precede qualifying.

    Gaps are percentages of the fastest lap of the session, so they are
    comparable between circuits. Disabled or missing sessions yield NaN, which
    XGBoost handles natively.
    """
    weekend = df['weekend_idx']
    nan = pd.Series(np.nan, index=df.index)

    best_cols, soft_cols = [], []
    for k in FP_NUMBERS:
        enabled = features_config.get(f'use_fp{k}', True)
        best = df[f'fp{k}_best'].astype(float) if enabled else nan
        soft = df[f'fp{k}_soft_best'].astype(float) if enabled else nan

        df[f'fp{k}_gap_pct'] = _gap_pct(best, weekend)
        best_cols.append(best)
        soft_cols.append(soft)

    fp_best = pd.concat(best_cols, axis=1).min(axis=1)
    df['fp_best_gap_pct'] = _gap_pct(fp_best, weekend)
    df['fp_best_rank'] = fp_best.groupby(weekend).rank(method='min')

    # Soft-tyre runs are the closest thing to a qualifying push lap in FP.
    fp_soft = pd.concat(soft_cols, axis=1).min(axis=1)
    df['fp_soft_gap_pct'] = _gap_pct(fp_soft, weekend)
    df['fp_soft_rank'] = fp_soft.groupby(weekend).rank(method='min')

    team_key = [weekend, df['team']]
    df['team_fp_gap_pct'] = df['fp_best_gap_pct'].groupby(team_key).transform('min')

    gap = df['fp_best_gap_pct']
    team_sum = gap.groupby(team_key).transform('sum')
    team_count = gap.groupby(team_key).transform('count')
    others_count = (team_count - gap.notna().astype(int)).replace(0, np.nan)
    others_mean = (team_sum - gap.fillna(0)) / others_count
    df['fp_gap_vs_teammate'] = gap - others_mean

    use_sprint = features_config.get('use_sprint', True)
    sq_best = df['sq_best'].astype(float) if use_sprint else nan
    sprint_best = df['sprint_best'].astype(float) if use_sprint else nan
    df['sq_gap_pct'] = _gap_pct(sq_best, weekend)
    df['sq_rank'] = sq_best.groupby(weekend).rank(method='min')
    df['sprint_gap_pct'] = _gap_pct(sprint_best, weekend)
    if not use_sprint:
        df['sprint_position'] = np.nan

    return df


def window_stats(values, recent, older, older_weight):
    """
    Rolling form statistics from STRICTLY earlier observations.

    For observation i only observations before i are used (NaN ignored):
    - `last3`: mean of the last 3 (momentum / streak)
    - `recent`: mean of the last `recent` observations (the strongest signal)
    - `older`: mean of the `older` observations before those
    - `form`: mean of the last `recent + older` observations where the older
      block counts `older_weight` times as much per observation
    - `n_prior`: how many observations back `form` rests on

    The window runs over the group's own observations, so it crosses season
    boundaries: round 4 of a season uses the last rounds of the previous one.

    Parameters
    ----------
    values : pd.Series
        Chronologically ordered observations of one driver or team.

    Returns
    -------
    pd.DataFrame
        Indexed like `values`.
    """
    data = values.to_numpy(dtype=float)
    rows = []
    for i in range(len(data)):
        prior = data[:i]
        prior = prior[~np.isnan(prior)]
        recent_vals = prior[-recent:] if recent else prior[:0]
        older_vals = prior[-(recent + older):-recent] if len(prior) > recent else prior[:0]

        weight = len(recent_vals) + older_weight * len(older_vals)
        rows.append({
            'last3': prior[-3:].mean() if len(prior) else np.nan,
            'recent': recent_vals.mean() if len(recent_vals) else np.nan,
            'older': older_vals.mean() if len(older_vals) else np.nan,
            'form': (
                (recent_vals.sum() + older_weight * older_vals.sum()) / weight if weight else np.nan
            ),
            'n_prior': len(recent_vals) + len(older_vals),
        })
    return pd.DataFrame(rows, index=values.index)


def _group_window_stats(df, key, value, recent, older, older_weight):
    """`window_stats` for every group of `key`, aligned with `df`'s index."""
    parts = [
        window_stats(group[value], recent, older, older_weight)
        for _, group in df.groupby(key, sort=False)
    ]
    return pd.concat(parts).reindex(df.index)


def add_form_features(df, features_config):
    """
    Add driver, team and track history from STRICTLY earlier weekends.

    Every statistic only sees earlier weekends, so the qualifying result being
    predicted can never leak into its own features. `df` must be sorted
    chronologically.

    - Driver and team form over the last `recent_window` races (strongest),
      plus the `older_window` races before those (weaker), ignoring season
      boundaries. This captures momentum and in-season car upgrades.
    - `form_blend`: expected position mixing driver and team form, trusting
      each one in proportion to how much history it has. A rookie therefore
      leans on the team, a brand-new team on its drivers.
    - Driver track record at the same circuit over ALL earlier seasons.
    """
    recent = int(features_config.get('recent_window', DEFAULT_RECENT_WINDOW))
    older = int(features_config.get('older_window', DEFAULT_OLDER_WINDOW))
    older_weight = float(features_config.get('older_weight', DEFAULT_OLDER_WEIGHT))
    shrinkage = float(features_config.get('circuit_shrinkage', DEFAULT_CIRCUIT_SHRINKAGE))
    if recent < 1 or older < 0:
        raise ValueError("features.recent_window must be >= 1 and features.older_window >= 0")

    driver = _group_window_stats(df, 'driver', 'qual_position', recent, older, older_weight)
    for column in driver.columns:
        df[f'driver_{column}'] = driver[column]

    team_weekend = (
        df.groupby(['team', 'weekend_idx'], sort=False)['qual_position']
        .mean()
        .reset_index()
        .sort_values(['team', 'weekend_idx'])
    )
    team = _group_window_stats(team_weekend, 'team', 'qual_position', recent, older, older_weight)
    team.columns = [f'team_{c}' for c in team.columns]
    team_weekend = pd.concat([team_weekend[['team', 'weekend_idx']], team], axis=1)
    df = df.merge(team_weekend, on=['team', 'weekend_idx'], how='left')

    reliability_driver = (df['driver_n_prior'].clip(upper=recent) / recent).fillna(0)
    reliability_team = (df['team_n_prior'].clip(upper=recent) / recent).fillna(0)
    total = (reliability_driver + reliability_team).replace(0, np.nan)
    df['form_blend'] = (
        reliability_driver * df['driver_form'].fillna(0)
        + reliability_team * df['team_form'].fillna(0)
    ) / total

    # How much better/worse than usual the driver is at this circuit
    df['_circuit_residual'] = df['qual_position'] - df['driver_form']
    track = df.groupby(['driver', 'circuit_key'], sort=False)
    prior_visits = track['qual_position'].transform(lambda s: s.shift(1).notna().cumsum())
    prior_residuals = track['_circuit_residual'].transform(lambda s: s.shift(1).notna().cumsum())
    prior_residual_sum = track['_circuit_residual'].transform(lambda s: s.shift(1).fillna(0).cumsum())

    df['driver_circuit_visits'] = prior_visits
    df['driver_circuit_mean'] = track['qual_position'].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df['driver_circuit_delta'] = (prior_residual_sum / (prior_residuals + shrinkage)).where(
        prior_residuals > 0
    )

    return df


def build_feature_table(weekends, features_config):
    """
    Turn the per-driver weekend table into a leak-free feature table.

    One row = one driver in one qualifying. Rows are ordered chronologically
    and carry a `weekend_idx` (0, 1, 2, ...) used to split train/test by time.
    The target weekend may have NaN `qual_position` (not run yet).

    Parameters
    ----------
    weekends : pd.DataFrame
        Output of `fetch_weekends`.
    features_config : dict
        `use_fp1/2/3`, `use_sprint`, `recent_window`, `older_window`,
        `older_weight`, `circuit_shrinkage`.

    Returns
    -------
    pd.DataFrame
        Identifier columns, `qual_position` (label) and the feature columns
        returned by `get_feature_columns`.
    """
    df = weekends.copy()
    df['team'] = df['team'].map(normalize_team)
    df['circuit_key'] = df['circuit'].map(normalize_circuit)
    df = df.sort_values(['season', 'round', 'driver']).reset_index(drop=True)

    df['weekend_idx'] = df.groupby(['season', 'round']).ngroup()
    df['is_sprint_weekend'] = (df['event_format'] != 'conventional').astype(int)

    df = add_session_features(df, features_config)
    df = add_form_features(df, features_config)

    id_cols = ['season', 'round', 'event_name', 'weekend_idx', 'driver', 'team', 'qual_position']
    return df[id_cols + get_feature_columns(features_config)]


def split_train_target(table, target_season, target_round):
    """
    Split the feature table into labelled training rows and the target weekend.

    Training rows are only weekends strictly before the target.

    Returns
    -------
    train : pd.DataFrame
    target : pd.DataFrame
        All drivers of the target weekend (labels may be NaN).
    """
    is_target = (table['season'] == target_season) & (table['round'] == target_round)
    if not is_target.any():
        raise ValueError(f"Target weekend {target_season} round {target_round} not found")

    target = table[is_target].copy()
    target_idx = target['weekend_idx'].iloc[0]
    train = table[(table['weekend_idx'] < target_idx) & table['qual_position'].notna()].copy()

    return train, target
