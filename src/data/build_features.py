import numpy as np
import pandas as pd

FP_NUMBERS = (1, 2, 3)

# FastF1 reports the same franchise under different names across seasons.
TEAM_ALIASES = {
    'Audi': 'Sauber',
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


def normalize_team(name):
    """Map a FastF1 team name to a stable franchise name."""
    return TEAM_ALIASES.get(name, name)


def get_feature_columns(features_config):
    """
    Names of the model features for the given feature configuration.

    Every feature is available BEFORE the target qualifying starts: free
    practice data of the same weekend and form computed from earlier weekends.
    """
    use_fp = [features_config.get(f'use_fp{k}', True) for k in FP_NUMBERS]

    columns = [f'fp{k}_gap_pct' for k, use in zip(FP_NUMBERS, use_fp) if use]
    if any(use_fp):
        columns += [
            'fp_best_gap_pct',
            'fp_best_rank',
            'team_fp_gap_pct',
            'fp_gap_vs_teammate',
        ]
    columns += [
        'driver_prev_qual_mean',
        'driver_last3_qual_mean',
        'team_form_ewm',
        'is_sprint_weekend',
    ]
    return columns


def available_feature_columns(target_rows, feature_cols):
    """
    Features the target weekend can actually provide.

    A feature that is NaN for every target driver (e.g. all FP features
    before FP1) is dropped, and the model is trained without it. Training a
    model on features that are never present at prediction time would make it
    rely on a signal it will not get.
    """
    return [c for c in feature_cols if target_rows[c].notna().any()]


def add_fp_features(df, features_config):
    """
    Add free practice pace features, computed inside each race weekend.

    Gaps are percentages of the fastest lap of the session, so they are
    comparable between circuits. Disabled or missing sessions yield NaN, which
    XGBoost handles natively.
    """
    weekend = df['weekend_idx']
    best_cols = []

    for k in FP_NUMBERS:
        best = df[f'fp{k}_best'].astype(float)
        if not features_config.get(f'use_fp{k}', True):
            best = pd.Series(np.nan, index=df.index)

        session_best = best.groupby(weekend).transform('min')
        df[f'fp{k}_gap_pct'] = (best - session_best) / session_best * 100
        best_cols.append(best)

    fp_best = pd.concat(best_cols, axis=1).min(axis=1)
    overall_best = fp_best.groupby(weekend).transform('min')
    df['fp_best_gap_pct'] = (fp_best - overall_best) / overall_best * 100
    df['fp_best_rank'] = fp_best.groupby(weekend).rank(method='min')

    team_key = [weekend, df['team']]
    df['team_fp_gap_pct'] = df['fp_best_gap_pct'].groupby(team_key).transform('min')

    gap = df['fp_best_gap_pct']
    team_sum = gap.groupby(team_key).transform('sum')
    team_count = gap.groupby(team_key).transform('count')
    others_count = (team_count - gap.notna().astype(int)).replace(0, np.nan)
    others_mean = (team_sum - gap.fillna(0)) / others_count
    df['fp_gap_vs_teammate'] = gap - others_mean

    return df


def add_form_features(df, form_halflife):
    """
    Add driver and team form from STRICTLY earlier weekends.

    Every statistic is shifted by one weekend before aggregating, so the
    qualifying result being predicted can never leak into its own features.
    `df` must be sorted chronologically.
    """
    by_driver = df.groupby('driver')['qual_position']

    df['driver_prev_qual_mean'] = by_driver.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df['driver_last3_qual_mean'] = by_driver.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    team_weekend = (
        df.groupby(['team', 'weekend_idx'], sort=True)['qual_position']
        .mean()
        .reset_index()
        .sort_values(['team', 'weekend_idx'])
    )
    team_weekend['team_form_ewm'] = team_weekend.groupby('team')['qual_position'].transform(
        lambda s: s.shift(1).ewm(halflife=form_halflife, min_periods=1).mean()
    )

    return df.merge(
        team_weekend[['team', 'weekend_idx', 'team_form_ewm']],
        on=['team', 'weekend_idx'],
        how='left',
    )


def build_feature_table(weekends, features_config, form_halflife=6.0):
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
        `use_fp1`, `use_fp2`, `use_fp3` flags.
    form_halflife : float
        Half-life, in weekends, of the exponentially weighted team form.

    Returns
    -------
    pd.DataFrame
        Identifier columns, `qual_position` (label) and the feature columns
        returned by `get_feature_columns`.
    """
    df = weekends.copy()
    df['team'] = df['team'].map(normalize_team)
    df = df.sort_values(['season', 'round', 'driver']).reset_index(drop=True)

    df['weekend_idx'] = df.groupby(['season', 'round']).ngroup()

    df['is_sprint_weekend'] = (df['event_format'] != 'conventional').astype(int)

    df = add_fp_features(df, features_config)
    df = add_form_features(df, form_halflife)

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
