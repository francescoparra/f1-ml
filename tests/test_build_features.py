import numpy as np
import pandas as pd
import pytest

from src.data.build_features import (
    available_feature_columns,
    build_feature_table,
    get_feature_columns,
    normalize_circuit,
    normalize_team,
    split_train_target,
    window_stats,
)
from tests.conftest import make_weekends

ALL_FP = {'use_fp1': True, 'use_fp2': True, 'use_fp3': True}
SPEC = {'recent_window': 12, 'older_window': 12, 'older_weight': 0.4}


def one_weekend(**columns):
    """Single-weekend frame with the columns the session features need."""
    n = len(columns['driver'])
    base = {
        'season': 2024, 'round': 1, 'event_name': 'X', 'event_format': 'conventional',
        'circuit': 'Monaco', 'qual_position': list(range(1, n + 1)),
        'sq_best': [np.nan] * n, 'sprint_position': [np.nan] * n, 'sprint_best': [np.nan] * n,
    }
    for k in (1, 2, 3):
        base[f'fp{k}_best'] = [np.nan] * n
        base[f'fp{k}_laps'] = [np.nan] * n
        base[f'fp{k}_soft_best'] = [np.nan] * n
    base.update(columns)
    return pd.DataFrame(base)


# ---------------------------------------------------------------- window_stats

def test_window_stats_uses_only_earlier_observations():
    stats = window_stats(pd.Series(np.arange(1.0, 31.0)), 12, 12, 0.4)

    assert stats.iloc[0].isna().drop('n_prior').all()
    assert stats['n_prior'].iloc[0] == 0
    # observation 30 (value 31 would be next): prior = 1..29... check index 29
    row = stats.iloc[29]               # prior = values 1..29
    assert row['last3'] == pytest.approx(np.mean([27, 28, 29]))
    assert row['recent'] == pytest.approx(np.mean(range(18, 30)))
    assert row['older'] == pytest.approx(np.mean(range(6, 18)))
    expected_form = (sum(range(18, 30)) + 0.4 * sum(range(6, 18))) / (12 + 0.4 * 12)
    assert row['form'] == pytest.approx(expected_form)
    assert row['n_prior'] == 24


def test_window_stats_short_history_and_nan_are_ignored():
    stats = window_stats(pd.Series([5.0, np.nan, 7.0, 9.0]), 12, 12, 0.4)

    assert stats['recent'].iloc[3] == pytest.approx(6.0)   # (5 + 7) / 2, NaN skipped
    assert np.isnan(stats['older'].iloc[3])
    assert stats['form'].iloc[3] == pytest.approx(6.0)
    assert stats['n_prior'].iloc[3] == 2


def test_older_races_weigh_less_than_recent_ones():
    # 12 recent races at 2 and 12 older races at 20: form is far closer to 2
    stats = window_stats(pd.Series([20.0] * 12 + [2.0] * 12 + [np.nan]), 12, 12, 0.4)
    form = stats['form'].iloc[-1]
    assert form == pytest.approx((12 * 2 + 0.4 * 12 * 20) / (12 + 0.4 * 12))
    assert form < (2 + 20) / 2


def test_window_crosses_season_boundaries(weekends):
    table = build_feature_table(weekends, ALL_FP)
    first_of_second_season = table[(table['season'] == 2025) & (table['round'] == 1)]

    assert (first_of_second_season['driver_n_prior'] == 6).all()
    assert first_of_second_season['driver_recent'].notna().all()


def test_invalid_window_raises(weekends):
    with pytest.raises(ValueError, match='recent_window'):
        build_feature_table(weekends, {**ALL_FP, 'recent_window': 0})


# ------------------------------------------------------------------- leakage

def test_first_weekend_has_no_form_and_second_uses_only_first(weekends):
    table = build_feature_table(weekends, ALL_FP)

    first = table[table['weekend_idx'] == 0]
    second = table[table['weekend_idx'] == 1].set_index('driver')

    assert first['driver_form'].isna().all()
    assert first['team_form'].isna().all()
    expected = first.set_index('driver')['qual_position']
    np.testing.assert_array_equal(
        second['driver_recent'].to_numpy(), expected.reindex(second.index).to_numpy()
    )


def test_target_label_never_leaks_into_its_own_features():
    labeled = make_weekends(last_unlabeled=False)
    unlabeled = make_weekends(last_unlabeled=True)

    cols = get_feature_columns(ALL_FP)
    a = build_feature_table(labeled, ALL_FP)
    b = build_feature_table(unlabeled, ALL_FP)

    last = a['weekend_idx'].max()
    pd.testing.assert_frame_equal(
        a.loc[a['weekend_idx'] == last, cols].reset_index(drop=True),
        b.loc[b['weekend_idx'] == last, cols].reset_index(drop=True),
    )


def test_changing_future_labels_does_not_change_past_features(weekends):
    altered = weekends.copy()
    future = (altered['season'] == 2025) & (altered['round'] >= 4)
    altered.loc[future, 'qual_position'] = 1.0

    cols = get_feature_columns(ALL_FP)
    a = build_feature_table(weekends, ALL_FP)
    b = build_feature_table(altered, ALL_FP)

    cutoff = a.loc[(a['season'] == 2025) & (a['round'] == 4), 'weekend_idx'].iloc[0]
    pd.testing.assert_frame_equal(
        a.loc[a['weekend_idx'] <= cutoff, cols], b.loc[b['weekend_idx'] <= cutoff, cols]
    )


# ------------------------------------------------------------ session features

def test_fp_gap_is_percentage_of_session_fastest_lap():
    df = one_weekend(
        driver=['AAA', 'BBB'], team=['T1', 'T2'], fp1_best=[100.0, 101.0], fp1_laps=[10, 10]
    )
    table = build_feature_table(df, ALL_FP).set_index('driver')

    assert table.loc['AAA', 'fp1_gap_pct'] == 0
    assert table.loc['BBB', 'fp1_gap_pct'] == pytest.approx(1.0)
    assert table.loc['AAA', 'fp_best_rank'] == 1
    assert table.loc['BBB', 'fp_best_rank'] == 2
    assert table['fp2_gap_pct'].isna().all()


def test_soft_tyre_gap_only_compares_soft_laps():
    df = one_weekend(
        driver=['AAA', 'BBB', 'CCC'], team=['T1', 'T2', 'T3'],
        fp1_best=[99.0, 100.0, 101.0],
        fp1_soft_best=[np.nan, 100.0, 100.5],  # AAA's best lap was on another tyre
    )
    table = build_feature_table(df, ALL_FP).set_index('driver')

    assert table['fp_soft_gap_pct'].isna().tolist() == [True, False, False]
    assert table.loc['BBB', 'fp_soft_gap_pct'] == 0
    assert table.loc['CCC', 'fp_soft_gap_pct'] == pytest.approx(0.5)
    assert table.loc['CCC', 'fp_soft_rank'] == 2
    assert table.loc['AAA', 'fp_best_rank'] == 1


def test_teammate_gap_is_relative_to_the_other_driver():
    df = one_weekend(
        driver=['AAA', 'BBB'], team=['T1', 'T1'], fp1_best=[100.0, 101.0], fp1_laps=[10, 10]
    )
    table = build_feature_table(df, ALL_FP).set_index('driver')

    assert table.loc['AAA', 'fp_gap_vs_teammate'] == pytest.approx(-1.0)
    assert table.loc['BBB', 'fp_gap_vs_teammate'] == pytest.approx(1.0)
    assert table['team_fp_gap_pct'].tolist() == [0.0, 0.0]


def test_disabled_fp_sessions_are_not_features_nor_used(weekends):
    none = {'use_fp1': False, 'use_fp2': False, 'use_fp3': False}
    cols = get_feature_columns(none)
    assert not any('fp' in c.split('_') for c in cols)
    assert 'fp_best_rank' not in cols and 'fp_soft_rank' not in cols

    only_fp1 = {'use_fp1': True, 'use_fp2': False, 'use_fp3': False}
    table = build_feature_table(weekends, only_fp1)
    fp1_only_rank = (
        weekends.assign(r=weekends.groupby(['season', 'round'])['fp1_best'].rank(method='min'))
        .sort_values(['season', 'round', 'driver'])['r']
        .to_numpy()
    )
    np.testing.assert_array_equal(table['fp_best_rank'].to_numpy(), fp1_only_rank)
    assert 'fp2_gap_pct' not in table.columns


# -------------------------------------------------------------- sprint features

def test_sprint_features_exist_only_on_sprint_weekends(weekends):
    table = build_feature_table(weekends, ALL_FP)
    sprint = table[table['weekend_idx'] == 4]
    normal = table[table['weekend_idx'] == 0]

    assert (sprint['is_sprint_weekend'] == 1).all()
    assert sprint[['sq_gap_pct', 'sq_rank', 'sprint_position', 'sprint_gap_pct']].notna().all().all()
    assert normal[['sq_gap_pct', 'sq_rank', 'sprint_position', 'sprint_gap_pct']].isna().all().all()
    assert sprint['sq_gap_pct'].min() == 0
    assert sorted(sprint['sq_rank']) == list(range(1, 7))


def test_sprint_features_can_be_disabled(weekends):
    cols = get_feature_columns({**ALL_FP, 'use_sprint': False})
    assert not {'sq_gap_pct', 'sq_rank', 'sprint_position', 'sprint_gap_pct'} & set(cols)

    table = build_feature_table(weekends, {**ALL_FP, 'use_sprint': False})
    assert 'sq_gap_pct' not in table.columns


# ------------------------------------------------------------ circuit features

def test_circuit_history_counts_only_earlier_visits_to_the_same_track():
    weekends = make_weekends(n_weekends=9)       # circuits repeat every 4 weekends
    table = build_feature_table(weekends, ALL_FP)

    monaco = table[table['driver'] == 'AAA'].set_index('weekend_idx')
    assert monaco.loc[0, 'driver_circuit_visits'] == 0
    assert np.isnan(monaco.loc[0, 'driver_circuit_mean'])
    assert np.isnan(monaco.loc[0, 'driver_circuit_delta'])
    assert monaco.loc[1, 'driver_circuit_visits'] == 0   # first time at Baku
    assert monaco.loc[4, 'driver_circuit_visits'] == 1   # Monaco again
    assert monaco.loc[8, 'driver_circuit_visits'] == 2
    assert monaco.loc[4, 'driver_circuit_mean'] == monaco.loc[0, 'qual_position']
    assert monaco.loc[8, 'driver_circuit_mean'] == pytest.approx(
        monaco.loc[[0, 4], 'qual_position'].mean()
    )


def test_circuit_delta_is_shrunk_residual_against_own_form():
    n = 30
    rows = []
    for w in range(n):
        rows.append(one_weekend(
            driver=['AAA', 'BBB'], team=['T1', 'T2'],
            season=[2024 + w // 12] * 2, round=[w % 12 + 1] * 2,
            circuit=['Monaco' if w in (20, 29) else 'Baku'] * 2,
            # AAA is a 5th-place driver, except at Monaco where he always starts 1st
            qual_position=[1.0 if w in (20, 29) else 5.0, 2.0],
        ))
    table = build_feature_table(pd.concat(rows, ignore_index=True), {**ALL_FP, 'circuit_shrinkage': 1.0})

    aaa = table[table['driver'] == 'AAA'].set_index('weekend_idx')
    first_visit_residual = 1.0 - aaa.loc[20, 'driver_form']
    assert first_visit_residual < -3                                   # much better than usual
    assert aaa.loc[29, 'driver_circuit_visits'] == 1
    # 1 prior visit, shrinkage 1 -> half of the residual
    assert aaa.loc[29, 'driver_circuit_delta'] == pytest.approx(first_visit_residual / 2)


def test_circuit_and_team_aliases(weekends):
    assert normalize_team('Kick Sauber') == normalize_team('Alfa Romeo') == normalize_team('Audi') == 'Sauber'
    assert normalize_team('Ferrari') == 'Ferrari'
    assert normalize_circuit('Monte Carlo') == normalize_circuit('Monaco') == 'Monaco'
    assert normalize_circuit('Miami Gardens') == 'Miami'
    assert normalize_circuit('Singapore') == normalize_circuit('Marina Bay') == 'Marina Bay'
    assert normalize_circuit('Kuala Lumpur') == normalize_circuit('Sakhir') == 'Sakhir'


# ------------------------------------------------------------------ form blend

def test_rookie_leans_on_team_and_new_team_leans_on_driver():
    rows = []
    for w in range(14):
        rows.append(one_weekend(
            driver=['VET', 'ROO', 'NEW'], team=['Old', 'Old', 'Fresh' if w >= 13 else 'Old'],
            season=[2024] * 3, round=[w + 1] * 3,
            # ROO (rookie) only races from the last weekend on
            qual_position=[2.0, 4.0 if w == 13 else np.nan, 12.0],
        ))
    table = build_feature_table(pd.concat(rows, ignore_index=True), ALL_FP)
    last = table[table['weekend_idx'] == 13].set_index('driver')

    rookie = last.loc['ROO']
    assert np.isnan(rookie['driver_form']) and rookie['driver_n_prior'] == 0
    assert rookie['form_blend'] == pytest.approx(rookie['team_form'])      # team only

    newcomer = last.loc['NEW']
    assert newcomer['team_n_prior'] == 0 and np.isnan(newcomer['team_form'])
    assert newcomer['form_blend'] == pytest.approx(newcomer['driver_form'])  # driver only

    veteran = last.loc['VET']
    low, high = sorted([veteran['driver_form'], veteran['team_form']])
    assert low <= veteran['form_blend'] <= high


def test_blend_is_nan_without_any_history(weekends):
    table = build_feature_table(weekends, ALL_FP)
    assert table.loc[table['weekend_idx'] == 0, 'form_blend'].isna().all()


# ------------------------------------------------------------------- splitting

def test_split_train_target_only_uses_earlier_labeled_weekends():
    weekends = make_weekends(last_unlabeled=True)
    table = build_feature_table(weekends, ALL_FP)
    last = weekends.loc[weekends['qual_position'].isna(), ['season', 'round']].iloc[0]

    train, target = split_train_target(table, last['season'], last['round'])

    assert len(target) == 6
    assert target['qual_position'].isna().all()
    assert train['weekend_idx'].max() < target['weekend_idx'].iloc[0]
    assert train['qual_position'].notna().all()


def test_split_unknown_target_raises(weekends):
    table = build_feature_table(weekends, ALL_FP)
    with pytest.raises(ValueError, match='not found'):
        split_train_target(table, 1999, 1)


def test_features_missing_for_the_whole_target_are_dropped():
    weekends = make_weekends(last_unlabeled=True)
    last = (weekends['season'] == 2025) & (weekends['round'] == 6)
    weekends.loc[last, [c for c in weekends if c.startswith('fp')]] = np.nan  # before FP1

    table = build_feature_table(weekends, ALL_FP)
    _, target = split_train_target(table, 2025, 6)
    cols = available_feature_columns(target, get_feature_columns(ALL_FP))

    assert not any(c.startswith(('fp', 'team_fp')) or c == 'fp_gap_vs_teammate' for c in cols)
    assert {'driver_form', 'team_form', 'form_blend', 'driver_circuit_mean'} <= set(cols)
    assert 'sq_gap_pct' not in cols                    # conventional weekend


def test_partial_fp_keeps_available_sessions():
    weekends = make_weekends(last_unlabeled=True)
    last = (weekends['season'] == 2025) & (weekends['round'] == 6)
    weekends.loc[last, ['fp2_best', 'fp3_best', 'fp2_soft_best', 'fp3_soft_best']] = np.nan

    table = build_feature_table(weekends, ALL_FP)
    _, target = split_train_target(table, 2025, 6)
    cols = available_feature_columns(target, get_feature_columns(ALL_FP))

    assert 'fp1_gap_pct' in cols and 'fp_best_rank' in cols and 'fp_soft_gap_pct' in cols
    assert 'fp2_gap_pct' not in cols and 'fp3_gap_pct' not in cols


def test_sprint_target_before_and_after_sprint_qualifying():
    weekends = make_weekends(n_weekends=15, last_unlabeled=True)   # weekend 14 is a sprint
    key = tuple(weekends[['season', 'round']].iloc[-1])
    last = (weekends['season'] == key[0]) & (weekends['round'] == key[1])

    before = weekends.copy()
    before.loc[last, ['sq_best', 'sprint_position', 'sprint_best']] = np.nan   # only FP1 so far
    cols = get_feature_columns(ALL_FP)

    _, target = split_train_target(build_feature_table(before, ALL_FP), *key)
    assert not {'sq_gap_pct', 'sprint_position'} & set(available_feature_columns(target, cols))

    after = weekends.copy()
    after.loc[last, ['sprint_position', 'sprint_best']] = np.nan               # SQ done, sprint not
    _, target = split_train_target(build_feature_table(after, ALL_FP), *key)
    available = set(available_feature_columns(target, cols))
    assert {'sq_gap_pct', 'sq_rank'} <= available
    assert 'sprint_position' not in available
