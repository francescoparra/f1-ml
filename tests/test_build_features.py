import numpy as np
import pandas as pd
import pytest

from src.data.build_features import (
    build_feature_table,
    get_feature_columns,
    normalize_team,
    split_train_target,
)
from tests.conftest import make_weekends

ALL_FP = {'use_fp1': True, 'use_fp2': True, 'use_fp3': True}


def test_first_weekend_has_no_form_and_second_uses_only_first(weekends):
    table = build_feature_table(weekends, ALL_FP)

    first = table[table['weekend_idx'] == 0]
    second = table[table['weekend_idx'] == 1].set_index('driver')

    assert first['driver_prev_qual_mean'].isna().all()
    assert first['team_form_ewm'].isna().all()
    expected = first.set_index('driver')['qual_position']
    np.testing.assert_array_equal(
        second['driver_prev_qual_mean'].to_numpy(), expected.reindex(second.index).to_numpy()
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


def test_fp_gap_is_percentage_of_session_fastest_lap():
    df = pd.DataFrame({
        'season': 2024, 'round': 1, 'event_name': 'X', 'event_format': 'conventional',
        'driver': ['AAA', 'BBB'], 'team': ['T1', 'T2'], 'qual_position': [1.0, 2.0],
        'fp1_best': [100.0, 101.0], 'fp1_laps': [10, 10],
        'fp2_best': [np.nan, np.nan], 'fp2_laps': [np.nan, np.nan],
        'fp3_best': [np.nan, np.nan], 'fp3_laps': [np.nan, np.nan],
    })
    table = build_feature_table(df, ALL_FP).set_index('driver')

    assert table.loc['AAA', 'fp1_gap_pct'] == 0
    assert table.loc['BBB', 'fp1_gap_pct'] == pytest.approx(1.0)
    assert table.loc['AAA', 'fp_best_rank'] == 1
    assert table.loc['BBB', 'fp_best_rank'] == 2
    assert table['fp2_gap_pct'].isna().all()


def test_teammate_gap_is_relative_to_the_other_driver():
    df = pd.DataFrame({
        'season': 2024, 'round': 1, 'event_name': 'X', 'event_format': 'conventional',
        'driver': ['AAA', 'BBB'], 'team': ['T1', 'T1'], 'qual_position': [1.0, 2.0],
        'fp1_best': [100.0, 101.0], 'fp1_laps': [10, 10],
        'fp2_best': [np.nan] * 2, 'fp2_laps': [np.nan] * 2,
        'fp3_best': [np.nan] * 2, 'fp3_laps': [np.nan] * 2,
    })
    table = build_feature_table(df, ALL_FP).set_index('driver')

    assert table.loc['AAA', 'fp_gap_vs_teammate'] == pytest.approx(-1.0)
    assert table.loc['BBB', 'fp_gap_vs_teammate'] == pytest.approx(1.0)
    assert table['team_fp_gap_pct'].tolist() == [0.0, 0.0]


def test_disabled_fp_sessions_are_not_features_nor_used(weekends):
    none = {'use_fp1': False, 'use_fp2': False, 'use_fp3': False}
    cols = get_feature_columns(none)
    assert not any('fp' in c for c in cols)
    assert build_feature_table(weekends, none).columns.tolist()[-len(cols):] == cols

    only_fp1 = {'use_fp1': True, 'use_fp2': False, 'use_fp3': False}
    table = build_feature_table(weekends, only_fp1)
    fp1_only_rank = (
        weekends.assign(r=weekends.groupby(['season', 'round'])['fp1_best'].rank(method='min'))
        .sort_values(['season', 'round', 'driver'])['r']
        .to_numpy()
    )
    np.testing.assert_array_equal(table['fp_best_rank'].to_numpy(), fp1_only_rank)
    assert 'fp2_gap_pct' not in table.columns


def test_team_aliases_and_sprint_flag(weekends):
    assert normalize_team('Kick Sauber') == normalize_team('Alfa Romeo') == 'Sauber'
    assert normalize_team('Ferrari') == 'Ferrari'

    table = build_feature_table(weekends, ALL_FP)
    sprint = table.loc[table['weekend_idx'] == 4, 'is_sprint_weekend']
    assert (sprint == 1).all()
    assert (table.loc[table['weekend_idx'] == 0, 'is_sprint_weekend'] == 0).all()


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
    from src.data.build_features import available_feature_columns

    weekends = make_weekends(last_unlabeled=True)
    last = (weekends['season'] == 2025) & (weekends['round'] == 6)
    weekends.loc[last, [c for c in weekends if c.startswith('fp')]] = np.nan  # before FP1

    table = build_feature_table(weekends, ALL_FP)
    _, target = split_train_target(table, 2025, 6)
    cols = available_feature_columns(target, get_feature_columns(ALL_FP))

    assert not any('fp' in c for c in cols)
    assert {'driver_prev_qual_mean', 'driver_last3_qual_mean', 'team_form_ewm'} <= set(cols)


def test_partial_fp_keeps_available_sessions():
    from src.data.build_features import available_feature_columns

    weekends = make_weekends(last_unlabeled=True)
    last = (weekends['season'] == 2025) & (weekends['round'] == 6)
    weekends.loc[last, ['fp2_best', 'fp3_best']] = np.nan  # only FP1 has run

    table = build_feature_table(weekends, ALL_FP)
    _, target = split_train_target(table, 2025, 6)
    cols = available_feature_columns(target, get_feature_columns(ALL_FP))

    assert 'fp1_gap_pct' in cols and 'fp_best_rank' in cols
    assert 'fp2_gap_pct' not in cols and 'fp3_gap_pct' not in cols
