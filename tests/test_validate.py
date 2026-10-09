import pandas as pd

from src.data.build_features import build_feature_table, get_feature_columns
from src.models.predict_model import predict
from src.models.validate import walk_forward_validation

ALL_FP = {'use_fp1': True, 'use_fp2': True, 'use_fp3': True}
MODEL = {'type': 'xgboost', 'params': {'n_estimators': 10, 'max_depth': 2, 'random_state': 0}}
METRICS = ['mae', 'spearman', 'top3_accuracy']


def run(table, **kwargs):
    return walk_forward_validation(
        table, get_feature_columns(ALL_FP), MODEL, METRICS, **kwargs
    )


def test_evaluates_last_n_weekends_with_enough_history(weekends):
    table = build_feature_table(weekends, ALL_FP)
    per_weekend, summary = run(table, n_rounds=4, min_train_weekends=5)

    model_rows = per_weekend[per_weekend['method'] == 'model']
    assert len(model_rows) == 4
    assert model_rows['n_train_weekends'].min() >= 5
    assert set(per_weekend['method']) == {
        'model', 'baseline_fp_rank', 'baseline_driver_form', 'baseline_form_blend'
    }
    assert set(summary['method']) == set(per_weekend['method'])
    assert (summary['n_weekends'] == 4).all()


def test_only_earlier_weekends_are_used_for_training(weekends):
    table = build_feature_table(weekends, ALL_FP)
    per_weekend, _ = run(table, n_rounds=20, min_train_weekends=5)

    model_rows = per_weekend[per_weekend['method'] == 'model'].reset_index(drop=True)
    # weekend i is evaluated after training on exactly the i earlier weekends
    assert model_rows['n_train_weekends'].tolist() == list(range(5, 12))


def test_future_labels_do_not_change_a_past_score(weekends):
    altered = weekends.copy()
    last = (altered['season'] == 2025) & (altered['round'] == 6)
    altered.loc[last, 'qual_position'] = 1.0

    a, _ = run(build_feature_table(weekends, ALL_FP), n_rounds=20)
    b, _ = run(build_feature_table(altered, ALL_FP), n_rounds=20)

    keep = ~((a['season'] == 2025) & (a['round'] == 6))
    pd.testing.assert_frame_equal(
        a[keep].reset_index(drop=True), b[keep].reset_index(drop=True)
    )


def test_before_idx_excludes_the_target(weekends):
    table = build_feature_table(weekends, ALL_FP)
    per_weekend, _ = run(table, n_rounds=20, before_idx=8)

    assert per_weekend['n_train_weekends'].max() < 8


def test_not_enough_history_returns_empty(weekends):
    table = build_feature_table(weekends, ALL_FP)
    per_weekend, summary = run(table, min_train_weekends=50)

    assert per_weekend.empty and summary.empty


def test_model_beats_random_on_synthetic_data(weekends):
    table = build_feature_table(weekends, ALL_FP)
    _, summary = run(table, n_rounds=6)

    model = summary.set_index('method').loc['model']
    assert model['spearman'] > 0.5


def test_predict_returns_unique_ranks(weekends):
    table = build_feature_table(weekends, ALL_FP)
    cols = get_feature_columns(ALL_FP)
    train = table[table['weekend_idx'] < 10]
    test = table[table['weekend_idx'] == 10]

    from src.models.train_model import train_model
    model = train_model(train[cols], train['qual_position'], MODEL)
    result = predict(model, test[cols])

    assert sorted(result['predicted_rank']) == list(range(1, len(test) + 1))
    assert result.index.equals(test.index)


def test_fp_baseline_is_skipped_when_model_has_no_fp_features(weekends):
    table = build_feature_table(weekends, ALL_FP)
    cols = [c for c in get_feature_columns(ALL_FP) if 'fp_' not in c and not c.startswith('fp')]
    per_weekend, summary = walk_forward_validation(table, cols, MODEL, METRICS, n_rounds=3)

    assert set(per_weekend['method']) == {'model', 'baseline_driver_form', 'baseline_form_blend'}


def test_eval_sprint_scores_only_weekends_of_that_kind(weekends):
    table = build_feature_table(weekends, ALL_FP)
    kind = table.groupby('weekend_idx')['is_sprint_weekend'].first()

    sprint_only, _ = run(table, n_rounds=20, min_train_weekends=3, eval_sprint=1)
    normal_only, _ = run(table, n_rounds=20, min_train_weekends=3, eval_sprint=0)

    assert not sprint_only.empty and not normal_only.empty
    assert len(sprint_only[sprint_only['method'] == 'model']) == int(kind.iloc[3:].sum())
    # training still uses every earlier weekend, whatever its kind
    assert sprint_only['n_train_weekends'].max() > len(sprint_only['n_train_weekends'].unique())
