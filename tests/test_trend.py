import numpy as np
import pandas as pd
from debris_research.trend import trend_score, quality_groups, group_recall


def test_trend_uses_one_day_and_falls_back_without_history():
    frame = pd.DataFrame({'current_log_risk': [-8., -2., -7., -6., -5.],
                          'risk_slope_per_day': [2., 4., 100., 100., np.nan],
                          'history_reports': [3, 3, 1, 2, 3],
                          'history_span_days': [2., 2., 0., 0., 1.]})
    score, fallback = trend_score(frame)
    np.testing.assert_array_equal(score, [-6., 0., -7., -6., -5.])
    np.testing.assert_array_equal(fallback, [False, False, True, True, True])
    frame['target_final_log_risk'] = 0.
    np.testing.assert_array_equal(trend_score(frame)[0], score)


def test_quality_boundaries_and_absent_positive_group():
    frame = pd.DataFrame({'history_reports': [2, 3], 'latest_report_age_days': [1., 1.01],
                          'context_missing_fraction': [0., .1]})
    groups = quality_groups(frame)
    np.testing.assert_array_equal(groups['sparse_history_1_or_2_reports'], [True, False])
    np.testing.assert_array_equal(groups['latest_report_over_1_day_old'], [False, True])
    absent = group_recall(np.array([0, 1]), np.array([3, 4]), np.array([2, 1]), np.array([True, False]))
    assert absent['recall'] is None
    result = group_recall(np.array([0, 1]), np.array([3, 4]), np.array([2, 1]), np.array([False, True]))
    assert result['recall'] == .25
    assert result['missed_positive_appearances'] == 3
