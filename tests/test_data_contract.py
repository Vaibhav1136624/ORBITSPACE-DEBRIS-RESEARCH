import numpy as np
import pandas as pd
import pytest
from debris_research.data import Experiment, audit_and_clean, build_cohort, feature_columns, history_features, make_partitions
from debris_research.baseline import evaluate_baselines, queue_metrics


def encounter(event_id=1):
    return pd.DataFrame({
        'event_id': [event_id] * 4,
        'time_to_tca': [4.0, 2.0, 1.5, 0.5],
        'risk': [-12.0, -10.0, -8.0, -5.0],
        'miss_distance': [900.0, 800.0, 600.0, 300.0],
    })


def test_future_reports_cannot_change_features():
    original = encounter()
    modified = original.copy()
    modified.loc[modified.time_to_tca < 2, ['risk', 'miss_distance']] = [-2, 1]
    a, _ = build_cohort(original, Experiment())
    b, _ = build_cohort(modified, Experiment())
    pd.testing.assert_frame_equal(a[feature_columns(a)], b[feature_columns(b)])
    assert a.target_final_log_risk.iloc[0] != b.target_final_log_risk.iloc[0]


def test_cutoff_includes_exactly_48_hours_and_slope_uses_elapsed_time():
    features = history_features(encounter(), 2.0)
    assert features['history_reports'] == 2
    assert features['current_log_risk'] == -10
    assert features['risk_slope_per_day'] == pytest.approx(1)
    assert features['latest_report_age_days'] == 0


def test_uneven_observation_gaps_are_not_treated_as_equal():
    event = pd.DataFrame({'time_to_tca': [7., 6., 2.], 'risk': [-10., -9., -5.]})
    assert history_features(event, 2.)['risk_slope_per_day'] == pytest.approx(1.)


def test_threshold_is_in_log10_units():
    event = encounter()
    event.loc[event.time_to_tca == .5, 'risk'] = -6.
    cohort, _ = build_cohort(event, Experiment())
    assert cohort.target_high_risk.iloc[0] == 1
    event.loc[event.time_to_tca == .5, 'risk'] = -6.001
    cohort, _ = build_cohort(event, Experiment())
    assert cohort.target_high_risk.iloc[0] == 0


def test_ineligible_histories_and_final_reports_are_excluded():
    late = encounter(2)
    late['time_to_tca'] = [1.9, 1.8, 1.5, .5]
    early_final = encounter(3)
    early_final['time_to_tca'] = [5., 4., 3., 1.]
    cohort, audit = build_cohort(pd.concat([encounter(), late, early_final]), Experiment())
    assert cohort.event_id.tolist() == [1]
    assert audit['no_history_at_cutoff'] == 1
    assert audit['no_final_report_within_horizon'] == 1


def test_invalid_final_report_excludes_event_without_relabelling():
    bad = encounter(2)
    bad.loc[bad.time_to_tca == .5, 'risk'] = np.nan
    clean, audit = audit_and_clean(pd.concat([encounter(), bad]))
    assert clean.event_id.unique().tolist() == [1]
    assert audit['encounters_excluded_invalid_core_values'] == 1


def test_duplicates_do_not_inflate_history_and_conflicts_are_not_arbitrarily_ordered():
    original = encounter()
    clean, audit = audit_and_clean(pd.concat([original, original.iloc[[1]]]))
    assert len(clean) == 4
    assert audit['exact_duplicate_rows_removed'] == 1
    conflicting = original.iloc[[1]].copy()
    conflicting['risk'] = -9.
    clean, audit = audit_and_clean(pd.concat([original, conflicting, encounter(2)]))
    assert clean.event_id.unique().tolist() == [2]
    assert audit['encounters_excluded_conflicting_same_time_reports'] == 1


def test_partition_manifest_is_disjoint_complete_and_reproducible():
    cohort = pd.DataFrame({'event_id': np.arange(400), 'target_high_risk': np.arange(400) % 2})
    a = make_partitions(cohort, 17)
    b = make_partitions(cohort.sample(frac=1, random_state=9), 17)
    pd.testing.assert_frame_equal(a, b)
    assert a.event_id.nunique() == len(cohort) == len(a)
    assert a.partition.value_counts().to_dict() == {'train': 240, 'evaluation': 60, 'validation': 60, 'calibration': 40}


def test_target_and_identifiers_never_become_features():
    cohort, _ = build_cohort(encounter(), Experiment())
    features = feature_columns(cohort)
    assert all(not name.startswith('target_') for name in features)
    assert not set(features) & {'event_id', 'mission_id', 'final_time_to_tca'}


def test_queue_scores_have_known_answer():
    metrics = queue_metrics(np.array([1, 0, 1, 0]), np.array([.9, .8, .7, .1]), np.array([[0, 1, 2, 3]]), 2)
    assert metrics['recall_at_k'] == .5
    assert metrics['precision_at_k'] == .5
    assert metrics['mean_missed_high_risk_per_queue'] == 1


def test_no_positive_queue_has_undefined_recall_not_perfect_recall():
    metrics = queue_metrics(np.array([0, 0]), np.array([.9, .1]), np.array([[0, 1]]), 1)
    assert metrics['recall_at_k'] is None
    assert metrics['precision_at_k'] == 0


def test_baseline_refuses_evaluation_partition():
    with pytest.raises(ValueError, match='validation partition'):
        evaluate_baselines(pd.DataFrame({'partition': ['evaluation']}), Experiment())
