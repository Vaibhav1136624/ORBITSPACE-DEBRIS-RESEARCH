import numpy as np
import pandas as pd
import pytest
from debris_research.uncertainty import inner_split, conformal_radius, intervals, coverage_metrics, reliability_bins


def test_conformal_finite_sample_rank_and_domain():
    radius, rank = conformal_radius(np.arange(1., 11.))
    assert rank == 10 and radius == 10.
    lower, upper = intervals(np.array([-2., -20.]), radius)
    np.testing.assert_array_equal(lower, [-12., -30.])
    np.testing.assert_array_equal(upper, [0., -10.])
    m = coverage_metrics(np.array([-12., -9.]), lower, upper, np.array([True, True]))
    assert m['coverage'] == .5
    assert coverage_metrics(np.array([-12., -9.]), lower, upper, np.array([False, False]))['coverage'] is None
    with pytest.raises(ValueError, match='Insufficient'):
        conformal_radius([1.])


def test_inner_split_is_disjoint_and_order_independent():
    train = pd.DataFrame({'event_id': np.arange(100), 'partition': 'train', 'target_high_risk': np.arange(100)%5 == 0})
    fit, cal = inner_split(train)
    fit2, cal2 = inner_split(train.sample(frac=1, random_state=9))
    assert set(fit.event_id).isdisjoint(cal.event_id)
    assert len(fit) == 80 and len(cal) == 20
    assert set(fit.event_id) == set(fit2.event_id)
    assert set(cal.event_id) == set(cal2.event_id)
    with pytest.raises(ValueError, match='training'):
        inner_split(train.assign(partition='calibration'))


def test_reliability_bins_include_one_and_preserve_empty_bins():
    bins = reliability_bins(np.array([0, 1, 1]), np.array([0., .2, 1.]))
    assert [b['count'] for b in bins] == [1, 1, 0, 0, 1]
    assert bins[2]['observed_fraction'] is None
    assert sum(b['positives'] for b in bins) == 2
