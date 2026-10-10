import numpy as np
import pandas as pd
import pytest
from debris_research.analyze import paired_bootstrap, encounter_queue_counts


def test_paired_bootstrap_identical_models_have_zero_difference():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([.1, .8, .7, .9])
    result = paired_bootstrap(labels, {'latest_reported_risk': scores, 'copy': scores}, 100)
    assert result['copy']['paired_ap_difference_percentile_95'] == [0., 0.]
    assert result == paired_bootstrap(labels, {'latest_reported_risk': scores, 'copy': scores}, 100)


def test_encounter_counts_preserve_queue_ties_and_repeated_appearances():
    p = pd.DataFrame({'event_id': [10, 20, 30], 'score': [.8, .8, .1]})
    appearances, selected = encounter_queue_counts(p, np.array([[20, 10], [10, 30]]), 1, 'score')
    np.testing.assert_array_equal(appearances, [2, 1, 1])
    np.testing.assert_array_equal(selected, [1, 1, 0])
    with pytest.raises(ValueError, match='unknown'):
        encounter_queue_counts(p, np.array([[10, 99]]), 1, 'score')
    with pytest.raises(ValueError, match='Duplicate'):
        encounter_queue_counts(p, np.array([[10, 10]]), 1, 'score')
