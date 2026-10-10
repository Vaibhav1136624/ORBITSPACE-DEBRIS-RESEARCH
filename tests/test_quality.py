import numpy as np
import pandas as pd
from debris_research.data import CONTEXT
from debris_research.train import GROUPS
from debris_research.quality import assess


class Classifier:
    def decision_function(self, frame):
        assert np.isfinite(frame.to_numpy()).all()
        return np.zeros(len(frame))


class Calibrator:
    def predict_proba(self, scores):
        return np.tile([.9, .1], (len(scores), 1))


def fixture():
    frame = pd.DataFrame(0., index=[3, 7, 8], columns=GROUPS['plus_context'])
    frame['event_id'] = [10, 20, 30]
    frame['current_log_risk'] = -6.
    frame['history_reports'] = 2.
    return frame, {'features': GROUPS['plus_context'], 'classifier': Classifier(), 'sigmoid': Calibrator()}


def test_missing_and_stale_are_retained_without_scores():
    frame, bundle = fixture()
    frame.loc[7, CONTEXT] = np.nan
    frame.loc[7, 'context_missing_fraction'] = 1.
    frame.loc[8, 'latest_report_age_days'] = 1.01
    result = assess(frame, bundle)
    assert result.event_id.tolist() == [10, 20, 30]
    assert result.manual_review_required.tolist() == [False, True, True]
    assert result.model_score.iloc[0] == .1
    assert result.model_score.iloc[1:].isna().all()
    assert result.reported_log_risk.eq(-6.).all()
    assert 'missing_or_invalid_context' in result.reason_codes.iloc[1]


def test_invalid_risk_withholds_score_and_untrusted_value():
    frame, bundle = fixture()
    frame['current_log_risk'] = [np.nan, 1., -6.]
    frame.loc[8, 'latest_report_age_days'] = 1.
    result = assess(frame, bundle)
    assert result.reported_risk_valid.tolist() == [False, False, True]
    assert result.reported_log_risk.iloc[:2].isna().all()
    assert result.model_score.iloc[:2].isna().all()
    assert result.status.iloc[2] == 'available'
