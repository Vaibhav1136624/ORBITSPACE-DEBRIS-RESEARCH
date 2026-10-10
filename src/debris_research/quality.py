"""Prototype availability gate; unavailable encounters are retained for review."""
import numpy as np
import pandas as pd
from .data import CONTEXT
from .train import HISTORY

POLICY_VERSION = 'prototype-quality-v1'


def assess(frame, bundle):
    if frame.event_id.isna().any() or frame.event_id.duplicated().any():
        raise ValueError('Present, unique encounter IDs required')
    features = frame.reindex(columns=bundle['features']).apply(pd.to_numeric, errors='coerce')
    reasons = [[] for _ in range(len(frame))]
    def flag(mask, reason):
        for i in np.flatnonzero(np.asarray(mask)):
            reasons[i].append(reason)
    finite = np.isfinite(features)
    flag(~finite[HISTORY].all(axis=1), 'invalid_or_missing_history')
    flag(~finite[CONTEXT].all(axis=1), 'missing_or_invalid_context')
    current = features.current_log_risk
    age = features.latest_report_age_days
    flag(~np.isfinite(current) | (current > 0), 'invalid_reported_risk')
    flag(age > 1., 'report_older_than_one_day_at_cutoff')
    flag((age < 0) | (features.history_reports < 1) |
         (features.history_reports % 1 != 0) | (features.history_span_days < 0), 'invalid_history_metadata')
    expected_missing = 1.-finite[CONTEXT].mean(axis=1)
    flag(~np.isfinite(features.context_missing_fraction) |
         ~np.isclose(features.context_missing_fraction, expected_missing), 'inconsistent_missingness_metadata')
    available = np.array([not r for r in reasons])
    score = np.full(len(frame), np.nan)
    if available.any():
        decision = bundle['classifier'].decision_function(features.loc[available]).reshape(-1, 1)
        score[available] = bundle['sigmoid'].predict_proba(decision)[:, 1]
        invalid = available & (~np.isfinite(score) | (score < 0) | (score > 1))
        flag(invalid, 'invalid_model_output')
        available[invalid] = False
        score[invalid] = np.nan
    valid_risk = np.isfinite(current) & (current <= 0)
    return pd.DataFrame({'event_id': frame.event_id.to_numpy(),
                         'status': np.where(available, 'available', 'model_assessment_unavailable'),
                         'model_score': score,
                         'reported_log_risk': current.where(valid_risk).to_numpy(),
                         'reported_risk_valid': valid_risk.to_numpy(),
                         'manual_review_required': ~available,
                         'reason_codes': [';'.join(r) for r in reasons],
                         'policy_version': POLICY_VERSION})
