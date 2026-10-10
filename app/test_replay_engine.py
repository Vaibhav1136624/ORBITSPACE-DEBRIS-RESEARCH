import pytest
from replay_engine import Replay, Policy


def report(event=1, time=4., risk=-8., missing=False):
    return dict(event_id=event, time_to_tca=time, risk=risk, context_missing=missing)


def test_duplicate_and_conflict_are_atomic(tmp_path):
    r = Replay(tmp_path/'state.db')
    assert r.ingest(report()) == 'inserted'
    assert r.ingest(report()) == 'duplicate'
    with pytest.raises(ValueError, match='Conflicting'):
        r.ingest(report(risk=-3.))
    assert r.snapshot(4.)['rows'][0]['reported_log_risk'] == -8.
    r.close()


def test_future_exclusion_out_of_order_and_immutable_snapshot(tmp_path):
    r = Replay(tmp_path/'state.db')
    r.ingest(report(time=2., risk=-3.))
    r.ingest(report(time=4., risk=-8.))
    s = r.snapshot(3.)
    assert s['rows'][0]['reported_log_risk'] == -8.
    assert s['rows'][0]['available_reports'] == 1
    assert r.snapshot(3.) == s
    with pytest.raises(ValueError, match='immutable'):
        r.snapshot(3., Policy(capacity=1))
    assert r.snapshot(2.)['rows'][0]['reported_log_risk'] == -3.
    r.close()


def test_capacity_preserves_deferrals_and_mandatory_overflow(tmp_path):
    r = Replay(tmp_path/'state.db')
    r.ingest(report(1, risk=-4.))
    r.ingest(report(2, risk=None))
    r.ingest(report(3, risk=-7.))
    s = r.snapshot(4., Policy(capacity=1))
    assert s['encounters'] == 3 and s['selected'] == 1
    assert s['mandatory_demand'] == 2 and s['mandatory_overflow'] == 1
    assert s['rows'][1]['event_id'] == 2
    assert s['rows'][1]['model_score'] is None
    assert s['rows'][1]['manual_review_required']
    assert s['rows'][1]['capacity_overflow']
    r.close()


def test_restart_does_not_duplicate_notifications_and_records_recovery(tmp_path):
    path = tmp_path/'state.db'
    r = Replay(path)
    r.ingest(report(time=4., risk=-3.))
    r.ingest(report(time=3., risk=-8.))
    first = r.snapshot(4.)
    assert len(r.notifications()) == 1
    r.close()
    r = Replay(path)
    assert r.snapshot(4.) == first
    assert len(r.notifications()) == 1
    r.snapshot(3.)
    assert [n['kind'] for n in r.notifications()] == ['escalation', 'recovered']
    r.snapshot(2.)  # exactly one-day-old report is still fresh
    assert len(r.notifications()) == 2
    r.close()


def test_staleness_appears_without_new_message_and_zero_capacity(tmp_path):
    r = Replay(tmp_path/'state.db')
    r.ingest(report())
    r.snapshot(4., Policy(capacity=0))
    s = r.snapshot(2., Policy(capacity=0))
    assert s['selected'] == 0 and s['mandatory_overflow'] == 1
    assert s['rows'][0]['reasons'] == ['stale_report']
    assert r.notifications()[0]['kind'] == 'review_required'
    r.close()
