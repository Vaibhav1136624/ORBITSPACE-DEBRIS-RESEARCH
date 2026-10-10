import pytest
from orbitlab.messages import normalize, utc
from orbitlab.store import Store
from orbitlab.research import evaluate
from orbitlab.sources import poll_one, SourceFailure
from orbitlab.notifications import update

NOW = '2026-09-22T12:00:00+00:00'


def report(**changes):
    raw = dict(message_id='m1',created_at='2026-09-22T11:00:00Z',tca='2026-09-23T12:00:00Z',
               object_ids=['100','200'],encounter_key='encounter1',collision_probability=1e-5,
               masses_kg=[100,100],relative_speed_m_s=10000)
    raw.update(changes)
    return normalize(raw,'synthetic_fixture',NOW)


def test_contract_units_clocks_probability():
    with pytest.raises(ValueError): utc('2026-09-22T12:00:00')
    for probability in [-1,2,float('nan'),True]:
        with pytest.raises(ValueError): report(collision_probability=probability)
    with pytest.raises(ValueError): report(created_at='2026-09-22T13:00:00Z')
    zero = report(collision_probability=0)
    assert zero['zero_pc_reported'] and zero['reported_log_risk'] is None
    raw = dict(MESSAGE_ID='native',CREATION_DATE='2026-09-22T11:00:00',TCA='2026-09-23T12:00:00',
               SAT1_OBJECT_DESIGNATOR=100,SAT2_OBJECT_DESIGNATOR=200,RELATIVE_SPEED=10,RELATIVE_SPEED_UNIT='km/s')
    assert normalize(raw,'space_track',NOW)['relative_speed_m_s']==10000
    raw.pop('RELATIVE_SPEED_UNIT')
    with pytest.raises(ValueError): normalize(raw,'space_track',NOW)


def test_durable_dedup_conflict_lease_and_audit(tmp_path):
    path=tmp_path/'extension.db'; store=Store(path)
    assert store.ingest(report())
    assert not store.ingest(dict(report(),received_at='2026-09-22T13:00:00+00:00'))
    with pytest.raises(ValueError): store.ingest(report(collision_probability=1e-3))
    lease=store.reserve('test',100,now=10)
    assert lease and store.reserve('test',100,now=20) is None
    store.finish('test',lease,False,'http_429',cooldown=500,now=20)
    store.audit({'action':'test'}); assert store.verify_audit()
    store.close(); store=Store(path)
    assert store.reserve('test',100,now=510) is None
    assert store.reserve('test',100,now=521)
    store.db.execute("UPDATE audit SET payload='tampered'"); store.db.commit()
    assert not store.verify_audit(); store.close()


def test_cutoff_capacity_disagreement_and_proxies():
    a=report(); b=dict(report(message_id='m2',collision_probability=1e-8),provider='authorized_import')
    future=dict(report(message_id='future',encounter_key='future'),received_at='2026-09-22T13:00:00+00:00')
    other=report(message_id='other',encounter_key='other')
    decision=evaluate([a,b,future,other],NOW,capacity=1)
    assert len(decision['rows'])==2 and decision['mandatory_overflow']==1
    row=next(r for r in decision['rows'] if r['encounter_key']=='encounter1')
    assert row['escalation'] and row['abstain'] and row['provider_spread_log10']==3
    assert row['expected_relative_energy_j_proxy'] is not None
    assert decision['graph']['nodes'][0]['encounter_degree']==2
    assert decision['operationally_validated'] is False


def test_unassociated_sources_do_not_create_false_agreement():
    a=report(encounter_key=None)
    b=dict(a,provider='authorized_import')
    rows=evaluate([a,b],NOW)['rows']
    assert len(rows)==2 and all(r['provider_spread_log10'] is None for r in rows)


def test_notifications_restart_repeat_and_deadline(tmp_path):
    path=tmp_path/'extension.db'; store=Store(path)
    decision=evaluate([report()],NOW)
    assert len(update(store,decision))==1
    assert update(store,decision)==[]
    store.close(); store=Store(path)
    assert update(store,decision)==[]
    due=evaluate([report()],'2026-09-22T18:00:00Z')
    assert update(store,due)[0]['reason']=='reminder_due'
    with pytest.raises(ValueError): update(store,decision)
    store.close()


def test_provider_failures_and_validation(tmp_path):
    store=Store(tmp_path/'state.db')
    assert 'required' in poll_one(store,'space_track_cdm',env={})['status']
    def fail(_): raise SourceFailure('http_429',7200)
    assert poll_one(store,'nasa_cara',env={},transport=fail)['status']=='http_429'
    assert poll_one(store,'nasa_cara',env={},transport=fail)['status']=='cooldown_or_worker_lease'
    env={'SPACETRACK_IDENTITY':'fixture','SPACETRACK_PASSWORD':'fixture','SPACETRACK_CDM_QUERY':'fixture'}
    assert poll_one(store,'space_track_cdm',env,transport=lambda _: [{}])['status']=='payload_validation_failed'
    assert store.all('cdm')==[]; store.close()


def test_auth_roles_session_and_rate_limit(tmp_path):
    store=Store(tmp_path/'state.db')
    store.add_user('analyst','fixture-password-123','viewer')
    token=store.login('analyst','fixture-password-123')
    assert store.identity(token)['role']=='viewer'
    assert not store.identity('forged')
    store.logout(token); assert not store.identity(token)
    for _ in range(5):
        with pytest.raises(PermissionError): store.login('analyst','wrong')
    with pytest.raises(PermissionError,match='rate-limited'): store.login('analyst','fixture-password-123')
    store.close()
