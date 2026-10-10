"""Migration contract tests; require requirements-api.txt to be installed."""
import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from fastapi_api import create_app
from replay_engine import Replay
from orbitlab.store import Store


@pytest.fixture
def client(tmp_path):
    database = tmp_path/'state.db'
    replay = Replay(database)
    replay.ingest(dict(event_id=7,time_to_tca=4.,risk=-4.,context_missing=False))
    replay.snapshot(4.); replay.snapshot(3.); replay.close()
    app = create_app(database)
    store = Store(app.state.extension_database)
    store.add_user('analyst','test-password-long')
    store.add_user('viewer','test-password-long','viewer')
    store.close()
    with TestClient(app,base_url='http://127.0.0.1:8770') as connection:
        yield connection


def review(**changes):
    return dict(dict(action_id='first',checkpoint_days=3.,event_id=7,
                     action='acknowledge',operator='forged name',note='fixture'),**changes)


def login(client,username='analyst'):
    response = client.post('/api/login',json={'username':username,'password':'test-password-long'})
    assert response.status_code == 200
    assert 'HttpOnly' in response.headers['set-cookie']
    assert 'SameSite=strict' in response.headers['set-cookie']


def test_routes_data_and_schema(client):
    assert client.get('/api/health').json()['framework']=='FastAPI'
    assert client.get('/api/queue/latest').json()['encounters']==1
    assert client.get('/api/snapshots').json()['snapshots']==[4.,3.]
    assert client.get('/api/research').json()['decision']['rows']==[]
    assert client.get('/api/queue/2').status_code==404
    assert client.get('/api/queue/nan').status_code==400
    assert client.get('/api/history/3/999').status_code==404
    assert all(r['time_to_tca']>=3 for r in client.get('/api/history/3/7').json()['reports'])
    for path in ('/','/app.js','/research.js','/style.css'):
        response=client.get(path)
        assert response.status_code==200
        assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert '/api/reviews' in client.get('/openapi.json').json()['paths']


def test_auth_attribution_idempotency_and_logout(client):
    assert client.post('/api/reviews',json=review()).status_code==401
    login(client,'viewer')
    assert client.post('/api/reviews',json=review()).status_code==403
    login(client)
    first=client.post('/api/reviews',json=review())
    assert first.status_code==201 and first.json()['review']['operator']=='analyst'
    assert client.post('/api/reviews',json=review()).status_code==200
    assert client.post('/api/reviews',json=review(note='changed')).status_code==409
    assert client.get('/api/queue/latest').json()['acknowledged']==1
    assert client.get('/api/queue/4').json()['acknowledged']==0
    assert client.post('/api/logout',json={}).status_code==200
    assert client.get('/api/session').json()['identity'] is None
    assert client.post('/api/reviews',json=review()).status_code==401


def test_boundary_and_invalid_bodies(client):
    assert client.get('/api/health',headers={'Host':'evil.example'}).status_code==403
    assert client.post('/api/login',json={},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/api/queue/latest?unexpected=1').status_code==400
    assert client.post('/api/login',content='{}',headers={'Content-Type':'text/plain'}).status_code==415
    for raw in ('[]','{"number":NaN}','{bad json'):
        assert client.post('/api/login',content=raw,headers={'Content-Type':'application/json'}).status_code==400
    assert client.post('/api/login',content='x'*16385,headers={'Content-Type':'application/json'}).status_code==413
    assert client.post('/api/login',json={'username':'analyst','password':'wrong'}).status_code==401
