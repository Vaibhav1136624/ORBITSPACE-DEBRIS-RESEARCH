import http.client
import json
import threading
import pytest
from replay_engine import Replay
from local_api import Server
from orbitlab.store import Store


@pytest.fixture
def database(tmp_path):
    path = tmp_path/'state.db'
    replay = Replay(path)
    replay.ingest(dict(event_id=7, time_to_tca=4., risk=-4., context_missing=False))
    replay.snapshot(4.); replay.snapshot(3.)
    replay.close()
    return path


def action(**changes):
    return dict(dict(action_id='request-1', checkpoint_days=3., event_id=7,
                     action='acknowledge', operator='test operator', note='test fixture'), **changes)


def test_review_is_append_only_idempotent_and_scoped(database):
    replay = Replay(database)
    original = replay.snapshot(3.)
    first, created = replay.record_review(action())
    assert created
    same, created = replay.record_review(action())
    assert not created and first == same
    assert replay.queue()['rows'][0]['review_status'] == 'acknowledged'
    assert replay.queue(4.)['rows'][0]['review_status'] == 'unacknowledged'
    assert replay.snapshot(3.) == original
    with pytest.raises(FileExistsError):
        replay.record_review(action(note='changed'))
    replay.close()
    replay = Replay(database)
    replay.record_review(action(action_id='request-2', action='reopen'))
    assert replay.queue()['rows'][0]['review_status'] == 'unacknowledged'
    assert len(replay.reviews()) == 2
    assert replay.queue()['rows'][0]['escalation']
    with pytest.raises(KeyError):
        replay.record_review(action(action_id='bad-event', event_id=8))
    assert len(replay.reviews()) == 2
    replay.close()


def test_http_queue_acknowledgement_errors_and_origin(database):
    server = Server(('127.0.0.1', 0), database)
    store = Store(server.extension_database)
    store.add_user('test operator','test-password-long')
    token = store.login('test operator','test-password-long')
    store.add_user('viewer','test-password-long','viewer')
    viewer_token = store.login('viewer','test-password-long')
    store.close()
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        payload = json.dumps(body) if body is not None else None
        connection.request(method, path, body=payload, headers=headers or {'Content-Type': 'application/json','Cookie':'orbit_session='+token})
        response = connection.getresponse()
        raw = response.read()
        result = response.status, json.loads(raw) if 'application/json' in response.getheader('Content-Type', '') else raw.decode()
        connection.close()
        return result
    try:
        status, queue = request('GET', '/api/queue/latest')
        assert status == 200 and queue['encounters'] == 1
        assert 'Encounter queue' in request('GET', '/')[1]
        assert request('GET', '/app.js')[0] == 200
        assert request('GET','/api/research')[1]['decision']['rows'] == []
        assert request('POST','/api/reviews',action(),{'Content-Type':'application/json'})[0] == 401
        assert request('POST','/api/reviews',action(),{'Content-Type':'application/json','Cookie':'orbit_session='+viewer_token})[0] == 403
        assert request('POST','/api/login',{'username':'viewer','password':'wrong'})[0] == 401
        assert request('POST','/api/login',{'username':'viewer','password':'test-password-long'})[0] == 200
        history = request('GET', '/api/history/3/7')[1]
        assert all(r['time_to_tca'] >= 3 for r in history['reports'])
        assert request('GET', '/api/history/3/999')[0] == 404
        assert request('POST', '/api/reviews', action())[0] == 201
        assert request('POST', '/api/reviews', action())[0] == 200
        assert request('POST', '/api/reviews', action(note='different'))[0] == 409
        assert request('GET', '/api/queue/latest')[1]['acknowledged'] == 1
        assert request('GET', '/api/queue/2')[0] == 404
        assert request('GET', '/api/queue/nan')[0] == 400
        assert request('POST', '/api/reviews', action(event_id=999, action_id='unknown'))[0] == 404
        assert request('POST', '/api/reviews', action(), {'Content-Type': 'text/plain'})[0] == 415
        assert request('POST', '/api/reviews', action(), {'Content-Type': 'application/json', 'Origin': 'https://example.org'})[0] == 403
        assert len(request('GET', '/api/reviews')[1]['reviews']) == 1
        assert len(request('GET', '/api/notifications')[1]['notifications']) == 1
        assert request('POST','/api/logout',{})[0] == 200
        assert request('POST','/api/reviews',action())[0] == 401
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
