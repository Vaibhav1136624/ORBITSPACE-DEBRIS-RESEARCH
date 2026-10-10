"""Loopback-only research API. Uses the existing SQLite replay database."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlsplit
from replay_engine import Replay, finite
from orbitlab.store import Store
from orbitlab.sources import readiness
from orbitlab.research import evaluate, provider_statistics


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, database):
        if address[0] != '127.0.0.1':
            raise ValueError('Prototype API binds to loopback only')
        self.database = Path(database).resolve()
        if not self.database.is_file():
            raise ValueError('Run replay_engine.py first to create a replay database')
        # Apply additive schema migration before accepting requests.
        Replay(self.database).close()
        self.extension_database = self.database.with_name(self.database.stem+'-advanced.sqlite')
        Store(self.extension_database).close()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, status, value, cookie=None):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if cookie: self.send_header('Set-Cookie', cookie)
        self.end_headers()
        self.wfile.write(body)

    def token(self):
        cookie = SimpleCookie()
        try: cookie.load(self.headers.get('Cookie',''))
        except Exception: return ''
        return cookie['orbit_session'].value if 'orbit_session' in cookie else ''

    def allowed_host(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            self.send_json(403, {'error': 'Local host required'}); return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{h}' for h in allowed}:
            self.send_json(403, {'error': 'Cross-origin request rejected'}); return False
        return True

    def do_GET(self):
        if not self.allowed_host():
            return
        assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/research.js': ('research.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
        if self.path in assets:
            name, mime = assets[self.path]
            body = (Path(__file__).parent/'web'/name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mime+'; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(body)
            return
        replay = Replay(self.server.database)
        try:
            path = urlsplit(self.path).path
            if urlsplit(self.path).query:
                raise ValueError('Query parameters are not supported')
            if path == '/api/health':
                result = {'status': 'ok', 'mode': 'local_research', 'authentication': 'session_required_for_mutations'}
            elif path in ('/api/research', '/api/session'):
                store = Store(self.server.extension_database)
                try:
                    if path == '/api/session': result = {'identity':store.identity(self.token())}
                    else:
                        reports = store.all('cdm')
                        result = {'sources':readiness(), 'poll_history':store.source_profiles(),
                                  'provider_statistics':provider_statistics(reports),
                                  'decision':evaluate(reports, datetime.now(timezone.utc).isoformat()),
                                  'audit_chain_valid':store.verify_audit(),
                                  'limitations':['No operational validation','NASA algorithms not integrated',
                                                 'No SGP4 propagation or covariance collision calculation',
                                                 'SQLite local runtime; production stack blocked by unavailable dependencies']}
                finally: store.close()
            elif path == '/api/snapshots':
                result = {'snapshots': [json.loads(r[0])['checkpoint_days'] for r in replay.db.execute('SELECT payload FROM snapshots ORDER BY checkpoint DESC')]}
            elif path == '/api/queue/latest':
                result = replay.queue()
            elif path.startswith('/api/queue/'):
                checkpoint = float(path.removeprefix('/api/queue/'))
                if not finite(checkpoint) or checkpoint < 0:
                    raise ValueError('Invalid checkpoint')
                result = replay.queue(checkpoint)
            elif path == '/api/notifications':
                result = {'notifications': replay.notifications()}
            elif path == '/api/reviews':
                result = {'reviews': replay.reviews()}
            elif path.startswith('/api/history/'):
                parts = path.split('/')
                if len(parts) != 5:
                    raise ValueError('Use /api/history/checkpoint/event_id')
                checkpoint, event = float(parts[3]), int(parts[4])
                if not finite(checkpoint) or checkpoint < 0:
                    raise ValueError('Invalid checkpoint')
                queue = replay.queue(checkpoint)
                if event not in {r['event_id'] for r in queue['rows']}:
                    raise KeyError('Encounter not present in snapshot')
                result = {'event_id': event, 'checkpoint_days': checkpoint,
                          'reports': [json.loads(row[0]) for row in replay.db.execute(
                              'SELECT payload FROM reports WHERE event_id=? AND time_to_tca>=? ORDER BY time_to_tca DESC', (event, checkpoint))]}
            else:
                raise KeyError('Route not found')
            self.send_json(200, result)
        except KeyError as error:
            self.send_json(404, {'error': str(error.args[0])})
        except ValueError as error:
            self.send_json(400, {'error': str(error)})
        finally:
            replay.close()

    def do_POST(self):
        if not self.allowed_host():
            return
        if self.path not in ('/api/reviews','/api/login','/api/logout','/api/feedback'):
            self.send_json(404, {'error': 'Route not found'}); return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self.send_json(415, {'error': 'application/json required'}); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                self.send_json(413, {'error': 'Body must contain 1 to 16384 bytes'}); return
            request = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
            if not isinstance(request, dict):
                raise ValueError('JSON object required')
        except (ValueError, UnicodeError):
            self.send_json(400, {'error': 'Invalid JSON body or length'}); return
        store = Store(self.server.extension_database)
        try:
            if self.path == '/api/login':
                token = store.login(request.get('username'),request.get('password'))
                self.send_json(200, {'identity':store.identity(token)}, 'orbit_session='+token+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800')
                return
            identity = store.identity(self.token())
            if not identity:
                self.send_json(401, {'error':'Sign in required'}); return
            if self.path == '/api/logout':
                store.logout(self.token())
                self.send_json(200, {'status':'signed_out'}, 'orbit_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0'); return
            if identity['role'] not in ('operator','admin'):
                self.send_json(403, {'error':'Operator role required'}); return
            if self.path == '/api/feedback':
                if set(request) != {'feedback_id','decision_sha256','encounter_key','verdict','reason'}:
                    raise ValueError('Feedback identifier, decision hash, encounter, verdict and reason required')
                if request['verdict'] not in ('agree','raise_priority','lower_priority','insufficient_data'):
                    raise ValueError('Unknown verdict')
                if any(not isinstance(v,str) or not 1 <= len(v) <= 1000 for v in request.values()):
                    raise ValueError('Nonempty bounded feedback strings required')
                decisions = store.all('decision')
                match = next((d for d in decisions if d['decision_sha256']==request['decision_sha256']),None)
                if not match or request['encounter_key'] not in {r['encounter_key'] for r in match['rows']}:
                    raise ValueError('Feedback must reference a persisted decision and encounter')
                created = store.put('feedback',request['feedback_id'],dict(request,operator=identity['username'],learning_status='not_used_for_training'))
                self.send_json(201 if created else 200, {'created':created}); return
            request['operator'] = identity['username']
        except PermissionError as error:
            self.send_json(401, {'error':str(error)}); return
        except ValueError as error:
            self.send_json(400, {'error':str(error)}); return
        finally: store.close()
        replay = Replay(self.server.database)
        try:
            action, created = replay.record_review(request)
            self.send_json(201 if created else 200, {'review': action, 'created': created})
        except FileExistsError as error:
            self.send_json(409, {'error': str(error)})
        except KeyError as error:
            self.send_json(404, {'error': str(error.args[0])})
        except ValueError as error:
            self.send_json(400, {'error': str(error)})
        finally:
            replay.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path(__file__).parent/'artifacts/replay.sqlite')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = Server(('127.0.0.1', args.port), args.database)
    print(f'Local research API: http://127.0.0.1:{server.server_port}/api/queue/latest', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
