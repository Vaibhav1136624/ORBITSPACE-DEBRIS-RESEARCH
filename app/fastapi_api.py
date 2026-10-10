"""FastAPI/Uvicorn backend for the existing Orbit dashboard and SQLite data."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from replay_engine import Replay, finite
from orbitlab.store import Store
from orbitlab.sources import readiness
from orbitlab.research import evaluate, provider_statistics

ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT/'artifacts/replay.sqlite'
CSP = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"


def create_app(database=DEFAULT_DATABASE, port=8770):
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError('Run replay_engine.py first to create a replay database')
    Replay(database).close()
    extension = database.with_name(database.stem+'-advanced.sqlite')
    Store(extension).close()
    # Keep documentation offline: the OpenAPI schema has no CDN dependency.
    app = FastAPI(title='Orbit Research API', version='2.0.0', docs_url=None, redoc_url=None)
    app.state.database = database
    app.state.extension_database = extension
    allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}

    @app.middleware('http')
    async def local_boundary(request, call_next):
        if request.headers.get('host') not in allowed:
            response = JSONResponse({'error':'Local host required'}, status_code=403)
        elif request.headers.get('origin') and request.headers['origin'] not in {'http://'+h for h in allowed}:
            response = JSONResponse({'error':'Cross-origin request rejected'}, status_code=403)
        elif request.url.query:
            response = JSONResponse({'error':'Query parameters are not supported'}, status_code=400)
        else:
            response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = CSP
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({'error':error.detail}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({'error':'Invalid request parameters'}, status_code=400)

    @app.exception_handler(KeyError)
    async def not_found(request, error):
        return JSONResponse({'error':str(error.args[0])}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({'error':str(error)}, status_code=400)

    @app.exception_handler(FileExistsError)
    async def conflict(request, error):
        return JSONResponse({'error':str(error)}, status_code=409)

    @app.exception_handler(PermissionError)
    async def unauthenticated(request, error):
        return JSONResponse({'error':str(error)}, status_code=401)

    def asset(name, media_type):
        return FileResponse(ROOT/'web'/name, media_type=media_type)

    @app.get('/', include_in_schema=False)
    def index(): return asset('index.html','text/html')

    @app.get('/app.js', include_in_schema=False)
    def javascript(): return asset('app.js','text/javascript')

    @app.get('/research.js', include_in_schema=False)
    def research_javascript(): return asset('research.js','text/javascript')

    @app.get('/style.css', include_in_schema=False)
    def stylesheet(): return asset('style.css','text/css')

    @app.get('/api/health')
    def health():
        return {'status':'ok','mode':'local_research','framework':'FastAPI',
                'authentication':'session_required_for_mutations'}

    @app.get('/api/session')
    def session(request: Request):
        with closing(Store(extension)) as store:
            return {'identity':store.identity(request.cookies.get('orbit_session',''))}

    @app.get('/api/research')
    def research():
        with closing(Store(extension)) as store:
            reports = store.all('cdm')
            return {'sources':readiness(),'poll_history':store.source_profiles(),
                    'provider_statistics':provider_statistics(reports),
                    'decision':evaluate(reports,datetime.now(timezone.utc).isoformat()),
                    'audit_chain_valid':store.verify_audit(),
                    'limitations':['No operational validation','NASA algorithms not integrated',
                                   'No SGP4 propagation or covariance collision calculation',
                                   'SQLite local runtime; PostgreSQL and production deployment remain pending']}

    @app.get('/api/snapshots')
    def snapshots():
        with closing(Replay(database)) as replay:
            return {'snapshots':[json.loads(r[0])['checkpoint_days'] for r in replay.db.execute('SELECT payload FROM snapshots ORDER BY checkpoint DESC')]}

    @app.get('/api/queue/latest')
    def latest():
        with closing(Replay(database)) as replay: return replay.queue()

    def checkpoint_value(value):
        checkpoint = float(value)
        if not finite(checkpoint) or checkpoint < 0: raise ValueError('Invalid checkpoint')
        return checkpoint

    @app.get('/api/queue/{checkpoint}')
    def queue(checkpoint: str):
        with closing(Replay(database)) as replay: return replay.queue(checkpoint_value(checkpoint))

    @app.get('/api/notifications')
    def notifications():
        with closing(Replay(database)) as replay: return {'notifications':replay.notifications()}

    @app.get('/api/reviews')
    def reviews():
        with closing(Replay(database)) as replay: return {'reviews':replay.reviews()}

    @app.get('/api/history/{checkpoint}/{event_id}')
    def history(checkpoint: str, event_id: int):
        checkpoint = checkpoint_value(checkpoint)
        with closing(Replay(database)) as replay:
            if event_id not in {r['event_id'] for r in replay.queue(checkpoint)['rows']}:
                raise KeyError('Encounter not present in snapshot')
            return {'event_id':event_id,'checkpoint_days':checkpoint,
                    'reports':[json.loads(r[0]) for r in replay.db.execute(
                        'SELECT payload FROM reports WHERE event_id=? AND time_to_tca>=? ORDER BY time_to_tca DESC',(event_id,checkpoint))]}

    async def body(request):
        if request.headers.get('content-type','').split(';')[0].strip().lower() != 'application/json':
            raise HTTPException(415,'application/json required')
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 16384: raise HTTPException(413,'Body must contain 1 to 16384 bytes')
        if not raw: raise HTTPException(413,'Body must contain 1 to 16384 bytes')
        try:
            payload = json.loads(raw,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
            if not isinstance(payload,dict): raise ValueError('Object required')
            return payload
        except (ValueError,UnicodeError): raise HTTPException(400,'Invalid JSON body') from None

    def mutate(kind, payload, token):
        # Every connection is created/used/closed in this same worker thread.
        with closing(Store(extension)) as store:
            if kind == 'login':
                token = store.login(payload.get('username'),payload.get('password'))
                response = JSONResponse({'identity':store.identity(token)})
                response.set_cookie('orbit_session',token,max_age=28800,httponly=True,samesite='strict')
                return response
            identity = store.identity(token)
            if not identity: raise HTTPException(401,'Sign in required')
            if kind == 'logout':
                store.logout(token)
                response = JSONResponse({'status':'signed_out'})
                response.delete_cookie('orbit_session',httponly=True,samesite='strict')
                return response
            if identity['role'] not in ('operator','admin'): raise HTTPException(403,'Operator role required')
            if kind == 'feedback':
                if set(payload) != {'feedback_id','decision_sha256','encounter_key','verdict','reason'}:
                    raise ValueError('Feedback identifier, decision hash, encounter, verdict and reason required')
                if any(not isinstance(v,str) or not 1 <= len(v) <= 1000 for v in payload.values()):
                    raise ValueError('Nonempty bounded feedback strings required')
                if payload['verdict'] not in ('agree','raise_priority','lower_priority','insufficient_data'):
                    raise ValueError('Unknown verdict')
                match = next((d for d in store.all('decision') if d['decision_sha256']==payload['decision_sha256']),None)
                if not match or payload['encounter_key'] not in {r['encounter_key'] for r in match['rows']}:
                    raise ValueError('Feedback must reference a persisted decision and encounter')
                created = store.put('feedback',payload['feedback_id'],dict(payload,operator=identity['username'],learning_status='not_used_for_training'))
                return JSONResponse({'created':created},status_code=201 if created else 200)
            payload['operator'] = identity['username']
        with closing(Replay(database)) as replay:
            action, created = replay.record_review(payload)
            return JSONResponse({'review':action,'created':created},status_code=201 if created else 200)

    # Publish concrete routes in OpenAPI while sharing the bounded-body/auth flow.
    def mutation_route(kind):
        async def endpoint(request: Request):
            payload = await body(request)
            return await run_in_threadpool(mutate,kind,payload,request.cookies.get('orbit_session',''))
        endpoint.__name__ = kind
        return endpoint

    for kind in ('login','logout','reviews','feedback'):
        app.add_api_route('/api/'+kind,mutation_route(kind),methods=['POST'],
                         openapi_extra={'requestBody':{'required':True,'content':{'application/json':{'schema':{'type':'object'}}}}})
    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=DEFAULT_DATABASE)
    parser.add_argument('--port',type=int,default=8770)
    args = parser.parse_args()
    uvicorn.run(create_app(args.database,args.port),host='127.0.0.1',port=args.port,
                proxy_headers=False,access_log=False)
