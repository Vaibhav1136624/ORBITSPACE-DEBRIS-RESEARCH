"""Conservative provider adapters. Configuration is not evidence of live access."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.cookiejar
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from .messages import normalize, utc, digest

ORDER = ['space_track_cdm', 'space_track_gp', 'esa_discos', 'nasa_cara', 'isro_authorized']
INTERVAL = {'space_track_cdm': 28800, 'space_track_gp': 3600, 'esa_discos': 86400, 'nasa_cara': 86400, 'isro_authorized': 3600}
DOCS = {'space_track_cdm':'https://www.space-track.org/documentation',
        'space_track_gp':'https://www.space-track.org/documentation',
        'esa_discos':'https://discosweb.esoc.esa.int/apidocs/',
        'nasa_cara':'https://github.com/nasa/CARA_Analysis_Tools',
        'isro_authorized':'https://www.isro.gov.in/Indian_Space_Situational_Awareness_Report_2025.html'}


class SourceFailure(Exception):
    def __init__(self, status, cooldown=0):
        super().__init__(status); self.status=status; self.cooldown=cooldown


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise SourceFailure('redirect_requires_review')


def request(url, headers=None, data=None, opener=None, as_json=True):
    host = urllib.parse.urlsplit(url)
    if host.scheme != 'https' or host.netloc not in ('www.space-track.org','discosweb.esoc.esa.int','raw.githubusercontent.com'):
        raise SourceFailure('endpoint_not_allowlisted')
    client = opener or urllib.request.build_opener(NoRedirect())
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with client.open(req, timeout=30) as response:
            raw = response.read(20_000_001)
            if len(raw) > 20_000_000: raise SourceFailure('response_exceeds_limit')
            return json.loads(raw) if as_json else raw.decode('utf-8')
    except urllib.error.HTTPError as error:
        cooldown = 0
        if error.code == 429:
            retry = error.headers.get('Retry-After', '')
            try: cooldown = max(60, int(retry))
            except ValueError:
                try: cooldown = max(60, (parsedate_to_datetime(retry)-datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError): cooldown = 3600
        raise SourceFailure('http_'+str(error.code), cooldown) from None
    except (urllib.error.URLError, OSError):
        raise SourceFailure('network_unavailable', 3600) from None
    except (ValueError, UnicodeError):
        raise SourceFailure('invalid_provider_response') from None


def ids(env):
    raw = env.get('ORBIT_OBJECT_IDS', '')
    if not re.fullmatch(r'\d+(,\d+)*', raw): raise SourceFailure('object_ids_required')
    values = sorted(set(int(n) for n in raw.split(',')))
    if len(values) > 100: raise SourceFailure('at_most_100_objects_per_configuration')
    return ','.join(map(str, values))


def readiness(env=None):
    env = os.environ if env is None else env
    return [
        {'source': 'space_track_cdm', 'state': 'configured_unverified' if all(env.get(k) for k in ['SPACETRACK_IDENTITY','SPACETRACK_PASSWORD','SPACETRACK_CDM_QUERY']) else 'credentials_and_authorised_query_required'},
        {'source': 'space_track_gp', 'state': 'configured_unverified' if all(env.get(k) for k in ['SPACETRACK_IDENTITY','SPACETRACK_PASSWORD','ORBIT_OBJECT_IDS']) else 'credentials_and_object_ids_required'},
        {'source': 'esa_discos', 'state': 'configured_unverified' if env.get('DISCOS_API_TOKEN') and env.get('ORBIT_OBJECT_IDS') else 'token_and_object_ids_required'},
        {'source': 'nasa_cara', 'state': 'public_reference_available_algorithm_not_integrated'},
        {'source': 'isro_authorized', 'state': 'configured_import_only' if env.get('ISRO_AUTHORIZED_INBOX') else 'authorised_export_or_agreement_required'},
    ]


def spacetrack(env, query):
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPCookieProcessor(cookies))
    form = urllib.parse.urlencode({'identity': env['SPACETRACK_IDENTITY'], 'password': env['SPACETRACK_PASSWORD']}).encode()
    request('https://www.space-track.org/ajaxauth/login', data=form, headers={'Content-Type':'application/x-www-form-urlencoded'}, opener=opener, as_json=False)
    if not list(cookies): raise SourceFailure('authentication_not_confirmed')
    if not query.startswith(('/expandedspacedata/query/', '/basicspacedata/query/')) or '?' in query or '#' in query or '\\' in query:
        raise SourceFailure('invalid_authorised_query_path')
    result = request('https://www.space-track.org'+query, opener=opener)
    if not isinstance(result, list): raise SourceFailure('expected_record_array')
    return result


def poll_one(store, source, env=None, transport=None):
    env = os.environ if env is None else env
    state = next(r['state'] for r in readiness(env) if r['source'] == source)
    if 'required' in state:
        return {'source':source, 'status':state, 'retrieved':0}
    lease = store.reserve(source, INTERVAL[source])
    if not lease: return {'source':source, 'status':'cooldown_or_worker_lease', 'retrieved':0}
    now = datetime.now(timezone.utc).isoformat()
    try:
        if transport is not None:
            payloads = transport(source)
        elif source == 'space_track_cdm':
            payloads = spacetrack(env, env['SPACETRACK_CDM_QUERY'])
        elif source == 'space_track_gp':
            payloads = spacetrack(env, '/basicspacedata/query/class/gp/NORAD_CAT_ID/'+ids(env)+'/decay_date/null-val/epoch/%3Enow-10/format/json')
        elif source == 'esa_discos':
            query = urllib.parse.urlencode({'filter':f'in(satno,({ids(env)}))', 'page[size]':'100'})
            response = request('https://discosweb.esoc.esa.int/api/objects?'+query, headers={'Authorization':'Bearer '+env['DISCOS_API_TOKEN'],'DiscosWeb-Api-Version':'2'})
            if response.get('links',{}).get('next'): raise SourceFailure('unexpected_pagination_narrow_object_list')
            payloads = response['data']
        elif source == 'nasa_cara':
            payloads = [{'reference_url': DOCS[source], 'readme': request('https://raw.githubusercontent.com/nasa/CARA_Analysis_Tools/master/README.md', as_json=False),
                         'status':'reference_only_no_algorithm_validation'}]
        elif source == 'isro_authorized':
            root = Path(env['ISRO_AUTHORIZED_INBOX']).resolve()
            if not root.is_dir(): raise SourceFailure('authorised_inbox_missing')
            payloads = []
            for path in sorted(root.glob('*.json')):
                data = json.loads(path.read_text(encoding='utf-8'))
                payloads.extend(data if isinstance(data, list) else [data])
        else: raise SourceFailure('unknown_source')
        if not isinstance(payloads, list): raise SourceFailure('expected_record_array')
        normalized = []
        for payload in payloads:
            if source in ('space_track_cdm','isro_authorized'):
                provider = 'space_track' if source == 'space_track_cdm' else 'isro_authorized'
                normalized.append(normalize(payload, provider, now))
            elif source == 'space_track_gp':
                utc(payload['EPOCH'], provider_utc=True)
                if 'NORAD_CAT_ID' not in payload: raise ValueError('GP missing object identity')
                normalized.append(payload)
            else: normalized.append(payload)
        inserted = 0
        for payload in normalized:
            inserted += store.ingest(payload) if source in ('space_track_cdm','isro_authorized') else store.put(source, digest(payload), payload)
        store.finish(source, lease, True, 'retrieved_and_validated')
        store.audit({'action':'provider_ingestion','source':source,'received':len(normalized),'inserted':inserted})
        return {'source':source,'status':'retrieved_and_validated','retrieved':len(normalized),'inserted':inserted}
    except SourceFailure as error:
        store.finish(source, lease, False, error.status, error.cooldown)
        return {'source':source, 'status':error.status, 'retrieved':0}
    except (KeyError, ValueError, TypeError, OSError):
        store.finish(source, lease, False, 'payload_validation_failed')
        return {'source':source,'status':'payload_validation_failed','retrieved':0}


def run_ordered(store, env=None):
    return [poll_one(store, source, env) for source in ORDER]
