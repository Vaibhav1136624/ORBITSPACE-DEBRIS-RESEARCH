"""Strict normalized message contract with explicit clock, units, and identity."""
from datetime import datetime, timezone
import hashlib
import json
import math


def utc(value, provider_utc=False):
    if not isinstance(value, str):
        raise ValueError('Timestamp must be an ISO string')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        if not provider_utc:
            raise ValueError('Explicit timezone required')
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec='microseconds')


def number(value, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError('Boolean is not numeric data')
    value = float(value)
    if not math.isfinite(value) or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError('Numeric value outside allowed domain')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalize(payload, provider, received_at, encounter_key=None):
    """Accept explicit normalized JSON or the published flattened Space-Track CDM format."""
    if provider not in ('space_track', 'isro_authorized', 'authorized_import', 'synthetic_fixture'):
        raise ValueError('Unsupported provider')
    received = utc(received_at)
    native = 'CREATION_DATE' in payload
    if native:
        if provider != 'space_track':
            raise ValueError('Flattened CDM mapping is specific to Space-Track')
        message_id = payload.get('MESSAGE_ID') or payload.get('CDM_ID')
        created = utc(payload['CREATION_DATE'], provider_utc=True)
        tca = utc(payload['TCA'], provider_utc=True)
        objects = [str(payload['SAT1_OBJECT_DESIGNATOR']), str(payload['SAT2_OBJECT_DESIGNATOR'])]
        pc = payload.get('COLLISION_PROBABILITY')
        speed = payload.get('RELATIVE_SPEED')
        units = payload.get('RELATIVE_SPEED_UNIT')
        if speed not in (None, ''):
            if units not in ('m/s', 'km/s'):
                raise ValueError('Explicit supported relative-speed units required')
            speed = number(speed, 0)*(1000 if units == 'km/s' else 1)
        else:
            speed = None
        supplied_key = encounter_key
    else:
        message_id = payload['message_id']
        created, tca = utc(payload['created_at']), utc(payload['tca'])
        objects = payload['object_ids']
        pc = payload.get('collision_probability')
        speed = payload.get('relative_speed_m_s')
        speed = None if speed is None else number(speed, 0)
        supplied_key = encounter_key or payload.get('encounter_key')
    if not message_id or not isinstance(objects, list) or len(objects) != 2 or any(not str(o).strip() for o in objects) or str(objects[0]) == str(objects[1]):
        raise ValueError('Message and two distinct object identifiers required')
    if created > tca or created > received:
        raise ValueError('Message creation must not follow receipt or closest approach')
    pc = None if pc in (None, '') else number(pc, 0, 1)
    # TCA can move between updates. Cross-provider/shifted-TCA association must be explicit.
    key = supplied_key or 'unassociated:'+digest({'objects': sorted(map(str, objects)), 'tca': tca})[:24]
    if not isinstance(key, str) or not 1 <= len(key) <= 200:
        raise ValueError('Invalid encounter key')
    masses = payload.get('masses_kg', [None, None])
    if not isinstance(masses, list) or len(masses) != 2:
        raise ValueError('Expected two masses')
    masses = [None if m is None else number(m, 0) for m in masses]
    return {'provider': provider, 'message_id': str(message_id), 'encounter_key': key,
            'association': 'explicit' if supplied_key else 'exact_pair_tca_only',
            'created_at': created, 'received_at': received, 'tca': tca,
            'object_ids': list(map(str, objects)), 'collision_probability': pc,
            'reported_log_risk': math.log10(pc) if pc is not None and pc > 0 else None,
            'zero_pc_reported': pc == 0, 'relative_speed_m_s': speed, 'masses_kg': masses,
            'raw_sha256': digest(payload), 'clock': 'absolute_utc',
            'fixture': provider == 'synthetic_fixture'}
