"""Persistent research replay on an aligned, synthetic time-to-approach clock."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Policy:
    capacity: int = 5
    escalation_log_risk: float = -6.
    stale_days: float = 1.
    version: str = 'replay-v1'

    def __post_init__(self):
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 0:
            raise ValueError('Capacity must be a nonnegative integer')
        if not finite(self.stale_days) or self.stale_days < 0 or not finite(self.escalation_log_risk) or self.escalation_log_risk > 0:
            raise ValueError('Invalid policy threshold')


class Replay:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS reports (
          event_id INTEGER, time_to_tca REAL, payload TEXT, hash TEXT,
          PRIMARY KEY(event_id, time_to_tca));
        CREATE TABLE IF NOT EXISTS snapshots (
          checkpoint REAL PRIMARY KEY, payload TEXT);
        CREATE TABLE IF NOT EXISTS notifications (
          checkpoint REAL, event_id INTEGER, kind TEXT, prior_state TEXT, state TEXT,
          PRIMARY KEY(checkpoint, event_id));
        CREATE TABLE IF NOT EXISTS event_state (event_id INTEGER PRIMARY KEY, state TEXT);
        CREATE TABLE IF NOT EXISTS review_actions (
          action_id TEXT PRIMARY KEY, checkpoint REAL, event_id INTEGER,
          action TEXT, operator TEXT, note TEXT, created_utc TEXT, request_payload TEXT);
        ''')

    def close(self):
        self.db.close()

    def ingest(self, report):
        required = {'event_id', 'time_to_tca', 'risk', 'context_missing'}
        if set(report) != required:
            raise ValueError('Unexpected report fields')
        if isinstance(report['event_id'], bool) or not isinstance(report['event_id'], int) or report['event_id'] < 0:
            raise ValueError('Invalid event ID')
        if not finite(report['time_to_tca']) or report['time_to_tca'] < 0:
            raise ValueError('Invalid report time')
        if not isinstance(report['context_missing'], bool):
            raise ValueError('Missingness flag must be boolean')
        if report['risk'] is not None and (not finite(report['risk']) or report['risk'] > 0):
            raise ValueError('Invalid reported risk: use null for unavailable')
        payload = canonical(report)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            old = self.db.execute('SELECT hash FROM reports WHERE event_id=? AND time_to_tca=?',
                                  (report['event_id'], report['time_to_tca'])).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError('Conflicting report at same event/time; replay halted for reconciliation')
                return 'duplicate'
            self.db.execute('INSERT INTO reports VALUES (?,?,?,?)',
                            (report['event_id'], report['time_to_tca'], payload, digest))
        return 'inserted'

    def snapshot(self, checkpoint, policy=Policy()):
        if not finite(checkpoint) or checkpoint < 0:
            raise ValueError('Invalid checkpoint')
        stored = self.db.execute('SELECT payload,hash FROM reports WHERE time_to_tca>=? ORDER BY event_id,time_to_tca DESC', (checkpoint,)).fetchall()
        histories = {}
        for text, digest in stored:
            report = json.loads(text)
            histories.setdefault(report['event_id'], []).append((report, digest))
        rows = []
        for event, history in histories.items():
            latest, digest = history[-1]
            risk = latest['risk']; reasons = []
            if risk is None:
                reasons.append('reported_risk_unavailable')
            if latest['context_missing']:
                reasons.append('missing_context')
            if latest['time_to_tca']-checkpoint > policy.stale_days:
                reasons.append('stale_report')
            escalation = risk is not None and risk >= policy.escalation_log_risk
            state = 'escalation' if escalation else ('review_required' if reasons else 'normal')
            rows.append({'event_id': event, 'reported_log_risk': risk,
                         'report_time_to_tca': latest['time_to_tca'], 'report_age_days': latest['time_to_tca']-checkpoint,
                         'available_reports': len(history), 'latest_report_hash': digest,
                         'history_hash': hashlib.sha256(canonical([h for _, h in history]).encode()).hexdigest(),
                         'assessment_status': 'assessment_unavailable' if reasons else 'baseline_available',
                         'model_score': None, 'ranking_method': 'latest_reported_risk',
                         'reasons': reasons, 'escalation': escalation,
                         'manual_review_required': bool(reasons), 'state': state})
        # Escalations, then unavailable assessments, then the baseline queue.
        # Every row remains visible, including all unmet mandatory demand.
        rows.sort(key=lambda r: (0 if r['escalation'] else 1 if r['manual_review_required'] else 2,
                                  -(r['reported_log_risk'] if r['reported_log_risk'] is not None else -math.inf), r['event_id']))
        for i, row in enumerate(rows):
            row.update(rank=i+1, selected=i < policy.capacity, capacity_overflow=i >= policy.capacity,
                       mandatory_overflow=i >= policy.capacity and (row['escalation'] or row['manual_review_required']))
        result = {'checkpoint_days': checkpoint, 'clock': 'synthetic aligned time-to-approach',
                  'policy': asdict(policy), 'rows': rows, 'encounters': len(rows),
                  'selected': sum(r['selected'] for r in rows),
                  'mandatory_demand': sum(r['escalation'] or r['manual_review_required'] for r in rows),
                  'mandatory_overflow': sum(r['mandatory_overflow'] for r in rows)}
        payload = canonical(result)
        with self.db:
            old = self.db.execute('SELECT payload FROM snapshots WHERE checkpoint=?', (checkpoint,)).fetchone()
            if old:
                if old[0] != payload:
                    raise ValueError('Snapshot is immutable: use a new replay database for changed inputs/policy')
                return result
            last = self.db.execute('SELECT MIN(checkpoint) FROM snapshots').fetchone()[0]
            if last is not None and checkpoint >= last:
                raise ValueError('Replay checkpoints must advance toward closest approach')
            for row in rows:
                previous = self.db.execute('SELECT state FROM event_state WHERE event_id=?', (row['event_id'],)).fetchone()
                state = row['state']; prior = previous[0] if previous else 'normal'
                if state != prior:
                    kind = 'recovered' if state == 'normal' else state
                    self.db.execute('INSERT INTO notifications VALUES (?,?,?,?,?)',
                                    (checkpoint, row['event_id'], kind, prior, state))
                self.db.execute('INSERT OR REPLACE INTO event_state VALUES (?,?)', (row['event_id'], state))
            self.db.execute('INSERT INTO snapshots VALUES (?,?)', (checkpoint, payload))
        return result

    def notifications(self):
        return [dict(zip(['checkpoint_days', 'event_id', 'kind', 'prior_state', 'state'], row))
                for row in self.db.execute('SELECT * FROM notifications ORDER BY checkpoint DESC,event_id')]

    def queue(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = self.db.execute('SELECT MIN(checkpoint) FROM snapshots').fetchone()[0]
        stored = self.db.execute('SELECT payload FROM snapshots WHERE checkpoint=?', (checkpoint,)).fetchone()
        if not stored:
            raise KeyError('Snapshot not found')
        result = json.loads(stored[0])
        actions = self.reviews(checkpoint)
        latest = {a['event_id']: a for a in actions}
        for row in result['rows']:
            action = latest.get(row['event_id'])
            row['review_status'] = 'acknowledged' if action and action['action'] == 'acknowledge' else 'unacknowledged'
            row['last_review_action'] = action
        result['acknowledged'] = sum(r['review_status'] == 'acknowledged' for r in result['rows'])
        return result

    def reviews(self, checkpoint=None):
        sql = 'SELECT action_id,checkpoint,event_id,action,operator,note,created_utc FROM review_actions'
        params = ()
        if checkpoint is not None:
            sql += ' WHERE checkpoint=?'; params = (checkpoint,)
        return [dict(zip(['action_id', 'checkpoint_days', 'event_id', 'action', 'operator', 'note', 'created_utc'], row))
                for row in self.db.execute(sql+' ORDER BY rowid', params)]

    def record_review(self, request):
        required = {'action_id', 'checkpoint_days', 'event_id', 'action', 'operator', 'note'}
        if set(request) != required:
            raise ValueError('Exact review fields required: '+', '.join(sorted(required)))
        for field, limit in [('action_id', 128), ('operator', 100), ('note', 1000)]:
            value = request[field]
            if not isinstance(value, str) or len(value) > limit or (field != 'note' and not value.strip()):
                raise ValueError('Invalid '+field)
        if request['action'] not in ('acknowledge', 'reopen'):
            raise ValueError('Action must be acknowledge or reopen')
        if not finite(request['checkpoint_days']) or request['checkpoint_days'] < 0:
            raise ValueError('Invalid checkpoint')
        if isinstance(request['event_id'], bool) or not isinstance(request['event_id'], int) or request['event_id'] < 0:
            raise ValueError('Invalid event ID')
        normalized = dict(request, checkpoint_days=float(request['checkpoint_days']))
        payload = canonical(normalized)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT request_payload FROM review_actions WHERE action_id=?', (request['action_id'],)).fetchone()
            if old:
                if old[0] != payload:
                    raise FileExistsError('Action ID already used for a different request')
                return next(a for a in self.reviews() if a['action_id'] == request['action_id']), False
            queue = self.queue(request['checkpoint_days'])
            if request['event_id'] not in {r['event_id'] for r in queue['rows']}:
                raise KeyError('Encounter not present in this snapshot')
            created = datetime.now(timezone.utc).isoformat()
            self.db.execute('INSERT INTO review_actions VALUES (?,?,?,?,?,?,?,?)',
                            (request['action_id'], request['checkpoint_days'], request['event_id'], request['action'],
                             request['operator'], request['note'], created, payload))
            return {'action_id': request['action_id'], 'checkpoint_days': float(request['checkpoint_days']),
                    'event_id': request['event_id'], 'action': request['action'], 'operator': request['operator'],
                    'note': request['note'], 'created_utc': created}, True


def write_demo(source, output):
    """Read one saved validation queue and its reports; never use final labels."""
    import pandas as pd
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queue_path = source/'artifacts/step-2b/queue-event-ids.csv'
    ids = set(pd.read_csv(queue_path).iloc[0].astype(int))
    context = ['miss_distance', 'relative_speed', 'max_risk_estimate', 't_sigma_r', 't_sigma_t', 't_sigma_n',
               'c_sigma_r', 'c_sigma_t', 'c_sigma_n', 't_obs_available', 't_obs_used', 'c_obs_available',
               'c_obs_used', 't_time_lastob_end', 'c_time_lastob_end', 't_weighted_rms', 'c_weighted_rms']
    raw_path = source/'work/data/train_data.csv'
    with raw_path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != json.loads((source/'artifacts/run-manifest.json').read_text())['csv_sha256']:
        raise ValueError('Dataset hash mismatch')
    reports = []
    for chunk in pd.read_csv(raw_path, usecols=['event_id', 'time_to_tca', 'risk']+context, chunksize=20000):
        chunk = chunk.loc[chunk.event_id.isin(ids) & (chunk.time_to_tca >= 2.)]
        for _, row in chunk.iterrows():
            reports.append({'event_id': int(row.event_id), 'time_to_tca': float(row.time_to_tca),
                            'risk': float(row.risk), 'context_missing': any(not finite(row[c]) for c in context)})
    reports.sort(key=lambda r: (-r['time_to_tca'], r['event_id']))
    replay = Replay(output/'replay.sqlite')
    try:
        ingest_counts = {'inserted': 0, 'duplicate': 0}
        for report in reports:
            ingest_counts[replay.ingest(report)] += 1
        duplicate_result = replay.ingest(reports[0])
        snapshots = [replay.snapshot(day) for day in (4., 3., 2.)]
        notices = replay.notifications()
        for snap in snapshots:
            (output/f"queue-{int(snap['checkpoint_days']*24)}h.json").write_text(json.dumps(snap, indent=2, allow_nan=False))
        (output/'notifications.json').write_text(json.dumps(notices, indent=2))
        manifest = {'raw_sha256': digest, 'queue_source_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest(),
                    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'cohort': 'first saved validation queue, selected without labels', 'queue_event_ids': sorted(ids),
                    'ingest_counts_this_run': ingest_counts, 'explicit_duplicate_result': duplicate_result,
                    'report_count': len(reports), 'notifications': len(notices),
                    'reserved_calibration_used': False, 'evaluation_used': False}
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        lines = ['# Step 4A: persistent replay and review queue', '',
                 'Research demonstration using the first saved 50-encounter validation queue. A common time-to-approach clock is synthetic; it does not establish that these encounters happened simultaneously.', '',
                 'Default capacity is five. Reported-risk escalations (log-risk >= -6) are ordered first, then unavailable assessments, then normal baseline-ranked encounters. Every encounter with an available report remains visible, including mandatory overflow. Selection means assignment to a review slot, not a completed review.', '',
                 '| Checkpoint | Visible encounters | Selected | Mandatory demand | Mandatory overflow |', '|---|---:|---:|---:|---:|']
        for snap in snapshots:
            lines.append(f"| {snap['checkpoint_days']*24:.0f} hours | {snap['encounters']} | {snap['selected']} | {snap['mandatory_demand']} | {snap['mandatory_overflow']} |")
        lines += ['', f'{len(reports)} pre-48-hour reports ingested; explicit repeated delivery returned `{duplicate_result}`. {len(notices)} state-transition notifications persisted.', '',
                  'Report payloads and histories are hashed. Duplicate delivery is idempotent; conflicting event/time content is rejected. Snapshots are immutable; identical replay after restart does not generate duplicate notifications. A changed policy/input at an existing checkpoint requires a separate replay run.', '',
                  'Notifications mark escalation, review-required, and recovery transitions at these checkpoints. Repeated unchanged states are suppressed. These coarse checkpoints do not measure per-report notification delay, missed transient escalations, or notification-reduction efficacy.', '',
                  'This milestone uses reported-risk ranking only. No learned model is scored at these changing horizons; model_score is null. Reasons reflect report risk availability, selected context missingness, and freshness, not a complete scientific validity check.', '',
                  'SQLite stores reports, immutable decisions, state, and notifications. Final labels are not inputs. Operator acknowledgements, API/dashboard, durable background scheduling, and full replay efficacy evaluation remain to be implemented.']
        (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines))
    finally:
        replay.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', required=True)
    parser.add_argument('--output-dir', default=str(Path(__file__).parent/'artifacts'))
    args = parser.parse_args()
    write_demo(args.source_project, args.output_dir)
