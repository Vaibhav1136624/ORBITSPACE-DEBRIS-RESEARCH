"""Persistent local alert outbox with bounded suppression, never external delivery."""
import json
from .messages import digest
from .research import seconds


def update(store, decision, reminder_hours=6):
    if reminder_hours <= 0: raise ValueError('Positive reminder interval required')
    store.db.executescript('''
    CREATE TABLE IF NOT EXISTS alert_state(encounter TEXT PRIMARY KEY,payload TEXT);
    CREATE TABLE IF NOT EXISTS alert_outbox(id TEXT PRIMARY KEY,payload TEXT);
    ''')
    emitted = []
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        for row in decision['rows']:
            key, now = row['encounter_key'], decision['as_of']
            old = store.db.execute('SELECT payload FROM alert_state WHERE encounter=?',(key,)).fetchone()
            old = json.loads(old[0]) if old else None
            if old and now < old['observed_at']: raise ValueError('Notification clock cannot run backwards')
            state = 'escalation' if row['escalation'] else 'review' if row['abstain'] else 'normal'
            reason = None
            if old is None and state != 'normal': reason = 'new_'+state
            elif old and old['state'] != state: reason = 'state_transition'
            elif old and state != 'normal':
                if row['log_risk'] is not None and (old['risk'] is None or row['log_risk'] >= old['risk']+0.5): reason = 'risk_increase'
                elif row['hours_to_tca'] <= 6 and old['hours_to_tca'] > 6: reason = 'deadline_crossing'
                elif old['last_sent'] and seconds(now,old['last_sent']) >= reminder_hours*3600: reason = 'reminder_due'
                elif row['mandatory_overflow'] and not old['overflow']: reason = 'mandatory_capacity_overflow'
            last_sent = old['last_sent'] if old else None
            # Risk reference is last alert, so several small increases cannot evade notification.
            reference_risk = old['risk'] if old else row['log_risk']
            if reason:
                alert = {'encounter_key':key,'created_at':now,'reason':reason,'state':state,
                         'decision_sha256':decision['decision_sha256'],'delivery':'local_outbox_only'}
                alert_id = digest(alert)
                cursor = store.db.execute('INSERT OR IGNORE INTO alert_outbox VALUES(?,?)',(alert_id,json.dumps(alert)))
                if cursor.rowcount: emitted.append(dict(alert,id=alert_id))
                last_sent, reference_risk = now, row['log_risk']
            payload = {'observed_at':now,'last_sent':last_sent,'risk':reference_risk,'state':state,
                       'hours_to_tca':row['hours_to_tca'],'overflow':row['mandatory_overflow']}
            store.db.execute('INSERT OR REPLACE INTO alert_state VALUES(?,?)',(key,json.dumps(payload)))
    return emitted
