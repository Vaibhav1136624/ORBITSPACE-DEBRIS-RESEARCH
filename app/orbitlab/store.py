"""Durable extension store: immutable messages, jobs, feedback and audit chain."""
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import time
from .messages import digest


class Store:
    def __init__(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS documents(kind TEXT,key TEXT,payload TEXT,hash TEXT,PRIMARY KEY(kind,key));
        CREATE TABLE IF NOT EXISTS source_state(source TEXT PRIMARY KEY,next_poll REAL DEFAULT 0,lease TEXT,lease_until REAL DEFAULT 0,successes INTEGER DEFAULT 0,failures INTEGER DEFAULT 0,status TEXT,last_success REAL);
        CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT,payload TEXT,previous_hash TEXT,hash TEXT);
        CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY,password_hash TEXT,salt TEXT,role TEXT);
        CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,username TEXT,expires REAL);
        CREATE TABLE IF NOT EXISTS login_limits(username TEXT PRIMARY KEY,attempts INTEGER,until REAL);
        ''')

    def close(self): self.db.close()

    def put(self, kind, key, payload):
        value = json.dumps(payload, sort_keys=True, allow_nan=False)
        hashed = digest(payload)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT hash FROM documents WHERE kind=? AND key=?', (kind, key)).fetchone()
            if old:
                if old['hash'] != hashed: raise ValueError('Immutable document conflict')
                return False
            self.db.execute('INSERT INTO documents VALUES(?,?,?,?)', (kind, key, value, hashed))
        return True

    def all(self, kind):
        return [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM documents WHERE kind=? ORDER BY key', (kind,))]

    def ingest(self, report):
        # Receipt time may differ on a redelivery; original receipt remains authoritative.
        key = report['provider']+':'+report['message_id']
        old = self.db.execute('SELECT payload FROM documents WHERE kind=? AND key=?', ('cdm', key)).fetchone()
        if old:
            original = json.loads(old[0])
            candidate = dict(report, received_at=original['received_at'])
            return self.put('cdm', key, candidate)
        return self.put('cdm', key, report)

    def reserve(self, source, interval, now=None):
        now = time.time() if now is None else now
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            self.db.execute('INSERT OR IGNORE INTO source_state(source) VALUES (?)', (source,))
            row = self.db.execute('SELECT * FROM source_state WHERE source=?', (source,)).fetchone()
            if row['next_poll'] > now or row['lease_until'] > now: return None
            token = secrets.token_hex(16)
            self.db.execute('UPDATE source_state SET next_poll=?,lease=?,lease_until=?,status=? WHERE source=?',
                            (now+interval, token, now+300, 'in_progress', source))
            return token

    def finish(self, source, lease, success, status, cooldown=0, now=None):
        now = time.time() if now is None else now
        with self.db:
            row = self.db.execute('SELECT * FROM source_state WHERE source=? AND lease=?', (source, lease)).fetchone()
            if not row: raise ValueError('Source lease no longer owned')
            self.db.execute('UPDATE source_state SET lease=NULL,lease_until=0,next_poll=?,successes=?,failures=?,status=?,last_success=? WHERE source=?',
                (max(row['next_poll'], now+cooldown), row['successes']+int(success), row['failures']+int(not success), status,
                 now if success else row['last_success'], source))

    def source_profiles(self):
        return [dict(r) for r in self.db.execute('SELECT source,next_poll,successes,failures,status,last_success FROM source_state ORDER BY source')]

    def audit(self, event):
        payload = json.dumps(dict(event, recorded_at=datetime.now(timezone.utc).isoformat()), sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            last = self.db.execute('SELECT hash FROM audit ORDER BY seq DESC LIMIT 1').fetchone()
            previous = last[0] if last else '0'*64
            hashed = hashlib.sha256((previous+payload).encode()).hexdigest()
            self.db.execute('INSERT INTO audit(payload,previous_hash,hash) VALUES(?,?,?)', (payload, previous, hashed))
        return hashed

    def verify_audit(self):
        previous = '0'*64
        for row in self.db.execute('SELECT * FROM audit ORDER BY seq'):
            if row['previous_hash'] != previous or row['hash'] != hashlib.sha256((previous+row['payload']).encode()).hexdigest(): return False
            previous = row['hash']
        return True

    def add_user(self, username, password, role='operator'):
        if not isinstance(username, str) or not 1 <= len(username) <= 100 or not isinstance(password, str) or len(password) < 12 or role not in ('viewer','operator','admin'):
            raise ValueError('Username, password of at least 12 characters, and valid role required')
        salt = secrets.token_hex(16)
        password_hash = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        with self.db:
            self.db.execute('INSERT INTO users VALUES(?,?,?,?)', (username, password_hash, salt, role))
        self.audit({'action': 'user_created', 'username': username, 'role': role})

    def login(self, username, password):
        now = time.time()
        if not isinstance(username, str) or not isinstance(password, str) or len(password) > 1024: raise ValueError('Invalid credentials')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            limit = self.db.execute('SELECT * FROM login_limits WHERE username=?', (username,)).fetchone()
            if limit and limit['until'] > now and limit['attempts'] >= 5: raise PermissionError('Login temporarily rate-limited')
            count = limit['attempts'] if limit and limit['until'] > now else 0
            self.db.execute('INSERT OR REPLACE INTO login_limits VALUES(?,?,?)', (username, count+1, now+900))
        user = self.db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        salt = user['salt'] if user else '0'*32
        value = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        if not user or not hmac.compare_digest(user['password_hash'], value): raise PermissionError('Invalid credentials')
        token = secrets.token_urlsafe(32)
        with self.db:
            self.db.execute('DELETE FROM login_limits WHERE username=?', (username,))
            self.db.execute('INSERT INTO sessions VALUES(?,?,?)', (hashlib.sha256(token.encode()).hexdigest(), username, now+28800))
        self.audit({'action': 'login', 'username': username})
        return token

    def identity(self, token):
        if not token: return None
        row = self.db.execute('SELECT u.username,u.role FROM sessions s JOIN users u ON s.username=u.username WHERE token_hash=? AND expires>?',
                              (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        return dict(row) if row else None

    def logout(self, token):
        with self.db: self.db.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
