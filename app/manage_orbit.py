"""Local management CLI. Secrets are read from environment or a hidden prompt."""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
from orbitlab.store import Store
from orbitlab.sources import run_ordered, readiness
from orbitlab.research import evaluate
from orbitlab.notifications import update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path(__file__).parent/'artifacts/replay-advanced.sqlite')
    sub = parser.add_subparsers(dest='command',required=True)
    user = sub.add_parser('add-user'); user.add_argument('username'); user.add_argument('--role',choices=['viewer','operator','admin'],default='operator')
    sub.add_parser('status'); sub.add_parser('ingest-once'); sub.add_parser('evaluate')
    args = parser.parse_args(); store = Store(args.database)
    try:
        if args.command == 'add-user':
            password = getpass.getpass('Password (at least 12 characters): ')
            if password != getpass.getpass('Repeat password: '): raise ValueError('Passwords do not match')
            store.add_user(args.username,password,args.role); print('Account created.')
        elif args.command == 'status':
            print(json.dumps({'sources':readiness(),'history':store.source_profiles()},indent=2))
        elif args.command == 'ingest-once':
            print(json.dumps(run_ordered(store),indent=2))
        else:
            result = evaluate(store.all('cdm'),datetime.now(timezone.utc).isoformat())
            store.put('decision',result['decision_sha256'],result)
            alerts = update(store,result)
            print(json.dumps({'decision':result,'new_local_notifications':alerts},indent=2))
    finally: store.close()


if __name__ == '__main__': main()
