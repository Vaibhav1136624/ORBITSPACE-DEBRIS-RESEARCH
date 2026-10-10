"""Create a small SYNTHETIC dashboard fixture without downloading research data."""
from pathlib import Path
from replay_engine import Replay


def main():
    destination = Path(__file__).parent/'artifacts/replay.sqlite'
    if destination.exists():
        raise SystemExit('Database already exists; refusing to modify it.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    replay = Replay(destination)
    try:
        for event in range(1,13):
            for day in (4.,3.,2.):
                replay.ingest(dict(event_id=event,time_to_tca=day,
                                   risk=-4.5 if event <= 3 else -7.-event/10,
                                   context_missing=event in (4,5,6)))
        for day in (4.,3.,2.): replay.snapshot(day)
    finally: replay.close()
    print('Created 12 SYNTHETIC encounters, not live telemetry.')


if __name__ == '__main__': main()
