# ORBITSPACE-DEBRIS-RESEARCH

Orbit — Space-Debris Alert Prioritiser

A research workbench for ranking conjunction reports under limited human-review capacity. Includes encounter-level machine-learning experiments, a persistent replay engine, and a local dashboard.

## Features

- Time-cutoff filtering, deterministic priority queues, explicit mandatory overflow and immutable decision snapshots.
- SQLite persistence, authenticated operator reviews and idempotent review actions.
- HTML/CSS/JavaScript dashboard with hover effects and reduced-motion support.
- UTC ingestion adapters for authorized Space-Track messages, GP data and ESA DISCOS metadata; authorized ISRO JSON import and NASA CARA reference retrieval.
- Experimental trend, consequence, disagreement, abstention, queue sensitivity, connectivity and provider-quality indicators. Feedback is collected, not automatically used for training.
- FastAPI/Uvicorn migration source alongside the existing standard-library HTTP backend.

## Status

Research prototype, not an operational spacecraft collision-avoidance system. Live provider access is unverified. Models and experimental proxies are not operationally calibrated. NASA algorithms, orbital propagation, production deployment and additional rare-event validation remain pending.

The FastAPI migration was written but could not be runtime-validated in the development environment because dependency downloads were blocked. Its tests explicitly skip when dependencies are unavailable. The legacy HTTP API remains available. Do not interpret skipped tests as FastAPI validation.

## Quick start (Python 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]" -r app/requirements-api.txt
.\.venv\Scripts\python app/seed_demo.py
.\.venv\Scripts\python app/fastapi_api.py --port 8770
```

Open http://127.0.0.1:8770/. `/api/health` identifies the FastAPI backend and `/openapi.json` provides the API schema. The seed creates clearly synthetic example encounters; it does not download or imply live telemetry. For the previously tested backend, run `python app/local_api.py --port 8770` instead.

To enable review actions, create an account using `python app/manage_orbit.py add-user analyst --role operator`. The password is entered in a hidden terminal prompt; there are no default credentials.

## Tests

```powershell
python -m pytest tests -q
python -m pytest app --rootdir app -q
```

Optional sequence experiments require PyTorch; see the sequence dependency in `pyproject.toml`. Large benchmark datasets, trained models, databases, credentials, local accounts and generated artifacts are intentionally excluded. Research data must be obtained separately under its applicable terms. Use `debris-research --help` for the research CLI.

## Source configuration

Only configure provider access you are authorized to use. Set credentials in the process environment, never in source control:

- Space-Track: `SPACETRACK_IDENTITY`, `SPACETRACK_PASSWORD`, `SPACETRACK_CDM_QUERY`, `ORBIT_OBJECT_IDS`.
- DISCOS: `DISCOS_API_TOKEN`, `ORBIT_OBJECT_IDS`.
- ISRO exports: `ISRO_AUTHORIZED_INBOX`.

`python app/manage_orbit.py status` reports configuration; `ingest-once` fetches eligible sources with persistent cooldowns; `evaluate` persists research decisions and local alerts. NASA integration currently retrieves reference material only. No external alert delivery is enabled.

## Layout

- `src/debris_research/`: reproducible research pipeline.
- `tests/`: research tests.
- `app/`: replay, backends, adapters and dashboard.
- `scripts/`: environment utilities.
- `advanced-roadmap.md`: research roadmap.
