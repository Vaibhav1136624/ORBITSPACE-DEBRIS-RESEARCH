"""Validate isolation and load the numerical packages used by the project."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

cache_dir = Path(sys.prefix).parent / 'cache' / 'matplotlib'
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(cache_dir))

if sys.prefix == sys.base_prefix:
    raise SystemExit('Run this check using the project virtual environment')
cfg = (Path(sys.prefix) / 'pyvenv.cfg').read_text()
if 'include-system-site-packages = false' not in cfg.lower():
    raise SystemExit('Global packages must be disabled in the reference environment')
versions = {}
for distribution, module in [('numpy','numpy'), ('pandas','pandas'), ('scipy','scipy'),
                             ('scikit-learn','sklearn'), ('matplotlib','matplotlib'),
                             ('pytest','pytest'), ('space-debris-prioritiser','debris_research')]:
    imported = importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)
    if module != 'debris_research' and not Path(imported.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit(f'{distribution} was loaded from outside the virtual environment')
print(json.dumps({'python': sys.version.split()[0], 'environment': sys.prefix,
                  'isolated': True, 'packages': versions}, indent=2))
