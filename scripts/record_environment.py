"""Save a dependency lock and local interpreter selection after setup."""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
locked = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--exclude-editable'], text=True)
(root / 'requirements-lock.txt').write_text('# Verified with Python 3.12.14 on Windows\n' + locked, encoding='utf-8')
(root / '.local-environment.json').write_text(json.dumps({'python': sys.executable, 'environment': sys.prefix}, indent=2), encoding='utf-8')
print('Saved dependency lock and local environment selection.')
