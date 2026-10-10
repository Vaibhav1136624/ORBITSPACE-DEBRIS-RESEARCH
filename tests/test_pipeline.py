import json
from pathlib import Path
import pandas as pd
from debris_research import cli


def test_offline_pipeline_produces_reports_and_preserves_evaluation(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    rows = []
    for event in range(400):
        for days, risk in [(4., -12.), (3., -11.), (2., -10.), (.5, -5. if event % 2 else -8.)]:
            rows.append({'event_id': event, 'time_to_tca': days, 'risk': risk, 'mission_id': event % 3})
    csv = data_dir / 'train_data.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    (data_dir / 'download_manifest.json').write_text(json.dumps({
        'source_doi': 'synthetic-unit-test', 'license': 'test fixture',
        'attribution': 'Deterministic synthetic test data, not ESA observations.'
    }))
    monkeypatch.setattr(cli, 'download', lambda _: csv)
    output = tmp_path / 'artifacts'
    cli.main(['--data-dir', str(data_dir), '--output-dir', str(output)])
    metrics = json.loads((output / 'baseline-metrics.json').read_text())
    assert metrics['partition'] == 'validation'
    assert metrics['evaluation_performance_computed'] is False
    assert metrics['validation_encounters'] == 60
    splits = pd.read_csv(output / 'split-manifest.csv')
    assert len(splits) == splits.event_id.nunique() == 400
    assert (output / 'encounter-history.png').stat().st_size > 1000
    assert (output / 'baseline-ranking.png').stat().st_size > 1000
    assert 'No AI training has occurred' in (output / 'milestone-report.md').read_text(encoding='utf-8')
    for partition in ['train', 'validation', 'calibration', 'evaluation']:
        saved = pd.read_csv(tmp_path / 'processed' / f'{partition}.csv')
        assert saved.partition.eq(partition).all()
