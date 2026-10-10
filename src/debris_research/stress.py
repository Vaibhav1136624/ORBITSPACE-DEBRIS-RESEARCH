"""Step 3B: feature-level missing/stale-input scenarios, not orbital replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import CONTEXT, sha256
from .quality import assess, POLICY_VERSION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, default=Path('work/processed/validation.csv'))
    parser.add_argument('--prototype-dir', type=Path, default=Path('artifacts/step-3a'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/step-3b'))
    args = parser.parse_args(argv)
    manifest = json.loads((args.prototype_dir/'experiment.json').read_text())
    if sha256(args.validation) != manifest['input_sha256']['validation']:
        raise ValueError('Validation hash mismatch')
    frame = pd.read_csv(args.validation).sort_values('event_id').reset_index(drop=True)
    if not frame.partition.eq('validation').all():
        raise ValueError('Only validation is permitted')
    bundle = joblib.load(args.prototype_dir/'prototype.joblib')
    rng = np.random.default_rng(20260912)
    chosen = rng.choice(len(frame), round(.2*len(frame)), replace=False)
    scenarios = {'original': frame.copy()}
    missing = frame.copy(); missing.loc[chosen, CONTEXT] = np.nan
    missing.loc[chosen, 'context_missing_fraction'] = 1.
    scenarios['remove_context_20pct'] = missing
    stale = frame.copy(); stale.loc[chosen, 'latest_report_age_days'] += 2.
    stale.loc[chosen, 'latest_time_to_tca'] += 2.
    scenarios['add_two_days_age_20pct'] = stale
    combined = missing.copy()
    combined.loc[chosen, 'latest_report_age_days'] += 2.
    combined.loc[chosen, 'latest_time_to_tca'] += 2.
    scenarios['missing_and_stale_20pct'] = combined
    invalid = frame.copy(); invalid.loc[chosen, 'current_log_risk'] = np.nan
    scenarios['invalid_reported_risk_20pct'] = invalid
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, inputs in scenarios.items():
        assessed = assess(inputs, bundle)
        assert assessed.event_id.tolist() == frame.event_id.tolist()
        deferred = assessed.manual_review_required.to_numpy()
        assert assessed.loc[deferred, 'model_score'].isna().all()
        assessed.to_csv(out/f'{name}-assessments.csv', index=False)
        rows.append({'scenario': name, 'encounters_retained': len(assessed),
                     'available': int((~deferred).sum()), 'deferred': int(deferred.sum()),
                     'high_risk_deferred': int(frame.loc[deferred, 'target_high_risk'].sum()),
                     'invalid_reported_risk': int((~assessed.reported_risk_valid).sum()),
                     'deferral_fraction': float(deferred.mean())})
    pd.DataFrame({'event_id': frame.event_id.iloc[chosen]}).sort_values('event_id').to_csv(out/'perturbed-event-ids.csv', index=False)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'policy_version': POLICY_VERSION,
              'seed': 20260912, 'perturbed_encounters': len(chosen), 'scenarios': rows,
              'policy': 'withhold score for any missing/invalid context, invalid history/risk, or report age >1 day; retain every row for review',
              'interpretation': 'Synthetic feature perturbations, not reconstructed historical reports or measured natural degradation',
              'reserved_calibration_used': False, 'evaluation_used': False,
              'validation_sha256': sha256(args.validation), 'model_sha256': sha256(args.prototype_dir/'prototype.joblib'),
              'source_sha256': sha256(Path(__file__)), 'gate_source_sha256': sha256(Path(__file__).with_name('quality.py'))}
    (out/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Step 3B: input-quality stress tests', '',
             'Prototype gate: withhold the model score for missing/invalid context, invalid history/risk, or a report older than one day at the prediction cutoff. Retain every encounter, its valid reported risk, and reason codes. The gate is a conservative development rule, not a validated safety policy.', '',
             '| Scenario | Retained | Available | Manual review | High-risk needing review | Invalid reported risk |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['encounters_retained']} | {r['available']} | {r['deferred']} | {r['high_risk_deferred']} | {r['invalid_reported_risk']} |")
    lines += ['', f'The same seeded {len(chosen)} encounters (20%) are perturbed in each scenario. Selection does not use labels. Missing-context metadata is recomputed; the stale scenario changes age features only and is not a physical reconstruction of a stale report history.', '',
              'Unavailable scores are blank, never zero. Invalid reported risk is blank and explicitly flagged; other valid reported risks are retained. Deferred cases count as review demand, not successful detections. This experiment does not claim recall after deferral, simulate operator decisions, or implement capacity overflow.', '',
              'Per-scenario CSVs contain every encounter, status, reason codes, reported risk, score, and policy version. Labels are used only for aggregate retrospective diagnostics.', '',
              'The all-or-nothing context rule may create substantial review demand. It prevents silent scoring of these unsupported inputs but does not detect every form of unfamiliar data or model error. A status of available is not a reliability guarantee.', '',
              'No models were retrained; reserved calibration and evaluation remain untouched. Next: prepare the bounded sequence-model experiment; capacity, deferral overflow, and notifications belong to the later replay engine.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
