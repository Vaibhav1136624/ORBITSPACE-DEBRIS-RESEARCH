"""Step 2D: fixed one-day trend comparator and descriptive quality groups."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .analyze import MODELS, encounter_queue_counts, paired_bootstrap
from .data import sha256


def trend_score(frame):
    """Extrapolate available log-risk slope one day; single-report fallback is persistence."""
    current = frame.current_log_risk.to_numpy(dtype=float)
    slope = frame.risk_slope_per_day.to_numpy(dtype=float)
    usable = (frame.history_reports.to_numpy() >= 2) & (frame.history_span_days.to_numpy() > 0) & np.isfinite(slope)
    if not np.isfinite(current).all() or (current > 0).any():
        raise ValueError('Current log-risk must be finite and <= 0')
    score = current.copy()
    score[usable] += slope[usable]  # Exactly one day, without validation tuning.
    return np.minimum(score, 0.), ~usable


def quality_groups(frame):
    return {
        'all': np.ones(len(frame), dtype=bool),
        'sparse_history_1_or_2_reports': frame.history_reports.to_numpy() <= 2,
        'history_3_or_more_reports': frame.history_reports.to_numpy() >= 3,
        'latest_report_over_1_day_old': frame.latest_report_age_days.to_numpy() > 1.,
        'latest_report_at_most_1_day_old': frame.latest_report_age_days.to_numpy() <= 1.,
        'any_missing_context': frame.context_missing_fraction.to_numpy() > 0.,
        'complete_context': frame.context_missing_fraction.to_numpy() == 0.,
    }


def group_recall(labels, appearances, hits, mask):
    positives = (np.asarray(labels) == 1) & mask
    count = int(appearances[positives].sum())
    selected = int(hits[positives].sum())
    return {'encounters': int(mask.sum()), 'high_risk_encounters': int(positives.sum()),
            'positive_queue_appearances': count, 'selected_positive_appearances': selected,
            'missed_positive_appearances': count-selected,
            'recall': selected/count if count else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('artifacts/step-2b'))
    parser.add_argument('--validation', type=Path, default=Path('work/processed/validation.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/step-2d'))
    args = parser.parse_args(argv)
    source = json.loads((args.input_dir/'experiment.json').read_text())
    if sha256(args.validation) != source['input_sha256']['validation']:
        raise ValueError('Validation hash mismatch')
    v = pd.read_csv(args.validation)
    if not v.partition.eq('validation').all():
        raise ValueError('Only validation is permitted')
    p = pd.read_csv(args.input_dir/'validation-predictions.csv')
    frame = p.merge(v, on='event_id', suffixes=('', '_source'), validate='one_to_one')
    if len(frame) != len(v) or len(frame) != len(p) or not frame.target_high_risk.eq(frame.target_high_risk_source).all():
        raise ValueError('Prediction and validation identities/labels differ')
    frame['one_day_trend'], fallback = trend_score(frame)
    queues = pd.read_csv(args.input_dir/'queue-event-ids.csv').to_numpy()
    labels = frame.target_high_risk.to_numpy()
    records = []
    for model in MODELS + ['one_day_trend']:
        for k in (5, 10, 20):
            appearances, hits = encounter_queue_counts(frame, queues, k, model)
            for group, mask in quality_groups(frame).items():
                records.append({'model': model, 'k': k, 'group': group,
                                **group_recall(labels, appearances, hits, mask)})
    uncertainty = paired_bootstrap(labels, {m: frame[m].to_numpy() for m in ['latest_reported_risk', 'one_day_trend']})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'rule': 'min(0, current_log_risk + risk_slope_per_day * 1 day); persistence when trend unavailable',
              'horizon_days': 1, 'fallback_encounters': int(fallback.sum()),
              'group_definitions': {'sparse': 'history_reports <= 2', 'stale': 'latest_report_age_days > 1', 'missing': 'context_missing_fraction > 0'},
              'design_status': 'Fixed before this run, but after earlier validation inspection; exploratory, not a preregistered independent test.',
              'calibration_used': False, 'evaluation_used': False,
              'input_sha256': {'validation': sha256(args.validation), 'predictions': sha256(args.input_dir/'validation-predictions.csv'), 'queues': sha256(args.input_dir/'queue-event-ids.csv')},
              'source_sha256': sha256(Path(__file__)), 'ap_uncertainty': uncertainty, 'group_metrics': records}
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    pd.DataFrame(records).to_csv(out/'quality-group-metrics.csv', index=False)
    saved = frame[['event_id', 'target_high_risk', 'current_log_risk', 'risk_slope_per_day', 'one_day_trend']].copy()
    saved['persistence_fallback'] = fallback
    saved.to_csv(out/'trend-predictions.csv', index=False)
    def percent(x):
        return 'N/A' if x is None else f'{x:.1%}'
    lines = ['# Step 2D: trend and data-quality comparisons', '',
             'Rule: current log10 reported risk + available elapsed-time slope x one day, capped at zero. A single report, zero history span, or unavailable slope falls back to current risk. This is a ranking comparator, not a validated physical forecast.', '',
             f'{int(fallback.sum())} encounters use persistence fallback. The one-day horizon and descriptive group thresholds were fixed before this run, after earlier validation inspection; results remain exploratory. No tuning was performed.', '',
             '## Overall comparison', '', '| Model | Recall@5 | Recall@10 | Recall@20 |', '|---|---:|---:|---:|']
    for model in MODELS + ['one_day_trend']:
        cells = [percent(r['recall']) for r in records if r['model'] == model and r['group'] == 'all']
        lines.append('| '+model+' | '+' | '.join(cells)+' |')
    ap = uncertainty['one_day_trend']; lo, hi = ap['paired_ap_difference_percentile_95']
    lines += ['', f"Trend average precision: {ap['average_precision']:.3f}; paired difference vs baseline 95% bootstrap interval: {lo:+.3f} to {hi:+.3f}.", '',
              'Bootstrap: 1,000 paired stratified encounter draws, seed 20260912; fixed models and observed class balance. Intervals exclude training variability, distribution shift, and multiple-comparison adjustment.', '',
              '## Descriptive groups at review capacity 10', '',
              '| Group | Encounters | High-risk encounters | Baseline recall | Trend recall |', '|---|---:|---:|---:|---:|']
    for group in quality_groups(frame):
        if group == 'all':
            continue
        base = next(r for r in records if r['group'] == group and r['model'] == 'latest_reported_risk' and r['k'] == 10)
        trend = next(r for r in records if r['group'] == group and r['model'] == 'one_day_trend' and r['k'] == 10)
        lines.append(f"| {group} | {base['encounters']} | {base['high_risk_encounters']} | {percent(base['recall'])} | {percent(trend['recall'])} |")
    lines += ['', 'All methods rank the original full queues. Group recall counts only positives in the group; queues are not rebuilt per group. N/A means no positive queue appearances, not zero performance. Groups overlap across dimensions and should not be added together.', '',
              'History, freshness, and missingness are descriptive indicators, not validated reliability scores. Complete context does not imply reliable measurements. With only nine positive validation encounters, subgroup results cannot establish robust differences or causation.', '',
              'All model/group details are in quality-group-metrics.csv. Calibration and final evaluation were not used.', '',
              'Next: define the regression experiment for final reported log-risk, compare persistence with a fixed regressor, and retain the baseline as the working ranking reference until evidence supports a change.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
