"""Step 2C: validation error analysis and paired encounter bootstrap."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import sha256

MODELS = ['latest_reported_risk', 'plus_context',
          'random_forest_plus_context', 'gradient_boosting_plus_context']


def paired_bootstrap(labels, scores, trials=1000, seed=20260912):
    """Resample encounters within each class; use identical draws for every model."""
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both binary classes are required')
    if trials < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    if any(len(s) != len(labels) or not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Scores must be finite and aligned with labels')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    samples = {name: [] for name in scores}
    for _ in range(trials):
        ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for name, values in scores.items():
            samples[name].append(average_precision_score(labels[ids], np.asarray(values)[ids]))
    baseline = np.asarray(samples['latest_reported_risk'])
    return {name: {
        'average_precision': float(average_precision_score(labels, scores[name])),
        'ap_percentile_95': np.quantile(values, [.025, .975]).tolist(),
        'paired_ap_difference_percentile_95': np.quantile(np.asarray(values)-baseline, [.025, .975]).tolist(),
    } for name, values in samples.items()}


def encounter_queue_counts(predictions, queue_ids, k, model):
    """Count appearances and selections, preserving saved queue order for ties."""
    if predictions.event_id.duplicated().any():
        raise ValueError('Duplicate encounter ID')
    index = pd.Index(predictions.event_id)
    locations = index.get_indexer(np.asarray(queue_ids).ravel()).reshape(np.asarray(queue_ids).shape)
    if (locations < 0).any():
        raise ValueError('Queue contains unknown encounter')
    if not 1 <= k <= locations.shape[1]:
        raise ValueError('Invalid review capacity')
    appearances = np.zeros(len(predictions), dtype=int)
    hits = appearances.copy()
    scores = predictions[model].to_numpy()
    for queue in locations:
        if len(np.unique(queue)) != len(queue):
            raise ValueError('Duplicate encounter within queue')
        appearances[queue] += 1
        chosen = queue[np.argsort(-scores[queue], kind='stable')[:k]]
        hits[chosen] += 1
    return appearances, hits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('artifacts/step-2b'))
    parser.add_argument('--validation', type=Path, default=Path('work/processed/validation.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/step-2c'))
    parser.add_argument('--bootstrap-trials', type=int, default=1000)
    args = parser.parse_args(argv)
    manifest = json.loads((args.input_dir/'experiment.json').read_text())
    if sha256(args.validation) != manifest['input_sha256']['validation']:
        raise ValueError('Validation hash differs from training experiment')
    validation = pd.read_csv(args.validation)
    if not validation.partition.eq('validation').all():
        raise ValueError('Only validation is permitted')
    predictions = pd.read_csv(args.input_dir/'validation-predictions.csv')
    joined = predictions.merge(validation, on='event_id', suffixes=('', '_source'), validate='one_to_one')
    if len(joined) != len(validation) or len(joined) != len(predictions):
        raise ValueError('Prediction IDs differ from validation')
    if not joined.target_high_risk.eq(joined.target_high_risk_source).all():
        raise ValueError('Prediction labels differ from validation')
    queues = pd.read_csv(args.input_dir/'queue-event-ids.csv').to_numpy()
    labels = predictions.target_high_risk.to_numpy()
    print(f'Paired bootstrap: {args.bootstrap_trials} draws, {len(predictions)} encounters', flush=True)
    uncertainty = paired_bootstrap(labels, {m: predictions[m].to_numpy() for m in MODELS}, args.bootstrap_trials)
    records, sensitivity = [], {}
    for model in MODELS:
        sensitivity[model] = {}
        for k in (5, 10):
            appearances, hits = encounter_queue_counts(predictions, queues, k, model)
            positive = labels == 1
            total, found = appearances[positive].sum(), hits[positive].sum()
            recalls = []
            for i in np.flatnonzero(positive):
                denominator = total - appearances[i]
                recall_without = float((found-hits[i])/denominator) if denominator else None
                if recall_without is not None:
                    recalls.append(recall_without)
                records.append({'event_id': int(predictions.event_id.iloc[i]), 'model': model,
                                'k': k, 'appearances': int(appearances[i]), 'selected': int(hits[i]),
                                'missed': int(appearances[i]-hits[i]),
                                'recall_excluding_this_positive_from_denominator': recall_without})
            sensitivity[model][str(k)] = {'recall': float(found/total),
                                          'leave_one_positive_out_range': [min(recalls), max(recalls)]}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out/'high-risk-queue-errors.csv', index=False)
    columns = ['event_id', 'current_log_risk', 'target_final_log_risk', 'history_reports',
               'risk_change', 'latest_report_age_days', 'context_missing_fraction'] + MODELS
    joined.loc[joined.target_high_risk == 1, columns].to_csv(out/'high-risk-encounters.csv', index=False)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'bootstrap_trials': args.bootstrap_trials,
              'seed': 20260912, 'method': 'paired stratified encounter percentile bootstrap, fixed trained models',
              'ap_uncertainty': uncertainty, 'queue_sensitivity': sensitivity,
              'source_hashes': {f: sha256(args.input_dir/f) for f in ['experiment.json', 'validation-predictions.csv', 'queue-event-ids.csv']},
              'analysis_source_sha256': sha256(Path(__file__)), 'evaluation_used': False, 'calibration_used': False}
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Step 2C: errors and uncertainty', '',
             f'Analysed {len(predictions)} validation encounters, including {int(labels.sum())} high-risk labels. No models were refitted.', '',
             '## Overall ranking uncertainty', '',
             '| Method | Average precision | 95% bootstrap interval | Paired AP difference vs baseline, 95% interval |',
             '|---|---:|---:|---:|']
    for model, values in uncertainty.items():
        lo, hi = values['ap_percentile_95']; dlo, dhi = values['paired_ap_difference_percentile_95']
        lines.append(f"| {model} | {values['average_precision']:.3f} | {lo:.3f} to {hi:.3f} | {dlo:+.3f} to {dhi:+.3f} |")
    lines += ['', f'{args.bootstrap_trials} paired bootstrap draws resample encounters within each class. Models share each draw. These exploratory intervals condition on the observed class balance and fixed trained models; they do not include training uncertainty, distribution shift, or adjustment for examining multiple candidates. Nine positive encounters provide limited information.', '',
              '## Queue sensitivity', '', '| Method | Recall@5 | Leave-one-positive-out range | Recall@10 | Leave-one-positive-out range |', '|---|---:|---:|---:|---:|']
    for model, values in sensitivity.items():
        row = []
        for k in ('5', '10'):
            lo, hi = values[k]['leave_one_positive_out_range']
            row += [f"{values[k]['recall']:.1%}", f'{lo:.1%} to {hi:.1%}']
        lines.append('| '+model+' | '+' | '.join(row)+' |')
    lines += ['', 'The ranges remove one high-risk encounter at a time from recall accounting while leaving saved queues and rankings fixed. They are sensitivity checks, **not confidence intervals** and not reranked queues. Repeated queues are not treated as independent observations.', '',
              'Inspect high-risk-queue-errors.csv for each positive encounter\'s appearances, selections, and misses; high-risk-encounters.csv provides its observed history and retrospective final risk. Final risk is used only for analysis.', '',
              '## Decision', '', 'Keep latest-risk sorting as the working reference. These validation experiments do not establish an improvement in top-5/top-10 prioritisation. No production model is selected. Calibration and locked evaluation were not used.', '',
              'Next: add the planned transparent trend baseline and analyse quality/sparse-history groups before extending the forecasting experiments.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
