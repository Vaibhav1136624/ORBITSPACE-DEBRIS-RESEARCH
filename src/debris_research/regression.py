"""Step 2E: fixed final-log-risk regression compared with persistence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline

from .baseline import queue_metrics
from .data import sha256
from .train import GROUPS, validate_partitions
from .trend import trend_score


def fit_regressor(train, columns):
    if train.empty or not train.partition.eq('train').all():
        raise ValueError('Fit accepts training only')
    if not set(columns) <= set(GROUPS['plus_context']):
        raise ValueError('Only predeclared prediction features are allowed')
    target = train.target_final_log_risk.to_numpy()
    if not np.isfinite(target).all() or (target > 0).any():
        raise ValueError('Final log-risk must be finite and <= 0')
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                          HistGradientBoostingRegressor(loss='squared_error', max_iter=150,
                            learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=20,
                            l2_regularization=1., early_stopping=False, random_state=20260912))
    model.fit(train[columns], target)
    return model


def error_metrics(target, predicted, mask=None):
    target, predicted = np.asarray(target), np.asarray(predicted)
    if target.shape != predicted.shape or not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError('Targets and predictions must be finite and aligned')
    error = predicted-target
    if mask is not None:
        error = error[np.asarray(mask)]
    if not len(error):
        return {'encounters': 0, 'mae': None, 'rmse': None, 'mean_signed_error': None}
    return {'encounters': len(error), 'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.square(error).mean())), 'mean_signed_error': float(error.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=Path('work/processed'))
    parser.add_argument('--reference-dir', type=Path, default=Path('artifacts/step-2b'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/step-2e'))
    args = parser.parse_args(argv)
    reference = json.loads((args.reference_dir/'experiment.json').read_text())
    for name in ('train', 'validation'):
        if sha256(args.processed_dir/f'{name}.csv') != reference['input_sha256'][name]:
            raise ValueError('Input hash mismatch')
    train = pd.read_csv(args.processed_dir/'train.csv')
    val = pd.read_csv(args.processed_dir/'validation.csv').sort_values('event_id').reset_index(drop=True)
    validate_partitions(train, val)
    queue_ids = pd.read_csv(args.reference_dir/'queue-event-ids.csv').to_numpy()
    queues = pd.Index(val.event_id).get_indexer(queue_ids.ravel()).reshape(queue_ids.shape)
    if (queues < 0).any():
        raise ValueError('Unknown encounter in reference queue')
    forecasts = {'persistence': val.current_log_risk.to_numpy(), 'one_day_trend': trend_score(val)[0]}
    models = {}
    for group, columns in GROUPS.items():
        print(f'Training fixed regressor: {group}', flush=True)
        models[group] = fit_regressor(train, columns)
        forecasts[f'regression_{group}'] = np.minimum(models[group].predict(val[columns]), 0.)
    target = val.target_final_log_risk.to_numpy()
    labels = val.target_high_risk.to_numpy(dtype=int)
    metrics = {name: {'all': error_metrics(target, scores),
                      'high_risk': error_metrics(target, scores, labels == 1),
                      'average_precision': float(average_precision_score(labels, scores)),
                      'queues': [queue_metrics(labels, scores, queues, k) for k in (5, 10, 20)]}
               for name, scores in forecasts.items()}
    # Paired encounter bootstrap of MAE differences; fixed class balance and fitted models.
    rng = np.random.default_rng(20260912)
    classes = [np.flatnonzero(labels == c) for c in (0, 1)]
    draws = np.stack([np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in classes]) for _ in range(1000)])
    base_error = np.abs(forecasts['persistence']-target)
    for name, scores in forecasts.items():
        delta = np.abs(scores-target)-base_error
        metrics[name]['paired_mae_difference_95'] = np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump({'pipeline': model, 'features': GROUPS[name], 'prediction_cap': 0.,
                     'target': 'final reported log10 collision probability'}, out/f'{name}.joblib')
    predictions = val[['event_id', 'target_final_log_risk', 'target_high_risk']].copy()
    for name, scores in forecasts.items():
        predictions[name] = scores
    predictions.to_csv(out/'validation-forecasts.csv', index=False)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'step': '2E',
              'target': 'final reported log10 probability, not collision outcome',
              'input_sha256': reference['input_sha256'], 'queue_sha256': sha256(args.reference_dir/'queue-event-ids.csv'),
              'source_sha256': sha256(Path(__file__)), 'feature_groups': GROUPS,
              'parameters': models['plus_context'].steps[-1][1].get_params(),
              'train_encounters': len(train), 'validation_encounters': len(val),
              'calibration_used': False, 'evaluation_used': False, 'metrics': metrics,
              'bootstrap': {'trials': 1000, 'seed': 20260912, 'method': 'paired stratified encounter percentile, fixed fitted models'}}
    (out/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Step 2E: final reported log-risk forecasts', '',
             'A fixed histogram gradient-boosting regressor learns final reported log10 risk. All three feature groups use identical settings, training-only median imputation, squared-error loss, and no hyperparameter search. Predictions are capped at zero. No class weighting is used.', '',
             f'Training: {len(train):,} encounters. Validation: {len(val):,}, including {int(labels.sum())} high-risk labels. Existing validation queues are reused.', '',
             '| Method | MAE | RMSE | High-risk MAE | Average precision | Recall@5 | Recall@10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['all']['mae']:.3f} | {m['all']['rmse']:.3f} | {m['high_risk']['mae']:.3f} | {m['average_precision']:.3f} | {m['queues'][0]['recall_at_k']:.1%} | {m['queues'][1]['recall_at_k']:.1%} |")
    lines += ['', 'Errors are in log10 probability units: one unit corresponds to a factor of ten in reported probability. Average log-error cannot be interpreted as an average physical collision probability.', '',
              '## Paired MAE difference versus persistence', '', 'Negative values favour the comparator. These are 95% percentile intervals from 1,000 paired stratified encounter bootstrap draws, conditional on fixed fitted models and observed class balance.', '']
    for name, m in metrics.items():
        lo, hi = m['paired_mae_difference_95']
        lines.append(f'- {name}: {lo:+.3f} to {hi:+.3f} log10 units.')
    lines += ['', 'The target is a later reported assessment, not whether a collision occurred. Prediction intervals and calibrated probabilities are not implemented in this step. The bootstrap excludes training variation, distribution shift, and multiple-comparison adjustment.', '',
              'Overall error can hide failures on rare high-risk encounters; inspect both high-risk error and queue recall. Nine high-risk validation encounters cannot support a strong reliability claim. No operational model is selected here. Calibration/evaluation remain reserved.', '',
              'Next: consolidate the model comparisons and define candidate selection and uncertainty experiments before adding complexity.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
