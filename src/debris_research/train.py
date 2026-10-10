"""Step 2A: fixed logistic-regression experiments on train/validation only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import queue_metrics
from .data import CONTEXT, Experiment, sha256

HISTORY = ['current_log_risk', 'history_reports', 'history_span_days',
           'latest_report_age_days', 'latest_time_to_tca', 'risk_change',
           'risk_std', 'risk_min', 'risk_max', 'risk_slope_per_day', 'last_risk_change']
GROUPS = {'latest_only': HISTORY[:1], 'plus_history': HISTORY,
          'plus_context': HISTORY + list(CONTEXT) + ['context_missing_fraction']}


def validate_partitions(train, validation):
    for frame, name in [(train, 'train'), (validation, 'validation')]:
        if frame.empty or not frame.partition.eq(name).all():
            raise ValueError(f'Expected only {name} encounters')
        if frame.event_id.isna().any() or frame.event_id.duplicated().any():
            raise ValueError('Encounter IDs must be present and unique')
        if set(frame.target_high_risk.unique()) != {0, 1}:
            raise ValueError('Each partition must contain both binary classes')
    if set(train.event_id) & set(validation.event_id):
        raise ValueError('Train and validation encounters overlap')


def fit_model(train, columns, seed, kind='logistic'):
    if not train.partition.eq('train').all():
        raise ValueError('Fit accepts only the training partition')
    if not set(columns) <= set(GROUPS['plus_context']):
        raise ValueError('Only predeclared prediction features are allowed')
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    if kind == 'logistic':
        model = make_pipeline(imputer, StandardScaler(),
                              LogisticRegression(C=1.0, class_weight='balanced',
                                                 max_iter=3000, random_state=seed))
    elif kind == 'random_forest':
        model = make_pipeline(imputer, RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight='balanced_subsample', random_state=seed, n_jobs=2))
    elif kind == 'gradient_boosting':
        model = make_pipeline(imputer, HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=20, l2_regularization=1.0, class_weight='balanced',
            early_stopping=False, random_state=seed))
    else:
        raise ValueError(f'Unknown model kind: {kind}')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train[columns], train.target_high_risk)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=Path('work/processed'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--include-trees', action='store_true', help='Run step 2B with fixed tree competitors')
    args = parser.parse_args(argv)
    config = Experiment()
    # Deliberately open only these two files; reserved partitions are not inputs.
    train = pd.read_csv(args.processed_dir / 'train.csv').sort_values('event_id')
    validation = pd.read_csv(args.processed_dir / 'validation.csv').sort_values('event_id').reset_index(drop=True)
    validate_partitions(train, validation)
    if len(validation) < config.queue_size:
        raise ValueError('Validation must contain at least one full queue')
    rng = np.random.default_rng(config.seed)
    queues = np.stack([rng.choice(len(validation), config.queue_size, replace=False)
                       for _ in range(config.queue_trials)])
    labels = validation.target_high_risk.to_numpy(dtype=int)
    scores = {'latest_reported_risk': validation.current_log_risk.to_numpy()}
    models, features = {}, {}
    kinds = ['logistic', 'random_forest', 'gradient_boosting'] if args.include_trees else ['logistic']
    for kind in kinds:
        for group, columns in GROUPS.items():
            name = group if kind == 'logistic' else f'{kind}_{group}'
            print(f'Training {kind}: {group} ({len(columns)} features)', flush=True)
            models[name] = fit_model(train, columns, config.seed, kind)
            features[name] = columns
            scores[name] = models[name].predict_proba(validation[columns])[:, 1]
    metrics = {}
    for name, values in scores.items():
        metrics[name] = {'average_precision': float(average_precision_score(labels, values)),
                         'queues': [queue_metrics(labels, values, queues, k) for k in (5, 10, 20)]}
    step = '2B' if args.include_trees else '2A'
    out = args.output_dir or Path(f'artifacts/step-{step.lower()}')
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump({'pipeline': model, 'features': features[name],
                     'score_definition': 'Uncalibrated score for final reported risk >= 1e-6'},
                    out / f'{name}.joblib')
    predictions = validation[['event_id', 'target_high_risk']].copy()
    for name, values in scores.items():
        predictions[name] = values
    predictions.to_csv(out / 'validation-predictions.csv', index=False)
    # Keep the queue identities so every comparison can be reproduced exactly.
    pd.DataFrame(validation.event_id.to_numpy()[queues]).to_csv(out / 'queue-event-ids.csv', index=False)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'step': step, 'seed': config.seed, 'feature_groups': GROUPS,
                'model_parameters': {name: model.steps[-1][1].get_params() for name, model in models.items()},
                'configuration': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 3000,
                                  'queue_size': config.queue_size, 'queue_trials': config.queue_trials},
                'train_encounters': len(train), 'train_high_risk': int(train.target_high_risk.sum()),
                'validation_encounters': len(validation), 'validation_high_risk': int(labels.sum()),
                'input_sha256': {name: sha256(args.processed_dir / f'{name}.csv') for name in ('train', 'validation')},
                'source_sha256': sha256(Path(__file__)),
                'calibration_used': False, 'evaluation_performance_computed': False,
                'metrics': metrics}
    (out / 'experiment.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    lines = [f'# Step {step}: trained model comparison', '',
             'Models predict whether the final reported collision probability exceeds the benchmark threshold (1e-6).', '',
             'The class-weighted outputs are **uncalibrated ranking scores**, not physical collision probabilities.', '',
             f'Training: {len(train):,} encounters / {int(train.target_high_risk.sum())} high-risk labels. Validation: {len(validation):,} encounters / {int(labels.sum())} high-risk labels.', '',
             'Median imputation and logistic-regression scaling were fitted on training only. All configurations were fixed before this run; no hyperparameter search. Full parameters are in experiment.json.', '',
             '| Method | Average precision | Recall@5 | Recall@10 | Recall@20 |',
             '|---|---:|---:|---:|---:|']
    for name, result in metrics.items():
        recalls = ' | '.join(f"{m['recall_at_k']:.1%}" if m['recall_at_k'] is not None else 'N/A' for m in result['queues'])
        lines.append(f"| {name} | {result['average_precision']:.4f} | {recalls} |")
    lines += ['', 'All methods use the same 200 simulated queues of 50 validation encounters. Encounters can repeat across queues; these are not 200 independent trials.', '',
              f'Only {int(labels.sum())} high-risk validation encounters exist in this cohort. Results are exploratory; no confidence or superiority claim follows from these point estimates.', '',
              'History/context ablations compare feature groups. No production model is selected in this step. Calibration and locked evaluation data were not opened.', '',
              'Next: investigate encounter-level errors and uncertainty before selecting a candidate.' if args.include_trees else 'Next: compare fixed Random Forest and gradient-boosting candidates.']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
