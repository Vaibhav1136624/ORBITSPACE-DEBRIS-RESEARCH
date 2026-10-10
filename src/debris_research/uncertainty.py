"""Step 3A: training-internal calibration and split-conformal diagnostics."""
import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from .data import sha256
from .train import GROUPS, fit_model, validate_partitions
from .regression import fit_regressor

SEED = 20260912


def inner_split(train):
    if train.empty or not train.partition.eq('train').all() or train.event_id.duplicated().any():
        raise ValueError('Unique training encounters required')
    train = train.sort_values('event_id')
    return train_test_split(train, test_size=.2, random_state=SEED,
                            stratify=train.target_high_risk)


def conformal_radius(residuals, coverage=.9):
    residuals = np.asarray(residuals, dtype=float)
    if not len(residuals) or not np.isfinite(residuals).all() or (residuals < 0).any() or not 0 < coverage < 1:
        raise ValueError('Nonnegative finite residuals and valid coverage required')
    rank = int(np.ceil((len(residuals)+1)*coverage))
    if rank > len(residuals):
        raise ValueError('Insufficient calibration samples for finite interval')
    return float(np.sort(residuals)[rank-1]), rank


def intervals(predictions, radius):
    center = np.minimum(np.asarray(predictions, dtype=float), 0.)
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius < 0:
        raise ValueError('Finite predictions and nonnegative radius required')
    return center-radius, np.minimum(center+radius, 0.)


def coverage_metrics(target, lower, upper, mask):
    target, lower, upper = [np.asarray(x)[mask] for x in (target, lower, upper)]
    return {'encounters': len(target), 'covered': int(((target >= lower) & (target <= upper)).sum()),
            'coverage': float(((target >= lower) & (target <= upper)).mean()) if len(target) else None,
            'mean_width': float((upper-lower).mean()) if len(target) else None}


def reliability_bins(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Aligned probabilities in [0,1] required')
    bins = np.minimum((probabilities*5).astype(int), 4)
    return [{'lower': i/5, 'upper': (i+1)/5, 'count': int((bins == i).sum()),
             'positives': int(labels[bins == i].sum()),
             'mean_probability': float(probabilities[bins == i].mean()) if (bins == i).any() else None,
             'observed_fraction': float(labels[bins == i].mean()) if (bins == i).any() else None} for i in range(5)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=Path('work/processed'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/step-3a'))
    args = parser.parse_args(argv)
    train = pd.read_csv(args.processed_dir/'train.csv')
    val = pd.read_csv(args.processed_dir/'validation.csv').sort_values('event_id').reset_index(drop=True)
    validate_partitions(train, val)
    fit, cal = inner_split(train)
    if set(fit.event_id) & set(cal.event_id):
        raise ValueError('Internal calibration overlap')
    print(f'Inner fit: {len(fit)} / {int(fit.target_high_risk.sum())} positive; inner calibration: {len(cal)} / {int(cal.target_high_risk.sum())} positive', flush=True)
    columns = GROUPS['plus_context']
    classifier = fit_model(fit, columns, SEED)
    sigmoid = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=SEED)
    sigmoid.fit(classifier.decision_function(cal[columns]).reshape(-1, 1), cal.target_high_risk)
    probabilities = {'training_prevalence': np.full(len(val), fit.target_high_risk.mean()),
                     'raw_logistic': classifier.predict_proba(val[columns])[:, 1],
                     'sigmoid_calibrated': sigmoid.predict_proba(classifier.decision_function(val[columns]).reshape(-1, 1))[:, 1]}
    y = val.target_high_risk.to_numpy()
    classification = {name: {'brier': float(brier_score_loss(y, prob)),
                              'log_loss': float(log_loss(y, prob, labels=[0, 1])),
                              'average_precision': float(average_precision_score(y, prob)),
                              'bins': reliability_bins(y, prob)} for name, prob in probabilities.items()}
    regressor = fit_regressor(fit, columns)
    centers = {'persistence': (cal.current_log_risk.to_numpy(), val.current_log_risk.to_numpy()),
               'context_regressor': (np.minimum(regressor.predict(cal[columns]), 0.),
                                     np.minimum(regressor.predict(val[columns]), 0.))}
    output = val[['event_id', 'target_high_risk', 'target_final_log_risk']].copy()
    for name, prob in probabilities.items():
        output[name] = prob
    regression = {}
    for name, (cal_center, val_center) in centers.items():
        radius, rank = conformal_radius(np.abs(cal.target_final_log_risk.to_numpy()-cal_center))
        lower, upper = intervals(val_center, radius)
        regression[name] = {'radius': radius, 'quantile_rank': rank,
                            'all': coverage_metrics(val.target_final_log_risk, lower, upper, np.ones(len(val), dtype=bool)),
                            'high_risk': coverage_metrics(val.target_final_log_risk, lower, upper, y == 1)}
        output[name+'_lower'] = lower; output[name+'_upper'] = upper
        output[name+'_center'] = val_center
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split = pd.concat([fit[['event_id']].assign(role='inner_fit'), cal[['event_id']].assign(role='inner_calibration')]).sort_values('event_id')
    split.to_csv(out/'inner-split.csv', index=False)
    output.to_csv(out/'validation-diagnostics.csv', index=False)
    joblib.dump({'classifier': classifier, 'sigmoid': sigmoid, 'regressor': regressor,
                 'features': columns, 'interval_parameters': regression, 'prototype_only': True}, out/'prototype.joblib')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'fit_encounters': len(fit), 'fit_positives': int(fit.target_high_risk.sum()),
                'inner_calibration_encounters': len(cal), 'inner_calibration_positives': int(cal.target_high_risk.sum()),
                'reserved_calibration_used': False, 'evaluation_used': False, 'nominal_coverage': .9,
                'input_sha256': {n: sha256(args.processed_dir/f'{n}.csv') for n in ('train', 'validation')},
                'split_sha256': sha256(out/'inner-split.csv'), 'source_sha256': sha256(Path(__file__)),
                'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
                'parameters': {'classifier': classifier.steps[-1][1].get_params(), 'sigmoid': sigmoid.get_params(), 'regressor': regressor.steps[-1][1].get_params()},
                'classification': classification, 'regression_intervals': regression}
    (out/'experiment.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    lines = ['# Step 3A: uncertainty prototype', '',
             f'Inner training: {len(fit)} encounters, {int(fit.target_high_risk.sum())} positives. Inner calibration: {len(cal)} encounters, {int(cal.target_high_risk.sum())} positives. All are from the original training partition.', '',
             'Both learned models were refitted on inner training only. Reserved calibration and evaluation were not opened. Validation is reused for exploratory diagnostics; it is not an independent final test.', '',
             '## Classification diagnostics', '', '| Method | Brier score | Log loss | Average precision |', '|---|---:|---:|---:|']
    for name, m in classification.items():
        lines.append(f"| {name} | {m['brier']:.5f} | {m['log_loss']:.5f} | {m['average_precision']:.4f} |")
    lines += ['', 'Lower Brier/log loss is better. Under rare labels, a constant low probability can score well; discrimination and positive counts must also be inspected. Scores estimate the final reported-risk threshold label, not physical collision probability.', '',
              'Five fixed-width reliability bins and their counts are recorded in experiment.json. Sparse/empty bins cannot establish reliability. Sigmoid calibration is fixed, regularized C=1, unweighted, with no tuning.', '',
              '## Nominal 90% regression intervals', '', '| Center | Radius (log10) | Overall coverage | High-risk coverage | Mean width |', '|---|---:|---:|---:|---:|']
    for name, m in regression.items():
        lines.append(f"| {name} | {m['radius']:.3f} | {m['all']['covered']}/{m['all']['encounters']} ({m['all']['coverage']:.1%}) | {m['high_risk']['covered']}/{m['high_risk']['encounters']} ({m['high_risk']['coverage']:.1%}) | {m['all']['mean_width']:.3f} |")
    lines += ['', 'Absolute residuals use the finite-sample order statistic ceil((n+1)*0.90). Intervals are intersected with log-risk <= 0. Widths are log10 units and can span many orders of magnitude; coverage alone does not make an interval useful.', '',
              'Conformal coverage depends on exchangeability and is marginal, not a guarantee for the high-risk subgroup or a real fleet. Seven inner-calibration positives and nine validation positives provide little evidence about rare cases. These reduced-training results are not a direct model comparison with step 2.', '',
              'No model is promoted and no operational confidence claim is made. Next: inspect controlled missing/stale-input behaviour and define explicit unavailable-assessment handling; the compact sequence comparison remains outstanding.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
