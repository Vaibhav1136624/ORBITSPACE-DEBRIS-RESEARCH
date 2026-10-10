"""Step 3C: one fixed CPU GRU experiment on pre-cutoff encounter histories."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import CONTEXT, sha256
from .train import validate_partitions
from .baseline import queue_metrics
from .trend import trend_score

CONFIG = {'seed': 20260912, 'max_reports': 32, 'hidden_size': 16, 'epochs': 20,
          'batch_size': 128, 'learning_rate': .001, 'weight_decay': .0001,
          'cutoff_days': 2., 'threads': 2}
FIELDS = ['risk', 'time_to_tca', 'delta_time_days'] + CONTEXT


def histories(raw, event_ids, max_reports=32):
    if max_reports < 1:
        raise ValueError('Need at least one report')
    prefixes = raw.loc[raw.event_id.isin(event_ids) & (raw.time_to_tca >= 2.)].copy()
    if prefixes.duplicated(['event_id', 'time_to_tca']).any():
        raise ValueError('Ambiguous duplicate report time')
    lookup = {int(e): g.sort_values('time_to_tca', ascending=False).tail(max_reports)
              for e, g in prefixes.groupby('event_id')}
    sequences = []
    for event in event_ids:
        if event not in lookup:
            raise ValueError(f'No pre-cutoff history for encounter {event}')
        group = lookup[event].copy()
        group['delta_time_days'] = -group.time_to_tca.diff().fillna(0.)
        array = group[FIELDS].to_numpy(dtype=float)
        array[~np.isfinite(array)] = np.nan
        if not np.isfinite(array[:, :3]).all() or (array[:, 0] > 0).any():
            raise ValueError('Invalid core sequence values')
        sequences.append(array)
    return sequences


class SequenceScaler:
    def fit(self, sequences):
        # Fit on real training tokens only. Padding never enters preprocessing.
        tokens = np.concatenate(sequences)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(tokens)
        self.scaler = StandardScaler().fit(self.imputer.transform(tokens))
        return self

    def transform(self, sequences, max_reports=32):
        output = np.zeros((len(sequences), max_reports, 2*len(FIELDS)), dtype=np.float32)
        lengths = []
        for i, sequence in enumerate(sequences):
            if not 1 <= len(sequence) <= max_reports:
                raise ValueError('Sequence length outside padding limit')
            missing = ~np.isfinite(sequence)
            normalized = self.scaler.transform(self.imputer.transform(sequence))
            values = np.concatenate([normalized, missing.astype(float)], axis=1).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite transformed sequence')
            output[i, :len(sequence)] = values
            lengths.append(len(sequence))
        return torch.from_numpy(output), torch.tensor(lengths, dtype=torch.int64)


class EncounterGRU(nn.Module):
    def __init__(self, input_size=2*len(FIELDS), hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values, lengths):
        packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=Path('work/processed'))
    parser.add_argument('--raw-csv', type=Path, default=Path('work/data/train_data.csv'))
    parser.add_argument('--artifacts-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/step-3c'))
    args = parser.parse_args(argv)
    reference = json.loads((args.artifacts_dir/'step-2b/experiment.json').read_text())
    raw_manifest = json.loads((args.artifacts_dir/'run-manifest.json').read_text())
    if sha256(args.raw_csv) != raw_manifest['csv_sha256']:
        raise ValueError('Raw dataset hash mismatch')
    for name in ('train', 'validation'):
        if sha256(args.processed_dir/f'{name}.csv') != reference['input_sha256'][name]:
            raise ValueError('Partition hash mismatch')
    train = pd.read_csv(args.processed_dir/'train.csv').sort_values('event_id').reset_index(drop=True)
    val = pd.read_csv(args.processed_dir/'validation.csv').sort_values('event_id').reset_index(drop=True)
    validate_partitions(train, val)
    ids = set(train.event_id) | set(val.event_id)
    columns = ['event_id', 'risk', 'time_to_tca'] + CONTEXT
    retained = []
    for chunk in pd.read_csv(args.raw_csv, usecols=columns, chunksize=20000):
        retained.append(chunk.loc[chunk.event_id.isin(ids) & (chunk.time_to_tca >= 2.)])
    raw = pd.concat(retained, ignore_index=True)
    train_sequences = histories(raw, train.event_id.tolist())
    val_sequences = histories(raw, val.event_id.tolist())
    # Cross-check sequence endpoints against the established snapshot features.
    np.testing.assert_allclose([s[-1, 0] for s in train_sequences], train.current_log_risk)
    np.testing.assert_allclose([s[-1, 0] for s in val_sequences], val.current_log_risk)
    scaler = SequenceScaler().fit(train_sequences)
    x, lengths = scaler.transform(train_sequences)
    xv, lv = scaler.transform(val_sequences)
    torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['threads']); torch.use_deterministic_algorithms(True)
    model = EncounterGRU(hidden_size=CONFIG['hidden_size'])
    y = torch.tensor(train.target_high_risk.to_numpy(), dtype=torch.float32)
    positive_weight = float((len(y)-y.sum().item())/y.sum().item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    (out/'fixed-config.json').write_text(json.dumps(CONFIG, indent=2))
    loss_log = []; started = time.perf_counter()
    for epoch in range(CONFIG['epochs']):
        model.train(); order = torch.randperm(len(y)); total = 0.
        for batch in order.split(CONFIG['batch_size']):
            optimizer.zero_grad()
            loss = criterion(model(x[batch], lengths[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total += loss.item()*len(batch)
        loss_log.append({'epoch': epoch+1, 'training_weighted_bce': total/len(y)})
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: training loss {total/len(y):.4f}", flush=True)
    elapsed = time.perf_counter()-started
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(xv, lv)).numpy()
    if not np.isfinite(score).all():
        raise ValueError('Invalid validation predictions')
    prior = pd.read_csv(args.artifacts_dir/'step-2b/validation-predictions.csv').set_index('event_id').loc[val.event_id]
    np.testing.assert_array_equal(prior.target_high_risk, val.target_high_risk)
    models = {'latest_reported_risk': val.current_log_risk.to_numpy(), 'one_day_trend': trend_score(val)[0],
              'logistic_context': prior.plus_context.to_numpy(), 'fixed_gru': score}
    queue_path = args.artifacts_dir/'step-2b/queue-event-ids.csv'
    queue_ids = pd.read_csv(queue_path).to_numpy()
    queues = pd.Index(val.event_id).get_indexer(queue_ids.ravel()).reshape(queue_ids.shape)
    if (queues < 0).any():
        raise ValueError('Unknown queue encounter')
    labels = val.target_high_risk.to_numpy(dtype=int)
    metrics = {name: {'average_precision': float(average_precision_score(labels, scores)),
                      'queues': [queue_metrics(labels, scores, queues, k) for k in (5, 10, 20)]} for name, scores in models.items()}
    torch.save(model.state_dict(), out/'gru-state.pt')
    joblib.dump({'imputer': scaler.imputer, 'scaler': scaler.scaler,
                 'fields': FIELDS, 'max_reports': CONFIG['max_reports']}, out/'sequence-scaler.joblib')
    pd.DataFrame(loss_log).to_csv(out/'training-loss.csv', index=False)
    pd.DataFrame({'event_id': val.event_id, 'target_high_risk': labels, 'gru_score': score,
                  'sequence_length': lv.numpy()}).to_csv(out/'validation-predictions.csv', index=False)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'configuration': CONFIG,
              'positive_weight': positive_weight, 'parameters': sum(p.numel() for p in model.parameters()),
              'training_seconds': elapsed, 'sequence_fields': FIELDS, 'missingness_channels': True,
              'training_encounters': len(train), 'validation_encounters': len(val),
              'truncated_training_histories': int((train.history_reports > 32).sum()),
              'truncated_validation_histories': int((val.history_reports > 32).sum()),
              'input_sha256': reference['input_sha256'], 'raw_csv_sha256': raw_manifest['csv_sha256'],
              'queue_sha256': sha256(queue_path), 'source_sha256': sha256(Path(__file__)),
              'model_sha256': sha256(out/'gru-state.pt'),
              'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scikit-learn']},
              'reserved_calibration_used': False, 'evaluation_used': False, 'validation_used_for_early_stopping': False,
              'metrics': metrics}
    (out/'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# Step 3C: compact sequence-model experiment', '',
             f"One CPU GRU, 16 hidden units, {result['parameters']:,} parameters, 20 fixed epochs. Training took {elapsed:.1f} seconds. No architecture search, validation checkpoint selection, or early stopping.", '',
             'Inputs are the latest 32 reports available at least 48 hours before closest approach, ordered oldest to newest. Each token contains risk, time to closest approach, elapsed time since the previous retained token, selected context, and missingness flags. Single-report histories are supported. Packed sequences exclude padding from the recurrent state.', '',
             'Median imputation and scaling use real training tokens only. Final risk, labels, and identifiers are not sequence features. Endpoints were checked against existing snapshot features; the source archive and partition hashes match.', '',
             '| Method | Average precision | Recall@5 | Recall@10 | Recall@20 |', '|---|---:|---:|---:|---:|']
    for name, m in metrics.items():
        recalls = ' | '.join(f"{q['recall_at_k']:.1%}" for q in m['queues'])
        lines.append(f"| {name} | {m['average_precision']:.4f} | {recalls} |")
    lines += ['', 'The weighted classifier outputs are uncalibrated scores for a future reported-risk label, not physical impact probabilities. All methods use the same validation queues. There are only nine high-risk validation encounters. This single-seed comparison is exploratory and does not establish reliability or superiority.', '',
              'Reserved calibration and final evaluation were not used. No model is promoted automatically. Next: update the candidate decision using this bounded experiment, then implement the historical replay and capacity-aware review engine with visible deferrals.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
