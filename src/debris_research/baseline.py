"""Validation-only ranking baselines and reproducible simulated review queues."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .data import Experiment


def queue_metrics(labels: np.ndarray, scores: np.ndarray, queues: np.ndarray, k: int) -> dict:
    if not (1 <= k <= queues.shape[1]):
        raise ValueError("K must fit within every queue")
    total_true = total_found = 0
    zero_positive_queues = 0
    for indices in queues:
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("An encounter must not be repeated within a queue")
        ordered = indices[np.argsort(-scores[indices], kind="stable")]
        positives = int(labels[indices].sum())
        found = int(labels[ordered[:k]].sum())
        total_true += positives
        total_found += found
        zero_positive_queues += int(positives == 0)
    return {
        "k": k,
        "recall_at_k": total_found / total_true if total_true else None,
        "precision_at_k": total_found / (len(queues) * k),
        "mean_missed_high_risk_per_queue": (total_true - total_found) / len(queues),
        "mean_high_risk_per_queue": total_true / len(queues),
        "queues_without_high_risk": zero_positive_queues,
        "queue_trials": len(queues),
    }


def evaluate_baselines(validation: pd.DataFrame, config: Experiment) -> tuple[dict, pd.DataFrame]:
    if "partition" not in validation or not validation.partition.eq("validation").all():
        raise ValueError("This milestone only evaluates the validation partition")
    if len(validation) < config.queue_size:
        raise ValueError("Validation cohort must be at least as large as the queue")
    validation = validation.sort_values("event_id").reset_index(drop=True)
    labels = validation.target_high_risk.to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation cohort needs both classes")
    rng = np.random.default_rng(config.seed)
    queues = np.stack([
        rng.choice(len(validation), config.queue_size, replace=False)
        for _ in range(config.queue_trials)
    ])
    latest = validation.current_log_risk.to_numpy(dtype=float)
    # Transparent persistence forecast: final log-risk is unchanged from current log-risk.
    latest_ranking = latest
    models = {"latest_reported_risk": latest_ranking, "random_order_reference": rng.random(len(labels))}
    metrics = {
        "partition": "validation",
        "evaluation_performance_computed": False,
        "validation_encounters": len(validation),
        "validation_high_risk_encounters": int(labels.sum()),
        "queue_size": config.queue_size,
        "queue_trials": config.queue_trials,
        "queue_seed": config.seed,
        "interpretation": "Synthetic queues from an enriched historical benchmark, not simultaneous operational traffic. Reused encounters make trial variability unsuitable as a confidence interval.",
        "baselines": {},
    }
    for name, scores in models.items():
        metrics["baselines"][name] = {
            "average_precision": float(average_precision_score(labels, scores)),
            "queues": [queue_metrics(labels, scores, queues, k)
                       for k in (5, 10, 20) if k <= config.queue_size],
        }
    example = validation.iloc[queues[0]].copy()
    example = example.sort_values(["current_log_risk", "event_id"], ascending=[False, True])
    example.insert(0, "review_rank", np.arange(1, len(example) + 1))
    return metrics, example
