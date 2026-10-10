"""Audit reports and construct features using only information known at a cutoff."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

SOURCE_URL = "https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip"
REQUIRED = ("event_id", "time_to_tca", "risk")
CONTEXT = [
    "miss_distance", "relative_speed", "max_risk_estimate",
    "t_sigma_r", "t_sigma_t", "t_sigma_n",
    "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_obs_available", "t_obs_used", "c_obs_available", "c_obs_used",
    "t_time_lastob_end", "c_time_lastob_end", "t_weighted_rms", "c_weighted_rms",
]

@dataclass(frozen=True)
class Experiment:
    cutoff_days: float = 2.0
    final_report_max_days: float = 1.0
    log_risk_threshold: float = -6.0
    seed: int = 20260912
    queue_size: int = 50
    queue_trials: int = 200

    def __post_init__(self):
        if not (0 < self.final_report_max_days < self.cutoff_days):
            raise ValueError("Final-report horizon must be positive and before the prediction cutoff")
        if not np.isfinite(self.log_risk_threshold) or self.log_risk_threshold > 0:
            raise ValueError("Risk threshold must be a finite log10 probability <= 0")
        if self.queue_size < 1 or self.queue_trials < 1:
            raise ValueError("Queue dimensions must be positive")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing mandatory columns: {sorted(missing)}")
    frame = raw.copy()
    frame.columns = frame.columns.str.strip()
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    for column in REQUIRED:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid_ids = np.isfinite(frame.event_id) & (frame.event_id >= 0) & (frame.event_id % 1 == 0)
    invalid_id_rows = int((~valid_ids).sum())
    frame = frame.loc[valid_ids].copy()
    frame["event_id"] = frame.event_id.astype("int64")
    valid_core = (
        np.isfinite(frame.risk) & (frame.risk <= 0)
        & np.isfinite(frame.time_to_tca) & (frame.time_to_tca >= 0)
    )
    invalid_events = set(frame.loc[~valid_core, "event_id"])
    # Exclude the whole event, so deleting a bad final row cannot silently change its label.
    frame = frame.loc[~frame.event_id.isin(invalid_events)].copy()
    collisions = frame.duplicated(["event_id", "time_to_tca"], keep=False)
    ambiguous_events = set(frame.loc[collisions, "event_id"])
    frame = frame.loc[~frame.event_id.isin(ambiguous_events)].copy()
    for column in frame.select_dtypes(include="number").columns:
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
    audit = {
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "raw_encounters": int(raw.event_id.nunique()),
        "exact_duplicate_rows_removed": duplicates,
        "invalid_id_rows_removed": invalid_id_rows,
        "encounters_excluded_invalid_core_values": len(invalid_events),
        "encounters_excluded_conflicting_same_time_reports": len(ambiguous_events),
        "clean_rows": len(frame),
        "clean_encounters": int(frame.event_id.nunique()),
        "missing_fraction_by_column": {
            k: float(v) for k, v in raw.isna().mean().sort_values(ascending=False).items()
        },
        "time_unit": "days until closest approach",
        "risk_unit": "log10(reported collision probability)",
    }
    return frame.sort_values(["event_id", "time_to_tca"], ascending=[True, False]), audit


def history_features(history: pd.DataFrame, cutoff_days: float) -> dict:
    history = history.loc[history.time_to_tca >= cutoff_days].sort_values(
        "time_to_tca", ascending=False
    )
    if history.empty:
        raise ValueError("No reports available by this prediction cutoff")
    first, latest = history.iloc[0], history.iloc[-1]
    values = history.risk.to_numpy(dtype=float)
    elapsed = first.time_to_tca - history.time_to_tca.to_numpy(dtype=float)
    span = float(elapsed[-1])
    result = {
        "current_log_risk": float(latest.risk),
        "history_reports": len(history),
        "history_span_days": span,
        "latest_report_age_days": float(latest.time_to_tca - cutoff_days),
        "latest_time_to_tca": float(latest.time_to_tca),
        "risk_change": float(latest.risk - first.risk),
        "risk_std": float(np.std(values)),
        "risk_min": float(np.min(values)),
        "risk_max": float(np.max(values)),
        "risk_slope_per_day": float(np.polyfit(elapsed, values, 1)[0]) if span > 0 else 0.0,
        "last_risk_change": float(values[-1] - values[-2]) if len(values) > 1 else 0.0,
    }
    for column in CONTEXT:
        result[column] = float(latest[column]) if column in history and pd.notna(latest[column]) else np.nan
    result["context_missing_fraction"] = float(np.mean([pd.isna(result[c]) for c in CONTEXT]))
    return result


def build_cohort(frame: pd.DataFrame, config: Experiment) -> tuple[pd.DataFrame, dict]:
    records = []
    counts = {"no_final_report_within_horizon": 0, "no_history_at_cutoff": 0}
    for event_id, event in frame.groupby("event_id", sort=True):
        event = event.sort_values("time_to_tca", ascending=False)
        final = event.iloc[-1]
        if final.time_to_tca >= config.final_report_max_days:
            counts["no_final_report_within_horizon"] += 1
            continue
        history = event.loc[event.time_to_tca >= config.cutoff_days]
        if history.empty:
            counts["no_history_at_cutoff"] += 1
            continue
        features = history_features(history, config.cutoff_days)
        features.update({
            "event_id": int(event_id),
            "mission_id": history.iloc[-1].get("mission_id", np.nan),
            "final_time_to_tca": float(final.time_to_tca),
            "target_final_log_risk": float(final.risk),
            "target_high_risk": int(final.risk >= config.log_risk_threshold),
        })
        records.append(features)
    if not records:
        raise ValueError("No eligible encounters; inspect horizons and raw data")
    cohort = pd.DataFrame(records).sort_values("event_id").reset_index(drop=True)
    counts.update({
        "eligible_encounters": len(cohort),
        "high_risk_encounters": int(cohort.target_high_risk.sum()),
        "high_risk_fraction": float(cohort.target_high_risk.mean()),
        "single_report_histories": int((cohort.history_reports == 1).sum()),
    })
    return cohort, counts


def make_partitions(cohort: pd.DataFrame, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    if cohort.event_id.duplicated().any():
        raise ValueError("Partitions require exactly one record per encounter")
    cohort = cohort.sort_values("event_id").reset_index(drop=True)
    counts = cohort.target_high_risk.value_counts()
    if len(counts) != 2 or counts.min() < 20:
        raise ValueError("Need at least 20 encounters in each class for four stratified partitions")
    ids = cohort.event_id.to_numpy()
    target_by_id = cohort.set_index("event_id").target_high_risk
    def split(items, test_size, salt):
        return train_test_split(items, test_size=test_size, random_state=seed + salt,
                                stratify=target_by_id.loc[items])
    # Integer allocations avoid floating-point ceiling errors in sequential splits.
    rest, evaluation = split(ids, round(len(ids) * 0.15), 0)
    rest, calibration = split(rest, round(len(ids) * 0.10), 1)
    train, validation = split(rest, round(len(ids) * 0.15), 2)
    rows = [
        {"event_id": int(event), "partition": name}
        for name, group in [
            ("train", train), ("validation", validation),
            ("calibration", calibration), ("evaluation", evaluation)
        ] for event in group
    ]
    return pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)


def feature_columns(cohort: pd.DataFrame) -> list[str]:
    excluded = {"event_id", "mission_id", "final_time_to_tca", "partition"}
    return [c for c in cohort if c not in excluded and not c.startswith("target_")]
