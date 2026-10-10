"""Reproduce milestone 1 without evaluating the reserved final partition."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import pandas as pd
from .baseline import evaluate_baselines
from .data import Experiment, audit_and_clean, build_cohort, feature_columns, make_partitions, sha256
from .fetch import download
from .report import figures, markdown_report


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cutoff-hours", type=float, default=48)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args(argv)
    config = Experiment(cutoff_days=args.cutoff_hours / 24, seed=args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv = download(args.data_dir)
    print("Auditing reports...", flush=True)
    raw = pd.read_csv(csv)
    clean, audit = audit_and_clean(raw)
    cohort, counts = build_cohort(clean, config)
    partitions = make_partitions(cohort, config.seed)
    dataset = cohort.merge(partitions, on="event_id", validate="one_to_one")
    processed = args.data_dir.parent / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, group in dataset.groupby("partition"):
        group.to_csv(processed / f"{name}.csv", index=False)
    partitions.to_csv(output / "split-manifest.csv", index=False)
    split_summary = dataset.groupby("partition").agg(
        encounters=("event_id", "size"), high_risk=("target_high_risk", "sum")
    ).reset_index()
    validation = dataset.loc[dataset.partition == "validation"].copy()
    print(f"Eligible encounters: {len(cohort)}; evaluating validation only...", flush=True)
    metrics, example = evaluate_baselines(validation, config)
    example.to_csv(output / "validation-queue.csv", index=False)
    source = json.loads((args.data_dir / "download_manifest.json").read_text(encoding="utf-8"))
    save_json(output / "source-manifest.json", source)
    save_json(output / "data-audit.json", audit)
    save_json(output / "cohort-summary.json", counts)
    save_json(output / "baseline-metrics.json", metrics)
    save_json(output / "feature-schema.json", {
        "features": feature_columns(cohort),
        "excluded": ["event_id", "mission_id", "final_time_to_tca", "target_final_log_risk", "target_high_risk", "partition"],
        "cutoff_days": config.cutoff_days,
        "definition": "Features use only reports available at or before the cutoff.",
    })
    save_json(output / "run-manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "python": sys.version.split()[0],
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "space-debris-prioritiser"]},
        "csv_sha256": sha256(csv),
        "partition_manifest_sha256": sha256(output / "split-manifest.csv"),
        "evaluation_performance_computed": False,
    })
    selected = figures(clean, validation, metrics, config, output)
    markdown_report(output, audit, counts, split_summary, metrics, config, source, selected)
    print(json.dumps({"report": str(output / "milestone-report.md"), "cohort": counts, "partitions": split_summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()

