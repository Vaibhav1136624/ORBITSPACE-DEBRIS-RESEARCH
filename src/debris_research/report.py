"""Generate inspectable scientific figures and a milestone report."""
from pathlib import Path
import os
import sys

cache = Path(sys.prefix).parent / "cache" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def figures(clean, validation, metrics, config, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": "#0d1726", "axes.facecolor": "#121f32",
        "text.color": "#eaf2fc", "axes.labelcolor": "#d4e0ee",
        "xtick.color": "#a9bdd4", "ytick.color": "#a9bdd4",
        "axes.edgecolor": "#39506c", "grid.color": "#39506c",
        "legend.facecolor": "#182840", "legend.edgecolor": "#39506c",
    })
    candidates = validation.loc[(validation.target_high_risk == 1) & (validation.history_reports >= 3)]
    if candidates.empty:
        candidates = validation.loc[validation.history_reports >= 2]
    if candidates.empty:
        candidates = validation
    selected = candidates.sort_values("event_id").iloc[0]
    event_id = int(selected.event_id)
    event = clean.loc[clean.event_id == event_id].sort_values("time_to_tca", ascending=False)
    available = event.loc[event.time_to_tca >= config.cutoff_days]
    fig, axis = plt.subplots(figsize=(11.5, 6.5), layout="constrained")
    axis.plot(event.time_to_tca * 24, event.risk, "o-", color="#7287a3", label="Full historical record", lw=1.8, ms=4)
    axis.plot(available.time_to_tca * 24, available.risk, "o-", color="#64dfcd", label="Available at prediction cutoff", lw=2.5, ms=5)
    axis.axvline(config.cutoff_days * 24, color="#a3bafb", ls="--", lw=1.4, label=f"{config.cutoff_days * 24:g}-hour cutoff")
    axis.axhline(config.log_risk_threshold, color="#ffba74", ls=":", lw=1.5, label="Benchmark high-risk threshold")
    axis.axvspan(0, config.cutoff_days * 24, color="#4c5980", alpha=.18)
    axis.set_xlim(max(event.time_to_tca.max() * 24 + 4, config.cutoff_days * 24 + 4), 0)
    axis.set_xlabel("Hours remaining until closest approach")
    axis.set_ylabel("Reported log10 collision probability")
    axis.set_title(f"Encounter {event_id}: how its risk assessment changed", loc="left", fontsize=17, pad=15)
    axis.grid(alpha=.35)
    axis.legend(loc="best", fontsize=9)
    fig.suptitle("HISTORICAL EXAMPLE  |  Later reports are withheld from prediction inputs", fontsize=10, color="#a9bdd4")
    fig.savefig(output / "encounter-history.png", dpi=160)
    fig.savefig(output / "encounter-history.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.7), layout="constrained")
    entries = metrics["baselines"]
    reference = entries["latest_reported_risk"]["queues"]
    x = np.arange(len(reference))
    for offset, (key, label, color) in enumerate([
        ("latest_reported_risk", "Latest reported risk", "#64dfcd"),
        ("random_order_reference", "Random ordering reference", "#8195b5"),
    ]):
        values = [100 * (r["recall_at_k"] or 0) for r in entries[key]["queues"]]
        bars = axis.bar(x + (offset - .5) * .35, values, width=.35, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    axis.set_xticks(x, [f"Review {r['k']} of {config.queue_size}" for r in reference])
    axis.set_ylim(0, 110)
    axis.set_ylabel("Later high-risk encounters included (%)")
    axis.set_title("Baseline review-queue performance", loc="left", fontsize=17, pad=15)
    axis.grid(axis="y", alpha=.3)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", fontsize=9)
    fig.suptitle("VALIDATION ONLY  |  Simulated queues; descriptive results, not operational guarantees", fontsize=10, color="#a9bdd4")
    fig.savefig(output / "baseline-ranking.png", dpi=160)
    fig.savefig(output / "baseline-ranking.svg")
    plt.close(fig)
    return event_id


def markdown_report(output, audit, counts, splits, metrics, config, source, selected_event):
    rows = []
    for name, detail in metrics["baselines"].items():
        for result in detail["queues"]:
            recall = result["recall_at_k"]
            rows.append(f"| {name.replace('_', ' ')} | {result['k']} | {recall:.1%} | {result['precision_at_k']:.1%} | {result['mean_missed_high_risk_per_queue']:.3f} |")
    split_rows = [f"| {row.partition} | {row.encounters:,} | {row.high_risk:,} |" for row in splits.itertuples()]
    text = f"""# Space-Debris Alert Prioritiser: milestone 1

**Status:** Real-data audit, encounter partitions, and validation baseline implemented. Learned models, uncertainty calibration, and the full application are subsequent milestones.

**Data provenance:** [ESA author archive](https://zenodo.org/records/4463683), DOI {source['source_doi']}; {source['license']}. Archive MD5 verified against the publisher's metadata, with SHA-256 recorded locally. Attribution: {source['attribution']}

The input has **{audit['raw_rows']:,} reports**, **{audit['raw_encounters']:,} encounters**, and **{audit['raw_columns']} fields**.

**Cohort construction**

- Removed {audit['exact_duplicate_rows_removed']:,} exact duplicate reports.
- Excluded {audit['encounters_excluded_invalid_core_values']:,} encounters with invalid mandatory values.
- Excluded {audit['encounters_excluded_conflicting_same_time_reports']:,} encounters with conflicting reports at the same timestamp.
- Excluded {counts['no_final_report_within_horizon']:,} otherwise clean encounters without a final report less than {config.final_report_max_days * 24:g} hours before closest approach.
- Excluded {counts['no_history_at_cutoff']:,} remaining encounters without reports available at least {config.cutoff_days * 24:g} hours before closest approach.
- Retained **{counts['eligible_encounters']:,} encounters**, including **{counts['high_risk_encounters']:,} high-risk labels** ({counts['high_risk_fraction']:.2%}).
- {counts['single_report_histories']:,} retained encounters have only one available report at the cutoff.

The target is the final reported log10 collision probability. High risk means that reported probability is at least 1e-6 (log-risk >= -6), matching the benchmark definition. It does not mean that a collision occurred.

**Encounter-level partitions**

| Partition | Encounters | High-risk labels |
|---|---:|---:|
{chr(10).join(split_rows)}

All prefixes from an encounter will inherit its partition. Identifiers and final-report fields are excluded from model features. Calibration and evaluation performance have not been computed.

**Validation baseline**

The primary baseline simply orders events by their latest available reported risk. A seeded random ordering is provided for context. No AI training has occurred in this milestone.

| Method | Review capacity | Recall | Precision | Mean missed high-risk cases / queue |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

These numbers use {config.queue_trials} simulated queues of {config.queue_size} encounters, sampled from validation with seed {config.seed}. Encounters are unique within a queue but may recur across queues. Results are descriptive; the repeated queues are not independent trials for statistical confidence intervals.

![Baseline queue performance](baseline-ranking.png)

**An inspectable encounter**

Example encounter {selected_event} was selected from validation, preferring a high-risk encounter with at least three available reports. The shaded future region is shown for retrospective understanding and excluded from prediction inputs.

![Encounter history](encounter-history.png)

**What these results can establish**

They establish a reproducible reference for the next model comparison. The source deliberately enriches high-risk events, and relative encounter times do not establish which events occurred simultaneously. These are synthetic review queues, not a reconstruction of normal fleet traffic.

**Next milestone**

Train tabular models using only the training partition, compare on validation, and test whether history features improve ranking over this baseline. Preserve the evaluation partition until the final selection and policy rules are frozen.
"""
    (output / "milestone-report.md").write_text(text, encoding="utf-8")

