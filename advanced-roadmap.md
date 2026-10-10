# Space-Debris Alert Prioritiser

Advanced research and engineering roadmap  
Prepared 12 September 2026

The project will be a research prototype that ingests historical close-approach reports, forecasts later risk assessments, ranks encounters for human review, and explains changes. It preserves all five accepted features: review capacity, risk trends, data reliability, explanations, and notification reduction.

The advanced scope comes from sound forecasting experiments, uncertainty assessment, reproducible decisions, and dependable application behaviour. Research originality remains a hypothesis to investigate and demonstrate.

**Primary research question**

At a fixed review capacity, can encounter history and data-quality information improve the fraction of subsequently high-risk encounters reviewed, compared with sorting by the latest reported collision probability?

Secondary questions:
- Which features help beyond the latest reported risk?
- How well do prediction intervals and classification probabilities behave on unseen encounters?
- Can repeated notifications be reduced without delaying or missing important escalations?
- How does performance change with sparse, stale, missing, or unfamiliar data?

**What the application will show**

- A queue of encounters with selectable review capacity, recorded policy thresholds, and clear indication of cases outside that capacity.
- An encounter timeline showing reports available at each prediction checkpoint.
- The current reported collision probability.
- A forecast of the final reported log-probability, with an empirically evaluated prediction interval.
- A separately labelled probability that the final reported risk exceeds the selected benchmark threshold.
- Separate indicators for reported orbital uncertainty, model uncertainty, and data quality.
- Factual summaries of changes, plus model-attribution explanations.
- A historical replay with notifications, acknowledgements, model versions, and an audit trail.
- An evaluation dashboard comparing methods and showing failures.

The forecast is about a later assessment; its classification probability must never be labelled as the physical probability of impact. Historical encounters will be used in simulated review queues. The first application does not have an operational spacecraft-command interface.

**Data and scientific contract**

The initial source is ESA's Kelvins Collision Avoidance Challenge. Its published training table has 162,634 reports belonging to 13,154 encounters, with 103 fields. Reports from one encounter are correlated; the row count is not the number of independent examples. The table deliberately overrepresents high-risk events. The risk field is log10 of the reported collision probability. These properties determine splitting, metrics, and interpretation. [1]

Use the labelled training archive to construct development, validation, calibration, and locked evaluation partitions at the encounter level. Do not assume labels for the historical competition test file are available.

- Audit download access, provenance, usage terms, file hashes, schema, units, missing values, and report ordering.
- Define a primary cohort with useful history at least 48 hours before closest approach and a final report within 24 hours of closest approach, following the documented challenge conventions.
- Forecast final log-risk from reports at least 48 hours before closest approach.
- Add a 72-hour checkpoint only for encounters with sufficient history; report cohort changes and compare matched cohorts where needed.
- Derive slopes using elapsed time, because reports arrive irregularly.
- Use the ESA benchmark threshold of Pc >= 1e-6, equivalently log10(Pc) >= -6, for the initial high-risk label. This is an experimental definition, not a universal manoeuvre policy. [2]
- Keep all prefixes and checkpoints of an encounter in one partition.
- Fit imputers, transforms, feature selection, calibration, and policy thresholds using the appropriate development partitions.
- Keep event identifiers out of predictive features. Treat mission identifiers as grouping/context, with any use as a feature explicitly justified and tested.
- Add mission-held-out evaluation if mission counts and labels support a meaningful test.
- Claim a chronological split only if trustworthy absolute event dates are actually available.
- Record exclusions and class balance in every cohort.
- Treat duplicate reports and physically invalid units or reconstructable covariance inputs as ingestion-quality issues, not automatic evidence of a safe or dangerous encounter.

**Model progression**

| Stage | Model or method | Purpose |
|---|---|---|
| A | Latest reported risk and a simple trend rule | Establish useful transparent baselines. |
| B | Logistic regression and Random Forest | Establish trained tabular baselines. |
| C | Gradient-boosted trees with features from each available history | Main candidate for nonlinear relationships. |
| D | One compact GRU or temporal convolution model with time-gap and missingness information | Test whether sequence modelling adds value. |
| E | Calibrated event-classification probabilities and quantile/conformal intervals for final log-risk | Evaluate confidence and permit explicit model deferral. |

The sequence model is an experiment, not a predetermined winner. Select the application model on validation results and freeze that choice before opening the locked evaluation results.

Classification calibration and regression intervals have separate objectives. Assess reliability curves and Brier score for event classification. Assess empirical interval coverage and width for regression, including the high-risk subset. Conformal coverage depends on assumptions about calibration and future data; do not present it as an unconditional guarantee under distribution shift.

Unfamiliar or poor-quality inputs can trigger an explicit "model assessment unavailable" state while retaining the reported risk and review visibility. Deferrals count toward review burden and are reported in evaluation.

NASA CARA has documented substantial limitations in the AI/ML approaches it evaluated, including the unpredictability of future updates and inadequate operational reliability. Our claims therefore depend on measured benchmark results, not an assumption that a more complex model improves safety. [3]

**The five feature packages**

| Feature | Advanced implementation | Evaluation |
|---|---|---|
| Review capacity | Configurable capacity K; rank using validated forecasts; retain all encounters; keep explicit policy escalations visible and show capacity overflow. Begin with equal per-event review cost. | Recall@K, precision@K, missed high-risk events, and total review demand including escalations and deferrals. |
| Risk trends | Last risk, elapsed-time slopes, changes, variability, number of reports, and observation-quality changes. Compare different observation horizons. | Remove each feature group and measure changes relative to the strongest simple baseline. |
| Data reliability | Separate provenance, freshness, missingness, observation counts, reported orbital uncertainty, and model confidence. | Performance by quality group; controlled missing/stale-message experiments; deferral rate. |
| Explanations | Factual change summaries plus model attributions. Store the exact data, features, and model version behind every decision. | Check factual claims against records. Explain that attribution describes model behaviour and does not establish physical causation. |
| Notification reduction | Group reports by encounter; stateful change detection; thresholds for escalation/de-escalation; retained history. Keep replay outcomes reproducible. | Notification count, important escalations missed, and delay compared with notifying on every qualifying update. |

Repeated historical reports must not create duplicate encounters or notifications. Replaying the same input and configuration should reproduce the same decisions.

**Experiment design**

Predefine the following before final testing:
- Primary prediction checkpoint and eligible cohort.
- High-risk threshold and main K values.
- The baseline ranking, candidate models, and validation selection rule.
- Rules for synthetic queue construction and random seeds.
- Rules governing escalation, model deferral, and capacity overflow.
- Metrics, uncertainty estimates, and planned subgroup analyses.

For queue experiments, draw documented, fixed batches of unseen encounters and evaluate paired rankings on the same batches. ESA's relative encounter timelines do not by themselves establish which encounters occurred simultaneously. Batch construction is a simulation assumption.

Report:
- Recall@K and precision@K for K values appropriate to each queue size.
- PR-AUC and the count of missed high-risk encounters.
- Regression error on final log-risk, including the high-risk subset.
- Classification calibration and regression-interval coverage.
- Notification reductions alongside escalation delays and missed escalations.
- Inference latency on a named machine and fixed workload.
- Paired confidence intervals using the encounter as the independent unit.
- Ablations: latest risk only; plus trends; plus observation quality; plus the prioritisation policy.

A positive result requires improvement over a meaningful baseline, with quantified uncertainty. A well-executed negative result is still valuable and must be reported. The enriched benchmark does not establish fleet-wide alert frequencies or real-world probability calibration.

**Proposed application architecture**

Historical archive -> validated reports -> encounter store -> replay worker -> feature generation -> model forecasts -> ranking and notification policy -> API -> dashboard

An immutable decision record connects each step to its input snapshot, data version, model version, policy version, and replay time.

| Component | Proposed technology | Responsibility |
|---|---|---|
| Research and data processing | Python, pandas, NumPy | Data audit, features, experiments. |
| Models | scikit-learn; gradient-boosting library; PyTorch for the sequence experiment | Training, prediction, calibration. |
| Backend | FastAPI and Pydantic | Validated ingestion and application endpoints. |
| Persistent storage | PostgreSQL with SQLAlchemy and migrations | Reports, encounters, predictions, policies, acknowledgements, audit records. |
| Background processing | One replay worker initially; a durable job queue if needed | Ordered replay and forecast jobs. |
| Frontend | React and TypeScript with interactive charts | Queue, encounter details, replay, explanations, evaluation. |
| Reproducibility | Versioned datasets/configuration, model artifacts, experiment records | Traceable results and model selection. |
| Delivery | Docker Compose and automated checks | Repeatable local setup and regression prevention. |

This architecture reuses patterns familiar from GrowthPilot AI: event ingestion, duplicate handling, policy decisions, dashboards, and audit history. The new learning is in orbital-report interpretation and statistical evaluation.

A generative assistant is optional. If added later, it may summarise retrieved encounter records with citations to those records. The first version uses structured explanations so that every statement is verifiable.

**Implementation sequence**

Planning estimate: 12-16 weeks at approximately 10-12 hours per week. Allow 4-6 months if learning the statistical and software concepts alongside implementation.

| Milestone | Indicative timing | Deliverable and exit condition |
|---|---|---|
| 1. Data contract and baseline | Weeks 1-2 | Download/audit report, documented cohorts and partitions, one encounter plot, latest-risk ranking, reproducible baseline metrics. |
| 2. Forecasting experiments | Weeks 3-4 | Tabular models, history features, experiment log, feature-group comparisons, validation-selected candidate. |
| 3. Sequence and uncertainty experiments | Weeks 5-7 | One sequence competitor, classification calibration, evaluated prediction intervals, quality-based failure analysis. |
| 4. Prioritisation and replay engine | Weeks 8-9 | Capacity-aware queue, explicit deferral/escalation rules, reproducible replay, duplicate handling, notification metrics. |
| 5. Full application | Weeks 10-12 | API, database, React dashboard, encounter explanations, acknowledgements, model/policy audit history. |
| 6. Evaluation and reliability | Weeks 13-14 | Locked evaluation, confidence intervals, missing/stale-data tests, replay recovery tests, application smoke checks. |
| 7. Reproducible delivery | Weeks 15-16 | Local setup, documented demo, research report, model/data cards, installation instructions, short presentation. |

Independent experimental work can overlap dashboard development after the input and prediction schemas stabilise.

**Meaningful automated checks**

- Correct risk units and high-risk label transformation.
- No future report included in a prediction snapshot.
- No encounter shared across partitions.
- Consistent results after duplicate or out-of-order report delivery.
- Factual explanations that match the stored snapshot.
- Model deferral and capacity overflow remain visible.
- Notification state survives a replay-worker restart.
- Every decision resolves to its data, model, and policy versions.
- The end-to-end path from a sample report to the dashboard works.

**Completion criteria**

The advanced prototype is complete when:
- All five feature packages work in historical replay.
- A frozen, reproducible experiment compares the system with the latest-risk baseline.
- Prediction confidence and data quality are displayed and evaluated separately.
- Results include ranking utility, notification trade-offs, failure cases, and statistical uncertainty.
- The application runs from documented local setup instructions.
- Claims are limited to what the experiments demonstrate.

External live feeds, spacecraft manoeuvre optimisation, and a validated operational collision-avoidance system would be separate extensions with additional data and validation requirements.

**First implementation milestone**

Inspect the downloadable dataset and create the data contract, encounter-level split manifest, one encounter-history plot, and the latest-risk baseline. This establishes the evidence against which every advanced addition will be judged.

**Sources**

1. [ESA Kelvins Collision Avoidance Challenge: data](https://kelvins.esa.int/collision-avoidance-challenge/data/).
2. [ESA Kelvins Collision Avoidance Challenge: scoring](https://kelvins.esa.int/collision-avoidance-challenge/scoring/).
3. [NASA CARA compendium of AI/ML studies, 2025 conference paper](https://ntrs.nasa.gov/citations/20250008251).
4. [ESA challenge definition and prediction timeline](https://kelvins.esa.int/collision-avoidance-challenge/challenge/).

Source-derived facts above are distinct from the proposed implementation, experimental choices, and effort estimates.

