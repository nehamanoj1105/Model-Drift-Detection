# AGENTS.md

## Repository layout (important)

- The working Python project lives under the **sub-directory**
  `Ensemble Learning for Model Drift Detection/`. All `experiments/...`
  relative paths (data, results, plots) are written relative to *that*
  directory, not the git root.
- Always run experiment scripts from inside
  `Ensemble Learning for Model Drift Detection/`, e.g.
  `python experiments/exp9b/run_exp9b.py`.

## Environment

- Install dependencies from `Ensemble Learning for Model Drift Detection/requirements.txt`
  (`numpy pandas scipy scikit-learn matplotlib psutil joblib`).
- `pandas.DataFrame.to_latex` also needs `jinja2` (install it separately).
- No test suite; experiments are validated by re-running and comparing saved CSVs.

## Experiment 9B (RAPT on the 5G NR latency dataset)

- Existing baseline: `experiments/exp9b/run_exp9b.py` (Frozen, Event-Driven,
  Full Retraining, RAPT) over `data/processed_exp9b_stream.csv` (499 windows,
  regimes A/B/C/D, 20% initial-train prefix = 99 windows, seeds 42-46).
- Baseline is reproducible: RAPT Macro-F1 = 0.889447.
- Drift-severity extension: `experiments/exp9b/run_exp9b_drift.py`
  (9B-A natural reuse, 9B-B covariate, 9B-C concept, 9B-D recurring concept).
  Modules: `exp9b_drift_config.py`, `drift_construct.py`, `drift_harness.py`,
  `drift_metrics.py`, `exp9b_drift_figures.py`, `exp9b_drift_tables.py`,
  `exp9b_drift_report.py`, `exp9b_drift_checks.py`.
- Do NOT modify the RAPT algorithm (`rapt_9b.py`) for these experiments; the
  proposed model name stays simply **RAPT**.
- Key modelling facts:
  - Regimes A (149 windows, mixed classes) and B/C/D are largely single-class,
    so concept drift is constructed *within* regime A to isolate P(Y|X).
  - Covariate drift mixes source/target windows (labels untouched).
  - Concept drift uses a shared-permutation rank-reversal of the top-k
    predictive features (preserves P(X) exactly, reverses P(Y|X)).
  - RAPT keys reuse on `regime_id`; recurring concept drift (A->B->A') reuses
    the stale A policy, which is the central 9B-D finding.
- Outputs go to `results/experiment_9b/{natural_drift,covariate_drift,
  concept_drift,recurring_concept_drift,raw,figures,tables}` and
  `EXPERIMENT_9B_FINAL_REPORT.md`.

## Final cross-dataset pass (5 models incl. RAPT-Enhanced)

- All final experiments evaluate five models: Frozen, Event-Driven,
  Full_Retraining, RAPT, RAPT-Enhanced. RAPT itself is never modified.
- RAPT-Enhanced = base RAPT + the two protocol-agnostic mechanisms of
  `rapt/enhanced_hybrid_rapt.py`: buffer-blended novelty refitting (1500 vs 500
  recent samples) and selective parity refitting (refit a *reused* policy whose
  recent window accuracy < 0.5 over 3 windows). The reference online
  micro-learner and dynamic decision threshold are NOT applicable to the
  aggregated-window multi-class protocols here and are intentionally omitted.
- Revised 9A stream = INSECTS `incremental_reoccurring_balanced`
  (river.datasets.Insects), 500-sample windows, 150 windows, 11 recurring
  regime visits, initial_train_windows = 30 (20%). Harness in
  `experiments/exp9a/` (`exp9a_config.py`, `load_and_prepare_stream_9a.py`,
  `models_9a.py`, `event_driven_9a.py`, `rapt_9a.py`, `evaluation_9a.py`,
  `run_exp9a.py`).
- 9A headline (5 seeds): Frozen 0.237, Event-Driven 0.222, Full Retraining
  0.372, RAPT 0.297, RAPT-Enhanced 0.371 Macro-F1. RAPT is the cheapest adapter
  (~0.71 s) and performs 4 reuses; RAPT-Enhanced matches Full Retraining.
- 9B headline (natural, 5 seeds): Frozen 0.896, Event-Driven 0.890, Full
  Retraining 0.903, RAPT 0.889, RAPT-Enhanced 0.892; RAPT has the lowest adapt
  CPU (~0.40 s) and 6 reuses.
- Cross-dataset artefacts: `experiments/exp9a/cross_dataset_9a_9b.py` ->
  `results/experiment_9a/cross_dataset/`, `results/experiment_9a/figures/
  fig_cross_dataset_*.png`, `results/experiment_9a/tables/table_cross_dataset.*`.
- `experiments/exp9a/figures_tables_9a.py` -> fig9a_* figures and
  table_9a1..9a3. `experiments/exp9b/regenerate_9b_outputs.py` rebuilds 9B
  figures/tables/report from saved raw results without re-running models.
- `experiments/exp9a/write_combined_report.py` -> `EXPERIMENT_9A_9B_FINAL_REPORT.md`.
- Run everything from the `Ensemble Learning for Model Drift Detection`
  directory (all config paths are relative to it).

## Experiment 9A — Three telecom datasets (final cross-dataset pass)

9A was extended from a single INSECTS stream to **three independent telecom
datasets** to test whether RAPT exploits recurring network conditions to keep
predictive performance while cutting adaptation cost. Key files (in
`experiments/exp9a/`):

- `three_dataset_config.py` — all constants (seeds, window sizes, paths,
  dataset list); no hard-coded values elsewhere.
- `three_dataset_load.py` — loaders + sample-level windowing.
- `three_dataset_profile.py` — dataset screening + recurrence analysis
  (writes `dataset_screening/dataset_profile.csv` and `recurrence_analysis.csv`).
- `drift_detectors_9a.py` — ADWIN / Page-Hinkley / EDD / EDMA baselines; they
  decide WHEN to adapt on the shared class-anchored buffer.
- `three_dataset_run.py [pilot|full]` — prequential runner.
- `three_dataset_figures.py`, `three_dataset_tables.py`, `three_dataset_cross.py`,
  `three_dataset_report.py` — outputs and reports.

Datasets and documented choices:
- **5G Campus Network QoS** — reuses the processed stream
  (`experiments/exp9/data/…windowed.csv`); 3-class next-window p90-delay QoS.
- **UGR'16** — `data/ugr16/UGR16v1.mat`. The calibration split (`X`/`Y`) has
  only blacklist traffic and **no labelled attacks**, so the labelled 30-day
  block (`test`/`Yt`) is used; target = binary anomaly (any non-blacklist
  attack minute). Window = 240 min to align with the 4-hour regime block.
- **NordicDat** — `data/nordicdat/nordicdat.csv`; 3-class next-window
  median-delay QoS (prefix thresholds); regime = operator.

Streams are **sample-level** (one row per raw observation, `window_id` groups
`window_size` consecutive rows); models train/predict on raw samples, the window
is the adaptation step. Window sizes differ per dataset (Campus 10, UGR 240,
Nordic 500) because raw granularity differs; this is documented in the report.

9A headline (5 seeds, macro-F1 / adapt CPU s):
- 5G Campus: Frozen 0.936/0.00, Event 0.964/0.11, Full 0.984/1.45,
  RAPT 0.938/0.24 (2 retrains, 12 reuses).
- UGR'16: Frozen 0.969/0.00, Event 0.897/0.59, Full 0.960/24.1,
  RAPT 0.836/2.07 (11 retrains, 133 reuses), RAPT-Enhanced 0.928/2.80.
  Base RAPT's blind reuse is **harmful** under concept change coupled with
  regime recurrence; parity refit recovers it.
- NordicDat: Frozen 0.278/0.00, Event 0.255/0.26, Full 0.423/2.26,
  RAPT 0.382/0.28 (2 retrains, 16 reuses).

Outputs: `results/experiment_9a_three/raw/` (per-window + per-seed CSV);
`experiments/exp9a/{figures,tables,models,drift_detectors,reports}/`;
reports `EXPERIMENT_9A_THREE_TELECOM_DATASETS_REPORT.md` and
`EXPERIMENT_9A_9B_FINAL_REPORT.md`. `tabulate` must be installed for the report
generator.

## Revalidation (results/final) — corrections to earlier claims

The v2 revalidation (`results/final/v2/`, `results/final/FINAL_RESULTS.md`,
`results/final/SUBMISSION_AUDIT.md`) re-ran all four streams under one frozen
protocol. Corrections that supersede the 9A headline above:

- **Metric.** Paper Table II mixes scales: 5G Campus / UGR'16 / NordicDat use the
  per-window mean macro-F1, 5G NR uses the pooled value. Corrected tables:
  `results/final/latex_tables/C_primary_perwindow.tex` (recommended) and
  `..._pooled.tex`. `C_metric_reconciliation.csv` maps every cell.
- **UGR'16 mechanism.** The RAPT-Enhanced recovery (0.8360 -> 0.9276) is the
  novelty-refit buffer size (500 -> 1000 rows, +0.0917), NOT the parity refit.
  The parity branch never fires on UGR'16 (0 refits at thresholds 0.5-0.95).
  A second real contributor is checkpoint provenance (stored checkpoints were
  mis-keyed; correcting the keying is +0.0798).
- **Detectors.** ADWIN and Page-Hinkley never fire because river 0.26.1
  `update()` returns `None` and the harness does `bool(...)`; EDMA is NOT
  degenerate once its comparison-order bug is fixed (fires 2.8-16.2 events);
  EDD's old counts (114/115/117/344) were inflated by a first-window spurious
  fire. See `results/final/v2/A10_detector_corrected.csv`.
- **Cost.** The cheap-refresh advantage is 5G Campus-specific. On 5G NR
  RAPT-Cheap is slower (3.26 s vs 0.72 s, +351%) and lower-F1 (0.8830 vs 0.9110)
  because window size 1 makes "every 5 windows" fire every 5 samples.
- Still supported: RAPT below Full Retraining on all four streams (both metrics);
  RAPT worst on UGR'16; the UGR'16 RAPT-Enhanced recovery as a number.

Re-run commands (from `Ensemble Learning for Model Drift Detection/`):
`python results/final/v2/b_pooled.py` then `c_reconcile.py`, `d_figures.py`,
`a10_detector_recheck.py`.

## Experiment 9B drift-severity study

`experiments/exp9b/run_exp9b_drift.py` runs 9B-B/C/D and regenerates
`EXPERIMENT_9B_FINAL_REPORT.md`, the figures, and the tables. Do not hand-edit the
report: `exp9b_drift_report.py` overwrites it, so change the generator and re-run.

Harness scope that must not be misread as an algorithm result:

- The 9B RAPT controller acts **only at regime boundaries**. It has no in-regime
  monitoring or refresh path, so a single-regime stream is never adapted and
  `RAPT == Frozen` exactly. The manuscript's RAPT adds in-regime monitoring, so
  9B-C concept-drift inertness bounds the boundary-only controller, not the
  manuscript algorithm. The recurring A->B->A' result is the stronger finding
  because RAPT does fire there (1 reuse event, 100 reused trees) yet the reused
  policy is stale.
- Concept-drift recovery is **censored** for every model: the reversed P(Y|X) is
  permanent, so the 95%-of-pre-drift target is unreachable and every recovery
  value equals the 40-window horizon. Recovered-rate 0.00 (n=125) vs 0.48
  covariate (n=250) and 0.76 recurring (n=125). Concept recovery rows carry no
  discriminative signal.
- Concept-drift severity is feature-granular (`ceil(level*6)`, capped at 5), so
  20% and 30% both affect 2 features and their curves are identical.

Predictive metrics are deterministic across runs; only CPU/runtime columns drift.
Report numbers are derived from the saved CSVs, so do not hardcode timings.
