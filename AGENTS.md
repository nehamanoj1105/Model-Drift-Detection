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
