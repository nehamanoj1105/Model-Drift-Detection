# EXTRACT — RAPT paper-claim validation data

Branch: `extract-20261001`. Everything below is read from existing CSVs, JSON, logs and code; nothing was re-run. The literal token `MISSING` marks a quantity that does not exist in the repository, with the file that was searched.

Regenerate with: `python "results/extract/extract_all.py"` (run from the `Ensemble Learning for Model Drift Detection` directory).

## Files

| File | Rows | What it holds |
|---|---:|---|
| `E10_aprime_phase_f1.csv` | 75 | A' phase macro-F1 per method |
| `E10_pipeline_note.csv` | 1 | which 9B numbers use a different pipeline from Table II |
| `E10_recurring_drift_9b.csv` | 25 | 9B-D (A->B->A') per method and severity |
| `E11_dataset_facts.csv` | 4 | samples, usable samples, features, classes, windows, window size, prefix, regimes, recurrences, causal observability |
| `E11_nordic_duplicates.csv` | 1 | NordicDat raw rows and duplicate count |
| `E11_settlements.csv` | 4 | the four named dataset-count settlements |
| `E12_enhanced_disagreement.csv` | 2 | paper vs rapt_9a.py disagreements on the fingerprint gate and 1500 buffer |
| `E12_history.csv` | 5 | git provenance of the 0.97 floor, rel_drop 0.10, parity 0.5 and detector parameters |
| `E13_cpu_timing.csv` | 28 | repeated timing runs (experiment5 stage3 run1/run2) and single-run 9A/revalidation medians and ranges |
| `E13_timing_note.csv` | 1 | notes on Frozen CPU including its initial fit |
| `E1_metrics_all_streams.csv` | 575 | one row per pipeline/stream/method/seed with pooled + per-window macro-F1, accuracy, adaptation CPU, runtime, retrains, reuses, refreshes (5 models, 4 detectors, every Table III rung) |
| `E2_table_ii_vs_rerun.csv` | 70 | each Table II / Table III cell beside its rerun value, deltas, and whether the published cell matches the pooled or per-window metric |
| `E3_cost_matched.csv` | 296 | FR-Cheap, Periodic-Cheap-5/-10, Event-Driven-Cheap, RAPT-Cheap and the floor variant on all four streams with pooled F1, CPU, runtime, refreshes and a pareto flag |
| `E3_reconciliation.csv` | 3 | the named RAPT-Cheap Campus pipeline differences |
| `E4_gate_conflict_note.csv` | 1 | resolution of the 'gate never fires on Campus' vs 4-vs-2-retrain conflict |
| `E4_mechanism_counters.csv` | 555 | per rung/stream/seed gate accept/reject, parity refits, refresh/trigger firings, retrains, reuses, max refit rows |
| `E5_mismatch_fraction.csv` | 1 | fraction of stored checkpoints trained on a different regime than their key, with the denominator definition |
| `E5_provenance.csv` | 920 | every stored checkpoint: stream, seed, regime key, creation window, training regimes, reuse age |
| `E6_ugr16_diagnosis.csv` | 116 | UGR16 factorial (refit rows / provenance / parity) + oracle diagnostics, as far as artifacts exist |
| `E6_ugr16_paired_visits.csv` | 1 | number of paired UGR16 visits |
| `E6_ugr16_py_transfer.csv` | 143 | UGR16 P(Y|X) between-visit transfer per regime with n |
| `E7_detector_meta.csv` | 5 | harness provenance, river return-value-bug status, EDD counts, Event-Driven trigger rule |
| `E7_detectors.csv` | 207 | events/retrains per detector, setting, stream, pipeline; includes the paper-default settings flag |
| `E8_config_count.csv` | 1 | number of configurations in the Campus ladder |
| `E8_missing_bootstrap_types.csv` | 2 | day-block / visit-block bootstrap availability |
| `E8_stats.csv` | 80 | per comparison: window/seed Wilcoxon p, Cohen's d, bootstrap CI and bootstrap type |
| `E8_tost.csv` | 12 | TOST at margins 0.005 / 0.01 / 0.02 |
| `E9_label_delay.csv` | 31 | pooled macro-F1 at delay 0 and 1 per method for Campus, Nordic, 5G NR; UGR16 label type |
| `manifest.csv` | 28 | path, size, SHA-256 and commit for every file in this folder |

## MISSING items

| # | Item | Searched in |
|---|---|---|
| 1 | Synthetic step-change detector sanity result | results/revalidation/a6_detectors.py, results/revalidation/*.py, tests/, audit/ — no synthetic step-change detector test exists |
| 2 | Day-block bootstrap CI | results/revalidation/A7_block_bootstrap.csv, results/revalidation/run_a7_stats.py — blocks are contiguous regime visits, not calendar days |
| 3 | Visit-block bootstrap CI | results/revalidation/A7_block_bootstrap.csv — the A7 blocks are runs of equal regime_id and are labelled window-block; no separate visit-block variant |
| 4 | UGR16 label-delay run | results/revalidation/A8_label_delay.csv — UGR16 is not in the A8 dataset list |
| 5 | RAPT-Cheap floor variant on UGR16 / Nordic / 5G NR | Final_Experiments/results/raw/summary_full.csv, results/revalidation/A3_pareto.csv — RAPT_FLOOR/RAPT-Cheap-floor ran on Campus only |
| 6 | Plain 'Periodic-Cheap' on any stream | results/revalidation/A3_pareto.csv — the harness only emits Periodic-Cheap-5 and Periodic-Cheap-10 |
| 7 | Prediction CPU and initial-fit CPU medians for repeated runs | experiment5/results/stage3_summary_metrics_run{1,2}.csv — only aggregate mean adaptation/total CPU is stored; the RAPT harnesses have no repeated timing runs |
| 8 | Full 3-factor UGR16 cross (refit rows x provenance x parity) | results/revalidation/A2_*, A3_*, A5_* — only partial slices exist; the refit-500-vs-1500 comparison is in audit/PHASE2_FINDINGS.md (markdown), not a CSV |
| 9 | 5G NR feature count in its stream definition | experiments/exp9b/results/stream_definition.json — no feature_columns key; the 12 features are in results/revalidation/streams.py load_5g_nr |

## Open conflicts

Every place two files give different numbers for the same quantity. Both values and both sources are listed. (CPU differences across harnesses are expected — the paper documents adaptation CPU as machine-dependent — but they are recorded here.)

| Quantity | Value A | Source A | Value B | Source B | Why they differ |
|---|---|---|---|---|---|
| RAPT-Cheap Campus adaptation CPU | 0.8467 | Final_Experiments/results/tables/table_final_main.csv [Adapt CPU (s)] | 1.1073 | results/revalidation/A3_pareto.csv [adaptation_cpu_sec_mean] | Different harness: rapt_ladder.py (Final_Experiments) vs run_stream_v2.py RAPTV2 (revalidation). Also machine timing. |
| RAPT-Cheap Campus pooled macro-F1 | 0.9851 | Final_Experiments/results/tables/table_final_main.csv [Macro-F1, per-window mean] | 0.9944 | results/revalidation/A3_pareto.csv [pooled_macro_f1_mean] | Aggregation: per-window mean vs pooled over all evaluation samples. |
| RAPT-Cheap Campus per-window macro-F1 | 0.9851 | Final_Experiments/results/tables/table_final_main.csv [Macro-F1] | 0.9817 | results/revalidation/A3_summary.csv [per_window_macro_f1 mean] | Same aggregation, different code path (50-tree/1000-buffer ladder vs 20-tree/300-buffer RAPTV2). |
| Campus RAPT adaptation CPU | 0.2292 | experiments/exp9a/tables/table9a_main.csv [Adapt CPU] | 0.2387 | results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean] | Re-run timing; F1 reproduces exactly, CPU shifts. |
| UGR16 Full Retraining adaptation CPU | 23.6318 | experiments/exp9a/tables/table9a_main.csv [Adapt CPU] | 24.9407 | results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean] | Re-run timing; machine-dependent. |
| Nordic Frozen adaptation CPU | 0.0000 | experiments/exp9a/tables/table9a_main.csv [Adapt CPU] | 1.1454 | results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean] | In A1_summary Frozen's adaptation_cpu_sec equals its initial fit; Table II reports 0 for Frozen. |
| UGR16 Frozen adaptation CPU | 0.0000 | experiments/exp9a/tables/table9a_main.csv [Adapt CPU] | 1.2212 | results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean] | Same Frozen-includes-initial-fit artefact as Nordic. |
| 5G NR Event-Driven per-window macro-F1 | 0.8903 | results/experiment_9b/natural_drift/summary.csv [macro_f1_mean, published in Table II] | 0.9060 | results/revalidation/A1_summary_all_streams.csv [per_window_macro_f1 mean for 5g_nr] | 9B natural-drift aggregation (500-packet windows, prefix 99) vs revalidation A1 (window_size=1, prefix 99). Different window definition. |
| 5G NR prefix windows | 99 | experiments/exp9b/results/stream_definition.json [initial_train_windows] | 100 | results/revalidation/A10_dataset_facts.csv [prefix_round_499x0.2] | 9B uses floor(499*0.2)=99; A10 reports round(499*0.2)=100. The A1/A3 runner uses 99. |
| Fingerprint gate on Campus | 'gate never fires' (implied by the paper's ladder narrative) | Paper_Final/manuscript.tex [Sec. ablation] | gate_accept=9 / gate_reject=3 per seed (orig), 0/12 (fix) | results/revalidation/A5_gate.csv [gate_accept;gate_reject] | The gate does fire on Campus. RAPT_T2 has no gate; RAPT_FULL adds it, which is why RAPT_FULL retrains 4 times vs 2 (Final_Experiments/results/raw/summary_full.csv [retrains]). |
| Enhanced variant fingerprint gate | present | Paper_Final/manuscript.tex:273-275 | absent | experiments/exp9a/rapt_9a.py [no gate code in RAPTSystem/RAPTEnhancedSystem] | rapt_9a.py implements no fingerprint gate; reuse is keyed on regime_id only. |
| Base RAPT novelty refit size | 500 | Paper_Final/manuscript.tex:303 'where the original used 500' | buffer_capacity=1000 (no 500) | experiments/exp9a/rapt_9a.py [RAPTSystem._train_policy uses self.buffer_capacity] | Only RAPTEnhancedSystem sets novelty_refit_n=1500; the 500 value lives in the revalidation/audit harness, not in base RAPT. |
| UGR16 recurrence attribution | parity refit recovers 0.8360 -> 0.9276 | Paper_Final/manuscript.tex [Discussion] | recovery is the 1500-vs-500 novelty buffer; parity_refits=0 | audit/PHASE2_FINDINGS.md [E2] and results/experiment_9a_three/raw/summary_ugr16_full.csv [parity_refits=0] | The parity-refit branch never fires at threshold 0.5; the buffer size produces the gain. |
| UGR16 sample count | 43,200 | results/experiment_9a_three/raw/stream_def_ugr16_full.json [n_raw_samples] | 48,000 (the figure raised in the task) | no repository file contains 48,000 | The labelled 30-day block is 43,200 minutes; 48,000 does not appear in any loader or stream_def. |
| EDD event count | 38 retrains on Campus | experiments/exp9a/tables/table9a_drift_detectors.csv [Adaptation Events] | 114 detected events / 38 retrains on Campus | results/revalidation/A6_detector_events.csv [events;retrains] | Consistent: 114 raw firings collapse to 38 retrains under the 3-window minimum retrain interval. |

## Pipeline map

- **9A** — `experiments/exp9a/three_dataset_run.py` / `run_exp9a.py`; Table II source `experiments/exp9a/tables/table9a_main.csv`; raw per-window in `results/experiment_9a_three/raw/`.
- **9B** — `experiments/exp9b/run_exp9b.py` (natural) and `run_exp9b_drift.py` (severity); Table II 5G NR source `results/experiment_9b/natural_drift/summary.csv`.
- **Final_Experiments** — `Final_Experiments/run_final.py`; Table III source `Final_Experiments/results/tables/table_final_main.csv`.
- **revalidation v2** — `results/revalidation/run_stream_v2.py` + `run_a1.py` / `run_a3.py`; pooled metrics in `A1_pooled_all_streams.csv` / `A3_cost_matched.csv`.

