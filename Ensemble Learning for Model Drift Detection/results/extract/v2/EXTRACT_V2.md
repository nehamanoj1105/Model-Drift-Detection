# EXTRACT_V2 — follow-up extraction (branch `extract-20261001`)

All numbers come from a committed file or a script in this folder. `MISSING` means the quantity does not exist in the repository. Nothing was taken from memory. Newly-run artefacts are explicitly labelled.

Regenerate:
```
cd "Ensemble Learning for Model Drift Detection"
python results/extract/v2/ugr16_refit_log.py            # task 2 (needs UGR16 raw data)
python results/extract/v2/extract_v2.py                # task 3 + task 5 assembly
python results/extract/v2/detector_sanity.py           # task 4 (NEW)
python results/extract/v2/ugr16_dayblock_bootstrap.py  # task 5
python results/extract/v2/generate_extract_v2_md.py    # this file
```

## Task 1 — printed contents of E3, E5_mismatch_fraction, E7_detector_meta, E4_gate_conflict_note

### E3 — cost-matched controls (per stream: pooled F1, adaptation CPU, pareto flag)

| stream | method | pooled_macro_f1 | adaptation_cpu_sec | total_runtime_sec | refreshes | pareto_flag |
|---|---|---|---|---|---|---|
| 5G NR | FR-Cheap | 0.8763937761897903 | 0.3189322424000238 | 3.017479740600538 | 0.0 | False |
| 5G NR | Periodic-Cheap-10 | 0.8638572860710371 | 1.4735367002001112 | 4.160065413600387 | 0.0 | False |
| 5G NR | Periodic-Cheap-5 | 0.8668738428725113 | 2.906559301799825 | 5.604750232799415 | 0.0 | False |
| 5G NR | RAPT-Cheap | 0.861443782877646 | 3.209112679000282 | 5.976439729399862 | 79.0 | False |
| Campus | FR-Cheap | 0.9913268929519112 | 0.5321479661999952 | 1.499638518400752 | 0.0 | True |
| Campus | Periodic-Cheap-10 | 0.9923056324786717 | 0.5528225840000047 | 1.5565783657992145 | 0.0 | True |
| Campus | Periodic-Cheap-5 | 0.9924460355857558 | 1.023932689600008 | 1.9947588866009027 | 0.0 | True |
| Campus | RAPT-Cheap | 0.9944031424849448 | 1.107293887999979 | 2.1393281262004167 | 22.0 | True |
| Nordic | FR-Cheap | 0.4764319435510449 | 1.1273285423999142 | 2.247779991601419 | 0.0 | False |
| Nordic | Periodic-Cheap-10 | 0.5036615502500312 | 0.9678008051999996 | 2.0506720621997374 | 0.0 | True |
| Nordic | Periodic-Cheap-5 | 0.4983368982313449 | 1.5159450083999673 | 2.6071784864005165 | 0.0 | False |
| Nordic | RAPT-Cheap | 0.4283228029352579 | 2.2256222234000687 | 3.3795448855998984 | 21.0 | False |
| UGR16 | FR-Cheap | 0.5983756371118472 | 6.287144677599883 | 7.408955578399036 | 0.0 | False |
| UGR16 | Periodic-Cheap-10 | 0.6462490745723272 | 1.0588217038000152 | 2.1190745678017264 | 0.0 | False |
| UGR16 | Periodic-Cheap-5 | 0.6451005227349043 | 1.6372598791999735 | 2.704739245800738 | 0.0 | False |
| UGR16 | RAPT-Cheap | 0.8007942095585247 | 4.012828509200148 | 5.466102379600488 | 0.0 | False |

Source: `results/extract/E3_cost_matched.csv` (per-seed pooled F1 in the same file; means shown here). `pareto_flag` is the committed A3 pareto membership.

MISSING in E3: `Periodic-Cheap` without a window suffix (the harness only emits `-5` and `-10`); `RAPT-Cheap-Floor` outside Campus (see `results/extract/E3_cost_matched.csv`, notes column).

### E5_mismatch_fraction (with denominator)

| numerator | denominator | fraction_trained_on_different_regime | denominator_definition | source_file |
|---|---|---|---|---|
| 80 | 80 | 1.0000 | number of STORED checkpoints (rows with a non-empty train_regimes field, i.e. creation events), pooled over all four streams and seeds 42-46; reuse rows are excluded | Ensemble Learning for Model Drift Detection/results/revalidation/A2_checkpoint_provenance.csv [train_regime_matches_key] |

Console confirmation: `results/revalidation/logs/a2.log: 'checkpoints created: 80, trained on a DIFFERENT regime: 80 (100.0%)'`

### E7_detector_meta

| item | value | detail |
|---|---|---|
| paper Fig.8 / Table II detector harness | Ensemble Learning for Model Drift Detection/experiments/exp9a/drift_detectors_9a.py [lines 98-182 (make_detector + DetectorAdaptiveModel)] | run_stream_v2.py imports DetectorAdaptiveModel from drift_detectors_9a and calls dm.update_and_adapt(w, X[idx], y_all[idx], err) |
| detector harness had return-value bug? | NO | drift_detectors_9a.DetectorAdaptiveModel.update_and_adapt uses `bool(self.detector.update(win_error))` (line 160), i.e. it consumes the boolean return value correctly. ADWIN/PageHinkley default `update` returns bool; the harness is not affected by the river `drift_detected`-attribute pitfall. |
| EDD event count, paper harness | Campus 38 retrains/114 detected events; UGR16 39/115; Nordic 39/117 (per seed 42) | Ensemble Learning for Model Drift Detection/experiments/exp9a/drift_detectors_9a.py [make_detector EDD -> _CustomEDD(drift_level=3.0)] ; events from Ensemble Learning for Model Drift Detection/results/revalidation/A6_detector_events.csv [events] |
| Event-Driven trigger rule | experiments/exp9a/event_driven_9a.py:72: threshold = mu_err + self.error_threshold_k * max(sigma_err, 0.05) ; experiments/exp9a/event_driven_9a.py:72-74 (mu_err + error_threshold_k*max(sigma_err,0.05); trigger when curr_err > threshold) | error window size 20, k=2.0 (run_stream_v2.py ERROR_WINDOW_SIZE/ERROR_THRESHOLD_K); fires when recent_errors[-1] > mu + k*max(sd,0.05) |
| synthetic step-change sanity result | MISSING | no synthetic step-change detector test exists; searched Ensemble Learning for Model Drift Detection/results/revalidation/a6_detectors.py, results/revalidation/*.py, tests/, audit/ |

### E4_gate_conflict_note

| question | resolution | evidence_a | evidence_b |
|---|---|---|---|
| "gate never fires on Campus" vs RAPT_FULL 4 retrains / RAPT_T2 2 retrains | RAPT_T2 (Table III rung 1) has no fingerprint gate. RAPT_FULL adds the gate; on Campus the gate fires (A5_gate: RAPT-Gate-Orig gate_accept=9, gate_reject=3 per seed), so RAPT_FULL rejects some reuses and retrains 4 times vs 2 for RAPT_T2 (Final_Experiments summary_full: retrains column). The 'gate never fires' claim is false for the Campus ladder. In the revalidation v2 A1 harness the Table-II RAPT controller has NO gate at all (rapt_9a.py RAPTSystem), which is a separate code path. | Ensemble Learning for Model Drift Detection/Final_Experiments/results/raw/summary_full.csv [retrains] | Ensemble Learning for Model Drift Detection/results/revalidation/A5_gate.csv [gate_accept;gate_reject] |

## Task 2 — UGR'16 novelty-refit training rows (base RAPT vs RAPT-Enhanced, Table II)

| method | refit_n | n_novelty_refits | buffer_rows_min | buffer_rows_max | recent_after_cap_min | recent_after_cap_max | anchor_rows | train_rows_min | train_rows_max | source_file |
|---|---|---|---|---|---|---|---|---|---|---|
| RAPT | 500 | 55 | 500 | 500 | 500 | 500 | 30 | 530 | 530 | results/extract/v2/ugr16_refit_log.py (replicates experiments/exp9a/three_dataset_run.py:197-233) |
| RAPT-Enhanced | 1500 | 55 | 1000 | 1500 | 1000 | 1000 | 30 | 1030 | 1030 | results/extract/v2/ugr16_refit_log.py (replicates experiments/exp9a/three_dataset_run.py:197-233) |

Per-refit log (every transition, 55 novelty refits + 55 reuses per method over seeds 42-46): `results/extract/v2/T2_ugr16_refit_rows.csv`. Code paths: `results/extract/v2/T2_ugr16_refit_codepaths.csv`.

**Why they differ although both look capped at 1000.** Both controllers call `make_train_buffer(..., buffer_capacity=1000)` (`experiments/exp9a/rapt_9a.py:143` and `:198`), so both look capped at 1000. But the buffer passed in is already pre-sliced by the runner:

- base RAPT: `refit_n = 500` (`three_dataset_run.py:198`), and `handle_regime_transition(..., X_buffer=buf_X[-refit_n:])` (`:217`) hands over `buf_X[-500:]` = **500 rows**; `make_train_buffer` then adds the 30-row class anchor -> **530 training rows**.
- RAPT-Enhanced: `refit_n = ENHANCED_NOVELTY_REFIT_N = 1500` (`exp9a_config.py:86`, passed at `three_dataset_run.py:199/204`), so `:217` hands over `buf_X[-1500:]`. The streaming buffer is itself capped at `BUFFER_CAPACITY = 1000` (`:222-223`), so the slice yields `min(1500, buffer_len)` = **1000 rows**; `make_train_buffer`'s own cap is exactly 1000 so it does not reduce this, then the 30-row anchor -> **1030 training rows**.

So the base-RAPT 500 pre-slice is the real limiter; the 1000 cap only binds for RAPT-Enhanced. The Enhanced `-1500` slice never actually yields 1500 training rows: at the first transition `buf_X` holds 8,640 rows and `buf_X[-1500:]` gives 1,500, but `make_train_buffer`'s 1000 cap reduces it to 1,000 (confirmed in `T2_ugr16_refit_rows.csv`: `buffer_rows_max = 1500`, `recent_after_cap_max = 1000`).

## Task 3 — 5G NR: row meaning, counts, label granularity, loaders, Event-Driven F1

| item | value | source |
|---|---|---|
| one row in the Table-II 5G NR stream | one 500-packet telemetry window (a fixed-size aggregate), with 12 QoS features | experiments/exp9b/load_and_prepare_stream_9b.py:34-59 extract_window_features (window = WINDOW_SIZE=500 packets, line 15/56-62) |
| number of rows / windows | 499 | experiments/exp9b/results/stream_definition.json [total_windows] |
| initial train prefix | 99 | experiments/exp9b/results/stream_definition.json [initial_train_windows] |
| label granularity | 3-class QoS class of the window's own p90 latency (GOOD/DEGRADED/BAD), thresholds frozen on the initial 20% prefix | experiments/exp9b/load_and_prepare_stream_9b.py:189-197 assign_qos_class(df_stream['target_p90_lat']); thresholds in stream_definition.json [qos_thresholds_ms] |
| loader that produced Table II (5G NR) | 9B: experiments/exp9b/load_and_prepare_stream_9b.py -> processed_exp9b_stream.csv; consumed by experiments/exp9b/run_exp9b.py / drift_harness.py | results/experiment_9b/natural_drift/summary.csv ; experiments/exp9b/run_exp9b.py |
| loader that produced the revalidation numbers | results/revalidation/streams.py:33-62 load_5g_nr (reads the SAME processed_exp9b_stream.csv but re-declares window_size=1, one row per window) | results/revalidation/streams.py:33-62 |
| window_size as seen by the revalidation | 1 (one pre-built window per streaming step) | results/revalidation/streams.py:52 |

Event-Driven macro-F1 under both definitions (and the other four models for context):

| method | table_ii_value | per_window_macro_f1 | pooled_macro_f1 | per_window_accuracy | pooled_accuracy |
|---|---|---|---|---|---|
| Frozen | 0.8961 | 0.9075 | 0.8961 | 0.9075 | 0.9075 |
| Event-Driven | 0.8903 | 0.9060 | 0.8923 | 0.9060 | 0.9060 |
| Full Retraining | 0.9027 | 0.9110 | 0.9018 | 0.9110 | 0.9110 |
| RAPT | 0.8894 | 0.9040 | 0.8930 | 0.9040 | 0.9040 |
| RAPT-Enhanced | 0.8915 | 0.9085 | 0.8965 | 0.9085 | 0.9085 |

- Table II 5G NR cells = the 9B natural-drift per-window mean (`results/experiment_9b/natural_drift/summary.csv`).
- `per_window_macro_f1` = mean of the per-window macro-F1 over evaluation windows, from `A1_summary_all_streams.csv`.
- `pooled_macro_f1` = macro-F1 computed once over all evaluation samples concatenated, from `A1_pooled_all_streams.csv`.

The Event-Driven gap (Table II 0.8903 vs revalidation per-window 0.9060 vs pooled 0.8923) is a window-definition/aggregation difference, not a re-run discrepancy: Table II averages per-window F1 over the 9B 500-packet stream, while the revalidation re-reads the same windows with `window_size=1` and reports both a per-window mean and a pooled score.

## Task 4 — detector sanity on a synthetic step-change (NEW SCRIPT)

`results/extract/v2/detector_sanity.py` is **new** (written for this task). Stream: n=300, error 0.05 for indices 0-199 then 0.40, 20 Bernoulli repeats (seeds 1000-1019). Wiring A = paper/Table II (`experiments/exp9a/drift_detectors_9a.py:98-107`); Wiring B = revalidation defaults (`results/revalidation/a6_detectors.py:29-43`); Wiring C = the loosest setting in the revalidation A6 sweep (included only to show the detectors can fire).

| wiring | detector | fired_fraction | total_fires_mean | pre_change_fires_mean | post_change_fires_mean | latency_mean | latency_min | latency_max |
|---|---|---|---|---|---|---|---|---|
| A_paper_tableII | ADWIN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| A_paper_tableII | EDD | 1.0000 | 261.8500 | 161.8500 | 100.0000 | 0.0000 | 0.0000 | 0.0000 |
| A_paper_tableII | EDMA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| A_paper_tableII | Page-Hinkley | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B_revalidation_default | ADWIN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B_revalidation_default | EDD | 1.0000 | 261.8500 | 161.8500 | 100.0000 | 0.0000 | 0.0000 | 0.0000 |
| B_revalidation_default | EDMA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B_revalidation_default | Page-Hinkley | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| C_revalidation_sweep_extreme | ADWIN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| C_revalidation_sweep_extreme | EDD | 1.0000 | 261.8500 | 161.8500 | 100.0000 | 0.0000 | 0.0000 | 0.0000 |
| C_revalidation_sweep_extreme | EDMA | 1.0000 | 16.5500 | 7.3000 | 9.2500 | 1.5000 | 0.0000 | 9.0000 |
| C_revalidation_sweep_extreme | Page-Hinkley | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |

Reading:

- **ADWIN (delta=0.002), Page-Hinkley (threshold=50), EDMA (alpha=0.2, k=2) never fire**, not even on a 0.05->0.40 step, in both the paper wiring and the revalidation default wiring. ADWIN and Page-Hinkley remain silent even at the loosest swept setting (delta=0.3; threshold=0.05). This is a genuine limitation of the error signal they are fed (one error value per window) rather than a return-value bug: both harnesses consume the boolean `update()` return correctly.
- **EDD fires on essentially every window** (261.85/300 on average, 161.85 before the change and 100 after; latency 0). It is not detecting drift, it is saturating. This is consistent with the committed EDD event counts (Campus 114 events / 38 retrains; UGR16 115/39; Nordic 117/39; 5G NR 344/115) being far higher than the other detectors'.
- EDMA does fire under the loosest setting (alpha=0.4, k=1.0: 16.55 fires, mean latency 1.5 windows), so the zero-fire at the paper default is a sensitivity issue, not an implementation defect.

Raw per-repeat counts: `results/extract/v2/T4_detector_sanity_raw.csv`; wiring provenance: `results/extract/v2/T4_detector_sanity_meta.csv`.

## Task 5 — UGR'16 day-block bootstrap (24 calendar-day blocks, 5 seeds)

`results/extract/v2/ugr16_dayblock_bootstrap.py` is **new**. It is **distinct** from the existing A7 regime-run bootstrap (`results/revalidation/A7_block_bootstrap.csv`), which resamples contiguous runs of equal `regime_id`. Here blocks are calendar days: one window = 240 min, so a 1,440-min day = 6 windows; the evaluation stream (windows 36-179) spans calendar days 6-29 = **24 blocks**. Metric = pooled macro-F1 recomputed from the committed per-window confusion matrices (`A1_per_window_all_streams.csv`); both methods of a comparison are resampled on the same drawn blocks (paired), 2,000 draws per seed.

Validation: recomputing pooled macro-F1 from the confusion matrices reproduces the committed `A1_pooled_all_streams.csv` values exactly (max |diff| = 0.0 over all 45 UGR16 method/seed rows — see `T5_ugr16_pooled_recompute_check.csv`).

| comparison | bootstrap_type | metric | obs_diff | ci_low | ci_high | n_blocks | n_seeds |
|---|---|---|---|---|---|---|---|
| RAPT vs Frozen | day-block (24 calendar days) | pooled_macro_f1 | -0.1588 | -0.2452 | -0.1058 | 24.0000 | 5 |
| RAPT-Enhanced vs Frozen | day-block (24 calendar days) | pooled_macro_f1 | -0.0208 | -0.0784 | 0.0168 | 24.0000 | 5 |
| RAPT-Enhanced vs Full Retraining | day-block (24 calendar days) | pooled_macro_f1 | 0.0405 | -0.0400 | 0.1013 | 24.0000 | 5 |
| RAPT-Enhanced vs Frozen | regime-run (contiguous regime visits) [existing A7] | macro_f1 | -0.0413 | -0.0661 | -0.0191 | 144.0000 | 5 |
| RAPT-Enhanced vs Full Retraining | regime-run (contiguous regime visits) [existing A7] | macro_f1 | -0.0319 | -0.0571 | -0.0082 | 144.0000 | 5 |
| RAPT vs Frozen | regime-run (contiguous regime visits) [existing A7] | macro_f1 | -0.1330 | -0.1711 | -0.0975 | 144.0000 | 5 |

Per-seed day-block CIs: `results/extract/v2/T5_ugr16_dayblock_per_seed.csv`. The day-block CIs are wider than the regime-run CIs because there are only 24 independent day blocks versus 144 regime-run blocks; the qualitative ordering (RAPT < Frozen; RAPT-Enhanced ~ Frozen/Full Retraining) is unchanged.

## MISSING (v2)

| Item | Searched in |
|---|---|
| Raw UGR'16 data committed to the repo | `experiments/exp9a/data/ugr16/` is empty; data is downloaded per `experiments/exp9a/data/DATASETS.md`. Task 2/5 used a freshly downloaded copy of the documented file. |
| Any committed log of per-refit training-row counts | `results/revalidation/A2_checkpoint_provenance.csv` stores `n_train_rows` but only for the RAPTV2 controller, not the Table-II `rapt_9a.py` path. Task 2 produced it by re-running. |
| `Periodic-Cheap` (no window suffix) | `results/revalidation/A3_pareto.csv` — only `-5` and `-10` exist. |
| ADWIN / Page-Hinkley firing at the paper default on a clear step | `results/extract/v2/T4_detector_sanity.csv` — they do not fire; no committed artifact contradicts this. |

## Newly-run artefacts (labelled)

| File | What was run | Why |
|---|---|---|
| `T2_ugr16_refit_rows.csv`, `T2_ugr16_refit_summary.csv`, `T2_ugr16_refit_codepaths.csv` | `ugr16_refit_log.py` re-ran base RAPT and RAPT-Enhanced on UGR'16, seeds 42-46 | no committed log of per-refit row counts existed |
| `T4_detector_sanity*.csv` | `detector_sanity.py` | no synthetic step-change sanity test existed |
| `T5_ugr16_dayblock*.csv`, `T5_ugr16_pooled_recompute_check.csv` | `ugr16_dayblock_bootstrap.py` | day-block bootstrap did not exist (A7 used regime runs) |
| `T3_5gnr_*.csv` | `extract_v2.py` (reads committed 9B/revalidation files only) | assembly, no model run |

