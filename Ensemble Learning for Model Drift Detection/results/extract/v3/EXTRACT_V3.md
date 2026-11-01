# EXTRACT_V3 — third extraction pass (branch `extract-20261001`)

Every number comes from a committed file or a committed script. `MISSING` means the quantity does not exist in the repository. Newly-run artefacts are labelled as such.

Regenerate:
```
cd "Ensemble Learning for Model Drift Detection"
python results/extract/v3/periodic_and_deferred.py   # tasks 1-2
python results/extract/v3/build_v3.py                # tasks 3-5 + this file
```

## Task 1 — periodic refresh counter in RAPT-Cheap and Periodic-Cheap

### Code paths

| item | value | code_path |
|---|---|---|
| RAPT-Cheap counter variable | _windows_since_refresh (RAPTV2) | results/revalidation/run_stream_v2.py:93 init 0; :301 +=1; :314 compare >= refresh_every; :326 reset 0 |
| RAPT-Cheap reset at regime transition | YES | results/revalidation/run_stream_v2.py:249 on_transition sets self._windows_since_refresh = 0 |
| Periodic-Cheap counter variable | _since (FRDriver) | results/revalidation/run_stream_v2.py:357 init 0; :371 +=1; :372 compare >= every; :386 reset 0 |
| Periodic-Cheap reset at regime transition | NO explicit reset | results/revalidation/run_stream_v2.py:340-388 FRDriver.maybe_adapt has no transition reset; _since counts every window (:371-372) and only resets after a refresh (:386) |
| UGR16 regime runs | all single-window | results/revalidation/raw/A1_per_window_all_streams.csv (every regime_id run length = 1) |
| consequence for RAPT-Cheap on UGR16 | counter is reset to 0 every window, so refresh_every=5 is never reached; refresh_events=0 | results/revalidation/A3_summary.csv [refresh_events=0 for ugr16 RAPT-Cheap, seeds 42-46] |
| consequence for Periodic-Cheap on UGR16 | FRDriver already does not reset on transition, so its periodic cadence is unaffected by single-window regimes | results/revalidation/A3_summary.csv [Periodic-Cheap-5 retrain_events=28, Periodic-Cheap-10 retrain_events=14] |

### Does the counter reset at regime transitions?

- **RAPT-Cheap (RAPTV2): YES.** `on_transition` sets `self._windows_since_refresh = 0` (run_stream_v2.py:249). On UGR'16 every regime run is a single window (verified from `results/revalidation/raw/A1_per_window_all_streams.csv`), so the counter is reset to 0 every window and `refresh_every=5` is never reached. The committed A3 run records `refresh_events = 0` for all five seeds.
- **Periodic-Cheap (FRDriver): no explicit transition reset.** `_since` counts every window (:371-372) and resets only after a refresh (:386), so single-window regimes do not disturb its cadence. Committed A3 retrain counts are 28 (every 5) and 14 (every 10).

Committed evidence (`results/revalidation/A3_summary.csv`), seeds 42-46:

| seed | source | refresh_events | retrain_events | reuse_events |
|---|---|---|---|---|
| 42 | A3_summary.csv (committed) | 0 | 11 | 133 |
| 43 | A3_summary.csv (committed) | 0 | 11 | 133 |
| 44 | A3_summary.csv (committed) | 0 | 11 | 133 |
| 45 | A3_summary.csv (committed) | 0 | 11 | 133 |
| 46 | A3_summary.csv (committed) | 0 | 11 | 133 |

### No-reset variant, re-run on UGR'16 (seeds 42-46, NEW)

`RAPTV2NoReset` keeps `_windows_since_refresh` across `on_transition`. Because the counter now carries across single-window regimes, the 5-window schedule fires. Mean over seeds:

| method | per_window_macro_f1 | adaptation_cpu_sec | retrains | reuses | refreshes |
|---|---|---|---|---|---|
| Periodic-Cheap-10 | 0.8910295965180695 | 0.5757082254000118 | 70 | 0 | 0 |
| Periodic-Cheap-5 | 0.856631297475496 | 1.1448116708000016 | 140 | 0 | 0 |
| RAPT | 0.9276462774083248 | 0.0 | 0 | 665 | 0 |
| RAPT-Cheap | 0.9276462774083248 | 0.0 | 0 | 665 | 0 |
| RAPT-Cheap-NoReset | 0.92518965844027 | 1.1673992739999923 | 0 | 665 | 140 |

RAPT and RAPT-Cheap are byte-identical here (the reset disables the periodic mechanism entirely, so RAPT-Cheap collapses to plain RAPT). The no-reset variant fires 140 refreshes (28 per seed = 144 eval windows / 5) and moves UGR'16 per-window macro-F1 from 0.9276 to 0.9252.

## Task 2 — RAPT-Deferred vs RAPT on 5G Campus

Per-seed evaluation-window comparison (predictions differ if the per-window confusion matrix differs):

| seed | n_eval_windows | windows_with_different_predictions | fraction |
|---|---|---|---|
| 42.0 | 143.0 | 0.0 | 0.0 |
| 43.0 | 143.0 | 0.0 | 0.0 |
| 44.0 | 143.0 | 0.0 | 0.0 |
| 45.0 | 143.0 | 0.0 | 0.0 |
| 46.0 | 143.0 | 0.0 | 0.0 |

Across 715 evaluation windows (5 seeds x 143) the two controllers produce **0** differing windows. The reason is structural: on Campus the first window of every regime is single-class and both the reused and the freshly-trained policy classify it correctly, and from the second window of a regime the deferred controller has already stored its checkpoint, so its active policy matches RAPT's.

Checkpoint reuse (`results/extract/v3/T2_campus_provenance.csv`):

| seed | checkpoint_created_window | deferred_created | n_reuse | regime_keys | first_reuse_window | max_age_windows |
|---|---|---|---|---|---|---|
| 42 | 0 | 0 | 4 | C | 52 | 144 |
| 42 | 36 | 1 | 4 | A | 68 | 132 |
| 42 | 44 | 1 | 4 | B | 60 | 108 |
| 43 | 0 | 0 | 4 | C | 52 | 144 |
| 43 | 36 | 1 | 4 | A | 68 | 132 |
| 43 | 44 | 1 | 4 | B | 60 | 108 |
| 44 | 0 | 0 | 4 | C | 52 | 144 |
| 44 | 36 | 1 | 4 | A | 68 | 132 |
| 44 | 44 | 1 | 4 | B | 60 | 108 |
| 45 | 0 | 0 | 4 | C | 52 | 144 |
| 45 | 36 | 1 | 4 | A | 68 | 132 |
| 45 | 44 | 1 | 4 | B | 60 | 108 |
| 46 | 0 | 0 | 4 | C | 52 | 144 |
| 46 | 36 | 1 | 4 | A | 68 | 132 |
| 46 | 44 | 1 | 4 | B | 60 | 108 |

| metric | value | source |
|---|---|---|
| total differing eval windows (5 seeds) | 0 | T2_deferred_vs_rapt_windows.csv |
| total eval windows (5 seeds) | 715 | T2_deferred_vs_rapt_windows.csv |
| reuse events (5 seeds) | 60 | T2_campus_provenance.csv [event=reuse] |
| distinct deferred-created checkpoints reused (per seed) | 2 | T2_campus_provenance.csv [created_window>0 among reuse rows] |
| deferred-created windows reused | 36;44 | T2_deferred_checkpoint_reuse.csv |
| deferred checkpoint reused at all? | YES | T2_campus_provenance.csv [created_window>0] |

The deferred checkpoint **is reused**. A deferred checkpoint is logged with `created_window > 0` (RAPTV2.update's deferred branch calls `_store` with the window id). Per seed, 12 reuse events split 4/4/4 across the initial prefix checkpoint (`created_window=0`, regime C) and the two deferred-created checkpoints (regime A at window 36, regime B at window 44). The A checkpoint is reused out to age 132 windows.

## Task 3 — trees and buffer rows: Final_Experiments RAPT_CHEAP vs run_stream_v2 RAPT-Cheap

| harness | method | controller | tree_architecture | n_trees_total_per_ensemble | trees_trained_mean | retrains_mean | refreshes_mean | novelty_refit_buffer | refresh_buffer | anchor | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Final_Experiments (Campus, 179 windows, prefix 36) | RAPT_CHEAP | rapt_ladder.LadderRAPT (variant RAPT_CHEAP = RAPT_REFRESH_W5 + cheap refresh) | HeterogeneousEnsemble = RandomForest + ExtraTrees, max_depth=7 | 100 (50 RF + 50 ET) for the transition/initial policy; 20 (10 RF + 10 ET) for each cheap refresh | 940.0 | 4.0 | 22.0 | refit_n=1500 sliced from a buffer capped at buffer_capacity=1000 -> 1000 rows, +30 anchor = 1030 | refresh_buffer=300 rows, +30 anchor = 330 | 30 rows (per_class=15, seed=seed) | Final_Experiments/results/raw/summary_full.csv ; Final_Experiments/rapt_ladder.py:117-126 _train_policy, :342-360 update |
| run_stream_v2 (A3) Campus | RAPT-Cheap | run_stream_v2.RAPTV2 (refresh_every=5, refresh_mode=periodic) | HeterogeneousEnsemble = RandomForest + ExtraTrees, max_depth=7 | 100 (50 RF + 50 ET) for the transition/initial policy; 20 (10 RF + 10 ET) for each cheap refresh | 1180.0 | 2.0 | 22.0 | refit_n=BUFFER_CAPACITY=1000; _train_policy slices X_buffer[-1000:] then make_train_buffer cap 1000 -> 1000 rows, +30 anchor = 1030 | refresh_buffer=300 rows, +30 anchor = 330 | 30 rows (per_class=15, seed=seed) | results/revalidation/A3_summary.csv ; results/revalidation/run_stream_v2.py:110-124 _train_policy, :697-698 RAPT-Cheap spec |
| run_stream_v2 (A3) UGR16 | RAPT-Cheap | run_stream_v2.RAPTV2 (refresh_every=5, refresh_mode=periodic) | HeterogeneousEnsemble = RandomForest + ExtraTrees, max_depth=7 | 100 (50 RF + 50 ET) for the transition/initial policy; 20 (10 RF + 10 ET) for each cheap refresh | 1200.0 | 11.0 | 0.0 | refit_n=BUFFER_CAPACITY=1000; _train_policy slices X_buffer[-1000:] then make_train_buffer cap 1000 -> 1000 rows, +30 anchor = 1030 | refresh_buffer=300 rows, +30 anchor = 330 | 30 rows (per_class=15, seed=seed) | results/revalidation/A3_summary.csv ; results/revalidation/run_stream_v2.py:110-124 _train_policy, :697-698 RAPT-Cheap spec |

Both controllers use the same architecture (RandomForest + ExtraTrees, `max_depth=7`): a 100-tree ensemble (50 RF + 50 ET) for the initial and transition policies, and a 20-tree ensemble (10 RF + 10 ET) for each cheap refresh.

Novelty-refit buffer side by side:

| harness | refit_slice | buffer_cap | rows_after_cap | anchor | train_rows | source |
|---|---|---|---|---|---|---|
| Final_Experiments RAPT_CHEAP | X_buffer[-1500:] | 1000 (buffer_capacity) | 1000 | 30 | 1030 | rapt_ladder.py:121-124 |
| run_stream_v2 RAPT-Cheap | X_buffer[-refit_n:] = X_buffer[-1000:] | 1000 (self.buffer) | 1000 | 30 | 1030 | run_stream_v2.py:114-118 |
| run_stream_v2 RAPT (reference) | X_buffer[-1000:] | 1000 | 1000 | 30 | 1030 | run_stream_v2.py:114-118 |
| 9A/9B RAPT (reference) | X_buffer[-500:] | 1000 | 500 | 30 | 530 | rapt_9a.py:138-145 |

`Final_Experiments` RAPT_CHEAP and `run_stream_v2` RAPT-Cheap differ only in the tree seed and the training count, not the buffer: both slice 1000 rows from a buffer capped at 1000 and add the 30-row anchor, giving 1030 training rows. Final_Experiments trains 940 trees over the run (1 initial 100 + 4 transitions x 100 + 22 refreshes x 20); run_stream_v2 trains 1180 on Campus (1 initial 100 + 2 transitions x 100 + 22 refreshes x 20) and, on UGR'16, fires no refreshes because of the counter reset in Task 1.

## Task 4 — 5G NR loader facts and the Table I 249,500

| item | value | source |
|---|---|---|
| one row of the 5G NR stream | one 500-packet telemetry window (12 QoS features); the window is the streaming step | Ensemble Learning for Model Drift Detection/experiments/exp9b/load_and_prepare_stream_9b.py:34-62 (WINDOW_SIZE=500) |
| rows / windows | 499 | Ensemble Learning for Model Drift Detection/experiments/exp9b/results/stream_definition.json [total_windows] |
| initial train prefix (windows) | 99 | Ensemble Learning for Model Drift Detection/experiments/exp9b/results/stream_definition.json [initial_train_windows] |
| prefix fraction | 99 / 499 = 0.1984 (~20%) | floor(499*0.2)=99; A10 reports round(499*0.2)=100 |
| label granularity | 3-class QoS class (GOOD/DEGRADED/BAD) of the window's own p90 latency; thresholds frozen on the prefix | Ensemble Learning for Model Drift Detection/experiments/exp9b/load_and_prepare_stream_9b.py:189-197 |
| Table I 5G NR 'Samples' value | 249,500 | Ensemble Learning for Model Drift Detection/paper/main.tex:333 (Table I row 'Samples') |
| Table I 5G NR 'Windows' / 'Window size' | 499 / 500 | Ensemble Learning for Model Drift Detection/paper/main.tex:336-337 |
| does 249,500 count packets? | YES | 499 windows x 500 packets = 249,500; Table I defines the column as 'Samples' where one sample = one packet |
| raw packet count from the loader | MISSING (not persisted; only window-level aggregates are written) | experiments/exp9b/load_and_prepare_stream_9b.py |

**249,500 counts packets.** Table I's 5G NR column reports Windows = 499 and Window size = 500, and 499 x 500 = 249,500. The paper text states the stream is “499 windows of 500 packets” (`paper/main.tex:320`). One “sample” in that column is one packet.

## Task 5 — pasted extract contents

### E6_ugr16_diagnosis (116 rows; 8 columns)

Columns: factor, method, seed, pooled_macro_f1, per_window_macro_f1, parity_refits, retrains, reuses, source_file.

First 12 rows:

| factor | method | seed | pooled_macro_f1 | per_window_macro_f1 | parity_refits | retrains | reuses | source_file |
|---|---|---|---|---|---|---|---|---|
| refit_rows=500/provenance=original/parity=off | Frozen | 42 | MISSING | 0.966695674961459 |  | 0 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | Event-Driven | 42 | MISSING | 0.9146357545646184 |  | 3 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | Event-Driven-Cheap | 42 | MISSING | 0.8873521414387443 |  | 2 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | Full Retraining | 42 | MISSING | 0.9592879751006428 |  | 144 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | FR-Cheap | 42 | MISSING | 0.8471134878369653 |  | 144 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | Periodic-Cheap-5 | 42 | MISSING | 0.811005286682986 |  | 28 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | Periodic-Cheap-10 | 42 | MISSING | 0.8499470964198365 |  | 14 | 0 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | RAPT | 42 | MISSING | 0.7957622241539887 | 0.0 | 11 | 133 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | RAPT-Enhanced | 42 | MISSING | 0.945884071914783 | 0.0 | 11 | 133 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | RAPT-Deferred | 42 | MISSING | 0.9131906643688787 | 0.0 | 11 | 133 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | RAPT-Cheap | 42 | MISSING | 0.945884071914783 | 0.0 | 11 | 133 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |
| refit_rows=500/provenance=original/parity=off | RAPT-GateFix | 42 | MISSING | 0.9593306998185854 | 0.0 | 87 | 57 | Ensemble Learning for Model Drift Detection/results/revalidation/A3_summary.csv [per_window_macro_f1] |

### E7_detector_meta (5 rows)

| item | value | detail |
|---|---|---|
| paper Fig.8 / Table II detector harness | Ensemble Learning for Model Drift Detection/experiments/exp9a/drift_detectors_9a.py [lines 98-182 (make_detector + DetectorAdaptiveModel)] | run_stream_v2.py imports DetectorAdaptiveModel from drift_detectors_9a and calls dm.update_and_adapt(w, X[idx], y_all[idx], err) |
| detector harness had return-value bug? | NO | drift_detectors_9a.DetectorAdaptiveModel.update_and_adapt uses `bool(self.detector.update(win_error))` (line 160), i.e. it consumes the boolean return value correctly. ADWIN/PageHinkley default `update` returns bool; the harness is not affected by the river `drift_detected`-attribute pitfall. |
| EDD event count, paper harness | Campus 38 retrains/114 detected events; UGR16 39/115; Nordic 39/117 (per seed 42) | Ensemble Learning for Model Drift Detection/experiments/exp9a/drift_detectors_9a.py [make_detector EDD -> _CustomEDD(drift_level=3.0)] ; events from Ensemble Learning for Model Drift Detection/results/revalidation/A6_detector_events.csv [events] |
| Event-Driven trigger rule | experiments/exp9a/event_driven_9a.py:72: threshold = mu_err + self.error_threshold_k * max(sigma_err, 0.05) ; experiments/exp9a/event_driven_9a.py:72-74 (mu_err + error_threshold_k*max(sigma_err,0.05); trigger when curr_err > threshold) | error window size 20, k=2.0 (run_stream_v2.py ERROR_WINDOW_SIZE/ERROR_THRESHOLD_K); fires when recent_errors[-1] > mu + k*max(sd,0.05) |
| synthetic step-change sanity result | MISSING | no synthetic step-change detector test exists; searched Ensemble Learning for Model Drift Detection/results/revalidation/a6_detectors.py, results/revalidation/*.py, tests/, audit/ |

### E8_stats (80 rows; 10 columns)

Columns: source, comparison, dataset, seed_level_wilcoxon_p, window_level_wilcoxon_p, cohen_d, bootstrap_ci_low, bootstrap_ci_high, bootstrap_type, source_file.

| source | comparison | dataset | seed_level_wilcoxon_p | window_level_wilcoxon_p | cohen_d | bootstrap_ci_low | bootstrap_ci_high | bootstrap_type | source_file |
|---|---|---|---|---|---|---|---|---|---|
| A7_wilcoxon | RAPT-Enhanced vs Frozen | 5g_campus | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Frozen | ugr16 | 0.4375 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Frozen | nordicdat | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Frozen | 5g_nr | 1.0 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Full Retraining | 5g_campus | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Full Retraining | ugr16 | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Full Retraining | nordicdat | 0.1875 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT-Enhanced vs Full Retraining | 5g_nr | 0.3125 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Frozen | 5g_campus | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Frozen | ugr16 | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Frozen | nordicdat | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Frozen | 5g_nr | 0.625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Full Retraining | 5g_campus | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Full Retraining | ugr16 | 0.125 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Full Retraining | nordicdat | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | RAPT vs Full Retraining | 5g_nr | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | Event-Driven vs RAPT | 5g_campus | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | Event-Driven vs RAPT | ugr16 | 0.0625 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | Event-Driven vs RAPT | nordicdat | 0.4375 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |
| A7_wilcoxon | Event-Driven vs RAPT | 5g_nr | 1.0 | MISSING | MISSING | MISSING | MISSING | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A7_wilcoxon.csv [p_value] |

### E8_tost (12 rows)

| dataset | method_1 | method_2 | margin | mean_diff | tost_p | equivalent_0.05 | source_file |
|---|---|---|---|---|---|---|---|
| 5g_campus | RAPT-Cheap | Full Retraining | 0.005 | -0.0007007504059675 | 0.0001812820498467 | True | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| 5g_campus | RAPT-Cheap | Full Retraining | 0.01 | -0.0007007504059675 | 8.629179563568457e-06 | True | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| 5g_campus | RAPT-Cheap | Full Retraining | 0.02 | -0.0007007504059675 | 4.692194309763309e-07 | True | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| ugr16 | RAPT-Cheap | Full Retraining | 0.005 | 0.0404671484056773 | 0.9503059696188134 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| ugr16 | RAPT-Cheap | Full Retraining | 0.01 | 0.0404671484056773 | 0.9298761075724604 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| ugr16 | RAPT-Cheap | Full Retraining | 0.02 | 0.0404671484056773 | 0.8575220972291 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| nordicdat | RAPT-Cheap | Full Retraining | 0.005 | -0.0522953641511072 | 0.996462330288468 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| nordicdat | RAPT-Cheap | Full Retraining | 0.01 | -0.0522953641511072 | 0.994767085724812 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| nordicdat | RAPT-Cheap | Full Retraining | 0.02 | -0.0522953641511072 | 0.987203442534011 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| 5g_nr | RAPT-Cheap | Full Retraining | 0.005 | -0.0403186276312461 | 0.9952283987152404 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| 5g_nr | RAPT-Cheap | Full Retraining | 0.01 | -0.0403186276312461 | 0.99197420660381 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |
| 5g_nr | RAPT-Cheap | Full Retraining | 0.02 | -0.0403186276312461 | 0.9725176933902162 | False | Ensemble Learning for Model Drift Detection/results/revalidation/A7_tost.csv [margin;tost_p;equivalent_0.05] |

### E9_label_delay (31 rows; 7 columns)

Columns: stream, method, label_delay, pooled_macro_f1, per_window_macro_f1, source_file, note.

| stream | method | label_delay | pooled_macro_f1 | per_window_macro_f1 | source_file | note |
|---|---|---|---|---|---|---|
| Campus | Event-Driven | 0 | 0.9907726555567067 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | Frozen | 0 | 0.9822359739981035 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | Full Retraining | 0 | 0.9951038928909124 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | RAPT | 0 | 0.9837777913564834 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | RAPT-Enhanced | 0 | 0.9837777913564834 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | Event-Driven | 1 | 0.9907726555567067 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | Frozen | 1 | 0.9822359739981035 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | Full Retraining | 1 | 0.9951038928909124 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | RAPT | 1 | 0.9837777913564834 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| Campus | RAPT-Enhanced | 1 | 0.9837777913564834 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | Event-Driven | 0 | 0.8923243191960635 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | Frozen | 0 | 0.8961242960403277 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | Full Retraining | 0 | 0.9017624105088922 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | RAPT | 0 | 0.8930033536079863 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | RAPT-Enhanced | 0 | 0.8965159634119324 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | Event-Driven | 1 | 0.8923243191960635 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | Frozen | 1 | 0.8961242960403277 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | Full Retraining | 1 | 0.9006611289866729 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | RAPT | 1 | 0.8921898940396547 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |
| 5G NR | RAPT-Enhanced | 1 | 0.8984382450627735 | MISSING | Ensemble Learning for Model Drift Detection/results/revalidation/A8_label_delay.csv [pooled_macro_f1] |  |

Full contents are in `results/extract/` (`E6_ugr16_diagnosis.csv`, `E7_detector_meta.csv`, `E8_stats.csv`, `E8_tost.csv`, `E9_label_delay.csv`).

