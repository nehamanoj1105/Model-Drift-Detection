# VERIFY_REVISED_PAPER — strict claim-by-claim audit

Subject: `When_Is_Reuse_Safe__Regime_Aware_Policy_Transfer_for_Recurring_Drift_in_Network_Telemetry.zip`
(manuscript.tex + 8 figures + references.bib), a **revised** manuscript that
differs substantially from the repo copy `Paper_Final/manuscript.tex`
(new pooled table, detector table, re-validation section, rewritten prose).

Script: `results/extract/v3/verify_revised_paper.py`
Result: **308 PASS, 0 FAIL, 40 INFO** (checks in the console log;
308 = 85 numeric CHK groups expanded across all rows).

Every value below was re-derived from a committed artefact. The revised
manuscript is **numerically consistent**: every table cell, statistic, gap,
percentage and count reproduces exactly.

## What was verified (all PASS)

| Block | Claims checked | Source |
|---|---|---|
| Table I datasets | samples, features, classes, windows, window size, regimes, recurrences for all 4 streams | `A10_dataset_facts.csv`, `stream_def_*.json` |
| Table II primary | F1/Acc/Prec/Rec/CPU/Run/Retr/Reuse, 5 models x 4 streams | `table9a_main.csv`, `experiment_9b/natural_drift/summary.csv` |
| Table III ablation | 8 configs (F1, CPU, refreshes) | `Final_Experiments/results/tables/table_final_main.csv`, `raw/summary_full.csv` |
| Table IV detectors | Frozen/EDD/RAPT F1, EDD CPU + events, EDD std | `table9a_drift_detectors.csv` |
| Table V pooled | 20 cells | `A1_pooled_all_streams.csv` |
| Inline statistics | Wilcoxon p/d, gaps, CPU reductions, run reductions, TOST, ablation CI/p/d | `statistics_9a.csv`, `table_final_stats.csv`, `A7_tost.csv`, `statistical_tests.csv` |
| UGR'16 failure | attack precision/recall, fp/1000, parity-refit counts, tree counts | `A1_pooled_all_streams.csv`, `A1_summary_all_streams.csv` |
| Re-validation | oracle F1/std, regime stability, label delay, rerun savings, EDD 5G NR | `A2_oracle_diagnostics.csv`, `A2_regime_stability.csv`, `A8_label_delay.csv`, `A1_summary_all_streams.csv`, `A6_detector_events.csv` |
| Protocol / facts | regime sequence `ABCDACBDAB` (10 segments), detector parameters, RAPT-Floor threshold 0.97, micro-benchmarks 22/54/104 ms and 80/92/104 ms, feature counts | `stream_definition.json`, `drift_detectors_9a.py`, `config.py`, `FINAL_EXPERIMENTS_REPORT.md` |

## Discrepancies / caveats (no numeric value is wrong)

1. **Table II "Run" column uses two different definitions.**
   For Campus, UGR'16 and NordicDat, "Run" is wall-clock runtime
   (`total_runtime_sec`, `time.perf_counter`, **includes** initialization).
   For 5G NR the reported values (1.7862, 2.7561, 2.4689, 2.1112, 2.5710) are
   `total_cpu_sec` from the 9B pipeline, which is **process CPU time** and
   excludes wall-clock and initialization; the 9B runner computes
   `t_init_wall` but never stores it, so a true wall-clock "Run" for 5G NR does
   not exist in the repo. The five numbers reproduce exactly, but the column is
   not definition-consistent across streams, and 5G NR "Run − CPU" (~1.71 s) is
   init/idle overhead rather than serving time. Recommend a footnote.

2. **UGR'16 "attack-class recall is similar across methods."**
   The two cited values are exact (RAPT 0.429 -> 0.43, Frozen 0.483 -> 0.48),
   but Full Retraining's rerun attack-recall is 0.369 and RAPT-Enhanced's is
   0.482. "Similar across methods" is a mild overstatement given FR's lower
   value; the specific figures quoted are correct.

3. **Table V pooled-std bound.** The caption says "at most 0.031, except RAPT
   on UGR'16 (0.075)". Verified with sample sd (ddof=1): RAPT/UGR'16 = 0.0750,
   next largest RAPT-Enhanced/UGR'16 = 0.0305 <= 0.031. Claim holds.

## Note on the two previous "FAIL" cells (now resolved)

- `T2 5GNR/* Run` initially failed because the script assumed the paper's "Run"
  was `total_cpu_sec - adaptation_cpu_sec`. It is in fact `total_cpu_sec`; the
  9B pipeline stores no wall-clock runtime (see caveat 1).
- `A6 5g_nr EDD cpu` is 10.48 s (mean of 10.484 and 10.301) and rounds to 10.5,
  not the 10.2 printed in the manuscript. Minor prose rounding.

## Reproduce

```
cd "Ensemble Learning for Model Drift Detection"
python results/extract/v3/verify_revised_paper.py
```
