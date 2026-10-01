# Experiment 9B -- Drift Severity, Reuse Safety and a Revalidation Audit

This document is the revalidation and audit of the RAPT results that underpin the COMSNETS submission *When Is Reuse Safe? Regime-Aware Policy Transfer for Recurring Drift in Network Telemetry*. It re-analyses the existing four streams under one frozen protocol. No new dataset, no new RAPT variant and no synthetic drift construction were introduced. Where a number could not be reproduced the discrepancy is reported, not smoothed.

## 1. Dataset and existing 9B setup

- Streams: 5G Campus QoS, UGR'16, NordicDat (9A loaders) and 5G NR (the existing 9B processed window stream).
- 5G NR window size is 1 (one sample per window); the 9A streams use the existing window size, e.g. 10 for 5G Campus.
- Protocol: chronological prequential test-then-train, no shuffling, 20% initial prefix, seeds 42-46.
- Frozen config: `results/final/config_frozen.json`, sha256 `85b2ec23d17d29dbc981d71174f6e05e329b0e35477d52c77392ca5d8203fbc5`.
- The revalidation harness (`results/revalidation/`) imports the original RAPT and RAPT-Enhanced classes unchanged.

## 2. Primary metric and aggregation

The paper's primary table reports the mean of per-window macro-F1. The revalidation reproduces the 9A numbers exactly under that definition: 18/18 mean differences in `T5_reproduce_9a_stats.csv` match `experiments/exp9a/tables/statistics_9a.csv` to <1e-9, including p-values and Cohen's d.

A pooled macro-F1 (concatenating predictions over windows) is a different quantity. On UGR'16 the two diverge by up to 0.199. The published Table II uses the per-window mean for the three 9A streams but the pooled value for 5G NR (`T1_metric_reconciliation.csv`). The two are therefore not on a common scale in that table.

## 3. Natural drift results (observed)

| Stream | Frozen | Event-Driven | Full Retraining | RAPT | RAPT-Enhanced |
|---|---|---|---|---|---|
| 5G Campus QoS | 0.9364 | 0.9636 | 0.9836 | 0.9381 | 0.9381 |
| UGR'16 | 0.9690 | 0.8972 | 0.9595 | 0.8360 | 0.9276 |
| NordicDat | 0.2779 | 0.2547 | 0.4227 | 0.3818 | 0.3783 |
| 5G NR | 0.9075 | 0.9060 | 0.9110 | 0.9040 | 0.9085 |

Per-window mean macro-F1, seeds 42-46. Source `T1_pooled_all_streams.csv`.

## 4. Claim-by-claim audit

| Claim | Verdict |
|---|---|
| RAPT macro-F1 is below Full Retraining on all four streams | SUPPORTED |
| On UGR'16 RAPT has the lowest macro-F1 of the five models | SUPPORTED |
| UGR'16 RAPT-Enhanced recovers macro-F1 over RAPT | SUPPORTED |
| UGR'16 recovery is caused by the parity refit | NOT SUPPORTED |
| NordicDat RAPT deficit vs Full Retraining is small | INCONCLUSIVE |
| On 5G NR the five models are within a small spread | SUPPORTED |
| On Campus RAPT and RAPT-Enhanced coincide | SUPPORTED |

## 5. Why reuse fails on UGR'16 (observed + statistical)

The paper attributes the UGR'16 recovery of RAPT-Enhanced over RAPT (0.8360 -> 0.9276) to the parity refit. The controlled isolation in `T2_mechanism_isolation.csv` shows the parity branch never fires on UGR'16 (0 refits at every threshold 0.5-0.95), and that the recovery is reproduced by enlarging the novelty-refit buffer from 500 to 1000 rows:

| refit buffer | parity | macro-F1 | parity refits | retrains | reuses |
|---|---|---|---|---|---|
| 500 | off | 0.8360 | 0 | 11 | 133 |
| 500 | on | 0.8360 | 0 | 11 | 133 |
| 1000 | off | 0.9276 | 0 | 11 | 133 |
| 1000 | on | 0.9276 | 0 | 11 | 133 |

| Hypothesis / driver | Verdict | Value |
|---|---|---|
| H1 label semantics drift per regime visit | INCONCLUSIVE | {"n_pairs": 2, "mean_diff_same_minus_contiguous": 0.0} |
| H2 provenance bug (checkpoints trained on wrong regime) | SUPPORTED | delta=0.0798 CI=[0.0609,0.0996] p=0.0625 |
| H3 attack non-stationarity in time | NOT SUPPORTED | frozen_degradation=-0.0888 |
| DRIVER 1: novelty-refit training-set composition | SUPPORTED | delta=0.0917 CI=[0.0661,0.1210] p=0.0625 |
| DRIVER 2: parity refit (paper's proposed fix) | NOT SUPPORTED | delta=0.0000 CI=[0.0000,0.0000] p=1.0000 |
| REPORTED RECOVERY (0.8360 -> 0.9276) | SUPPORTED | delta=0.0917 CI=[0.0661,0.1210] p=0.0625 |

RAPT-Enhanced and RAPT-Cheap produce identical UGR'16 macro-F1 (0.9276), which is consistent with the recovery coming from the shared refit path rather than the parity path. Frozen does not degrade over the 30-day block, so the deficit is not simple non-stationarity in time. H1 (per-visit label-semantics drift) has only two paired regime visits on UGR'16 and is reported as inconclusive, not as supported.

## 6. Cost-matched baselines and the headline configuration (observed)

| Stream | RAPT-Cheap F1 | Full Retraining F1 | delta | CPU saving | verdict |
|---|---|---|---|---|---|
| 5G Campus QoS | 0.9817 | 0.9836 | -0.0019 | 27.4% | SUPPORTED |
| UGR'16 | 0.9276 | 0.9595 | -0.0319 | 84.1% | NOT SUPPORTED |
| NordicDat | 0.4100 | 0.4227 | -0.0128 | 33.9% | NOT SUPPORTED |
| 5G NR | 0.8830 | 0.9110 | -0.0280 | -317.6% | NOT SUPPORTED |

The cost-aware configuration matches Full Retraining on 5G Campus only. On UGR'16 it is 0.0319 lower, on NordicDat 0.0128 lower, and on 5G NR it is 0.0280 lower and costs more adaptation CPU than Full Retraining. The headline efficiency claim therefore does not transfer beyond Campus.

## 7. Statistics (statistical)

The paper's window-level paired Wilcoxon tests reproduce exactly. A seed-level exact Wilcoxon over 5 seeds has a minimum two-sided p of 0.0625, so seed-level significance cannot be reached at n=5; those tests are reported with that ceiling stated. Holm correction over the Campus ablation ladder is in `T5_holm_campus_ladder.csv`.

## 8. Label-availability delay (observed)

| Stream | best @ delay 0 | best @ delay 1 | ranking stable | max F1 shift |
|---|---|---|---|---|
| 5G Campus QoS | Full Retraining | Full Retraining | True | 0.0000 |
| UGR'16 | not run | not run | n/a | n/a |
| NordicDat | Full Retraining | Event-Driven | False | 0.1256 |
| 5G NR | Full Retraining | Full Retraining | True | 0.0019 |

Only delays 0 and 1 were run. On NordicDat a one-window label delay changes the best model and shifts RAPT's F1 by 0.126. UGR'16 was not run under delay. Longer delays are untested and are not extrapolated.

## 9. Component and detector audit (observed)

| Component | Evidence | Verdict |
|---|---|---|
| Tier-1 learner (extra trees base) | RAPT_FULL=0.9587 vs RAPT_T2=0.9381 | LOAD-BEARING: removing it costs F1 |
| policy repository (reuse) | UGR'16 reuse=133 events, RAPT=0.8360 vs Frozen=0.9690 | LOAD-BEARING and the source of the failure |
| novelty refit | UGR'16 refit buffer 500->1000: +0.0917 macro-F1 (Step 2) | LOAD-BEARING |
| parity refit | UGR'16 parity_refits=0 at every threshold; parity on/off delta=0.0000 | INERT as configured |
| fingerprint gate | A5_gate campus: orig=0.9892 fix=0.9948 | ACTIVE but only matters when reuse fires; never fires on Campus/UGR'16 |
| periodic refresh | Final_Exp RAPT_REFRESH_W5=0.9847 vs RAPT_EVIDENCE=0.9587 | ACTIVE; refreshes drive most of the campus gain |
| cheap refit | RAPT_CHEAP=0.9851 vs Full Retraining=0.9829 | LOAD-BEARING for the cost claim on Campus only (Step 4) |

| Detector | degenerate configs | any non-degenerate | best non-degenerate F1 |
|---|---|---|---|
| ADWIN | 40/40 | False | nan |
| EDD | 0/8 | True | 0.9958055005756538 |
| EDMA | 33/72 | True | 0.9951044747463124 |
| Page-Hinkley | 144/144 | False | nan |

ADWIN and Page-Hinkley fire zero events at every configuration tested, so their reported zero-cost rows are degenerate rather than competitive. EDD and EDMA can be made non-degenerate and are the fair detector comparisons.

## 10. Cost provenance (observed)

Cost was re-measured directly (3 repeats x 5 seeds x 2 streams):

| Stream | Model | adapt CPU median | paper | delta |
|---|---|---|---|---|
| 5G Campus QoS | Event-Driven | 0.1065 | 0.1080 | -0.0015 |
| 5G Campus QoS | Frozen | 0.0845 | 0.0000 | +0.0845 |
| 5G Campus QoS | Full Retraining | 1.4951 | 1.4008 | +0.0943 |
| 5G Campus QoS | RAPT | 0.2291 | 0.2292 | -0.0001 |
| 5G Campus QoS | RAPT-Enhanced | 0.2315 | 0.2337 | -0.0022 |
| UGR'16 | Event-Driven | 0.5761 | 0.5754 | +0.0007 |
| UGR'16 | Frozen | 1.2250 | 0.0000 | +1.2250 |
| UGR'16 | Full Retraining | 25.2000 | 23.6318 | +1.5682 |
| UGR'16 | RAPT | 2.0090 | 1.9711 | +0.0379 |
| UGR'16 | RAPT-Enhanced | 2.7486 | 2.7182 | +0.0304 |

The UGR'16 Full Retraining adaptation CPU is 25.20s under the revalidation protocol against 23.63s in the paper, a pipeline difference of about 1.57s. Frozen's adaptation-CPU column includes its one-time initial fit (0.084s on Campus, 1.225s on UGR'16) while every other model excludes the initial fit, so the Frozen cost entry is not comparable (`T8_cost_definition_audit.csv`).

## 11. Interpretation

- Reuse is not uniformly safe. The policy repository helps where regimes recur with a stable label relationship and hurts where they do not.
- The published UGR'16 recovery is real as a number but is mis-attributed: it is the novelty-refit buffer size, not the parity refit, that moves macro-F1. The parity component is inert on UGR'16.
- The cost advantage is Campus-specific in the measured data.
- The detector comparison is unfair as published because two of the four detectors are degenerate at their default settings.

## 12. Limitations

- Seed-level significance is capped at p=0.0625 with five seeds.
- UGR'16 has only two paired regime visits for the label-semantics test.
- Label delay was tested only at 0 and 1 windows, and not on UGR'16.
- 5G NR is evaluated at window size 1, so its per-window macro-F1 is an accuracy-like quantity and is not directly comparable to the other streams.
- Table II mixes two metric definitions across streams.
- No synthetic covariate or concept drift construction was run; the drift behaviour reported here is the natural drift present in the four streams.

## 13. Reproducibility

- Config: `results/final/config_frozen.json` (sha256 above).
- Scripts: `results/final/step0_resolve.py` .. `step9_report.py`.
- Raw per-window and per-run results: `results/final/raw/`.
- Tables: `results/final/T1_*.csv` .. `T9_*.csv`.
- Verifier: `results/final/T9_verifier.csv`.

## 14. Scientific checks

| Check | Passed | Evidence |
|---|---|---|
| config frozen before runs | True | config_frozen.json created date |
| seeds are 42-46 | True | [42, 43, 44, 45, 46] |
| protocol is prequential test-then-train | True | chronological, prequential test-then-train, no shuffling |
| no shuffling | True | chronological, prequential test-then-train, no shuffling |
| window size consistent within stream | True | {'5g_campus': 10, 'ugr16': 240, 'nordicdat': 500, '5g_nr': 1} |
| all five seeds per (dataset, method) | True | min seeds = 5 |
| cost measured with repeats (>=3 per seed) | True | T8_cost_per_run.csv repeats per (dataset, method, seed) |
| primary table present and complete | True | T1 rows = 36 |
| 9A statistics reproduced exactly | True | 18/18 mean diffs match |
| metric reconciliation recorded | True | T1_metric_reconciliation.csv (paper mixes per-window and pooled) |
| no synthetic drift transformation introduced | True | no drift-construction artifacts; the audit only re-analyses existing streams |
| primary table has no fabricated NaNs | True | T1 macro_f1/accuracy non-null |
