# Results Claims Check

Companion to `final_results_section.md`. Every major claim made in the Results
section is listed with its supporting experiment, result file, figure/table,
statistical status and required caveat.

**Legend for "Statistically supported":**
- **Yes** — a paired test reports $p < 0.05$ in the stated direction.
- **No** — a paired test did not establish a difference at $\alpha = 0.05$.
- **N/A** — descriptive value; no test applies.

**Authoritative source policy.** Where the same stream appears in more than one
result file, the authoritative source is the file produced by the final run of
that experiment. Two such cases exist and are documented in the
"Numerical consistency" section at the end.

---

## A. Overall predictive performance

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| A1 | Full Retraining recorded the highest Macro-F1 on 5G Campus QoS (0.9836 ± 0.0013) | 9A three-dataset | `experiments/exp9a/tables/table9a_main.csv` | Table `tab:results_main_performance` | N/A (descriptive) | Multi-seed mean; per-window RAPT-vs-FR test is significant and negative |
| A2 | RAPT recorded 0.9381 ± 0.0016 Macro-F1 on 5G Campus QoS, −0.0455 vs Full Retraining | 9A three-dataset | `experiments/exp9a/tables/table9a_main.csv`; `statistics_9a.csv` | Table `tab:results_main_performance`, `tab:results_stats` | Yes (negative) | Per-window pairing; do not generalise beyond this stream |
| A3 | RAPT recorded 0.8360 ± 0.0262 on UGR'16, −0.1236 vs Full Retraining | 9A three-dataset | `table9a_main.csv`; `statistics_9a.csv` | Table `tab:results_main_performance`, `tab:results_stats` | Yes (negative) | Largest RAPT deficit; coincides with strongest recurrence |
| A4 | RAPT-Enhanced recovered +0.0917 Macro-F1 over RAPT on UGR'16 (0.9276 vs 0.8360) | 9A three-dataset | `table9a_main.csv`; `statistics_9a.csv` | Table `tab:results_main_performance`, `tab:results_stats` | Yes (positive) | Enhancement effect is large on UGR'16, near zero on 5G Campus QoS |
| A5 | RAPT recorded the highest accuracy on NordicDat (0.6172) while Full Retraining recorded the highest Macro-F1 (0.4227) | 9A three-dataset | `table9a_main.csv` | Table `tab:results_main_performance` | N/A (descriptive) | Absolute performance is low; ordering inversion not explained by present data |
| A6 | RAPT does not improve Macro-F1 over Full Retraining on any of the four streams | 9A + 9B natural | `table9a_main.csv`; `results/experiment_9b/natural_drift/summary.csv` | Table `tab:results_main_performance` | Partially (3 of 4 significant, all negative) | NordicDat difference is not significant |
| A7 | On 9B the five models span only 0.0133 Macro-F1 | 9B natural | `results/experiment_9b/natural_drift/summary.csv` | Table `tab:results_main_performance` | N/A (descriptive) | Single stream |

## B. Adaptation cost

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| B1 | RAPT reduced adaptation CPU vs Full Retraining on all four streams (−83.6%, −91.7%, −88.2%, −48.2%) | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost`; Fig. `three_dataset_adaptation_cpu` | N/A (descriptive cost) | Adaptation CPU is single-run process time; not tested for significance |
| B2 | The runtime reduction is smaller than the adaptation-CPU reduction on every stream | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost`; Fig. `three_dataset_runtime` | N/A (descriptive) | Runtime includes fixed prediction/feature cost |
| B3 | Full Retraining retrains on every regime transition (14, 144, 18, 9) | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost` | N/A (descriptive) | Deterministic given the protocol |
| B4 | RAPT reuses 6–133 times against 2–11 retrains | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost`; Fig. `three_dataset_reuse`, `three_dataset_retraining` | N/A (descriptive) | Reuse count is not by itself evidence of benefit |

## C. Reuse behaviour

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| C1 | RAPT reuses far more often than it retrains on every stream | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost` | N/A | Descriptive |
| C2 | On UGR'16 RAPT reuses 133 times yet records the lowest Macro-F1; RAPT-Enhanced reuses the same number and records +0.0917 | 9A | `table9a_main.csv` | Table `tab:results_main_performance`, `tab:results_cost` | Yes (for the RAPT vs RAPT-Enhanced difference) | Difference is in the reuse decision, not reuse frequency |
| C3 | On 5G Campus QoS RAPT-Enhanced is numerically identical to RAPT (0.9381) | 9A | `table9a_main.csv` | Table `tab:results_main_performance` | N/A (identical values) | Enhancement inactive on this stream |

## D. Efficiency ablation

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| D1 | Refit cost is dominated by estimator count (20 trees ≈ 22 ms, 100 trees ≈ 104 ms) | Final_Experiments cost probe | `Final_Experiments/FINAL_EXPERIMENTS_REPORT.md` §4 | Text | N/A (measurement) | Configuration-specific, not a general law |
| D2 | Buffer size is a secondary factor (200 ≈ 80 ms, 1000 ≈ 104 ms) | Final_Experiments cost probe | `FINAL_EXPERIMENTS_REPORT.md` §4 | Text | N/A (measurement) | Configuration-specific |
| D3 | RAPT-Evidence fired zero refreshes and reproduced RAPT-Full (0.9587) | Final_Experiments | `Final_Experiments/results/tables/table_final_main.csv` | Table `tab:results_rapt_ablation` | N/A (identical values) | Cause: relative baseline converges toward the poor level |
| D4 | RAPT-Floor fired 7.6 refreshes, 0.9797 ± 0.0050, 0.5865 s | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (vs FR: p = 0.375) | Floor value 0.97 requires per-stream calibration |
| D5 | RAPT-Cheap 0.9851 ± 0.0029 at 0.8477 s; retained accuracy vs RAPT-Refresh-W5 (0.9847 at 2.7168 s) while cutting adapt CPU ≈ 69% | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (vs FR: p = 0.3125) | 20-tree refit may underfit on harder streams |
| D6 | RAPT-Incremental 0.9549 ± 0.0020 at 0.5813 s, dominated by RAPT-Full | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (vs FR: p = 0.0625) | Stream-specific negative result |

## E. RAPT-Cheap cost-efficiency

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| E1 | Δ Macro-F1 (Cheap − FR) = +0.0022 | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (p = 0.3125, d = 0.47, CI [−0.0036, +0.0080]) | CI contains zero; do not claim superiority |
| E2 | RAPT-Cheap reduced adaptation CPU ≈ 40% vs Full Retraining (0.8477 vs 1.4063 s) | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | N/A (cost) | Single configuration on one stream |
| E3 | RAPT-Cheap matched Full Retraining predictive performance | Final_Experiments | `table_final_main.csv`; `table_final_stats.csv` | Table `tab:results_rapt_ablation`, `tab:results_stats` | No — "matched", i.e. no significant difference established | Phrase as "matched", never "outperformed" |
| E4 | RAPT-Floor gives a larger CPU cut (−58.6%) at 0.9797, not significantly different from FR | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (p = 0.375) | Higher seed variance (±0.0050) |

## F. Historical drift detectors

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| F1 | ADWIN, Page-Hinkley and EDMA recorded identical results on all three 9A streams | 9A detectors | `experiments/exp9a/tables/table9a_drift_detectors.csv` | Table `tab:results_detectors`; Fig. `detectors_f1` | N/A (identical values) | Interpreted as non-triggering, not as equivalent algorithms |
| F2 | Those three detectors recorded zero adaptation events, i.e. reproduced Frozen | 9A detectors | `table9a_drift_detectors.csv`; `table9a_main.csv` | Table `tab:results_detectors` | N/A (descriptive) | Verified against Frozen rows of `table9a_main.csv` |
| F3 | EDD recorded the highest Macro-F1 in this comparison on 5G Campus QoS (0.9847) and NordicDat (0.4184) | 9A detectors | `table9a_drift_detectors.csv` | Table `tab:results_detectors`; Fig. `detectors_f1` | N/A (descriptive) | EDD adaptation cost is the highest (3.98–6.10 s) |
| F4 | RAPT adaptation CPU was lower than EDD on UGR'16 and NordicDat (1.97 vs 6.10 s; 0.26 vs 4.69 s) | 9A detectors | `table9a_drift_detectors.csv` | Table `tab:results_detectors`; Fig. `detectors_cpu` | N/A (descriptive cost) | RAPT Macro-F1 was also lower on both |
| F5 | The comparison is a trade-off, not a uniform ordering | 9A detectors | `table9a_drift_detectors.csv` | Table `tab:results_detectors` | N/A (interpretation) | No claim that RAPT universally outperforms detectors |

## G. Cross-dataset

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| G1 | Adaptation-CPU reduction repeats on UGR'16 and 9B | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost`; Fig. `cross_dataset_adapt_cpu` | N/A (descriptive) | Direction consistent on all four streams |
| G2 | Reuse exceeds retraining on both streams | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_cost`; Fig. `cross_dataset_reuse` | N/A (descriptive) | — |
| G3 | RAPT-Enhanced effect is large on UGR'16 (+0.0917) and small on 9B (+0.0021) | 9A + 9B | `table9a_main.csv`; `natural_drift/summary.csv` | Table `tab:results_main_performance`; Fig. `cross_dataset_f1` | Yes on UGR'16; not tested on 9B | Associated with strength of recurrence |
| G4 | The accuracy/Macro-F1 ordering inverts on NordicDat | 9A | `table9a_main.csv` | Table `tab:results_main_performance` | N/A (descriptive) | Not generalised; cause not isolated |

## H. Statistical results

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| H1 | RAPT vs Full Retraining is significant and negative on 5G Campus QoS and UGR'16, not significant on NordicDat | 9A | `experiments/exp9a/tables/statistics_9a.csv` | Table `tab:results_stats` | Yes/Yes/No | Per-window pairing |
| H2 | RAPT-Enhanced vs RAPT is significant and positive on UGR'16 (+0.0917) | 9A | `statistics_9a.csv` | Table `tab:results_stats` | Yes | Per-window pairing |
| H3 | 9B RAPT vs Full Retraining accuracy difference is significant and negative (−0.014) | 9B natural | `results/experiment_9b/natural_drift/statistical_tests.csv` | Table `tab:results_stats` | Yes (accuracy, not Macro-F1) | Unit is accuracy |
| H4 | Five-seed tests cannot attain p < 0.0625 | Methodology | `Final_Experiments/results/tables/table_final_stats.csv` | Table `tab:results_stats` | N/A (method note) | Limits strength of ablation claims |
| H5 | Non-significant results are reported as "no difference established", not as equivalence | Methodology | — | Text | N/A | Applied consistently in the section |

## I. Failure cases / negative findings

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| I1 | A relative trigger can miss persistent deficiency because its reference converges toward the poor level | Final_Experiments | `table_final_main.csv`; `FINAL_EXPERIMENTS_REPORT.md` §2 | Table `tab:results_rapt_ablation` | N/A (mechanism explanation, evidenced by 0 refreshes) | Explanation supported by the measured zero-refresh count |
| I2 | The absolute floor detected the condition (7.6 refreshes) | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (p = 0.375) | Threshold 0.97 not universal |
| I3 | Standalone incremental learner reached ≈0.945 vs ≈0.98 batch | Final_Experiments probe | `FINAL_EXPERIMENTS_REPORT.md` §5 (obs. 8) | Text | N/A (measurement) | 1 tree ≈0.946, 3 ≈0.945, 10 ≈0.941 |
| I4 | Blended incremental learner at weight 0.5 dropped to ≈0.963; at 0.3 ≈0.983 | Final_Experiments probe | `FINAL_EXPERIMENTS_REPORT.md` §5 (obs. 8) | Text | N/A (measurement) | Inert at the safe weight |
| I5 | RAPT-Incremental dominated on both axes (0.9549 at 0.5813 s) | Final_Experiments | `table_final_main.csv` | Table `tab:results_rapt_ablation` | No (p = 0.0625) | Stream-specific; not a general claim about incremental learning |
| I6 | 20-tree cheap refit may underfit on harder streams | Final_Experiments | `FINAL_EXPERIMENTS_REPORT.md` §7 | Text | N/A (limitation) | Not tested on harder streams |
| I7 | In recurring concept drift, RAPT and Frozen recorded identical Macro-F1 in every phase and severity | 9B recurring | `results/experiment_9b/recurring_concept_drift/per_phase_f1.csv` | Text | N/A (identical values) | RAPT reuse gave no advantage over frozen in the recurring scenario |
| I8 | No model recovered to 95% of pre-drift Macro-F1 within the window under concept drift | 9B concept | `results/experiment_9b/concept_drift/recovery_aggregated.csv` | Fig. `fig9b_concept_severity_recovery` | N/A (descriptive) | Recovery windows pinned at 40 (window maximum) |
| I9 | Five seeds provide limited statistical power | Methodology | — | Text | N/A (limitation) | Stated explicitly |

## J. Drift-severity results (9B)

| # | Claim | Experiment | Result file | Figure/Table | Statistically supported | Caveat |
|---|---|---|---|---|---|---|
| J1 | Under covariate drift RAPT exceeds Full Retraining by +0.0004 to +0.0096 Macro-F1 across severities | 9B covariate | `results/experiment_9b/covariate_drift/aggregated.csv` | Fig. `fig9b_covariate_severity_f1` | Not tested | Small, non-significant-looking margins; no per-severity test reported |
| J2 | RAPT-Enhanced reproduced RAPT exactly at every covariate severity | 9B covariate | `covariate_drift/aggregated.csv` | Text | N/A (identical values) | Consistent with enhancement inactive under pure covariate drift |
| J3 | Under concept drift, Frozen/Full Retraining/RAPT are identical at every severity because none adapted | 9B concept | `results/experiment_9b/concept_drift/aggregated.csv` | Fig. `fig9b_concept_severity_f1` | N/A (identical values) | Zero retrains/reuse recorded for all three |
| J4 | Event-Driven exceeded the non-adapting models by +0.0464 to +0.0696 at severities 0.10–0.30 | 9B concept | `concept_drift/aggregated.csv` | Fig. `fig9b_concept_severity_f1` | Not tested | Event-Driven is the only adapter under concept drift |
| J5 | Concept-drift construction changed P(Y|X) and not only P(X) | 9B concept diagnostics | `concept_drift/px_invariance.csv`; `feature_target_association.csv` | Text | N/A (diagnostic) | MI of p90_latency falls 0.2428 → 0.0238 while P(X) is preserved (KS = 0) |

---

## Numerical consistency check

The following cross-checks were performed between the Results section, the
generated tables and the raw result files.

**Consistent.** Tables 1 and 2 were generated by direct copy from
`experiments/exp9a/tables/table9a_main.csv` and
`results/experiment_9b/natural_drift/summary.csv`. Table 3 and the RAPT-Cheap /
RAPT-Floor / RAPT-Incremental figures were generated from
`Final_Experiments/results/tables/table_final_main.csv` and
`table_final_stats.csv`. Table 4 was copied from
`table9a_drift_detectors.csv`. Table 5 was copied from `statistics_9a.csv`,
`natural_drift/statistical_tests.csv` and `table_final_stats.csv`. The
RAPT-Cheap values quoted in the task specification
(0.9851 ± 0.0029, 0.8477 s; Full Retraining 0.9829 ± 0.0023, 1.4063 s) agree
exactly with `table_final_main.csv`.

**Discrepancy 1 — same 5G Campus QoS stream, two result sets.**
`Final_Experiments` reuses the existing exp9a preprocessing and evaluates the
*same* 5G Campus QoS stream, but reports different absolute values from the 9A
table. The stream definition is identical (`n_raw_samples` 1799, 179 total
windows, 19 features, 3 classes, initial train 36 windows). The difference is the
RAPT implementation: the 9A table reports the Tier-2 `rapt_9a.py` implementation
(0.9381 Macro-F1, 0.2292 s adapt CPU), while `Final_Experiments` reports the
ladder ablation (`RAPT_T2` = 0.9381 at 0.2292 s, confirming the match; `RAPT_FULL`
= 0.9587, which adds the similarity gate and refit).
- **Authoritative for Table 1 / Table 2 (main performance and cost):**
  `experiments/exp9a/tables/table9a_main.csv` — this is the 9A experiment's own
  output and defines "RAPT" in the main model set.
- **Authoritative for Table 3 / Section V-D / Section V-E (ablation):**
  `Final_Experiments/results/tables/table_final_main.csv`.
- The two are consistent: `RAPT_T2` in the ablation table equals `RAPT` in the
  9A table to four decimal places (0.9381, 0.2292 s). The ablation therefore does
  not contradict 9A; it extends it.

**Discrepancy 2 — 9B appears at two window counts.**
`results/experiment_9b/natural_drift/summary.csv` and
`experiments/exp9b/results/summary.csv` describe the same 9B stream but were
produced by different run scripts. The former (499 windows, 99 initial, 400
streaming) is the value used here and matches
`experiments/exp9b/results/stream_definition.json`. The
`experiments/exp9b/results/summary.csv` file reports the same Macro-F1 means but
different total-CPU columns and omits the tree counts.
- **Authoritative:** `results/experiment_9b/natural_drift/summary.csv`.
- **Reason:** it is the copy referenced by `EXPERIMENT_9B_FINAL_REPORT.md`, it
  carries the full metric set required by the manuscript, and its stream
  definition matches the published `stream_definition.json`.

**No fabricated values.** Every number in the Results section traces to one of
the files listed above. No value was estimated, interpolated or carried over
without a source file.

---

## Summary

1. **Primary-model findings.** Full Retraining records the highest Macro-F1 on
   three of four streams; base RAPT does not improve Macro-F1 over Full
   Retraining on any stream but reduces adaptation CPU on all four
   (−48% to −92%). RAPT-Enhanced closes most of the UGR'16 gap
   (0.8360 → 0.9276, significant).
2. **RAPT-Cheap.** 0.9851 ± 0.0029 Macro-F1 at 0.8477 s adaptation CPU versus
   Full Retraining 0.9829 ± 0.0023 at 1.4063 s. Matched predictive performance at
   approximately 40% lower adaptation CPU; ΔF1 = +0.0022 is **not** significant
   (p = 0.31).
3. **RAPT-Incremental negative finding.** 0.9549 ± 0.0020 at 0.5813 s —
   dominated by RAPT-Full (0.9587 at 0.4283 s). A standalone incremental learner
   reached only ≈0.945 versus ≈0.98 batch, and blending it in either hurt or was
   inert. Stream-specific, not a general claim.
4. **Historical detectors.** ADWIN, Page-Hinkley and EDMA recorded zero
   adaptation events and reproduced Frozen on all three 9A streams; EDD recorded
   the highest Macro-F1 of the comparison on two streams at the highest
   adaptation cost. The measured relationship is a trade-off, not a uniform
   ordering.
5. **Strongest statistically supported conclusions.** (i) RAPT-Enhanced vs RAPT
   on UGR'16, +0.0917, p = 5.4 × 10⁻⁹; (ii) RAPT vs Full Retraining on UGR'16,
   −0.1236, p = 2.6 × 10⁻¹¹; (iii) RAPT vs Full Retraining on 5G Campus QoS,
   −0.0455, p = 2.6 × 10⁻⁴. All are per-window comparisons on single streams.
   No five-seed ablation comparison reached significance.
6. **Important limitations.** Five seeds cap two-sided Wilcoxon p at 0.0625;
   per-window tests overstate effective sample size for cross-stream claims;
   the 0.97 floor and the 20-tree cheap refit are configuration-specific; the
   recurring-concept-drift experiment shows RAPT reuse gave no advantage over the
   frozen baseline; NordicDat's low absolute performance limits interpretation of
   its ordering.
7. **Files created.**
   - `Results_Final/reports/final_results_section.md`
   - `Results_Final/reports/results_claims_check.md`
   - `Results_Final/tables/final_results_tables.tex`
   - `Results_Final/figures/final_paper_figures/` (40 figures copied from the
     existing generated output; no new figures generated)
