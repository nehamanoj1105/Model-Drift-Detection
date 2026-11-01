# RAPT Revalidation — Consolidated Final Results

Scope: re-validation of the RAPT experiments and audit of the claims in the
COMSNETS submission *When Is Reuse Safe? Regime-Aware Policy Transfer for
Recurring Drift in Network Telemetry*. This file consolidates the v2 re-runs
(Steps A–D). It does not introduce a new dataset, a new RAPT variant or a new
synthetic drift construction. Every number below is drawn from a committed
artifact; where a number could not be reproduced the discrepancy is reported,
not smoothed.

Primary metric for reproducing the paper: **per-window mean macro-F1** (the
scale the paper uses for three of its four streams). Secondary metric:
**pooled macro-F1** (predictions concatenated over windows). The two are
different quantities and are reported side by side (Sections 2–4); the paper
mixes them in one table, which is itself a finding.

Seeds: `[42, 43, 44, 45, 46]`. Protocol: chronological prequential
test-then-train, no shuffling, initial prefix untouched, no future leakage.

---

## 1. Provenance

| Artifact | Contents |
|---|---|
| `results/final/v2/config_frozen_v2.json` | frozen v2 configuration |
| `results/final/v2/B_pooled_all_streams.csv` | pooled metrics, 4 streams × 9 models × 5 seeds |
| `results/final/v2/B_perwindow_all_streams.csv` | per-window metrics, same grid |
| `results/final/v2/raw/B_<stream>_seed<seed>.csv` | raw per-window rows (one file per stream × seed) |
| `results/final/v2/A10_detector_corrected.csv` | detector events, fixed vs original wiring |
| `results/final/v2/C_metric_reconciliation.csv` | paper value vs per-window vs pooled |
| `results/final/v2/C_primary_perwindow.tex` | corrected primary table (per-window) |
| `results/final/v2/C_primary_pooled.tex` | corrected primary table (pooled) |
| `results/final/figures/` | regenerated figures (PNG 300 dpi + PDF) |
| `results/final/latex_tables/` | corrected LaTeX tables |

Streams: 5G Campus QoS, UGR'16, NordicDat (9A loaders) and 5G NR (the existing
9B processed window stream). Window sizes: 5G Campus 10, UGR'16 240, NordicDat
500, 5G NR 1 (one sample per window, so its per-window macro-F1 is
accuracy-like). 5G NR has 499 windows; the 9A streams have 179–182.

---

## 2. Step B — pooled and per-window results (all four streams)

Per-window mean macro-F1, seeds 42–46 (`B_perwindow_all_streams.csv`):

| Stream | Frozen | Event-Driven | Full Retraining | RAPT | RAPT-Enhanced |
|---|---|---|---|---|---|
| 5G Campus | 0.9364 | 0.9636 | **0.9836** | 0.9381 | 0.9381 |
| UGR'16 | **0.9690** | 0.8972 | 0.9595 | 0.8360 | 0.9276 |
| NordicDat | 0.2779 | 0.2547 | **0.4227** | 0.3818 | 0.3783 |
| 5G NR | 0.9075 | 0.9060 | **0.9110** | 0.9040 | 0.9085 |

Pooled macro-F1, seeds 42–46 (`B_pooled_all_streams.csv`):

| Stream | Frozen | Event-Driven | Full Retraining | RAPT | RAPT-Enhanced |
|---|---|---|---|---|---|
| 5G Campus | 0.9822 | 0.9908 | **0.9951** | 0.9838 | 0.9838 |
| UGR'16 | 0.8216 | **0.8413** | 0.7603 | 0.6629 | 0.8008 |
| NordicDat | 0.3250 | 0.4399 | **0.4806** | 0.4618 | 0.4662 |
| 5G NR | 0.8961 | 0.8923 | **0.9018** | 0.8930 | 0.8965 |

UGR'16 pooled cross-check against the previously computed reference values
(`B_reference_crosscheck.csv`): all six reference values reproduce to
< 5×10⁻⁵ (Frozen 0.821638, Event-Driven 0.841253, Full Retraining 0.760327,
EDD 0.825633, RAPT 0.662866, RAPT-Enhanced 0.800794).

---

## 3. Claim-by-claim audit (under both metrics)

`B_statement_checks.csv` (restricted to the paper's five models; ties count):

| Statement | Pooled | Per-window |
|---|---|---|
| Frozen is best on UGR'16 | NOT SUPPORTED (Event-Driven 0.8413 > Frozen 0.8216) | SUPPORTED (0.9690) |
| RAPT is worst on UGR'16 | SUPPORTED (0.6629) | SUPPORTED (0.8360) |
| Full Retraining is best on three of four streams | SUPPORTED | SUPPORTED |
| RAPT is below Full Retraining on all four streams | SUPPORTED (4/4) | SUPPORTED (4/4) |
| RAPT has the highest accuracy on NordicDat | SUPPORTED (0.6172) | SUPPORTED (0.6172) |

The only metric-sensitive claim is "Frozen is best on UGR'16". Under the paper's
own (per-window) scale it holds; under the pooled scale Event-Driven is higher,
because Frozen's few catastrophic windows are diluted by pooling. This is a
scale artefact, not a new result.

---

## 4. Metric reconciliation (Step C)

`C_metric_reconciliation.csv` compares each paper Table II value against the
per-window and pooled re-runs:

| Stream | Paper used | Matches per-window | Matches pooled |
|---|---|---|---|
| 5G Campus | per-window | True (all 5 models) | False |
| UGR'16 | per-window | True (all 5 models) | False |
| NordicDat | per-window | True (all 5 models) | False |
| 5G NR | **pooled** | False | True (Frozen); ≤0.0015 otherwise |

Finding: the published Table II is on **two different scales**. Three streams use
the per-window mean; 5G NR uses the pooled value. On UGR'16 the two scales differ
by up to 0.199 (RAPT 0.8360 per-window vs 0.6629 pooled). The 5G NR block is
therefore not directly comparable to the other three in that table. Corrected
tables are in `C_primary_perwindow.tex` and `C_primary_pooled.tex`; the corrected
per-window table reproduces the paper's three per-window streams exactly.

---

## 5. Detector re-check (Step A10) — a corrected finding

The manuscript states: *"ADWIN, Page-Hinkley and EDMA recorded zero adaptation
events on all three streams and so reproduce the frozen baseline by
construction."* The original pipeline does produce those zeros, but they are
artefacts:

- **ADWIN / Page-Hinkley (river 0.26.1 wiring).** `river`'s `ADWIN.update()` and
  `PageHinkley.update()` return `None`; the original code does
  `bool(self.detector.update(x))`, which is therefore always `False`. These two
  detectors can never fire in the original pipeline.
- **EDMA (comparison-order bug).** `_CustomEDMA.update()` updates the EWMA
  *before* comparing, so the new sample is absorbed into its own baseline and the
  test almost never fires (at the default α=0.2, k=2 it never fires; only
  k=1 fires).
- **EDD** fires in the original pipeline, but on the very first window
  (`s_min`/`p_min` uninitialised, `min_instances` not enforced before the
  comparison).

With the wiring fixed (read `drift_detected` after `update`; compare EDMA against
the pre-update EWMA; enforce EDD `min_instances`), and the parameters held at the
paper's a-priori values (ADWIN δ=0.002, Page-Hinkley min 30/threshold 50, EDD
warning 2/drift 3, EDMA α=0.2/k=2), the corrected detector results are
(`A10_detector_corrected.csv`):

| Stream | Detector | Events (fixed) | Events (orig wiring) | Macro-F1 (per-window) | Macro-F1 (pooled) | Adapt CPU (s) |
|---|---|---|---|---|---|---|
| 5G Campus | ADWIN | 0 | 0 | 0.9364 | 0.9822 | 0.00 |
| 5G Campus | Page-Hinkley | 0 | 0 | 0.9364 | 0.9822 | 0.00 |
| 5G Campus | EDD | 38 | 114 | 0.9847 | 0.9951 | 4.15 |
| 5G Campus | EDMA | 5.2 | 5 | 0.9835 | 0.9951 | 0.57 |
| UGR'16 | ADWIN | 0 | 0 | 0.9690 | 0.8216 | 0.00 |
| UGR'16 | Page-Hinkley | 0 | 0 | 0.9690 | 0.8216 | 0.00 |
| UGR'16 | EDD | 39 | 115 | 0.9655 | 0.8256 | 6.21 |
| UGR'16 | EDMA | 2.8 | 3 | 0.9474 | 0.8203 | 0.52 |
| NordicDat | ADWIN | 0 | 0 | 0.2779 | 0.3250 | 0.00 |
| NordicDat | Page-Hinkley | 0 | 0 | 0.2779 | 0.3250 | 0.00 |
| NordicDat | EDD | 39 | 117 | 0.4184 | 0.4694 | 4.85 |
| NordicDat | EDMA | 3.0 | 29 | 0.2701 | 0.4255 | 0.37 |
| 5G NR | ADWIN | 0 | 0 | 0.9075 | 0.8961 | 0.00 |
| 5G NR | Page-Hinkley | 0 | 0 | 0.9075 | 0.8961 | 0.00 |
| 5G NR | EDD | 115 | 344 | 0.9015 | 0.8854 | 10.52 |
| 5G NR | EDMA | 16.2 | 39 | 0.8980 | 0.8813 | 1.49 |

Corrections to the manuscript's detector narrative:

1. **EDMA is not degenerate.** It fires 2.8–16.2 times under the fixed wiring
   (vs 0 at the default in the original) and no longer coincides with Frozen on
   the 9A streams: NordicDat 0.2701 vs Frozen 0.2779, UGR'16 0.9474 vs 0.9690,
   5G Campus 0.9835 vs 0.9364.
2. **ADWIN and Page-Hinkley are still zero**, so the "never fires" statement is
   correct for those two, but for a *wiring* reason, not because the stream is
   benign. This should be stated.
3. **EDD's event counts are inflated** in the original (114 vs 38 on Campus,
   344 vs 115 on 5G NR) because of the first-window spurious fire and a
   count difference in the harness; the qualitative EDD conclusion (most fires,
   highest F1 on Campus/NordicDat, highest cost) is unchanged.

Sanity check (`A8_detector_sanity.csv`): on a synthetic step change at index 200
the fixed detectors fire sensibly — ADWIN once (latency 55–183), Page-Hinkley
0–1, EDMA 15–26, EDD 345–371 — so the fix detects real change rather than firing
indiscriminately.

---

## 6. Why reuse fails on UGR'16 (mechanism)

`T2_mechanism_isolation.csv` isolates the UGR'16 recovery (0.8360 → 0.9276):

| refit buffer | parity | macro-F1 | parity refits | retrains | reuses |
|---|---|---|---|---|---|
| 500 | off | 0.8360 | 0 | 11 | 133 |
| 500 | on | 0.8360 | 0 | 11 | 133 |
| 1000 | off | 0.9276 | 0 | 11 | 133 |
| 1000 | on | 0.9276 | 0 | 11 | 133 |

- The recovery is produced entirely by enlarging the novelty-refit buffer
  (500 → 1000 rows): Δ = +0.0917, CI [0.0661, 0.1210], p = 0.0625.
- The parity refit — the paper's proposed fix — is **inert**: 0 refits at every
  threshold 0.5–0.95, and parity on/off changes macro-F1 by exactly 0.0000.
- The reported "parity recovers UGR'16" attribution is therefore **not
  supported**; the driver is the refit training-set composition.
- Frozen does not degrade over the 30-day block (third-1 minus third-3 =
  −0.0888), so the deficit is not simple non-stationarity in time.

A second real contributor is checkpoint provenance (`T2_hypothesis_verdict.csv`):
under the original keying, every stored checkpoint is trained on a *different*
regime than its key (`A5_provenance.csv`: fraction provenance-correct = 0.0);
correcting the keying (RAPT-Deferred) lifts UGR'16 macro-F1 by Δ = +0.0798,
CI [0.0609, 0.0996].

---

## 7. 5G NR cheap-refresh anomaly

The paper's Table III reports RAPT-Cheap as matching Full Retraining at much
lower cost. On 5G NR this reverses. `A3_5g_nr_table_ii.csv`:

| Model | per-window F1 | pooled F1 | Adapt CPU (s) | Refresh events | Trees trained |
|---|---|---|---|---|---|
| Full Retraining | 0.9110 | 0.9018 | 0.72 | 0 | 900 |
| RAPT | 0.9040 | 0.8930 | 0.28 | 0 | 400 |
| RAPT-Cheap | 0.8830 | 0.8614 | 3.26 | 79 | 3560 |

Cause (A3/A4): 5G NR has window size 1, so the periodic "refresh every 5 windows"
rule fires every 5 *samples* — 79 refreshes over 400 post-prefix samples — and
each refresh retrains 20 trees on a 300-row buffer, giving 3560 trees trained and
3.26 s adaptation CPU, i.e. **+351% over Full Retraining**, with 0.0280 lower
macro-F1. On 5G NR the cheap-refresh configuration is both slower and less
accurate than Full Retraining.

---

## 8. Statistical methodology (unchanged)

- Window-level paired Wilcoxon over 143–146 windows and seed-level exact
  Wilcoxon over n = 5 seeds, Cohen's d, 95% CIs.
- The seed-level minimum attainable two-sided p at n = 5 is 0.0625; tests that
  bind at this ceiling are reported as such, not as significant.
- No significance is manufactured: results with p ≥ 0.05 are reported as
  non-significant.
- The paper's window-level tests reproduce exactly; the five-seed ablation
  comparisons (p = 0.3125, p = 0.375) remain underpowered.

---

## 9. Generated figures and tables

Figures (`results/final/figures/`, PNG 300 dpi + PDF):

| Figure | File | Content |
|---|---|---|
| primary performance | `primary_performance` | per-window macro-F1, all four streams, 5 models |
| primary (pooled) | `primary_pooled` | pooled macro-F1, same grid |
| metric mixing | `metric_mixing` | bars = per-window, stars = pooled, per stream |
| UGR'16 streaming | `ugr16_streaming` | per-window F1 with RAPT reuse/retrain markers |
| UGR'16 rolling | `ugr16_rolling` | rolling accuracy (w = 10), all models |
| cost trade-off | `cost_tradeoff` | accuracy vs adaptation CPU, per (stream, model) |
| 5G NR refresh | `5g_nr_refresh` | CPU / trees / refresh events for the anomaly |

Tables (`results/final/latex_tables/`): `C_primary_perwindow.tex`,
`C_primary_pooled.tex`.

Existing 9B drift figures (`results/experiment_9b/figures/`) — natural drift,
covariate severity, concept severity, recurring drift, heatmaps, confusion
matrices, reuse-vs-retraining — are unchanged and remain the 9B drift artifacts.

---

## 10. Scientific checks

| Check | Result |
|---|---|
| config frozen before runs | `config_frozen_v2.json` present |
| seeds 42–46 on every (stream, model) | yes |
| prequential test-then-train, no shuffling | yes |
| initial prefix untouched | yes |
| no future leakage | yes (prequential; drift transforms applied only post-point) |
| window size consistent within stream | 10 / 240 / 500 / 1 |
| pooled and per-window both saved | yes |
| raw per-window rows saved | `v2/raw/B_*_seed*.csv` |
| UGR'16 reference values reproduce | 6/6 within 5×10⁻⁵ |
| no fabricated NaNs in primary table | yes |
| failed runs silently dropped | none |

---

## 11. Limitations

- Seed-level significance is capped at p = 0.0625 with five seeds.
- Table II mixes two metric definitions across streams; corrected tables are
  provided but the paper must pick one scale.
- UGR'16 has only two paired regime visits for the label-semantics test
  (reported INCONCLUSIVE).
- The detector conclusions are sensitive to the wiring; the corrected counts are
  reported here and the manuscript text must be updated (Section 5).
- 5G NR is evaluated at window size 1, so its per-window F1 is not directly
  comparable to the other streams.
- Timings come from one environment with parallelism disabled; comparable within
  the study, not portable.
- The 9B controlled covariate/concept/recurring drift experiments use the
  existing 9B construction (`experiments/exp9b/drift_construct.py`) and are not
  re-run here; their artifacts are unchanged.

---

## 12. Bottom line for the paper

1. The headline efficiency claim does **not** transfer beyond 5G Campus; on
   UGR'16, NordicDat and 5G NR the cheap configuration is lower-F1, and on 5G NR
   it is also more expensive.
2. The UGR'16 "parity refit" attribution is **not supported**; the recovery is
   the refit-buffer composition.
3. Table II is on **two scales**; one must be chosen.
4. The detector comparison must be restated: ADWIN/Page-Hinkley never fire due
   to a wiring artefact, EDMA is not degenerate, EDD's counts are inflated.
5. What survives: RAPT is below Full Retraining on all four streams (both
   metrics); RAPT is worst on UGR'16 (both metrics); the UGR'16 RAPT-Enhanced
   recovery is real as a number; the cost advantage is Campus-specific.
