# CHANGES — fast-final2 job

Scope: **experiments only**. No paper file (`paper/main.tex`, `paper/tables/*`)
was edited. All new numbers live under `results/final2/` and are reviewed in one
place in `results/final2/RESULTS_REVIEW.md`. This documents what changed, what
got worse, and which claims the new numbers do and do not support.

## New dataset (Task C)

NordicDat (hard, ~0.28 macro-F1, suspected label issue) is **replaced by
"Network operator KPIs time series"** (Zenodo 8147768, CC BY-NC 4.0). 14 real
aligned KPI series, 5-minute cadence, 37 days, 223 windows, 6 recurring
time-of-day regimes, producer-annotated binary anomaly target (positive rate
0.239). Details in `results/final2/new_dataset.md`. Loader: `final2_load.py`.

## New method: RAPT-ED-Guard (Task A)

`RAPTEDGuard` in `final2_methods.py`. The Event-Driven error trigger OR an
every-window check arms a cheap validation gate: score the stored regime policy
and the prefix policy on the last 50 labelled rows; reuse only if the stored
policy beats the prefix policy by `edguard_margin`. Margin chosen on dev
(5 seeds, first 60%) — 0.05 wins overall (0.8775 vs 0.8197 at margin 0.0),
frozen in `configs/final2.yaml`. No prefix-policy pretraining was added to keep
the run inside the time cap; this is a documented limitation, not a hidden one.

**Result (30 seeds, full stream, macro-F1 / total CPU s):**

| stream | Frozen | Event-Driven | RAPT | RAPT-Guard | **RAPT-ED-Guard** |
|---|---|---|---|---|---|
| Campus | 0.937 | 0.964 | 0.939 | 0.939 | **0.984** |
| 5G NR  | 0.911 | 0.908 | 0.897 | 0.915 | **0.905** |
| KPI-ops| 0.680 | 0.715 | 0.680 | 0.680 | **0.782** |
| UGR'16 | 0.969 | 0.898 | 0.914 | 0.936 | **0.952** |

RAPT-ED-Guard beats Frozen and Event-Driven on Campus and KPI-ops (Holm
$p<0.001$), and beats Event-Driven on UGR'16 (Holm $p<0.001$). It is
**significantly below Frozen on UGR'16** ($-0.017$, Holm $p<0.001$) and below
Event-Driven on 5G NR ($-0.003$). Its adaptation CPU is high because its gate
refits often (it matches the adapters' accuracy, not RAPT's cost).

## Reuse is sometimes harmful (honest negative)

On KPI-ops, base RAPT reuses a stale regime policy **173 times** and collapses
exactly to Frozen (0.680; window diff 0.0). On UGR'16 RAPT (0.914) and its
non-gated variants are **below even Frozen** (0.969). This is the paper's
central "identity-only reuse can be unsafe" point, now with KPI-ops as a second
instance. RAPT-Anchor's lookback oracle barely helps on either (0.680 / 0.925),
i.e. recent labelled rows are a weak signal when the regime's class prior
shifted.

## Gate accuracy (Task A reporting)

From `results/final2/gate_accuracy.csv` (30 seeds):

| stream | returns | allows | blocks | bad allows | bad-allow rate | missed reuse |
|---|---|---|---|---|---|---|
| Campus | 12.8 | 0.0 | 12.8 | 0.0 | — (all blocked) | 0.0 |
| 5G NR | 14.7 | 3.5 | 11.2 | 1.4 | 0.400 | 0.0 |
| KPI-ops | 173.0 | 40.4 | 132.6 | 0.6 | 0.016 | 1.1 |
| UGR'16 | 133.0 | 1.4 | 131.6 | 0.0 | 0.000 | 1.3 |

Overall bad-allow rate 4.5%; 2 of 288 blocks were missed reuses. The gate is
**very conservative** — it blocks almost everything on UGR'16 (1.4 allows of
133 returns), which is why RAPT-ED-Guard on UGR'16 lands near Frozen rather
than exploiting reuse.

## Long vs short window gate (Task D2)

`results/final2/gate_cadence.csv` + `figures/fig_gate_cadence.png`. Measured,
not asserted. On the constructed within-regime concept-drift stream the
boundary/short gate fires **0** refits and the every-window/long gate fires 8;
on the natural streams the long gate refits far more (UGR'16 421 vs 55,
KPI-ops 337 vs 25). Accuracy: Campus 0.973 vs 0.928, KPI-ops 0.707 vs 0.509,
UGR'16 0.936 vs 0.923, but 5G NR 0.895 vs 0.909 (the long gate is *worse*
there). So the feasibility result is **stream-dependent**, not uniformly
positive.

## Detector fixes (Task D3)

ADWIN and Page-Hinkley default thresholds never fired (== Frozen). Swept on dev
(`results/final2/dev_detector_sweep.csv`): froze `adwin_delta=0.2`,
`ph_threshold=0.002`. **ADWIN was added to the final method set.** Page-Hinkley
now fires 6–20 times/seed; ADWIN fires on UGR'16/KPI-ops (~9–11) but still near
zero on Campus/5G NR. Neither beats the best adapter anywhere
(ADWIN 0.89–0.95; Page-Hinkley 0.89–0.98). The old defaults are retained only in
the sweep table, not used.

## Concept-drift stress panel (Task D4)

Enabled the in-regime floor trigger for all methods. The four trajectories
**still coincide**: mean max−min spread across methods is 0.008 / 0.020 / 0.000
at severities 0.2 / 0.5 / 1.0 (`results/final2/concept_spread.csv`). Conclusion
to report: the panel is **degenerate even after the fix**, because the stream's
per-window macro-F1 is effectively determined by the class prior, not by which
method adapts. Recommendation: drop the panel from the paper and say why.

## Total CPU (Task E)

`total_cpu_sec = adaptation + prediction` is recorded for every method. The
doubled inference of RAPT-Anchor is visible (e.g. UGR'16 prediction 3.66 s vs
~1.27 s for single-inference methods). Frozen+Periodic has the *lowest* total
CPU while scoring well on several streams, so "low cost" alone does not
separate reuse from a refit schedule.

## Reuse-vs-trigger baselines (Task G)

`Event-Driven+Reuse` and `Frozen+Periodic` are in the final set. On KPI-ops
Frozen+Periodic (0.893) nearly matches Full Retraining (0.906) at ~1/10 the
adaptation CPU, and on UGR'16 it is the *worst* method (0.837). Event-Driven+Reuse
beats Event-Driven on UGR'16 (0.964 vs 0.898) but not on KPI-ops. The answer to
"is reuse or the trigger doing the work" is **both, and stream-dependent**.

## Class-weight / leakage diagnosis (Task F)

`results/final2/nordicdat_diagnosis.csv`. Class-weighting does not create the
KPI-ops signal: Frozen is 0.6797 both with and without class weights. The
strongest KPI-ops feature-target correlation is 0.837 (`kpi_r12`); top_feature
importance is not concentrated on `kpi_r9-r11` (the components of the target),
so no direct label leak. KPI-ops prefix is single-class (all normal), so Frozen
starts with a weak policy — the same hard, late-first-anomaly difficulty that
sank NordicDat. UGR'16 is strongly imbalanced (3.2% positives full, 8.1%
prefix).

## What got worse / claims to soften

1. **RAPT's efficiency headline shrinks on total CPU.** RAPT adaptation CPU is
   64–91% below full retraining, but its *total* CPU is only 1–7% below
   (prediction dominates). Report both; do not claim a large total-cost win.
2. **RAPT-ED-Guard is not a free lunch.** It recovers accuracy by refitting
   often, so it is an accuracy method, not a cheap one. On 5G NR it is below
   Event-Driven; on UGR'16 below Frozen.
3. **"Reuse is safe with a gate" is false on UGR'16** for both RAPT-ED-Guard
   ($-0.017$ vs Frozen) and RAPT-Guard ($-0.034$ vs Frozen).
4. **Frozen+Periodic** shows that much of RAPT's accuracy on KPI-ops is not
   attributable to reuse at all.
5. **The concept panel adds no evidence** — drop it.
6. RAPT-Anchor's small gain over RAPT (UGR'16 +0.011, 5G NR +0.007) plus its
   doubled inference makes the oracle look weak, not strong.

## Claims the new numbers do support

* Contrast UGR'16 (identity-only reuse is **actively harmful**, RAPT below
  Frozen) with Campus/KPI-ops (harmful in a different way — RAPT collapses to
  Frozen without a working gate).
* A cheap validation gate recovers **most** of the loss but not all of it,
  and is stream-dependent.
* Total CPU, not adaptation CPU, is the deployment-relevant cost, and on that
  metric reuse's advantage is small.

## Reproduction

See `experiments/final2/README.md` for the exact run order. Dev tuning uses 5
seeds (42–46), first 60% of post-prefix windows; the final run uses 30 seeds
(42–71) on the full streams once with `configs/final2.yaml`. Total compute for
this job ≈ 2.4 h wall (final run 54 min on 4 workers).
