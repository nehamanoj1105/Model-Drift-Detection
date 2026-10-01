# Phase 2 — New experiments (E1–E5)

Scripts: `audit/exp_e1_rapt_equivalence.py`, `exp_e2_ugr_mechanism.py`,
`exp_e3_ugr16.py`, `exp_e3b_conditional.py`, `exp_e4_tost.py`,
`exp_e5_safe_variant.py`. Raw console output in `audit/e1_out.txt` … `e5_out.txt`.

All experiments reuse the 9A harness unchanged (same loaders, windowing,
prefix, ensemble, buffer, anchor, seeds). Nothing in `experiments/` was edited.

---

## E1 — The three RAPT copies are the same algorithm

`experiments/exp9a/rapt_9a.RAPTSystem` and `Final_Experiments/rapt_ladder`
variant `RAPT_T2` were driven over the 5G Campus stream with identical inputs.

| seed | identical predictions |
|------|-----------------------|
| 42–46 | 1430 / 1430 windows |

Verdict: **bit-identical**. The ablation ladder's Tier-2 baseline is the same
algorithm as Table II "RAPT", so the ladder is interpretable. (Closes problem b.)

## E2 — UGR'16: the RAPT → RAPT-Enhanced gain is the buffer, not parity refitting

Exact reproduction of the committed summary (mean over seeds 42–46):

| controller | reproduction | committed |
|------------|--------------|-----------|
| novelty buffer 500 (Table II "RAPT") | 0.8360 | 0.8360 |
| novelty buffer 1500 (= "RAPT-Enhanced" with parity off) | 0.9276 | 0.9276 |

Parity-refit branch, threshold sweep (seed 42 shown; identical at 0.90/0.95/0.97):

| variant | macro-F1 | parity_refits |
|---------|----------|---------------|
| base (novelty 500) | 0.7958 | 0 |
| novelty 1500, parity off | 0.9459 | 0 |
| parity 0.50 (paper setting) | 0.7958 | 0 |
| parity 0.90 / 0.95 / 0.97 | 0.7958 | 0 |

The committed `summary_ugr16_full.csv` independently reports
`parity_refits = 0` for RAPT-Enhanced on all five seeds.

**Finding.** The 0.8360 → 0.9276 improvement is produced entirely by the larger
novelty-refit buffer (1500 vs 500). The parity-refit branch never fires, so it
contributes nothing to the reported result. The RAPT vs RAPT-Enhanced contrast
is, in the reported runs, "refit buffer 500 vs 1500", not "reuse without vs with
a degradation check".

## E3 — A frozen model does not show UGR'16 recurrence as harmful

Per regime recurrence, frozen-model macro-F1 at the first occurrence vs each
recurrence (all seeds pooled):

| stream | recurrences | P(store) | P(recur) | gap | gap > 0.10 |
|--------|-------------|----------|----------|-----|------------|
| Campus | 700 | 1.0000 | 0.9351 | +0.0649 | 15.3% |
| UGR'16 | 660 | 0.8960 | 0.9741 | **−0.0782** | 6.2% |
| Nordic | 715 | 0.3995 | 0.2786 | +0.1209 | 55.1% |

On UGR'16 recurrences score *higher* for the frozen model (gap negative), and the
frozen model's mean macro-F1 (0.9667) exceeds every adapting model. The frozen
baseline therefore does not support "blind reuse of a stale policy is harmful on
UGR'16"; the harm in Table II is specific to RAPT's 500-sample refit buffer.

## E3b — Conditional vs covariate character of UGR'16 recurrence

For each recurrence paired with the regime's first occurrence: change in anomaly
prevalence `dP(Y=1)`, change in the frozen model's label association `dAUC`, and
standardised centroid movement `dCentroid`, scaled by between-regime spread.

| stream | dCentroid / between-regime | mean |dP| | mean |dAUC| |
|--------|---------------------------|-----------|------------|
| UGR'16 | 2.23 | 0.073 | 0.0098 |
| Nordic | 0.87 | 0.140 | n/a |
| Campus | 2.52 | 0.342 | n/a |

For UGR'16 the covariate component dominates (ratio 2.23, i.e. a recurrence sits
further from its own first occurrence than typical between-regime distances) while
the label association is essentially unchanged (|dAUC| ≈ 0.01). This is the
opposite of the paper's characterisation ("regime identifiers recur but the
feature-to-label mapping does not"). Campus/Nordic are inconclusive here because
their 10- and 500-sample windows are single-class in most recurrences, so `dAUC`
is undefined; that limitation is stated rather than hidden.

## E4 — Equivalence of RAPT-Cheap and Full Retraining

RAPT-Cheap − Full Retraining on Campus, seed-paired (n = 5): Δ = +0.00221,
sd = 0.00468, Wilcoxon p = 0.3125.

| margin | TOST p | verdict |
|--------|--------|---------|
| ±0.005 | 0.1268 | not shown equivalent |
| ±0.010 | 0.0102 | equivalent |
| ±0.020 | 0.0005 | equivalent |

Window-paired (n = 143): equivalence holds only at ±0.020.

**Finding.** "Matches Full Retraining" is supported at a ±0.010 margin, not at
±0.005. The correct statement is bounded equivalence within about one F1 point,
not an unqualified "within measurement noise".

## E5 — A degradation check that actually fires

A reused policy refit whenever its rolling per-window accuracy fell below an
absolute floor:

| stream | floor | F1 | refits |
|--------|-------|----|--------|
| UGR'16 | none (base) | 0.8360 | 0 |
| UGR'16 | 0.60 / 0.70 / 0.80 / 0.90 / 0.95 | 0.8360 | 0 |
| Campus | 0.95 | 0.9646 (from 0.9381) | 5 |

**Finding.** An accuracy-based check — the mechanism the paper describes — cannot
fire on UGR'16 at any sensible floor, because per-window *accuracy* stays high
while *macro-F1* collapses (imbalanced, near-empty windows). Recovering UGR'16
requires monitoring a balanced metric (macro-F1), not accuracy. This explains why
the parity threshold of 0.5 is inert and shows the fix is not "lower the
threshold".

---

## Consolidated statement of what the reported UGR'16 result supports

1. RAPT's UGR'16 macro-F1 (0.8360) is set by its 500-sample refit buffer; a
   1500-sample buffer (0.9276) closes most of the gap to Full Retraining.
2. The parity-refit mechanism, as configured, never activates
   (`parity_refits = 0`), so it is not the cause of the reported recovery.
3. The paper's mechanism attribution (lines 304–305, 412–413, 441–443, 588–590)
   does not match the artifacts.
4. The "reuse is unsafe when label semantics drift" narrative is not what the
   UGR'16 recurrence data shows: recurrence there is predominantly covariate.

## Recommended paper changes (facts/numbers only; no prose rewording)

All four locations below state a cause that the artifacts contradict. Because
correcting them requires changing wording (not just a number), they are reported
for the author rather than edited, per the audit's hard rule against rewording.

| location | current claim | artifact fact |
|----------|---------------|---------------|
| caption, lines 441–443 | "the parity refit in RAPT-Enhanced pulls accuracy back … Vertical ticks mark parity refits" | `make_ugr_rolling_figure.py::novelty_refit_windows` draws 12 **novelty**-refit ticks; parity refits = 0 |
| lines 412–413 | parity refitting recovers macro-F1 to 0.9276 | recovery is the 1500 vs 500 novelty-refit buffer; `parity_refits = 0` |
| lines 304–305 | parity refit "refits a reused policy … whenever rolling accuracy falls under a parity threshold" | branch exists but never triggers at the configured threshold |
| lines 588–590 | "the parity refit earns its place by taking 0.8360 to 0.9276" | same as above; the buffer earns it |
| lines 62–63, 408 | "regime identifiers recur but the feature-to-label mapping does not" | UGR'16 recurrence is predominantly covariate (E3b ratio 2.23, \|dAUC\| ≈ 0.01); a frozen model scores higher on recurrences (E3) |

The numeric values themselves (0.8360, 0.9276, 11 retrains, 133 reuses) are
correct as measured; only their attribution is not.

