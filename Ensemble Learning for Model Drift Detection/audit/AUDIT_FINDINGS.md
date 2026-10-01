# Audit findings — claim validity vs artifacts

The number audit (`verify_paper_numbers.py`) only checks that prose matches the
committed CSVs. It cannot detect that a CSV-consistent sentence states the wrong
*cause*. This file records the cases where the interpretation, not the number, is
wrong. All figures below are read from committed artifacts.

## F1. The UGR'16 recovery is the refit buffer, not parity refitting

The paper attributes the RAPT → RAPT-Enhanced gain (0.8360 → 0.9276) to parity
refitting, in the abstract, the UGR'16 paragraph, a figure caption, the Discussion
and the Conclusion.

Artifacts:
- `e2_out.txt` / `PHASE2_FINDINGS.md`: parity threshold sweep at 0.50 / 0.90 / 0.95 /
  0.97 all give macro-F1 0.7958 with `parity_refits = 0`. The parity branch never
  fires at any threshold.
- `results/revalidation/A1_summary_all_streams.csv`: `parity_refits` is 0 for
  RAPT-Enhanced on every seed and every stream.
- `run_stream_v2.py:203` fingerprints `X_buffer[-50:]`, i.e. the pre-transition
  (previous-regime) buffer, against the checkpoint of the incoming regime. On a
  regime that has been seen before, reuse is accepted and the parity branch is only
  reachable when `_current_reused` is true with a falling rolling accuracy; measured,
  it is 0.

**Correct attribution:** the gain is the larger novelty-refit buffer (RAPT-Enhanced
refits on 1500 recent observations vs 500 for base RAPT).

## F2. The "1500" buffer is capped at 1000 rows

`rapt_9a.py` / `run_stream_v2.py` cap the refit buffer at `buffer_capacity = 1000`
(`exp9a_config.py:70`). `A5_novelty_buffer.csv` records the actual training rows at
each novelty refit:

| stream | max_refit_rows | pooled macro-F1 |
|--------|----------------|-----------------|
| 5g_campus | 485 | 0.9838 |
| ugr16 | **1030** | 0.8008 |
| nordicdat | **1045** | 0.4531 |
| 5g_nr | 245 | 0.8930 |

So the paper's "1500-observation buffer" text is inaccurate: on UGR'16 the effective
buffer is ~1000 rows. This resolves the conflict between the reviewer's "capped at
1000" and E2's reproduction of 0.9276 with a 1500 setting — the 1500 *setting* is
real, but the *effective* buffer is capped at 1000. (Note A5's RAPT-Novelty1500
reproduces 0.8008 for RAPT-Cheap-style config on UGR'16, matching `A3_pareto.csv`;
the 0.9276 figure is the RAPT-Enhanced configuration.)

## F3. A2 — the P(Y|X) hypothesis test is now run, and it is inconclusive

`A2_hypothesis_verdict.csv` now exists (produced by `a2_hypothesis_verdict.py`, which
reads `A2_regime_stability.csv`). It does not support the paper's label-drift narrative:

- H1 (regime identity tracks behaviour): NOT_SUPPORTED on every stream. Same-regime
  transfer is never significantly worse than the contiguous control. On UGR'16 the
  paired overlap is only n=2, so the hypothesis is INSUFFICIENT_DATA there.
- H2 (the UGR'16 deficit is a changed conditional): INCONCLUSIVE on UGR'16 — the
  class-conditional feature shift is large (covariate evidence) but the same-regime
  test is underpowered, so the conditional explanation can be neither confirmed nor
  ruled out.

**Action:** the paper now states that the UGR'16 failure is not attributed to a moved
label mapping, because the test does not separate that from a covariate shift.

- `A2_checkpoint_provenance.csv`: **55 of 720** stored UGR'16 policies (7.6%) were
  trained on a different regime than the key they were stored under — the provenance
  bug. This is a competing explanation for the UGR'16 deficit that no E-experiment
  has isolated.

## F4. Per-window vs pooled macro-F1 changes the UGR'16 story

`A1_summary_all_streams.csv` (per-window) and `A2_deferred_vs_base.csv` (pooled) give
opposite rankings on UGR'16:

| method | per-window F1 | pooled F1 | FP / 1000 attack-free windows | attack recall |
|--------|---------------|-----------|-------------------------------|---------------|
| Frozen | 0.9690 | 0.8216 | 0.10 | 0.483 |
| Event-Driven | 0.8972 | 0.8413 | 2.22 | 0.583 |
| Full Retraining | 0.9595 | 0.7603 | 0.55 | 0.369 |
| RAPT | 0.8360 | 0.6629 | **25.98** | 0.429 |

The Frozen → RAPT gap is 0.1330 per-window but 0.1588 pooled, and RAPT's false-alarm
rate on attack-free traffic is **259x** Frozen's. UGR'16 windows are mostly
single-class, so per-window macro-F1 is noisy; the ranking is metric-sensitive and the
UGR'16 section currently argues over that artifact.

## F5. Cost-matched baselines — the repository claim does not survive

`A3_pareto.csv`, 5G Campus (pooled macro-F1, adaptation CPU s):

| method | F1 | adapt CPU | retrains | refreshes |
|--------|-----|-----------|----------|-----------|
| RAPT-Cheap | 0.9944 | 1.107 | 2 | 22 |
| Periodic-Cheap-5 | **0.9924** | **1.024** | 28 | 0 |
| Periodic-Cheap-10 | 0.9923 | 0.553 | 14 | 0 |
| FR-Cheap | 0.9913 | 0.532 | 14 | 0 |
| Full Retraining | 0.9951 | 1.525 | 14 | 0 |

Periodic-Cheap-5 beats RAPT-Cheap on both axes (higher F1, lower CPU) on Campus. On
UGR'16 RAPT-Cheap is 0.8008 vs Periodic-Cheap-5 0.6451, so RAPT wins there. The
"cost-aware refit dominates" claim is not supported on Campus.

## F6. Equivalence margin

`e4_out.txt`: RAPT-Cheap vs Full Retraining on Campus is equivalent at ±0.010
(TOST p = 0.0102) but **not** at ±0.005 (p = 0.1268); window-paired equivalence holds
only at ±0.020. "Within measurement noise" overstates a ±0.010 margin.

## F7. Minor claim scoping

- "48–92% less adaptation CPU": the 92% figure is UGR'16 *base* RAPT; RAPT-Enhanced
  saves far less (25.30 → 2.72 s ≈ 89%; 5G NR 0.764 → 0.861 s is a *cost increase*).
  Scope the range to base RAPT.
- Dataset sample counts: paper says 43,200 for one stream; needs pinning to the
  loader (`exp9b/`).
- Event-Driven trigger: verify the text says μ + 2σ (`run_stream_v2.py:432`) rather
  than a fixed threshold.
