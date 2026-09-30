# FINAL EXPERIMENTS — RAPT mechanism ablation on 5G Campus QoS

_Prequential Test-Then-Train evaluation, seeds [42, 43, 44, 45, 46]. Auto-generated
tables and figures live in `results/`._

## 0. Scope and an important caveat on dataset choice

This folder contains a controlled evaluation of **three mechanisms** that the
existing 9A port of RAPT omits relative to the canonical two-tier RAPT
(`rapt/hybrid_rapt.py`):

1. a Tier-1 always-on online micro-learner with similarity-driven blending;
2. a parity-refit safety net for stale reused policies;
3. a fingerprint similarity gate on reuse.

**Dataset selection is scientific, not score-based.** The 5G Campus Network QoS
stream is used because it is a genuine telecom QoS dataset with semantic regime
labels and real recurrence, has moderate dimensionality (19 features) so a
fingerprint gate is meaningful, and runs the full ladder across 5 seeds cheaply.
It was *not* chosen because it flatters RAPT — as the results below show, RAPT
does **not** beat Full Retraining here, and that is reported as measured.

## 1. Protocol

- Stream: `experiments/exp9/data/processed_exp9_stream.csv` — 179 windows,
  3 classes (GOOD / DEGRADED / BAD), 3 regimes (A, B, C) with recurrence.
- Preprocessing: reused unchanged from `experiments/exp9a/three_dataset_load.py`
  (prefix-median imputation, documented 3-class next-window p90-delay target).
- Window size 10, initial training = first 20% of windows (36).
- Prequential Test-Then-Train; sequential chronological; no future leakage
  (test windows start at 36; the initial prefix is never evaluated).
- Identical class-anchored buffer for every adaptive model.
- Adaptation CPU measured with `time.process_time()`; wall time with
  `time.perf_counter()`; both measured per method on the same basis.

## 2. Models

Baselines (unchanged): **Frozen**, **Event-Driven**, **Full Retraining**.

RAPT mechanism ladder (each rung adds exactly one mechanism):

| Rung | Tier-1 online | Refit net | Similarity gate | Periodic refresh |
|---|---|---|---|---|
| RAPT_T2 | – | – | – | – |
| RAPT_T1 | ✔ | – | – | – |
| RAPT_T1_REFIT | ✔ | ✔ (absolute) | – | – |
| RAPT_FULL | ✔ | ✔ (absolute) | ✔ | – |
| RAPT_REL_REFIT | ✔ | ✔ (relative) | ✔ | – |
| RAPT_REFRESH_W5 | ✔ | ✔ (absolute) | ✔ | every 5 windows |
| RAPT_REFRESH_W10 | ✔ | ✔ (absolute) | ✔ | every 10 windows |

The periodic refresh is the mechanism that stops a *reused* policy from staying
frozen for the whole duration of a recurring regime — the identified root cause
of RAPT's deficit. The refresh interval (5 vs 10) is a cost/accuracy knob, not a
tuned hyper-parameter.

## 3. Results (mean ± sd over 5 seeds)

| Model | Macro-F1 | Accuracy | Precision | Recall | Adapt CPU (s) | Runtime (s) | Retrains | Reuse ev. | Trees trained | Trees reused |
|---|---|---|---|---|---|---|---|---|---|---|
| Frozen | 0.9367 ± 0.0016 | 0.9824 | 0.9383 | 0.9378 | 0.0000 | 1.279 | 0 | 0 | 100 | 0 |
| Event-Driven | 0.9639 ± 0.0000 | 0.9909 | 0.9641 | 0.9647 | 0.1059 | 1.392 | 1 | 0 | 200 | 0 |
| Full Retraining | 0.9829 ± 0.0023 | 0.9950 | 0.9830 | 0.9836 | 1.4072 | 2.705 | 14 | 0 | 1500 | 0 |
| RAPT_T2 | 0.9381 ± 0.0016 | 0.9838 | 0.9393 | 0.9392 | 0.2355 | 1.519 | 2 | 12 | 300 | 1200 |
| RAPT_T1 | 0.9404 ± 0.0020 | 0.9842 | 0.9415 | 0.9414 | 0.2368 | 1.827 | 2 | 12 | 300 | 1200 |
| RAPT_T1_REFIT | 0.9404 ± 0.0020 | 0.9842 | 0.9415 | 0.9414 | 0.2321 | 2.515 | 2 | 12 | 300 | 1200 |
| RAPT_FULL | 0.9587 ± 0.0000 | 0.9888 | 0.9592 | 0.9598 | 0.4296 | 2.739 | 4 | 10 | 500 | 1000 |
| RAPT_REL_REFIT | 0.9587 ± 0.0000 | 0.9888 | 0.9592 | 0.9598 | 0.4319 | 2.756 | 4 | 10 | 500 | 1000 |
| **RAPT_REFRESH_W5** | **0.9847 ± 0.0028** | 0.9950 | 0.9850 | 0.9852 | 2.7294 | 4.916 | 4 | 10 | 2700 | 1000 |
| RAPT_REFRESH_W10 | 0.9774 ± 0.0004 | 0.9926 | 0.9779 | 0.9783 | 0.9767 | 3.266 | 4 | 10 | 1000 | 1000 |

### Statistical comparison vs Full Retraining (paired Wilcoxon over seeds)

| Model | Δ F1 (mean) | 95% CI | Cohen's d | Wilcoxon p |
|---|---|---|---|---|
| Frozen | −0.0462 | [−0.0508, −0.0416] | −12.43 | 0.0625 |
| Event-Driven | −0.0190 | [−0.0219, −0.0161] | −8.13 | 0.0625 |
| RAPT_T2 | −0.0448 | [−0.0494, −0.0401] | −12.03 | 0.0625 |
| RAPT_T1 | −0.0425 | [−0.0463, −0.0388] | −14.20 | 0.0625 |
| RAPT_T1_REFIT | −0.0425 | [−0.0463, −0.0388] | −14.20 | 0.0625 |
| RAPT_FULL | −0.0242 | [−0.0271, −0.0212] | −10.32 | 0.0625 |
| RAPT_REL_REFIT | −0.0242 | [−0.0271, −0.0212] | −10.32 | 0.0625 |
| **RAPT_REFRESH_W5** | **+0.0018** | [−0.0002, +0.0038] | +1.10 | **0.2500** |
| RAPT_REFRESH_W10 | −0.0055 | [−0.0084, −0.0025] | −2.26 | 0.0625 |

p = 0.0625 is the smallest two-sided Wilcoxon p attainable with n = 5 seeds
(2/2^5); the effects are consistent in sign and large in standardised terms,
but **not significant at α = 0.05** and are reported as such. RAPT_REFRESH_W5
is the only variant that does not lose to Full Retraining (p = 0.25, i.e. no
detectable difference), and it does **not** win significantly either.

### Per-regime Macro-F1 (where the gain comes from)

| Regime | Frozen | Event-Driven | Full Retraining | RAPT_T2 | RAPT_FULL | RAPT_REFRESH_W5 |
|---|---|---|---|---|---|---|
| A | 0.930 | 0.979 | 0.988 | 0.930 | 0.930 | **0.990** |
| B | 0.976 | 0.980 | 0.980 | 0.980 | 0.980 | 0.982 |
| C | 0.885 | 0.911 | 0.979 | 0.885 | 0.977 | 0.979 |

## 4. Key observations (observed, not interpreted)

1. **The periodic refresh closes the gap and slightly exceeds Full Retraining**
   (0.9847 vs 0.9829), driven entirely by regime A: 0.930 → 0.990. Regime A was
   the case where RAPT_FULL served a weak reused policy identical to Frozen
   (0.930); refreshing it every 5 windows fixes it.
2. **The gain is not statistically significant** (Δ = +0.0018, p = 0.25). The
   correct claim is "RAPT_REFRESH_W5 matches Full Retraining", not "beats".
3. **The fingerprint similarity gate was the mechanism that fixed regime C**
   (0.885 → 0.977) in the previous run; the refresh fixes regime A. The two
   mechanisms address different failure modes and are complementary.
4. **Refresh frequency is a real cost/accuracy trade-off.**
   W10 (0.977, 0.98 s adaptation CPU) vs W5 (0.985, 2.73 s). W5 buys +0.007 F1
   for ~2.8× the adaptation CPU, which erases RAPT's cost advantage: at W5
   RAPT's runtime (4.92 s) exceeds Full Retraining's (2.71 s).
5. **Tier-1 online blending and both refit triggers remain inert here**
   (+0.002 F1 for Tier-1; the absolute trigger cannot fire on 3 classes and the
   relative trigger did not fire either). Documented as negative results.
6. **Cost ordering:** Frozen 1.28 s < Event-Driven 1.39 s < RAPT_T2 1.52 s <
   RAPT_T1 1.83 s < RAPT_T1_REFIT 2.52 s < Full Retraining 2.71 s <
   RAPT_REFRESH_W10 3.27 s < RAPT_REFRESH_W5 4.92 s.

## 5. Interpretation

RAPT's deficit on the 5G Campus QoS stream has two distinct causes, and each
needs its own mechanism:

- **Serving a stale policy for a *changed* regime** (regime C): fixed by the
  fingerprint similarity gate, which refuses reuse and retrains (+0.021 F1).
- **Serving a frozen policy for an *unchanged* regime** (regime A): fixed by the
  bounded periodic refresh (+0.021 F1 on top of the gate).

With both, RAPT reaches parity with Full Retraining (0.9847 vs 0.9829). The
honest framing is that **RAPT becomes as accurate as Full Retraining but loses
its cost advantage** at a 5-window refresh interval; at a 10-window interval it
keeps a modest cost advantage but is slightly worse (0.9774). There is no
setting in this sweep where RAPT is both more accurate *and* cheaper.

## 6. Limitations

- Single dataset, single target definition (3-class next-window p90-delay QoS).
- n = 5 seeds; Wilcoxon cannot reach p < 0.05 at this n, so "matches" is the
  strongest defensible claim for RAPT_REFRESH_W5.
- Refresh interval (5, 10) was chosen as a small cost/accuracy sweep, not tuned
  on the test stream for a maximum.
- The similarity gate uses a normalized mean-feature fingerprint with γ = 1.0;
  other scalings were not swept.
- The refit triggers were ported with their reference thresholds rather than
  tuned, to avoid test-stream leakage.
- Regime labels are treated as given; no detector-driven regime discovery.
- RAPT is **not** modified in the existing 9A/9B experiments; this folder is a
  separate ablation, and the "RAPT" in the main experiments remains the Tier-2
  algorithm.

## 7. Reproducibility

```bash
cd "Ensemble Learning for Model Drift Detection/Final_Experiments"
python run_final.py full      # 5 seeds -> results/raw/{per_window,summary}_full.csv
python figures.py full        # -> results/figures/*.png
python tables.py full         # -> results/tables/*
```

Configuration constants live only in `config.py`. Raw per-window results are
saved so every figure can be reproduced.

### Figures

| File | Content |
|---|---|
| `fig_final_f1_bars.png` | Macro-F1 by model |
| `fig_final_stream_f1.png` | Macro-F1 over the stream with regime transitions |
| `fig_final_cost_tradeoff.png` | Macro-F1 vs adaptation CPU (per seed) |
| `fig_final_reuse_retrain.png` | Reuse events / retrains / trees trained / reused |
| `fig_final_regime_f1.png` | Per-regime Macro-F1 |
| `fig_final_ladder.png` | Mechanism ladder: F1 and adaptation cost |
