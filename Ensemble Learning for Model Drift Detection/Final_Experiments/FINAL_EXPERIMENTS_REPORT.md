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

| Rung | Tier-1 online | Refit net | Sim. gate | Refresh |
|---|---|---|---|---|
| RAPT_T2 | – | – | – | – |
| RAPT_T1 | ✔ | – | – | – |
| RAPT_T1_REFIT | ✔ | ✔ (absolute) | – | – |
| RAPT_FULL | ✔ | ✔ (absolute) | ✔ | – |
| RAPT_REL_REFIT | ✔ | ✔ (relative) | ✔ | – |
| RAPT_REFRESH_W5 | ✔ | ✔ (absolute) | ✔ | periodic, every 5 windows |
| RAPT_REFRESH_W10 | ✔ | ✔ (absolute) | ✔ | periodic, every 10 windows |
| RAPT_EVIDENCE | ✔ | ✔ (absolute) | ✔ | relative evidence |
| RAPT_CHEAP | ✔ | ✔ (absolute) | ✔ | periodic, 5 windows, **cheap refit** |
| RAPT_COMBO | ✔ | ✔ (absolute) | ✔ | relative evidence + cheap refit |
| RAPT_FLOOR | ✔ | ✔ (absolute) | ✔ | absolute floor + cheap refit |

Three refresh *triggers* are compared:

- **periodic** — refresh every N windows;
- **evidence** (relative) — refresh when the policy's rolling accuracy falls
  `rel_drop` below its own decayed baseline;
- **floor** (absolute) — refresh when the rolling accuracy falls below a fixed
  value (`refresh_min_acc = 0.97`).

and two refresh *costs*:

- **full refit** — 100 trees (50 RF + 50 ET), 1000-sample buffer;
- **cheap refit** — 20 trees (10 RF + 10 ET), 300-sample buffer (measured ~4.7×
  cheaper per call: 22 ms vs 104 ms).

### Why the "evidence" trigger fails (measured, not assumed)

`RAPT_EVIDENCE` and `RAPT_COMBO` fire **zero** refreshes and are therefore
identical to `RAPT_FULL` (0.9587). The relative trigger cannot see regime A's
problem, because regime A's reused policy is *uniformly* bad (0.930) rather than
*degrading*: its decayed baseline converges down to match the bad accuracy, so
`mean(recent) < baseline − 0.10` is never satisfied. A relative trigger detects
degradation; it cannot detect persistent deficiency. The **absolute floor**
detects it (7.6 refreshes, +0.021 F1).

The cheap-refit mechanism was chosen from measured micro-benchmarks of the
existing ensemble, not guessed (see "Cost drivers" below).

## 3. Results (mean ± sd over 5 seeds)

| Model | Macro-F1 | Adapt CPU (s) | Runtime (s) | Retrains | Reuse ev. | Trees trained | Refreshes |
|---|---|---|---|---|---|---|---|
| Frozen | 0.9367 ± 0.0016 | 0.0000 | 1.278 | 0 | 0 | 100 | – |
| Event-Driven | 0.9639 ± 0.0000 | 0.1047 | 1.393 | 1 | 0 | 200 | – |
| Full Retraining | 0.9829 ± 0.0023 | 1.4035 | 2.692 | 14 | 0 | 1500 | – |
| RAPT_T2 | 0.9381 ± 0.0016 | 0.2299 | 1.517 | 2 | 12 | 300 | 0 |
| RAPT_T1 | 0.9404 ± 0.0020 | 0.2333 | 1.794 | 2 | 12 | 300 | 0 |
| RAPT_T1_REFIT | 0.9404 ± 0.0020 | 0.2316 | 2.528 | 2 | 12 | 300 | 0 |
| RAPT_FULL | 0.9587 ± 0.0000 | 0.4298 | 2.735 | 4 | 10 | 500 | 0 |
| RAPT_REL_REFIT | 0.9587 ± 0.0000 | 0.4295 | 2.742 | 4 | 10 | 500 | 0 |
| RAPT_REFRESH_W5 | 0.9847 ± 0.0028 | 2.7221 | 4.928 | 4 | 10 | 2700 | 22 |
| RAPT_REFRESH_W10 | 0.9774 ± 0.0004 | 0.9750 | 3.242 | 4 | 10 | 1000 | 5 |
| RAPT_EVIDENCE | 0.9587 ± 0.0000 | 0.4315 | 3.488 | 4 | 10 | 500 | **0** |
| **RAPT_CHEAP** | **0.9851 ± 0.0029** | **0.8467** | **2.350** | 4 | 10 | 940 | 22 |
| RAPT_COMBO | 0.9587 ± 0.0000 | 0.4368 | 3.462 | 4 | 10 | 500 | **0** |
| RAPT_FLOOR | 0.9797 ± 0.0050 | 0.5829 | 2.620 | 4 | 10 | 652 | 7.6 |

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
| RAPT_REFRESH_W5 | +0.0018 | [−0.0002, +0.0038] | +1.10 | 0.2500 |
| RAPT_REFRESH_W10 | −0.0055 | [−0.0084, −0.0025] | −2.26 | 0.0625 |
| RAPT_EVIDENCE | −0.0242 | [−0.0271, −0.0212] | −10.32 | 0.0625 |
| **RAPT_CHEAP** | **+0.0022** | [−0.0036, +0.0080] | +0.47 | **0.3125** |
| RAPT_COMBO | −0.0242 | [−0.0271, −0.0212] | −10.32 | 0.0625 |
| RAPT_FLOOR | −0.0032 | [−0.0108, +0.0043] | −0.53 | 0.3750 |

p = 0.0625 is the smallest two-sided Wilcoxon p attainable with n = 5 seeds
(2/2^5). **Nothing here is significant at α = 0.05.** RAPT_CHEAP (p = 0.31) and
RAPT_FLOOR (p = 0.38) show no detectable difference from Full Retraining.

### Per-regime Macro-F1

| Regime | Frozen | Full Retraining | RAPT_FULL | RAPT_CHEAP | RAPT_FLOOR | RAPT_REFRESH_W5 |
|---|---|---|---|---|---|---|
| A | 0.930 | 0.988 | 0.930 | **0.990** | 0.982 | 0.990 |
| B | 0.976 | 0.980 | 0.980 | 0.983 | 0.979 | 0.982 |
| C | 0.885 | 0.979 | 0.977 | 0.978 | 0.975 | 0.979 |

## 4. Cost drivers (measured micro-benchmarks)

Cost model: `adapt CPU ≈ (number of refreshes) × (cost per refit)`. Measured on
the existing ensemble:

| Factor | Setting | Cost per refit |
|---|---|---|
| n_estimators | 20 trees (10 RF + 10 ET) | **22 ms** |
| | 50 trees | 54 ms |
| | 100 trees (full) | **104 ms** |
| Buffer size | 200 samples | 80 ms |
| | 500 | 92 ms |
| | 1000 (full) | 104 ms |

Refit cost is dominated by `n_estimators`, not buffer size. The **cheap refit**
(20 trees, 300-sample buffer) is therefore the highest-leverage cost lever,
followed by reducing the *number* of refreshes.

## 5. Key observations (observed, not interpreted)

1. **RAPT_CHEAP is the only configuration in the sweep that is both at least as
   accurate as Full Retraining and cheaper.** F1 0.9851 vs 0.9829, adaptation
   CPU 0.85 s vs 1.40 s (**~40% lower**), runtime 2.35 s vs 2.69 s. This is the
   first setting that does not trade accuracy against cost.
2. **The difference is not statistically significant** (Δ = +0.0022, p = 0.31,
   d = 0.47). The defensible claim is "RAPT_CHEAP matches Full Retraining at
   ~60% of its adaptation cost", not "beats it".
3. **The relative evidence trigger is ineffective** (0 refreshes, 0.9587). It
   detects degradation, not persistent deficiency. This is a genuine negative
   result and is why the absolute floor was added.
4. **The absolute floor works but is weaker and less stable** (0.9797 ± 0.0050,
   7.6 refreshes). It still beats RAPT_FULL, and it costs less than
   RAPT_CHEAP (0.58 s vs 0.85 s), but its accuracy varies more across seeds.
5. **The gain comes from regime A**, the recurring regime whose reused policy
   was always bad: 0.930 → 0.990 (cheap) / 0.982 (floor). No regime is harmed
   by the cheap refit.
6. **Full-refit periodic refresh (W5) is now strictly dominated:** higher F1
   variance, 3.2× the adaptation CPU of RAPT_CHEAP (2.72 s vs 0.85 s), and the
   worst runtime of any model (4.93 s).
7. **Cost ordering (adaptation CPU):** Frozen 0.00 < Event-Driven 0.10 <
   RAPT_T2 0.23 < RAPT_T1_REFIT 0.23 < RAPT_FULL 0.43 < RAPT_FLOOR 0.58 <
   RAPT_CHEAP 0.85 < RAPT_REFRESH_W10 0.98 < Full Retraining 1.40 <
   RAPT_REFRESH_W5 2.72 s.

## 6. Interpretation

RAPT's deficit on the 5G Campus QoS stream has two distinct causes, each needing
its own mechanism:

- **Serving a stale policy for a *changed* regime** (regime C): fixed by the
  fingerprint similarity gate (0.885 → 0.977).
- **Serving a frozen policy that was *always* poor for a recurring regime**
  (regime A): fixed by a refresh, and specifically by a refresh that (a) is
  triggered on an **absolute** quality floor rather than a relative drop, and
  (b) refits **cheaply** so the cure costs less than the disease.

Combining both, RAPT_CHEAP reaches Full-Retraining accuracy at ~60% of its
adaptation CPU — i.e. the mechanism ladder can recover RAPT's cost advantage
without giving up accuracy. The important caveat is that this is a *single
dataset* result with n = 5 seeds and no statistical significance; the correct
summary is that the ladder removed the previously observed cost/accuracy
trade-off **on this stream**, not that RAPT is generally cheaper and better.

## 7. Limitations

- Single dataset, single target definition (3-class next-window p90-delay QoS).
- n = 5 seeds; nothing reaches p < 0.05, so "matches" is the strongest
  defensible claim for RAPT_CHEAP.
- The cheap refit (20 trees) reduces the refreshed policy's capacity. It happens
  to be sufficient here because the refresh only needs to correct a policy that
  is already close; on a harder stream it could underfit.
- The absolute floor (0.97) is a fixed constant. It was not tuned on the test
  stream, but a different stream would need a different floor; this is a
  documented hyper-parameter, not a universal constant.
- Refresh intervals / thresholds were chosen as a small sweep, not optimised.
- The similarity gate uses a normalized mean-feature fingerprint with γ = 1.0.
- Regime labels are treated as given; no detector-driven regime discovery.
- RAPT is **not** modified in the existing 9A/9B experiments; this folder is a
  separate ablation, and the "RAPT" in the main experiments remains the Tier-2
  algorithm.

## 8. Reproducibility

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
