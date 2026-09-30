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

| Rung | Tier-1 online | Refit net | Similarity gate |
|---|---|---|---|
| RAPT_T2 | – | – | – |
| RAPT_T1 | ✔ | – | – |
| RAPT_T1_REFIT | ✔ | ✔ (absolute) | – |
| RAPT_FULL | ✔ | ✔ (absolute) | ✔ |
| RAPT_REL_REFIT | ✔ | ✔ (relative) | ✔ |

## 3. Results (mean ± sd over 5 seeds)

| Model | Macro-F1 | Accuracy | Precision | Recall | Adapt CPU (s) | Runtime (s) | Retrains | Reuse ev. | Trees trained | Trees reused |
|---|---|---|---|---|---|---|---|---|---|---|
| Frozen | 0.9367 ± 0.0016 | 0.9824 | 0.9383 | 0.9378 | 0.0000 | 1.259 | 0 | 0 | 100 | 0 |
| Event-Driven | 0.9639 ± 0.0000 | 0.9909 | 0.9641 | 0.9647 | 0.1049 | 1.377 | 1 | 0 | 200 | 0 |
| Full Retraining | **0.9829 ± 0.0023** | 0.9950 | 0.9830 | 0.9836 | 1.4022 | 2.682 | 14 | 0 | 1500 | 0 |
| RAPT_T2 | 0.9381 ± 0.0016 | 0.9838 | 0.9393 | 0.9392 | 0.2296 | 1.504 | 2 | 12 | 300 | 1200 |
| RAPT_T1 | 0.9404 ± 0.0020 | 0.9842 | 0.9415 | 0.9414 | 0.2293 | 1.760 | 2 | 12 | 300 | 1200 |
| RAPT_T1_REFIT | 0.9404 ± 0.0020 | 0.9842 | 0.9415 | 0.9414 | 0.2325 | 2.482 | 2 | 12 | 300 | 1200 |
| RAPT_FULL | 0.9587 ± 0.0000 | 0.9888 | 0.9592 | 0.9598 | 0.4269 | 2.710 | 4 | 10 | 500 | 1000 |
| RAPT_REL_REFIT | 0.9587 ± 0.0000 | 0.9888 | 0.9592 | 0.9598 | 0.4299 | 2.705 | 4 | 10 | 500 | 1000 |

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

p = 0.0625 is the smallest two-sided Wilcoxon p attainable with n = 5 seeds
(2/2^5); the effects are consistent in sign and large in standardised terms,
but **not significant at α = 0.05** and are reported as such.

## 4. Key observations (observed, not interpreted)

1. **No RAPT variant beats Full Retraining** (best RAPT: 0.9587 vs 0.9829).
   The performance ordering is Full Retraining > Event-Driven > RAPT_FULL >
   RAPT_T1 > RAPT_T2 > Frozen.
2. **The fingerprint similarity gate is the only mechanism that moves F1**
   (+0.021 F1, 0.938 → 0.959), by refusing to reuse a stale policy for regime C
   and retraining instead (4 retrains vs 2; 10 reuses vs 12).
3. **Tier-1 online blending contributes almost nothing here** (+0.002 F1,
   and only on 2 of 5 seeds). The 3-class QoS task is well served by the
   batch ensemble, so the online learner adds no signal.
4. **Both refit triggers are inert on this dataset.** The absolute trigger
   (accuracy < 0.5) can never fire on a 3-class stream where degraded policies
   still score ≈ 0.95; the relative trigger (drop > 0.10) also never fires here.
   This is a genuine negative result about porting a binary-stream mechanism to
   a multi-class windowed stream.
5. **Cost ordering:** Frozen 1.26 s < Event-Driven 1.38 s < RAPT_T2 1.50 s <
   RAPT_T1 1.76 s < RAPT_T1_REFIT 2.48 s < Full Retraining 2.68 s ≈
   RAPT_FULL 2.71 s. RAPT's advantage is real at Tier-2 (0.23 s adaptation CPU
   vs 1.40 s for Full Retraining, ~6× cheaper), but the gate that fixes the F1
   also erases most of the runtime advantage.

## 5. Interpretation

On the 5G Campus QoS stream, RAPT's deficit versus Full Retraining is caused by
**stale policy reuse across regime recurrence**, not by slow drift reaction —
adding the similarity gate recovers about half the gap. The remaining gap is
structural: Full Retraining refreshes on every regime change and on fresh data,
while RAPT serves a fixed checkpoint for the duration of a regime. The two
mechanisms that were expected to help (Tier-1 blending, refit safety net) do not
help on this stream because the base batch ensemble is already strong and the
refit triggers do not fire.

## 6. Limitations

- Single dataset, single target definition (3-class next-window p90-delay QoS).
- n = 5 seeds; Wilcoxon cannot reach p < 0.05 at this n.
- The similarity gate uses a normalized mean-feature fingerprint with γ = 1.0;
  other scalings were not swept (and must not be swept on the test stream).
- The refit triggers were ported with their reference thresholds (0.5 absolute,
  0.10 relative drop) rather than tuned, to avoid test-stream leakage.
- Regime labels are treated as given; no detector-driven regime discovery.

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
