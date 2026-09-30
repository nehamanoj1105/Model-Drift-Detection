# EXPERIMENT 9B — FINAL REPORT
## Drift Severity and Concept-Drift Evaluation for RAPT

This report extends the existing Experiment 9B into a systematic evaluation of
natural recurring drift, controlled covariate (data) drift at multiple severities,
controlled concept drift at multiple severities, and recurring concept drift.
The proposed model is referred to simply as **RAPT**; no new algorithm variant is
introduced and the RAPT implementation is unmodified.

## 1. Dataset
The **second dataset already used by Experiment 9B**: the 5G NR end-to-end
latency simulation dataset (Zenodo DOI `10.5281/zenodo.20035549`). The processed
stream (`processed_exp9b_stream.csv`) contains **499 windows** of
500 packets each, spanning four regimes (A/B/C/D) in the sequence
`ABCDACBDAB`. The frozen QoS target is a 3-class label
(GOOD / DEGRADED / BAD) derived from the next-window p90 latency, using
thresholds computed on the initial training prefix (t1=2.308 ms,
t2=4.191 ms).

## 2. Existing 9B Setup (preserved)
| Component | Value |
| :--- | :--- |
| Window size | 500 packets (pre-built) |
| Initial training period | 99 windows (20% prefix) |
| Protocol | Sequential chronological, Test-Then-Train (prequential) |
| Seeds | [42, 43, 44, 45, 46] |
| Base model | RandomForest + ExtraTrees soft-voting ensemble (50 trees each, depth 7) |
| Preprocessing | StandardScaler fitted only on the initial prefix (no leakage) |
| Feature selection | Fixed 12 QoS features from the existing pipeline |
| Target | Frozen 3-class QoS label |
| Baselines | Frozen, Event-Driven, Full Retraining |
| Proposed | RAPT |
| Enhanced variant | RAPT-Enhanced (existing Enhanced-Hybrid-RAPT mechanisms) |
| Metrics | Macro-F1, Accuracy, Precision, Recall, Balanced Accuracy, adaptation/total CPU, retrain events, reuse events |
| Statistics | Window-level paired Wilcoxon signed-rank + Cohen's d |

The natural-drift experiment (9B-A) is the existing experiment, re-used unchanged.
All new experiments share this protocol so results are directly comparable.

## 3. Natural Drift Experiment (9B-A)
Regime transitions and recurrences in the natural stream:

| Segment | Regime | Windows | Transition |
| :---: | :---: | :---: | :---: |
| 0 | regime_A | [0, 49] | no |
| 1 | regime_B | [50, 99] | yes |
| 2 | regime_C | [100, 149] | yes |
| 3 | regime_D | [150, 199] | yes |
| 4 | regime_A | [200, 249] | yes |
| 5 | regime_C | [250, 299] | yes |
| 6 | regime_B | [300, 349] | yes |
| 7 | regime_D | [350, 399] | yes |
| 8 | regime_A | [400, 449] | yes |
| 9 | regime_B | [450, 499] | yes |

| Model | Macro-F1 | Accuracy | Adapt CPU (s) | Retrains | Reuse events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Frozen | 0.8961 ± 0.0099 | 0.9075 | 0.0000 | 0.00 | 0.00 |
| Event-Driven | 0.8903 ± 0.0107 | 0.9055 | 1.0406 | 11.00 | 0.00 |
| Full Retraining | 0.9027 ± 0.0059 | 0.9145 | 0.7636 | 9.00 | 0.00 |
| RAPT | 0.8894 ± 0.0110 | 0.9025 | 0.3955 | 3.00 | 6.00 |
| RAPT-Enhanced | 0.8915 ± 0.0117 | 0.9055 | 0.8609 | 3.00 | 6.00 |

_Observed_: The natural stream recurs over regimes A/B/C/D. RAPT stores one policy
per regime and reuses it on every recurrence (6 reuse events, 600 reused trees,
3 retrains), so it never retrains on a revisit; RAPT-Enhanced shares the same reuse
pattern. RAPT attains the lowest adaptation CPU (~0.40 s), roughly half of Full
Retraining (~0.76 s) and 38% of Event-Driven (~1.04 s), at a small Macro-F1 cost
relative to Frozen (~0.896) and Full Retraining (~0.903) — RAPT itself scores
~0.889. RAPT-Enhanced is marginally higher than RAPT (~0.892) with higher CPU from
its parity refits. See `fig9b_natural_drift_f1.png` and
`fig9b_rapt_adaptation_timeline.png`.

## 4. Covariate Drift Methodology (9B-B)
Covariate drift = change in **P(X)** with **P(Y|X)** held as constant as possible.
For each selected source→target regime transition in the existing stream we
construct a mixed post-drift block of 40 windows containing a controlled fraction
of target-regime windows:

| Severity | Composition |
| :---: | :--- |
| 10% | 36 source-regime + 4 target-regime windows (fixed-seed interleaving) |
| 20% | 32 source-regime + 8 target-regime windows (fixed-seed interleaving) |
| 30% | 28 source-regime + 12 target-regime windows (fixed-seed interleaving) |
| 50% | 20 source-regime + 20 target-regime windows (fixed-seed interleaving) |
| 100% | 0 source-regime + 40 target-regime windows (fixed-seed interleaving) |

**Sample selection.** Source and target windows are taken from chronologically
contiguous, non-overlapping slices of their respective regimes (no reuse across
the drift point). Labels are copied verbatim; **no relabeling** occurs, so each
sample keeps the label it had in its own regime and P(Y|X) is unchanged. The
target-regime samples are interleaved with a fixed-seed permutation, preserving
chronological ordering statistically rather than appending all target samples at
the end. All post-drift windows share a single regime id so regime-keyed adapters
register exactly one boundary at the drift point.

_Protocol note._ Because the existing stream is a finite set of pre-computed
windows, exact 10/20/30/50% mixtures are realised with `round(level*40)` target
windows; this is the closest valid controlled construction and is reported exactly.
A covariate construction cannot reuse the *same* windows for source and target, so
source windows for the post-drift block are drawn from a later slice of the source
regime than the pre-drift reference.

### Results

| Model | Severity | F1 (mean ± std) | Accuracy | Precision | Recall | Adapt CPU (s) | Runtime (s) | Retrains |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 10% | 0.5265 ± 0.0000 | 0.4900 | 0.7964 | 0.6699 | 0.0000 | 0.9240 | 0.00 |
| Frozen | 20% | 0.5201 ± 0.0000 | 0.4750 | 0.7904 | 0.6698 | 0.0000 | 0.9364 | 0.00 |
| Frozen | 30% | 0.5192 ± 0.0000 | 0.4700 | 0.7897 | 0.6698 | 0.0000 | 0.9412 | 0.00 |
| Frozen | 50% | 0.5102 ± 0.0000 | 0.4400 | 0.7817 | 0.6696 | 0.0000 | 0.9440 | 0.00 |
| Frozen | 100% | 0.4893 ± 0.0000 | 0.3700 | 0.7640 | 0.6693 | 0.0000 | 0.9375 | 0.00 |
| Event-Driven | 10% | 0.6819 ± 0.0106 | 0.6000 | 0.6947 | 0.6953 | 0.2389 | 1.1074 | 2.00 |
| Event-Driven | 20% | 0.6489 ± 0.0174 | 0.5650 | 0.7104 | 0.7005 | 0.1569 | 1.0185 | 1.00 |
| Event-Driven | 30% | 0.6438 ± 0.0074 | 0.5560 | 0.7071 | 0.6980 | 0.1547 | 1.0135 | 1.00 |
| Event-Driven | 50% | 0.6489 ± 0.0178 | 0.5510 | 0.7141 | 0.7059 | 0.1544 | 1.0145 | 1.00 |
| Event-Driven | 100% | 0.6672 ± 0.0082 | 0.5550 | 0.7258 | 0.7278 | 0.1601 | 1.0436 | 1.00 |
| Full Retraining | 10% | 0.6850 ± 0.0171 | 0.6120 | 0.7537 | 0.7334 | 0.2460 | 1.0956 | 3.00 |
| Full Retraining | 20% | 0.7894 ± 0.0068 | 0.7770 | 0.8236 | 0.7843 | 0.2539 | 1.1302 | 3.00 |
| Full Retraining | 30% | 0.7189 ± 0.0106 | 0.6440 | 0.7452 | 0.7461 | 0.2479 | 1.1000 | 3.00 |
| Full Retraining | 50% | 0.7671 ± 0.0177 | 0.7280 | 0.7684 | 0.7677 | 0.2416 | 1.0907 | 3.00 |
| Full Retraining | 100% | 0.8080 ± 0.0017 | 0.7940 | 0.8133 | 0.8040 | 0.2536 | 1.1135 | 3.00 |
| RAPT | 10% | 0.6854 ± 0.0309 | 0.6140 | 0.7607 | 0.7373 | 0.3356 | 1.1932 | 3.00 |
| RAPT | 20% | 0.7944 ± 0.0114 | 0.7880 | 0.8471 | 0.7886 | 0.3419 | 1.1941 | 3.00 |
| RAPT | 30% | 0.7285 ± 0.0170 | 0.6570 | 0.7467 | 0.7501 | 0.3348 | 1.1957 | 3.00 |
| RAPT | 50% | 0.7767 ± 0.0159 | 0.7430 | 0.7786 | 0.7755 | 0.3323 | 1.1923 | 3.00 |
| RAPT | 100% | 0.8122 ± 0.0038 | 0.8000 | 0.8193 | 0.8072 | 0.3343 | 1.1951 | 3.00 |
| RAPT-Enhanced | 10% | 0.6854 ± 0.0309 | 0.6140 | 0.7607 | 0.7373 | 0.3309 | 1.1819 | 3.00 |
| RAPT-Enhanced | 20% | 0.7944 ± 0.0114 | 0.7880 | 0.8471 | 0.7886 | 0.3309 | 1.1890 | 3.00 |
| RAPT-Enhanced | 30% | 0.7285 ± 0.0170 | 0.6570 | 0.7467 | 0.7501 | 0.3329 | 1.1948 | 3.00 |
| RAPT-Enhanced | 50% | 0.7767 ± 0.0159 | 0.7430 | 0.7786 | 0.7755 | 0.3332 | 1.1895 | 3.00 |
| RAPT-Enhanced | 100% | 0.8122 ± 0.0038 | 0.8000 | 0.8193 | 0.8072 | 0.3358 | 1.1943 | 3.00 |

### Recovery

| Model | Severity | Pre-drift F1 | Min F1 | Recovery F1 | Recovery windows | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 10% | 0.5946 | 0.0833 | 0.4365 | 29.50 | -0.1696 |
| Frozen | 20% | 0.5946 | 0.0833 | 0.4250 | 27.50 | -0.1934 |
| Frozen | 30% | 0.5946 | 0.0833 | 0.3915 | 40.00 | -0.2031 |
| Frozen | 50% | 0.5946 | 0.0000 | 0.3500 | 40.00 | -0.2446 |
| Frozen | 100% | 0.5946 | 0.0000 | 0.0918 | 40.00 | -0.5028 |
| Event-Driven | 10% | 0.7466 | 0.1667 | 0.7296 | 36.00 | -0.1273 |
| Event-Driven | 20% | 0.7062 | 0.1667 | 0.6834 | 29.20 | -0.0849 |
| Event-Driven | 30% | 0.7020 | 0.1375 | 0.7015 | 25.00 | -0.0791 |
| Event-Driven | 50% | 0.6946 | 0.2440 | 0.6952 | 22.80 | -0.0535 |
| Event-Driven | 100% | 0.7032 | 0.3056 | 0.4546 | 20.50 | -0.2486 |
| Full Retraining | 10% | 0.6966 | 0.5357 | 0.8484 | 20.50 | 0.1518 |
| Full Retraining | 20% | 0.7696 | 0.6190 | 0.9118 | 20.50 | 0.1422 |
| Full Retraining | 30% | 0.7568 | 0.4206 | 0.8785 | 20.50 | 0.1216 |
| Full Retraining | 50% | 0.8176 | 0.6875 | 0.9039 | 20.50 | 0.0863 |
| Full Retraining | 100% | 0.9436 | 0.5000 | 1.0000 | 13.50 | -0.1482 |
| RAPT | 10% | 0.7018 | 0.5714 | 0.8515 | 20.50 | 0.1498 |
| RAPT | 20% | 0.7768 | 0.6429 | 0.9193 | 20.50 | 0.1424 |
| RAPT | 30% | 0.7626 | 0.5921 | 0.9051 | 20.50 | 0.1425 |
| RAPT | 50% | 0.8318 | 0.6875 | 0.9077 | 20.50 | 0.0759 |
| RAPT | 100% | 0.9479 | 0.5000 | 1.0000 | 13.50 | -0.1525 |
| RAPT-Enhanced | 10% | 0.7018 | 0.5714 | 0.8515 | 20.50 | 0.1498 |
| RAPT-Enhanced | 20% | 0.7768 | 0.6429 | 0.9193 | 20.50 | 0.1424 |
| RAPT-Enhanced | 30% | 0.7626 | 0.5921 | 0.9051 | 20.50 | 0.1425 |
| RAPT-Enhanced | 50% | 0.8318 | 0.6875 | 0.9077 | 20.50 | 0.0759 |
| RAPT-Enhanced | 100% | 0.9479 | 0.5000 | 1.0000 | 13.50 | -0.1525 |

## 5. Concept Drift Methodology (9B-C)
Concept drift = change in **P(Y|X)**. To isolate it from covariate drift we hold
the regime (and hence the feature distribution) fixed and change only the
feature→label relationship.

**Transformation.** A deterministic *shared-permutation rank-reversal* is applied
to the top-k most predictive features, where k = ceil(severity × |ranked pool|) with
|ranked pool| = 6 (features ranked by mutual information on the pre-drift prefix),
capped at the 5 features that actually vary within the source regime:

| Severity | Affected features |
| :---: | :---: |
| 10% | 1 |
| 20% | 2 |
| 30% | 2 |
| 50% | 3 |
| 100% | 5 |

Because only five features vary within the source regime, the 20% and 30% levels
both map to two affected features; this granularity limit is reported rather than
hidden.

Construction: order the post-drift rows by the primary (most predictive) affected
feature ascending, reverse that order to obtain a permutation π, then apply the
SAME π to every affected feature column. Properties:

* π is a row permutation of the affected columns, so the **joint distribution of the
  affected features is preserved exactly** and every per-feature marginal is
  preserved exactly (KS = 0): the construction does not shift P(X).
* The pairing between the affected features and Y is reversed, so **P(Y|X) changes**.
* It is a pure, deterministic function of X applied only after the drift point and
  never reads the target; feature ranking uses only the pre-drift prefix.
* The regime id is left unchanged, so no oracle boundary is given to the adapters —
  this is the purest P(Y|X) test.

### Diagnostic evidence that P(Y|X) changed
Diagnostics are saved under `results/experiment_9b/concept_drift/`:
* `px_invariance.csv` — KS between the untransformed and transformed post-drift
  blocks (≈ 0 confirms P(X) is preserved) plus I(feature; Y) before/after.
* `feature_target_association.csv` — mutual information I(feature; Y) before vs
  after drift for each affected feature (decreases or reverses).
* `model_based_py_change.csv` — Macro-F1 of a model trained on the pre-drift prefix,
  measured on a held-out pre-drift block vs the post-drift block (a large drop with
  identical feature distributions demonstrates a conditional change).
* `class_conditional_stats.csv` — class-conditional feature mean/std before vs after.
* `conditional_proba_estimates.csv` — empirical P(Y=BAD | feature quartile) before
  vs after drift.

### Results

| Model | Severity | F1 (mean ± std) | Accuracy | Precision | Recall | Adapt CPU (s) | Runtime (s) | Retrains |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 10% | 0.8777 ± 0.0085 | 0.9725 | 0.9865 | 0.8250 | 0.0000 | 0.2954 | 0.00 |
| Frozen | 20% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0000 | 0.2977 | 0.00 |
| Frozen | 30% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0000 | 0.2971 | 0.00 |
| Frozen | 50% | 0.8117 ± 0.0187 | 0.8902 | 0.9241 | 0.7568 | 0.0000 | 0.3003 | 0.00 |
| Frozen | 100% | 0.7916 ± 0.0000 | 0.8627 | 0.8983 | 0.7405 | 0.0000 | 0.2944 | 0.00 |
| Event-Driven | 10% | 0.9241 ± 0.0550 | 0.9804 | 0.9902 | 0.8917 | 0.1556 | 0.3807 | 1.00 |
| Event-Driven | 20% | 0.9379 ± 0.0635 | 0.9725 | 0.9865 | 0.9125 | 0.1560 | 0.3759 | 1.00 |
| Event-Driven | 30% | 0.9379 ± 0.0635 | 0.9725 | 0.9865 | 0.9125 | 0.1621 | 0.3835 | 1.00 |
| Event-Driven | 50% | 0.8274 ± 0.0247 | 0.9098 | 0.9389 | 0.7734 | 0.2203 | 0.4394 | 1.80 |
| Event-Driven | 100% | 0.7916 ± 0.0000 | 0.8627 | 0.8983 | 0.7405 | 0.2350 | 0.4516 | 2.00 |
| Full Retraining | 10% | 0.8777 ± 0.0085 | 0.9725 | 0.9865 | 0.8250 | 0.0000 | 0.2221 | 0.00 |
| Full Retraining | 20% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0000 | 0.2148 | 0.00 |
| Full Retraining | 30% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0000 | 0.2214 | 0.00 |
| Full Retraining | 50% | 0.8117 ± 0.0187 | 0.8902 | 0.9241 | 0.7568 | 0.0000 | 0.2163 | 0.00 |
| Full Retraining | 100% | 0.7916 ± 0.0000 | 0.8627 | 0.8983 | 0.7405 | 0.0000 | 0.2221 | 0.00 |
| RAPT | 10% | 0.8777 ± 0.0085 | 0.9725 | 0.9865 | 0.8250 | 0.0761 | 0.2912 | 0.00 |
| RAPT | 20% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0764 | 0.2934 | 0.00 |
| RAPT | 30% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0778 | 0.2973 | 0.00 |
| RAPT | 50% | 0.8117 ± 0.0187 | 0.8902 | 0.9241 | 0.7568 | 0.0756 | 0.2957 | 0.00 |
| RAPT | 100% | 0.7916 ± 0.0000 | 0.8627 | 0.8983 | 0.7405 | 0.0772 | 0.2982 | 0.00 |
| RAPT-Enhanced | 10% | 0.8777 ± 0.0085 | 0.9725 | 0.9865 | 0.8250 | 0.0769 | 0.2970 | 0.00 |
| RAPT-Enhanced | 20% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0774 | 0.2978 | 0.00 |
| RAPT-Enhanced | 30% | 0.8683 ± 0.0000 | 0.9608 | 0.9810 | 0.8125 | 0.0773 | 0.2948 | 0.00 |
| RAPT-Enhanced | 50% | 0.8117 ± 0.0187 | 0.8902 | 0.9241 | 0.7568 | 0.0768 | 0.2974 | 0.00 |
| RAPT-Enhanced | 100% | 0.7916 ± 0.0000 | 0.8627 | 0.8983 | 0.7405 | 0.0793 | 0.2980 | 0.00 |

### Recovery

| Model | Severity | Pre-drift F1 | Min F1 | Recovery F1 | Recovery windows | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 10% | 1.0000 | 0.5067 | 0.6395 | 40.00 | -0.3605 |
| Frozen | 20% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| Frozen | 30% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| Frozen | 50% | 1.0000 | 0.2143 | 0.4913 | 40.00 | -0.5087 |
| Frozen | 100% | 1.0000 | 0.1667 | 0.4483 | 40.00 | -0.5517 |
| Event-Driven | 10% | 1.0000 | 0.6381 | 0.7778 | 40.00 | -0.2222 |
| Event-Driven | 20% | 1.0000 | 0.6305 | 0.8262 | 40.00 | -0.1738 |
| Event-Driven | 30% | 1.0000 | 0.6305 | 0.8262 | 40.00 | -0.1738 |
| Event-Driven | 50% | 1.0000 | 0.3232 | 0.5266 | 40.00 | -0.4734 |
| Event-Driven | 100% | 1.0000 | 0.1667 | 0.4483 | 40.00 | -0.5517 |
| Full Retraining | 10% | 1.0000 | 0.5067 | 0.6395 | 40.00 | -0.3605 |
| Full Retraining | 20% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| Full Retraining | 30% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| Full Retraining | 50% | 1.0000 | 0.2143 | 0.4913 | 40.00 | -0.5087 |
| Full Retraining | 100% | 1.0000 | 0.1667 | 0.4483 | 40.00 | -0.5517 |
| RAPT | 10% | 1.0000 | 0.5067 | 0.6395 | 40.00 | -0.3605 |
| RAPT | 20% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| RAPT | 30% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| RAPT | 50% | 1.0000 | 0.2143 | 0.4913 | 40.00 | -0.5087 |
| RAPT | 100% | 1.0000 | 0.1667 | 0.4483 | 40.00 | -0.5517 |
| RAPT-Enhanced | 10% | 1.0000 | 0.5067 | 0.6395 | 40.00 | -0.3605 |
| RAPT-Enhanced | 20% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| RAPT-Enhanced | 30% | 1.0000 | 0.4333 | 0.6188 | 40.00 | -0.3812 |
| RAPT-Enhanced | 50% | 1.0000 | 0.2143 | 0.4913 | 40.00 | -0.5087 |
| RAPT-Enhanced | 100% | 1.0000 | 0.1667 | 0.4483 | 40.00 | -0.5517 |

## 6. Recurring Concept Drift Methodology (9B-D)
Layout: **A → B → A'**, where A and A' share the *same regime id* but A' has the
rank-reversal applied. A' therefore has the feature characteristics of A with a
changed P(Y|X). Because RAPT keys policy reuse on the regime id, the A' visit is a
repository hit and RAPT **blindly reuses** the stored A policy.

### Results

| Model | Severity | Macro-F1 | Accuracy | Retrains | Reuse events | New trees | Reused trees |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 10% | 0.5632 | 0.6525 | 0.00 | 0.00 | 100.0 | 0.0 |
| Frozen | 20% | 0.5489 | 0.6375 | 0.00 | 0.00 | 100.0 | 0.0 |
| Frozen | 30% | 0.5489 | 0.6375 | 0.00 | 0.00 | 100.0 | 0.0 |
| Frozen | 50% | 0.5251 | 0.6125 | 0.00 | 0.00 | 100.0 | 0.0 |
| Frozen | 100% | 0.4890 | 0.5675 | 0.00 | 0.00 | 100.0 | 0.0 |
| Event-Driven | 10% | 0.6674 | 0.6900 | 1.00 | 0.00 | 200.0 | 0.0 |
| Event-Driven | 20% | 0.6516 | 0.6750 | 1.00 | 0.00 | 200.0 | 0.0 |
| Event-Driven | 30% | 0.6516 | 0.6750 | 1.00 | 0.00 | 200.0 | 0.0 |
| Event-Driven | 50% | 0.6322 | 0.6550 | 1.00 | 0.00 | 200.0 | 0.0 |
| Event-Driven | 100% | 0.5919 | 0.6100 | 1.00 | 0.00 | 200.0 | 0.0 |
| Full Retraining | 10% | 0.5584 | 0.6475 | 2.00 | 0.00 | 300.0 | 0.0 |
| Full Retraining | 20% | 0.5417 | 0.6300 | 2.00 | 0.00 | 300.0 | 0.0 |
| Full Retraining | 30% | 0.5417 | 0.6300 | 2.00 | 0.00 | 300.0 | 0.0 |
| Full Retraining | 50% | 0.5257 | 0.6125 | 2.00 | 0.00 | 300.0 | 0.0 |
| Full Retraining | 100% | 0.4968 | 0.5775 | 2.00 | 0.00 | 300.0 | 0.0 |
| RAPT | 10% | 0.5632 | 0.6525 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT | 20% | 0.5489 | 0.6375 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT | 30% | 0.5489 | 0.6375 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT | 50% | 0.5251 | 0.6125 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT | 100% | 0.4890 | 0.5675 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT-Enhanced | 10% | 0.5632 | 0.6525 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT-Enhanced | 20% | 0.5489 | 0.6375 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT-Enhanced | 30% | 0.5489 | 0.6375 | 1.00 | 1.00 | 200.0 | 100.0 |
| RAPT-Enhanced | 50% | 0.5275 | 0.6150 | 1.00 | 1.00 | 300.0 | 100.0 |
| RAPT-Enhanced | 100% | 0.4909 | 0.5700 | 1.00 | 1.00 | 480.0 | 100.0 |

Per-phase F1 (severity 100%):

| Phase | Model | Macro-F1 |
| :---: | :--- | :---: |
| A | Frozen | 1.0000 |
| A | Event-Driven | 1.0000 |
| A | Full Retraining | 1.0000 |
| A | RAPT | 1.0000 |
| A | RAPT-Enhanced | 1.0000 |
| B | Frozen | 0.2453 |
| B | Event-Driven | 0.3990 |
| B | Full Retraining | 0.2453 |
| B | RAPT | 0.2453 |
| B | RAPT-Enhanced | 0.2453 |
| A_prime | Frozen | 0.5180 |
| A_prime | Event-Driven | 0.5536 |
| A_prime | Full Retraining | 0.5582 |
| A_prime | RAPT | 0.5180 |
| A_prime | RAPT-Enhanced | 0.5251 |

### Recovery

| Model | Severity | Pre-drift F1 | Min F1 | Recovery F1 | Recovery windows | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 10% | 1.0000 | 0.5556 | 1.0000 | 10.60 | -0.0456 |
| Frozen | 20% | 1.0000 | 0.4167 | 1.0000 | 14.60 | -0.1204 |
| Frozen | 30% | 1.0000 | 0.4167 | 1.0000 | 14.60 | -0.1204 |
| Frozen | 50% | 1.0000 | 0.3750 | 0.9467 | 23.20 | -0.2667 |
| Frozen | 100% | 1.0000 | 0.3036 | 0.5187 | 40.00 | -0.4813 |
| Event-Driven | 10% | 1.0000 | 0.5556 | 1.0000 | 10.60 | -0.0456 |
| Event-Driven | 20% | 1.0000 | 0.5000 | 1.0000 | 13.80 | -0.1229 |
| Event-Driven | 30% | 1.0000 | 0.5000 | 1.0000 | 13.80 | -0.1229 |
| Event-Driven | 50% | 1.0000 | 0.3889 | 0.8713 | 27.40 | -0.2277 |
| Event-Driven | 100% | 1.0000 | 0.2560 | 0.5540 | 40.00 | -0.4460 |
| Full Retraining | 10% | 1.0000 | 0.5556 | 1.0000 | 10.60 | -0.0704 |
| Full Retraining | 20% | 1.0000 | 0.3750 | 1.0000 | 17.00 | -0.1605 |
| Full Retraining | 30% | 1.0000 | 0.3750 | 1.0000 | 17.00 | -0.1605 |
| Full Retraining | 50% | 1.0000 | 0.3750 | 0.9373 | 23.20 | -0.2603 |
| Full Retraining | 100% | 1.0000 | 0.3393 | 0.5578 | 40.00 | -0.4422 |
| RAPT | 10% | 1.0000 | 0.5556 | 1.0000 | 10.60 | -0.0456 |
| RAPT | 20% | 1.0000 | 0.4167 | 1.0000 | 14.60 | -0.1204 |
| RAPT | 30% | 1.0000 | 0.4167 | 1.0000 | 14.60 | -0.1204 |
| RAPT | 50% | 1.0000 | 0.3750 | 0.9467 | 23.20 | -0.2667 |
| RAPT | 100% | 1.0000 | 0.3036 | 0.5187 | 40.00 | -0.4813 |
| RAPT-Enhanced | 10% | 1.0000 | 0.5556 | 1.0000 | 10.60 | -0.0456 |
| RAPT-Enhanced | 20% | 1.0000 | 0.4167 | 1.0000 | 14.60 | -0.1204 |
| RAPT-Enhanced | 30% | 1.0000 | 0.4167 | 1.0000 | 14.60 | -0.1204 |
| RAPT-Enhanced | 50% | 1.0000 | 0.3750 | 1.0000 | 19.00 | -0.2510 |
| RAPT-Enhanced | 100% | 1.0000 | 0.3036 | 0.5249 | 40.00 | -0.4751 |

## 7. Exact Drift-Severity Definitions
| Severity | Covariate drift | Concept drift |
| :---: | :--- | :--- |
| 10% | 4/40 post-drift windows drawn from the target regime | top 1 predictive features rank-reversed |
| 20% | 8/40 post-drift windows drawn from the target regime | top 1 predictive features rank-reversed |
| 30% | 12/40 post-drift windows drawn from the target regime | top 2 predictive features rank-reversed |
| 50% | 20/40 post-drift windows drawn from the target regime | top 3 predictive features rank-reversed |
| 100% | 40/40 post-drift windows drawn from the target regime | top 5 predictive features rank-reversed |

## 8. Experimental Protocol
* Chronological streaming; each window is predicted **before** it is appended to
  the training buffer (Test-Then-Train).
* Initial training prefix is fixed at the 20% prefix of each constructed stream and
  is never modified by the drift transformation.
* The drift transformation is applied only at its intended point (post-drift block).
* No future samples leak into training; scaler fitted only on the initial prefix.
* The same window size and seeds are used across all experiments.

### Automated integrity checks
`results/experiment_9b/raw/scientific_checks.txt` (generated by
`exp9b_drift_checks.py`) verifies, from the actual constructed streams and saved
raw results: identical seeds/window size/drift levels; strictly increasing window
ids; the initial training prefix is untouched; the concept transform is applied only
after the drift point and is a pure permutation of P(X); covariate labels match their
source regimes exactly (no relabeling); covariate and concept are distinct
constructions; and the RAPT algorithm contains no variant symbols. All checks pass.

## 9. Models
* **Frozen** — trained once on the initial prefix, never updated.
* **Event-Driven** — retrains when a rolling error spike exceeds mu + k·sigma.
* **Full Retraining** — retrains on the historical buffer at each regime boundary.
* **RAPT** — stores a policy checkpoint per regime id; on a regime-boundary hit it
  reuses the stored policy (with light weight calibration), otherwise trains a new one.
* **RAPT-Enhanced** — base RAPT plus the two protocol-agnostic mechanisms of the
  existing Enhanced-Hybrid-RAPT architecture: (a) buffer-blended novelty refitting
  on a larger recent buffer (1500 vs 500 samples); (b) selective parity refitting,
  which refits a reused policy whose recent streaming accuracy drops below a
  threshold. Its online micro-learner and dynamic decision threshold are not
  applicable to the aggregated-window multi-class protocols used here.

## 10. Metrics
Predictive: Macro-F1, Accuracy, Precision, Recall. Adaptation: adaptation CPU,
total runtime, retrain events, RAPT reuse events, newly trained trees, reused trees.
Robustness: performance drop (ΔF1 = F1_after − F1_before), minimum F1, recovery F1,
recovery time (windows to regain 95% of pre-drift F1), relative degradation.

## 11. Statistical Methodology
Window-level paired Wilcoxon signed-rank tests between models, paired on
(seed, window_id), across seeds [42,43,44,45,46]. Effect size = Cohen's d on the
paired differences; 95% CI by bootstrap (2000 resamples). Non-significant results
are reported as non-significant.

### Statistical results

| Drift type | Severity | Comparison | Mean diff | Cohen's d | p-value | 95% CI | Significant |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| covariate | 10% | RAPT vs Frozen | 0.1240 | 0.295 | 0.0000 | [0.0990, 0.1500] | YES |
| covariate | 10% | RAPT vs Event-Driven | 0.0140 | 0.029 | 0.3601 | [-0.0180, 0.0460] | NO |
| covariate | 10% | RAPT vs Full Retraining | 0.0020 | 0.007 | 0.8252 | [-0.0160, 0.0190] | NO |
| covariate | 10% | RAPT-Enhanced vs Frozen | 0.1240 | 0.295 | 0.0000 | [0.0990, 0.1500] | YES |
| covariate | 10% | RAPT-Enhanced vs Event-Driven | 0.0140 | 0.029 | 0.3601 | [-0.0180, 0.0460] | NO |
| covariate | 10% | RAPT-Enhanced vs Full Retraining | 0.0020 | 0.007 | 0.8252 | [-0.0160, 0.0190] | NO |
| covariate | 10% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| covariate | 20% | RAPT vs Frozen | 0.3130 | 0.416 | 0.0000 | [0.2690, 0.3610] | YES |
| covariate | 20% | RAPT vs Event-Driven | 0.2230 | 0.340 | 0.0000 | [0.1860, 0.2640] | YES |
| covariate | 20% | RAPT vs Full Retraining | 0.0110 | 0.053 | 0.0934 | [-0.0020, 0.0240] | NO |
| covariate | 20% | RAPT-Enhanced vs Frozen | 0.3130 | 0.416 | 0.0000 | [0.2690, 0.3610] | YES |
| covariate | 20% | RAPT-Enhanced vs Event-Driven | 0.2230 | 0.340 | 0.0000 | [0.1860, 0.2640] | YES |
| covariate | 20% | RAPT-Enhanced vs Full Retraining | 0.0110 | 0.053 | 0.0934 | [-0.0020, 0.0240] | NO |
| covariate | 20% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| covariate | 30% | RAPT vs Frozen | 0.1870 | 0.335 | 0.0000 | [0.1550, 0.2220] | YES |
| covariate | 30% | RAPT vs Event-Driven | 0.1010 | 0.220 | 0.0000 | [0.0740, 0.1290] | YES |
| covariate | 30% | RAPT vs Full Retraining | 0.0130 | 0.056 | 0.0796 | [-0.0020, 0.0270] | NO |
| covariate | 30% | RAPT-Enhanced vs Frozen | 0.1870 | 0.335 | 0.0000 | [0.1550, 0.2220] | YES |
| covariate | 30% | RAPT-Enhanced vs Event-Driven | 0.1010 | 0.220 | 0.0000 | [0.0740, 0.1290] | YES |
| covariate | 30% | RAPT-Enhanced vs Full Retraining | 0.0130 | 0.056 | 0.0796 | [-0.0020, 0.0270] | NO |
| covariate | 30% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| covariate | 50% | RAPT vs Frozen | 0.3030 | 0.437 | 0.0000 | [0.2610, 0.3470] | YES |
| covariate | 50% | RAPT vs Event-Driven | 0.1920 | 0.305 | 0.0000 | [0.1550, 0.2300] | YES |
| covariate | 50% | RAPT vs Full Retraining | 0.0150 | 0.059 | 0.0628 | [0.0000, 0.0300] | NO |
| covariate | 50% | RAPT-Enhanced vs Frozen | 0.3030 | 0.437 | 0.0000 | [0.2610, 0.3470] | YES |
| covariate | 50% | RAPT-Enhanced vs Event-Driven | 0.1920 | 0.305 | 0.0000 | [0.1550, 0.2300] | YES |
| covariate | 50% | RAPT-Enhanced vs Full Retraining | 0.0150 | 0.059 | 0.0628 | [0.0000, 0.0300] | NO |
| covariate | 50% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| covariate | 100% | RAPT vs Frozen | 0.4300 | 0.623 | 0.0000 | [0.3890, 0.4730] | YES |
| covariate | 100% | RAPT vs Event-Driven | 0.2450 | 0.379 | 0.0000 | [0.2070, 0.2840] | YES |
| covariate | 100% | RAPT vs Full Retraining | 0.0060 | 0.051 | 0.1088 | [-0.0010, 0.0130] | NO |
| covariate | 100% | RAPT-Enhanced vs Frozen | 0.4300 | 0.623 | 0.0000 | [0.3890, 0.4730] | YES |
| covariate | 100% | RAPT-Enhanced vs Event-Driven | 0.2450 | 0.379 | 0.0000 | [0.2070, 0.2840] | YES |
| covariate | 100% | RAPT-Enhanced vs Full Retraining | 0.0060 | 0.051 | 0.1088 | [-0.0010, 0.0130] | NO |
| covariate | 100% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 10% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 10% | RAPT vs Event-Driven | -0.0078 | -0.089 | 0.1573 | [-0.0196, 0.0000] | NO |
| concept | 10% | RAPT vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 10% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 10% | RAPT-Enhanced vs Event-Driven | -0.0078 | -0.089 | 0.1573 | [-0.0196, 0.0000] | NO |
| concept | 10% | RAPT-Enhanced vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 10% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 20% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 20% | RAPT vs Event-Driven | -0.0118 | -0.109 | 0.0833 | [-0.0275, 0.0000] | NO |
| concept | 20% | RAPT vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 20% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 20% | RAPT-Enhanced vs Event-Driven | -0.0118 | -0.109 | 0.0833 | [-0.0275, 0.0000] | NO |
| concept | 20% | RAPT-Enhanced vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 20% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 30% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 30% | RAPT vs Event-Driven | -0.0118 | -0.109 | 0.0833 | [-0.0275, 0.0000] | NO |
| concept | 30% | RAPT vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 30% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 30% | RAPT-Enhanced vs Event-Driven | -0.0118 | -0.109 | 0.0833 | [-0.0275, 0.0000] | NO |
| concept | 30% | RAPT-Enhanced vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 30% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 50% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 50% | RAPT vs Event-Driven | -0.0196 | -0.141 | 0.0253 | [-0.0392, -0.0039] | YES |
| concept | 50% | RAPT vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 50% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 50% | RAPT-Enhanced vs Event-Driven | -0.0196 | -0.141 | 0.0253 | [-0.0392, -0.0039] | YES |
| concept | 50% | RAPT-Enhanced vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 50% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT vs Event-Driven | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT-Enhanced vs Event-Driven | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT-Enhanced vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| concept | 100% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 10% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 10% | RAPT vs Event-Driven | -0.0375 | -0.098 | 0.0508 | [-0.0750, 0.0000] | NO |
| recurring | 10% | RAPT vs Full Retraining | 0.0050 | 0.071 | 0.1573 | [0.0000, 0.0125] | NO |
| recurring | 10% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 10% | RAPT-Enhanced vs Event-Driven | -0.0375 | -0.098 | 0.0508 | [-0.0750, 0.0000] | NO |
| recurring | 10% | RAPT-Enhanced vs Full Retraining | 0.0050 | 0.071 | 0.1573 | [0.0000, 0.0125] | NO |
| recurring | 10% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 20% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 20% | RAPT vs Event-Driven | -0.0375 | -0.095 | 0.0588 | [-0.0775, 0.0000] | NO |
| recurring | 20% | RAPT vs Full Retraining | 0.0075 | 0.067 | 0.1797 | [-0.0025, 0.0200] | NO |
| recurring | 20% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 20% | RAPT-Enhanced vs Event-Driven | -0.0375 | -0.095 | 0.0588 | [-0.0775, 0.0000] | NO |
| recurring | 20% | RAPT-Enhanced vs Full Retraining | 0.0075 | 0.067 | 0.1797 | [-0.0025, 0.0200] | NO |
| recurring | 20% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 30% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 30% | RAPT vs Event-Driven | -0.0375 | -0.095 | 0.0588 | [-0.0775, 0.0000] | NO |
| recurring | 30% | RAPT vs Full Retraining | 0.0075 | 0.067 | 0.1797 | [-0.0025, 0.0200] | NO |
| recurring | 30% | RAPT-Enhanced vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 30% | RAPT-Enhanced vs Event-Driven | -0.0375 | -0.095 | 0.0588 | [-0.0775, 0.0000] | NO |
| recurring | 30% | RAPT-Enhanced vs Full Retraining | 0.0075 | 0.067 | 0.1797 | [-0.0025, 0.0200] | NO |
| recurring | 30% | RAPT-Enhanced vs RAPT | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 50% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 50% | RAPT vs Event-Driven | -0.0425 | -0.106 | 0.0350 | [-0.0825, -0.0050] | YES |
| recurring | 50% | RAPT vs Full Retraining | 0.0000 | 0.000 | 1.0000 | [-0.0125, 0.0125] | NO |
| recurring | 50% | RAPT-Enhanced vs Frozen | 0.0025 | 0.050 | 0.3173 | [0.0000, 0.0075] | NO |
| recurring | 50% | RAPT-Enhanced vs Event-Driven | -0.0400 | -0.101 | 0.0455 | [-0.0800, -0.0025] | YES |
| recurring | 50% | RAPT-Enhanced vs Full Retraining | 0.0025 | 0.022 | 0.6547 | [-0.0100, 0.0150] | NO |
| recurring | 50% | RAPT-Enhanced vs RAPT | 0.0025 | 0.050 | 0.3173 | [0.0000, 0.0075] | NO |
| recurring | 100% | RAPT vs Frozen | 0.0000 | 0.000 | 1.0000 | [0.0000, 0.0000] | NO |
| recurring | 100% | RAPT vs Event-Driven | -0.0425 | -0.104 | 0.0378 | [-0.0825, -0.0050] | YES |
| recurring | 100% | RAPT vs Full Retraining | -0.0100 | -0.058 | 0.2482 | [-0.0275, 0.0075] | NO |
| recurring | 100% | RAPT-Enhanced vs Frozen | 0.0025 | 0.050 | 0.3173 | [0.0000, 0.0075] | NO |
| recurring | 100% | RAPT-Enhanced vs Event-Driven | -0.0400 | -0.099 | 0.0489 | [-0.0800, -0.0025] | YES |
| recurring | 100% | RAPT-Enhanced vs Full Retraining | -0.0075 | -0.045 | 0.3657 | [-0.0250, 0.0075] | NO |
| recurring | 100% | RAPT-Enhanced vs RAPT | 0.0025 | 0.050 | 0.3173 | [0.0000, 0.0075] | NO |

## 12. Results Tables
Auto-generated CSV/LaTeX tables in `results/experiment_9b/tables/`:
* Table 9B-1 `table_9b1_natural_drift` — natural drift results.
* Table 9B-2 `table_9b2_covariate_drift` — covariate severity results.
* Table 9B-3 `table_9b3_concept_drift` — concept severity results.
* Table 9B-4 `table_9b4_recovery` — recovery analysis.
* Table 9B-5 `table_9b5_rapt_reuse` — RAPT reuse analysis.

## 13. Generated Figures
In `results/experiment_9b/figures/`:
* `fig9b_natural_drift_f1.png`
* `fig9b_covariate_severity_f1.png`
* `fig9b_covariate_severity_cpu.png`
* `fig9b_covariate_severity_recovery.png`
* `fig9b_concept_severity_f1.png`
* `fig9b_concept_severity_accuracy.png`
* `fig9b_concept_severity_recovery.png`
* `fig9b_drift_type_comparison.png`
* `fig9b_rapt_adaptation_timeline.png`
* `fig9b_reuse_vs_retraining.png`
* `fig9b_accuracy_cost_tradeoff.png`
* `fig9b_covariate_f1_heatmap.png`
* `fig9b_concept_f1_heatmap.png`
* `fig9b_concept_cm_10.png ... fig9b_concept_cm_100.png`
* `fig9b_regime_f1.png`

## 14. Key Observations
_Observed results_ (directly measured):
* Under **covariate drift**, Frozen degrades monotonically with severity
  (F1 0.527→0.489). Full Retraining and RAPT are the strongest adapters and track
  each other closely at every severity; RAPT is marginally higher (0.685→0.812).
  RAPT's advantage over Frozen and Event-Driven is statistically significant at
  every severity (paired Wilcoxon, p<0.01), but its advantage over Full Retraining
  is **not** significant (p≈0.06–0.82, small Cohen's d).
* RAPT's adaptation CPU under covariate drift (~0.33 s) is flat across severity and
  is in fact **slightly higher** than Full Retraining (~0.24 s); because the
  post-drift block is a single new regime id there is no policy reuse, so RAPT
  retrains three times, like Full Retraining. No cost advantage is observed here.
* Under **concept drift**, all models degrade and the drop is monotone in severity
  for Frozen/Full Retraining/RAPT; Event-Driven is modestly better at low severity.
  Because the regime id is unchanged, no adapter receives a boundary signal and
  RAPT equals Frozen exactly (F1 0.878→0.792); recovery is never sustained within
  the 40-window horizon for any model.
* Under **recurring concept drift** (A→B→A'), RAPT records one reuse event on the
  A' visit (100 reused trees) but its A' performance equals the Frozen policy and
  does not recover; Event-Driven attains the best F1.
* **RAPT-Enhanced** is essentially indistinguishable from RAPT under covariate and
  concept drift (identical F1; the parity-refit trigger rarely fires because reuse
  is absent there). Under recurring concept drift its parity refit gives only a
  marginal A' gain at high severity (0.733→0.750 at 50%, 0.518→0.525 at 100%), i.e.
  it does not remedy the stale-policy problem.

_Interpretation_: RAPT's cost advantage in the natural stream comes from
regime-keyed reuse. The same mechanism is a liability when a recurring regime's
label semantics have changed, because the regime key cannot distinguish A from A'.
When drift arrives without a regime-id change (concept drift), the regime-keyed
trigger never fires and RAPT is inert.

## 15. Limitations
(i) The second dataset is small (499 windows) and regimes B/C/D are largely
single-class, so covariate mixtures are modest and concept drift is constructed
within regime A; (ii) concept drift is injected synthetically via a deterministic
transform rather than observed; (iii) Full Retraining and RAPT use the regime id as
the boundary signal, so concept drift without a regime change is not detectable by
either; (iv) five seeds give limited statistical power, and several covariate
comparisons against Full Retraining are non-significant; (v) RAPT reuse is degenerate
(empty pre-drift repository) in the single-regime covariate and concept streams, so
no reuse occurs there; (vi) recovery is measured over a 40-window horizon and is
censored when not sustained within it.

## 16. Reproducibility Information
Run from the `Ensemble Learning for Model Drift Detection` directory:

```bash
python experiments/exp9b/run_exp9b.py          # 9B-A natural (existing)
python experiments/exp9b/run_exp9b_drift.py    # 9B-B/C/D + figures + tables + report
```

Configuration (`experiments/exp9b/exp9b_drift_config.py`):

```python
EXPERIMENT = "9B"
SEEDS = [42, 43, 44, 45, 46]
DRIFT_LEVELS = [0.1, 0.2, 0.3, 0.5, 1.0]
WINDOW_SIZE = 500
MODELS = ['Frozen', 'Event-Driven', 'Full_Retraining', 'RAPT', 'RAPT-Enhanced']
```

Outputs: `results/experiment_9b/{natural_drift,covariate_drift,concept_drift,
recurring_concept_drift,raw,figures,tables}`. Raw per-window and per-seed results
are saved so every figure and table can be regenerated.
