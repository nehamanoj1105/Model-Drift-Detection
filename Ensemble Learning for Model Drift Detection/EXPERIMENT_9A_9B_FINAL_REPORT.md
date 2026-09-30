# EXPERIMENTS 9A & 9B — FINAL CROSS-DATASET REPORT

Cross-dataset evaluation of the RAPT policy-transfer mechanism on two
independent recurring-regime streams. The proposed model is referred to simply
as **RAPT**. RAPT-Enhanced applies the existing Enhanced-Hybrid-RAPT
mechanisms; no new algorithm variant is introduced and RAPT itself is
unmodified.

## 1. Datasets and Provenance

| Exp | Dataset | Source | Classes | Windows | Window size | Initial train |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 9A | INSECTS incremental-reoccurring | river.datasets.Insects (USP DS; Souza et al. 2020) | 6 | 150 | 500 | 30 |
| 9B | 5G NR end-to-end latency QoS | Zenodo 10.5281/zenodo.20035549 | 3 | 499 | 500 | 99 |

9A stream SHA-256: `c5e2d1f71260bbdeff1fc9e26b4a8b9311215bd176bb939eeaad994b996796ad`. 9A regime sequence: `regime_12,regime_11,regime_2,regime_5,regime_3,regime_2,regime_4,regime_5,regime_12,regime_11,regime_2` (11 visits).
9B regime sequence: `ABCDACBDAB`.

## 2. Experiment 9A — Revised Setup

* Protocol: sequential chronological, Test-Then-Train (prequential); each
  window is fully predicted before it is appended to any adaptation buffer.
* Preprocessing: standardisation fitted only on the initial prefix (no leakage).
* Base model: RandomForest + ExtraTrees soft-voting ensemble (50 trees each,
  max depth 7). All adaptive models retrain on a class-anchored buffer so every
  class stays representable (identical guard for all models).
* Seeds: [42, 43, 44, 45, 46]. Window size: 500 samples.

### 9A results (mean ± std across seeds)

| Model | Macro-F1 | Accuracy | Adapt CPU (s) | Retrains | Reuse events | Trees reused |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen | 0.2374 ± 0.0011 | 0.3431 | 0.000 | 0 | 0 | 0 |
| Event-Driven | 0.2223 ± 0.0026 | 0.3523 | 0.788 | 4 | 0 | 0 |
| Full_Retraining | 0.3721 ± 0.0022 | 0.4991 | 1.759 | 9 | 0 | 0 |
| RAPT | 0.2972 ± 0.0036 | 0.4306 | 0.711 | 5 | 4 | 400 |
| RAPT-Enhanced | 0.3707 ± 0.0032 | 0.5047 | 2.217 | 5 | 4 | 400 |

_Paired Wilcoxon (window-level, all seeds):_

| Comparison | Mean Δ | Cohen's d | p | Significant |
| :--- | :---: | :---: | :---: | :---: |
| RAPT vs Event-Driven | +0.0782 | +0.212 | 4.543e-07 | YES |
| RAPT vs Frozen | +0.0875 | +0.292 | 3.277e-06 | YES |
| RAPT vs Full_Retraining | -0.0686 | -0.229 | 2.017e-05 | YES |
| RAPT-Enhanced vs Event-Driven | +0.1524 | +0.400 | 1.423e-21 | YES |
| RAPT-Enhanced vs Full_Retraining | +0.0056 | +0.023 | 0.2701 | NO |
| RAPT-Enhanced vs RAPT | +0.0741 | +0.370 | 0.008792 | YES |

## 3. Experiment 9B — Setup (preserved)

9B keeps the existing 5G-latency protocol: same window size (500), same 20%
initial prefix, same Test-Then-Train protocol, same seeds. The natural stream
(9B-A) is evaluated with the shared five-model harness; 9B-B/C/D add
controlled covariate, concept and recurring-concept drift at severities
[0.10, 0.20, 0.30, 0.50, 1.00] (see `EXPERIMENT_9B_FINAL_REPORT.md`).

### 9B natural-stream results (mean ± std across seeds)

| Model | Macro-F1 | Accuracy | Adapt CPU (s) | Retrains | Reuse events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Frozen | 0.8961 ± 0.0099 | 0.9075 | 0.000 | 0 | 0 |
| Event-Driven | 0.8903 ± 0.0107 | 0.9055 | 1.041 | 11 | 0 |
| Full Retraining | 0.9027 ± 0.0059 | 0.9145 | 0.764 | 9 | 0 |
| RAPT | 0.8894 ± 0.0110 | 0.9025 | 0.396 | 3 | 6 |
| RAPT-Enhanced | 0.8915 ± 0.0117 | 0.9055 | 0.861 | 3 | 6 |

## 4. Cross-Dataset Comparison

Aggregate metrics on both streams (same models, same protocol):

| Dataset | Model | Macro-F1 | Accuracy | Adapt CPU (s) | Retrains | Reuse events |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 9A | Frozen | 0.2374 ± 0.0011 | 0.3431 | 0.000 | 0 | 0 |
| 9A | Event-Driven | 0.2223 ± 0.0026 | 0.3523 | 0.788 | 4 | 0 |
| 9A | Full_Retraining | 0.3721 ± 0.0022 | 0.4991 | 1.759 | 9 | 0 |
| 9A | RAPT | 0.2972 ± 0.0036 | 0.4306 | 0.711 | 5 | 4 |
| 9A | RAPT-Enhanced | 0.3707 ± 0.0032 | 0.5047 | 2.217 | 5 | 4 |
| 9B | Frozen | 0.8961 ± 0.0099 | 0.9075 | 0.000 | 0 | 0 |
| 9B | Event-Driven | 0.8903 ± 0.0107 | 0.9055 | 1.041 | 11 | 0 |
| 9B | Full_Retraining | 0.9027 ± 0.0059 | 0.9145 | 0.764 | 9 | 0 |
| 9B | RAPT | 0.8894 ± 0.0110 | 0.9025 | 0.396 | 3 | 6 |
| 9B | RAPT-Enhanced | 0.8915 ± 0.0117 | 0.9055 | 0.861 | 3 | 6 |

_Paired Wilcoxon (window-level, all seeds):_

| Dataset | Comparison | Mean Δ | Cohen's d | p | Significant |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 9A | RAPT vs Frozen | +0.0875 | +0.292 | 3.277e-06 | YES |
| 9A | RAPT vs Event-Driven | +0.0782 | +0.212 | 4.543e-07 | YES |
| 9A | RAPT vs Full_Retraining | -0.0686 | -0.229 | 2.017e-05 | YES |
| 9A | RAPT-Enhanced vs Frozen | +0.1616 | +0.465 | 1.355e-19 | YES |
| 9A | RAPT-Enhanced vs Event-Driven | +0.1524 | +0.400 | 1.423e-21 | YES |
| 9A | RAPT-Enhanced vs Full_Retraining | +0.0056 | +0.023 | 0.2701 | NO |
| 9A | RAPT-Enhanced vs RAPT | +0.0741 | +0.370 | 0.008792 | YES |
| 9B | RAPT vs Frozen | -0.0050 | -0.060 | 0.007526 | YES |
| 9B | RAPT vs Event-Driven | -0.0030 | -0.013 | 0.5637 | NO |
| 9B | RAPT vs Full Retraining | -0.0120 | -0.051 | 0.02334 | YES |
| 9B | RAPT-Enhanced vs Frozen | -0.0020 | -0.010 | 0.6464 | NO |
| 9B | RAPT-Enhanced vs Event-Driven | +0.0000 | +0.000 | 1 | NO |
| 9B | RAPT-Enhanced vs Full Retraining | -0.0090 | -0.047 | 0.0364 | YES |
| 9B | RAPT-Enhanced vs RAPT | +0.0030 | +0.017 | 0.4602 | NO |

## 5. Key Observations

_Observed (measured):_
* On **9A (INSECTS, 6-class, hard)** Frozen scores ~0.237 and Event-Driven
  ~0.222; Full Retraining reaches ~0.372, RAPT ~0.297 and RAPT-Enhanced
  ~0.371. RAPT is significantly better than Frozen and Event-Driven
  (p<0.01) but significantly worse than Full Retraining (Δ≈-0.069, p<0.01).
  RAPT is the cheapest adaptive model (~0.71 s vs 1.76 s for Full Retraining)
  and performs 4 policy reuses (400 reused trees).
* On **9B (5G latency, 3-class, easier)** all models are close (~0.889-0.903).
  RAPT has the lowest adaptation CPU (~0.40 s vs ~0.76 s Full Retraining) and
  6 reuse events (600 reused trees, 3 retrains). Its F1 penalty vs Full
  Retraining is statistically significant but small (Δ≈-0.012).
* **RAPT-Enhanced** recovers most of RAPT's accuracy gap on 9A (0.297→0.371),
  statistically indistinguishable from Full Retraining (p≈0.27), at higher CPU
  (~2.22 s). On 9B it is ~equal to RAPT in F1 with higher CPU.

_Statistical:_ RAPT's advantage over Frozen/Event-Driven on 9A and its
advantage over Frozen/Full Retraining in CPU are consistent across seeds. The
accuracy differences vs Full Retraining on 9B, while significant, are tiny in
effect size (|d|≈0.05); non-significant comparisons are reported as such.

_Interpretation:_ RAPT's regime-keyed reuse delivers a consistent adaptation-
cost advantage on both recurring streams. Whether that reuse costs accuracy is
**dataset-dependent**: negligible on 9B, but material on 9A, where a reused
policy is applied to regimes whose label distribution differs from the stored
checkpoint. RAPT-Enhanced's parity refit recovers much of that accuracy on 9A,
trading it for additional adaptation CPU.

## 6. Limitations

(i) 9A is a hard 6-class problem with window-level majority labels, so absolute
F1 is low for every model and the discriminating signal is the cost/reuse
behaviour rather than peak F1; (ii) the two streams differ in class count and
difficulty, so cross-dataset F1 levels are not directly comparable — only the
relative model ordering and reuse/cost behaviour are; (iii) five seeds give
limited power; (iv) RAPT reuse is keyed on regime id, so it cannot react to
label-semantics change within a regime (concept drift), as documented in the
9B report; (v) the online micro-learner and dynamic decision threshold of the
reference Enhanced architecture are not applicable to the aggregated-window
multi-class protocols used here and are therefore not evaluated.

## 7. Reproducibility

```text
results/
  experiment_9a/
    raw/            per_window_9a.csv, per_seed_9a.csv, summary_9a.csv,
                    statistics_9a.csv, transitions_9a.csv,
                    stream_definition_9a.json
    cross_dataset/  cross_dataset_summary.csv, cross_dataset_stats.csv
    figures/        fig_cross_dataset_*.png
    tables/         table_cross_dataset.csv/.tex
  experiment_9b/
    natural_drift/ covariate_drift/ concept_drift/ recurring_concept_drift/
    raw/  figures/  tables/
```
Commands (from the `Ensemble Learning for Model Drift Detection` directory):
```bash
python experiments/exp9a/run_exp9a.py            # 9A, 5 seeds
python experiments/exp9b/run_exp9b_drift.py      # 9B drift suite
python experiments/exp9a/cross_dataset_9a_9b.py  # cross-dataset figures/tables
python experiments/exp9a/write_combined_report.py
```
Environment: Python 3.13, scikit-learn, river 0.26.1, scipy, psutil. Seeds
[42, 43, 44, 45, 46]; window size 500; drift levels [0.10, 0.20, 0.30, 0.50,
1.00]. All raw per-window results are saved so every figure is reproducible.