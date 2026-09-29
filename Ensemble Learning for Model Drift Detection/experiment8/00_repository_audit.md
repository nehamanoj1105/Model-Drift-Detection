# State 0: Repository Audit for Experiment 8

## Executive Summary
This document provides a comprehensive audit of existing codebase components, baseline model implementations, drift detectors, data loaders, metrics, statistical routines, and seed handling. This forms the foundation for Experiment 8 (Value-Based Selective Event-Driven Ensemble Adaptation).

---

## 1. Codebase Structure & Component Inventory

### A. Core Model Definitions (`code so far/models.py`, `experiment5/models.py`)
- Base models:
  - **Random Forest (RF)**: `n_estimators=50`, `max_depth=7`, `min_samples_split=4`, `n_jobs=-1`. Temporal horizon: 3,500 samples.
  - **Extra Trees (ET)**: `n_estimators=50`, `max_depth=7`, `min_samples_split=6`, `n_jobs=-1`. Temporal horizon: 2,500 samples.
  - **Gradient Boosting (GB)**: `n_estimators=50`, `max_depth=4`, `learning_rate=0.08`, `subsample=0.85`. Temporal horizon: 1,800 samples.
- Key function: `create_candidate_models(seed)` returns a dictionary `{'RandomForest': RF, 'ExtraTrees': ET, 'GradientBoosting': GB}` initialized with the specific random seed.

### B. Baseline Ensemble Implementation (`code so far/ensemble.py`, `experiment5/ensemble.py`)
- `AdaptiveEnsemble` class:
  - Manages component models (`RF`, `ET`, `GB`).
  - Maintains multi-temporal buffers for each model.
  - Performs diversity-aware softmax weighting over component predictions:
    $$Score_i = \text{EMA\_F1}_i + \gamma \cdot \text{Diversity}_i$$
    $$w_i = \frac{\exp(\beta \cdot Score_i)}{\sum_j \exp(\beta \cdot Score_j)}$$
  - In existing event-driven baselines, when drift is detected, `adapt()` retrains ALL models on their respective historical buffers.

### C. Drift Detection (`code so far/drift.py`, `experiment5/drift.py`)
- Statistical feature drift monitoring via normalized Wasserstein distance ($\text{threshold} = 0.12$).
- Performance-based drift monitoring via rolling macro F1 degradation ($\text{degradation threshold} = 0.05$).
- Output flags trigger adaptation events in the prequential loop.

### D. Data Generators & Loaders
1. **Synthetic Telemetry Stream (`code so far/data_generation.py`):**
   - 10,000 total samples (2,000 initial stationary training + 8,000 deployment stream = 16 windows of 500 samples).
   - Features: `speed`, `distance`, `delay`, `throughput` + nonlinear interactions `speed*delay`, `delay*throughput`.
   - Balanced drift schedule per seed covering `none`, `covariate`, `concept`, `mixed` drift.
2. **SEA Recurring Concepts Benchmark (`experiment5/sea_stream_generator.py`):**
   - Standard 3-feature stream ($f_1, f_2, f_3 \in [0, 10]$).
   - Concepts defined by $f_1 + f_2 \le \theta$ ($\theta_A=7.0, \theta_B=10.0, \theta_C=13.0$) with 10% label noise.
3. **Real-World ToN_IoT Weather Dataset (`experiment5/data_loader.py`):**
   - Chronologically sorted telemetry with 20,000 initial training samples + streaming deployment windows.

### E. Evaluation & Metrics (`code so far/metrics.py`, `experiment5/metrics.py`)
- Functions compute: `accuracy`, `precision`, `recall`, `f1` (macro average), `roc_auc`, `confusion_matrix`.
- Execution timing: CPU user time, system time, wall-clock time via `resource_monitor.py`.

### F. Statistical Testing (`code so far/statistics.py`)
- Wilcoxon signed-rank tests for paired window-level observations.
- 95% Confidence Intervals via bootstrap / parametric methods.
- Holm-Bonferroni correction for multiple hypothesis testing.

### G. Random Seed Control
- Established experimental seeds: `[42, 43, 44, 45, 46]`.
- All baseline and proposed methods use identical seeds per run.

---

## 2. Identified Baseline & Baseline Reproduction Protocol

The established event-driven baseline operates as follows:
```text
At window t:
1. Receive X_t, generate predictions using current ensemble state.
2. Receive y_t, evaluate macro F1 and metric performance.
3. Compute drift metrics (Wasserstein distance on X_t vs X_ref or performance drop).
4. If drift detected:
     Retrain ALL 3 models (RF, ET, GB) on their historical horizon buffers.
   Else:
     KEEP current models.
5. Update ensemble weights via diversity-aware softmax.
```

---

## 3. Scope of Experiment 8 Extensions

To move from full retraining to value-based selective adaptation:
1. **Per-Model Actions:** Define discrete actions `KEEP` ($C=0$), `PARTIAL` ($C=C_{\text{partial}}$), `FULL` ($C=C_{\text{full}}$).
2. **Probing:** Run a quick sub-sampled adaptation probe on $X_t$ to estimate gain $\hat{\delta}_{i,t}$.
3. **Gain Estimators:** Compare linear model, exponential diminishing-return model ($\hat{G}(c) = \alpha(1-e^{-\beta c})$), and empirical interpolation.
4. **Action Selection:** Enumerate all 27 action combinations for $M=3$ and optimize $\sum_i \hat{G}_i(a_i)$ under budget $\sum_i C_i(a_i) \le B$.
5. **Baselines & Ablations:** Frozen, Continuous, Event-Driven (Full), Random Selective, Weakest-Model, Equal-Budget, Oracle Allocation, Proposed Value-Based Allocation.
6. **Adaptation Difficulty Regimes:** Easy, Medium, Hard recovery regimes to test hypothesis dynamics.

---

## 4. Audit Validation Gate Checklist
- [x] Repository structure and file locations verified.
- [x] Core model hyperparameters and temporal horizons documented.
- [x] Reference baseline event-driven logic mapped.
- [x] Data generators and dataset schemas identified.
- [x] Prequential protocol constraints ($X_t$ pred -> $y_t$ eval -> drift -> adapt) verified.
- [x] Seed set `[42, 43, 44, 45, 46]` confirmed.

STATE 0 IS PASSED. Ready to create `experiment8/` folder structure and progress to State 1.
