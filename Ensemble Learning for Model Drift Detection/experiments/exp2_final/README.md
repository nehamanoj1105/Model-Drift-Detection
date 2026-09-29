# Experiment 2: Probability-Guided Regime Transfer for Drift-Adaptive Ensembles

## Overview & Goal
This study evaluates whether online probability estimation can reliably guide regime policy transfer in non-stationary data streams. At each decision point t, historical checkpoints c_1, ..., c_m (one per past regime) are candidate policies. The method uses an online rolling logistic regression classifier to predict P(positive transfer | features). If max_c P >= tau_t, the highest-probability candidate is transferred; otherwise, the system abstains and defaults to online local retraining.

## System Architecture & Data Flow

### 1. Data Stream Partitioning & Decision Intervals
- A data stream consists of consecutive instances partitioned into windows of size W.
- Regime blocks are fixed in advance and never derived from method outputs.
- At decision point t, the method evaluates the next decision interval I_t = [t, t+K) covering K windows.
- Information available at decision point t: features of recent windows [t-2K, t) and labels of all windows prior to t.
- Labels for interval I_t arrive strictly AFTER interval I_t has executed (no target/eval leakage).

### 2. Candidate Checkpoints & Baseline Policies
- When regime block b completes (all its windows precede t), candidate c_b is trained on X_init + block_b (Policy B) and stored in the candidate pool.
- Each candidate evaluates internal 3-fold stratified CV F1 score (HistF1) at creation.
- Checkpoint predictions on subsequent windows are cached ONCE upon candidate creation.

### 3. Online Labeling & Estimator Training
- Target variable y_candidate for candidate c at decision t:
  - Paired bootstrap (200 resamples over instances in I_t) computes dF1 = macroF1(c) - macroF1(LocalRetrain).
  - Positive label (y=1) if lower bound of dF1 90% CI is > 0 (with margin eps = 0.010). Otherwise y=0.
- Feature vector x_{c,t} per candidate at decision t:
  - Wasserstein: Mean 1-D Wasserstein distance between I_t covariate sample and candidate training block.
  - MMD: RBF MMD with median-heuristic bandwidth.
  - Cosine: Cosine similarity of feature mean vectors.
  - HistF1: Internal CV macro F1 score.
  - Age: Elapsed regime blocks since candidate closed.
  - PoolSize: Number of candidates available at t.
  - LocalRecentF1: Macro F1 of Local-Retrain on previous interval [t-K, t).
  - W_rel: W - min_c W.
  - MMD_rel: MMD - min_c MMD.

### 4. Online Estimator & Threshold Selection (tau_t)
- Online Logistic Regression (sklearn.linear_model.LogisticRegression(class_weight='balanced', C=1.0)).
- Standardized using online running mean and variance.
- Fitted on all accumulated historical candidate decision rows prior to t.
- Platt scaling calibration on held-out out-of-sample predictions once >= 200 rows exist.
- Dynamic threshold tau_t in {0.20, ..., 0.90} grid search on dev split (t < 0.30 * N_decisions) maximizing realized dF1.

### 5. Evaluated Methods (8 Main + Appendix)
1. Frozen: Initial model fit on X_init, never updated.
2. Event-Driven: Vendored Exp1 drift-detector ensemble.
3. Local-Retrain: Online model refit on most recent L labeled windows before t. Default fallback.
4. Similarity-Only: Argmax candidate by Wasserstein distance (always transfers).
5. Similarity-Weighted: Softmax over negative Wasserstein distance with dev-tuned temperature T.
6. Hist-Reliability: Highest HistF1 candidate (always transfers).
7. Probability-Guided: Argmax-P candidate if max_c P >= tau_t, else Local-Retrain.
8. Oracle: Best candidate or Local-Retrain per interval by realized F1.

## Training Policies
- Policy B (Primary Study): Models trained on X_init + X_regime (un-handicapped).
- Policy A (Ablation Study): Models trained strictly on target-regime data X_regime (Exp1 style).

## Stream Suite
- S1: Synthetic planted-concept recurrence stream (6,000 windows, 60 segments, 4 concepts, exact/partial/decoy recurrence).
- N1: Pure noise null stream (unlearnable).
- N2: Fresh unique concepts null stream (no recurrence, local retrain works, transfer fails).
- S2: 5G NR Latency real stream (Exp1 protocol, 499 windows, n_init=99).
- S4: Elec2 Electricity real stream (45,312 instances, river dataset).
- S3: INSECTS stream (if available).
