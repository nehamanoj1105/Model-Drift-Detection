# Experiment 2 — Probabilistic Regime Transfer

## Overview
Experiment 2 investigates whether an online meta-learning model can predict **transferability** ($P(\text{positive transfer} \mid X_{s,t})$) rather than relying solely on **telemetry similarity** ($S(R_s, R_t)$) when reusing historical policies for recurring network regimes.

### Core Research Question
> **Can we predict whether a historical regime policy will transfer successfully to the current regime, rather than assuming that the most similar historical regime is always the best source?**

---

## Directory Structure
```text
experiments/exp2/
    data/                      # Original & Enriched 9A/9B stream datasets
    configs/                   # Feature & threshold configurations
    results/                   # Detailed per-seed, per-window, transfer pairs, and summary results
    plots/                     # 13 publication figures + primary diagnostic overlay plot
    regime_representation.py   # Telemetry distribution, temporal, correlation & model state feature extractor
    similarity.py              # Euclidean, Wasserstein, JS, Correlation, & Temporal distance metrics
    transferability.py         # Directional feature builder X_{s,t}, online meta-model, 2-stage selection, & abstention gate
    exp2.py                    # Master streaming prequential system controller
    evaluation.py              # AUROC, AUPRC, Brier Score, ECE, & Wilcoxon signed-rank paired tests
    prepare_enriched_streams.py# Data stream generator (Original & 8-12 regime Enriched streams)
    run_exp2.py                # Master benchmark runner (11 methods x 5 seeds x 4 streams)
    plots_exp2.py              # Visualization generator
    README.md                  # System overview and usage instructions
    results/EXP2_REPORT.md     # Final comprehensive scientific report
```

---

## Key Methodological Innovations

### 1. Zero-Look-Ahead Anti-Leakage Protocol
At deployment time $t$:
- The transfer decision is computed using **strictly pre-transfer information** (unlabeled telemetry distribution, correlation structure, temporal dynamics, model disagreement, and past transfer history $D_{meta}(t)$).
- Future target labels, future F1, scenario IDs, and file names are strictly forbidden as model features.
- Post-hoc ground-truth outcomes ($\Delta F1_{s,t} = F1(\text{transfer } s \to t) - F1(\text{baseline on } t)$ over horizon $H=50$ windows) are logged to $D_{meta}(t)$ **only after** predictions are completed and true labels are observed.

### 2. 2-Stage Candidate Selection Architecture
- **Stage 1 (Cheap Similarity Filter)**: Rapidly filters historical checkpoint pool to top-$K$ candidates ($K=5$) using combined similarity $S_{combined}$.
- **Stage 2 (Transferability Estimation)**: Computes directional pair features $X_{s,t}$ and evaluates $P_{s,t} = P(\text{positive transfer} \mid X_{s,t})$ via online calibrated logistic regression.

### 3. Abstention Gate ($\tau = 0.60$)
If $\max_s P_{s,t} < \tau$, the system **rejects transfer** ("I don't trust any historical policy enough to transfer"), avoiding negative transfer and triggering local adaptation / retraining.

---

## Benchmark Methods (11 Total)
1. `Frozen`: Static initial model.
2. `Event-Driven`: Standard ADWIN / drift-detected retraining baseline.
3. `Full Retraining`: Continuous sliding-window retraining.
4. `RAPT-E`: Canonical baseline from Experiment 1 (Euclidean similarity thresholding).
5. `Similarity-Only`: Selects $s^* = \arg\max_s S_{combined}(R_s, R_t)$.
6. `Similarity-Weighted`: Blends candidate policies proportional to similarity.
7. `Historical Reliability`: Reuses checkpoint with highest past transfer success rate.
8. `Random Historical`: Selects random historical checkpoint.
9. `Probability-Guided Top-1` (**Proposed Primary Method**): Selects $s^* = \arg\max_s P_{s,t}$, abstains if $P_{s^*,t} < \tau$.
10. `Probability-Guided Weighted`: Blends policy weights proportional to $P_{i,t}$.
11. `Oracle Transfer`: Post-hoc upper bound selecting true best candidate on target (Evaluation headroom only).

---

## Execution Instructions

### 1. Prepare Telemetry Data Streams
```bash
python experiments/exp2/prepare_enriched_streams.py
```

### 2. Run Complete Benchmark (5 Seeds, 4 Streams, 11 Methods)
```bash
python experiments/exp2/run_exp2.py
```

### 3. Generate Diagnostic & Publication Plots
```bash
python experiments/exp2/plots_exp2.py
```
