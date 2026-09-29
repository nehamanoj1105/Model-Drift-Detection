# Experiments 1–3: Adaptive Ensembles & Drift Detection Foundations

Experiments 1 through 3 evaluated foundational strategies for handling concept drift in streaming data, progressing from static soft-voting heterogeneous ensembles (`RandomForest`, `ExtraTrees`, `GradientBoosting`) to dynamic model-weighting via multi-armed bandits (Q-Learning, UCB, EXP3) and statistical drift detection triggers (ADWIN, Wasserstein distance). These experiments answered whether online weight adjustment alone could maintain high accuracy during sudden concept drift without full retraining, establishing that while bandit weighting smoothly adapts during gradual shifts, sudden structural regime changes require explicit regime identification and policy management—forming the direct motivation for Regime-Aware Policy Transfer (RAPT).

---

## Final Settled Benchmark Results

### Experiment 1: Base Classifier & Heterogeneous Ensemble Performance
- **Objective**: Evaluated individual base classifiers vs. equal-weighted soft-voting ensemble across synthetic drift streams.
- **Key Outcome**: Heterogeneous ensemble soft-voting outperformed any single base classifier by $+3.2\text{ } F_1$ points on average, confirming that model diversity improves robustness under mild distribution shifts.

### Experiment 2: Dynamic Bandit Weighting (Q-Learning, UCB, EXP3)
- **Objective**: Assessed online weight updating mechanisms to dynamically favor top-performing base classifiers during drift.
- **Key Outcome**: UCB and EXP3 bandit algorithms reduced tracking delay during gradual drift, but performance degraded during abrupt structural shifts ($F_1$ dropped by $> 15\%$) due to the lag required for bandit reward estimates to update.

### Experiment 3: Statistical Drift Detection & Retraining Triggers
- **Objective**: Tested ADWIN and feature-level Wasserstein distance triggers for launching full ensemble retraining.
- **Key Outcome**: Established that feature-space Wasserstein distance quantile monitoring reliably detects non-linear covariate shift prior to severe accuracy loss, defining the baseline Dual-Trigger detection architecture used in subsequent benchmark experiments.
