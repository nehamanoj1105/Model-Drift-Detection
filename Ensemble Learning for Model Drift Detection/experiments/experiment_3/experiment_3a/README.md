# Experiment 3A — Frozen Model Degradation & Retraining Cost Under Drift

## Overview

This experiment establishes the empirical foundation for later adaptive ensemble
experiments (3B through 3H). It demonstrates:

1. **A frozen Model 1 degrades** under distribution shift
2. **Different drift types/severities** produce different degradation patterns
3. **Retraining can recover performance** at measurable computational cost
4. **Repeated retraining accumulates substantial resource cost**

## Quick Start

```bash
cd experiments/experiment_3/experiment_3a
python experiment_3a.py
```

## Experimental Design

### Dataset
- **Size**: 10,000 raw samples (2,000 training + 8,000 post-deployment)
- **Task**: Binary QoS violation classification
- **KPIs**: Speed, Distance, Delay, Throughput
- **Features**: Rolling window (W=20, step=5) × 6 statistics = 24 features

### Models
- **Frozen Model 1**: RandomForest (50 trees, depth 7) — trained ONCE on initial data
- **Retrained Baseline**: Same architecture, evaluated out-of-sample prequentially (Test-Then-Train) on each incoming window before cumulative retraining

### Drift Conditions
| ID | Drift Type | Severity | Description |
|----|-----------|----------|-------------|
| 3A-0 | None | — | Stationary baseline |
| 3A-1 | Covariate | Mild | Shifts P(X), preserves P(Y|X) |
| 3A-2 | Covariate | Moderate | Larger feature distribution shift |
| 3A-3 | Covariate | Severe | Maximum feature distribution shift |
| 3A-4 | Concept | Mild | Modifies P(Y|X) coefficients |
| 3A-5 | Concept | Moderate | Larger decision boundary shift |
| 3A-6 | Concept | Severe | Maximum decision boundary shift |

### Drift Measurement
- **Primary**: Wasserstein distance (normalized per feature)
- **Secondary**: KS statistic, Population Stability Index (PSI)

### Resource Monitoring
- Process-level CPU and memory tracking via `psutil`
- Metrics: wall-clock time, CPU time, average/peak CPU%, average/peak memory

### Statistical Robustness
- 5 evaluation seeds [42, 43, 44, 45, 46]
- Results reported as mean ± standard deviation

## Outputs

### Results (`results/`)
- `performance_results.csv` — Per-window performance for frozen and retrained models
- `resource_results.csv` — Per-retraining-event resource measurements
- `drift_results.csv` — Per-window drift score measurements
- `experiment_config.json` — Complete experiment configuration

### Plots (`plots/`)
1. `01_drift_score_over_time.png` — Drift score progression per condition
2. `02_frozen_model_f1_over_time.png` — Frozen model degradation over time
3. `03_drift_vs_frozen_f1.png` — Drift score vs. model performance
4. `04_degradation_vs_severity.png` — Performance degradation vs. severity
5. `05_frozen_vs_retrained_f1.png` — Frozen vs. retrained comparison
6. `06_cpu_by_severity.png` — Retraining CPU cost by condition
7. `07_retraining_time_by_severity.png` — Retraining time by condition
8. `08_cpu_vs_recovery.png` — CPU cost vs. performance recovery
9. `09_memory_vs_training_size.png` — Memory scaling with dataset size
10. `10_efficiency_gain_per_cpu.png` — Retraining efficiency
11. `11_cumulative_retraining_cost.png` — Accumulated retraining cost
12. `12_experiment_timeline.png` — Full experiment timeline
13. `13_resource_cost_curves.png` — Drift → cost trade-off curves

### Models (`models/`)
- Frozen model and scaler saved as `.joblib` files (seed 42 only)

## Dependencies
- numpy, pandas, scikit-learn, matplotlib, scipy, psutil, joblib
