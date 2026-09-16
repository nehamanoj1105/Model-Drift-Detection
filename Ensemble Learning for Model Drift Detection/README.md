# Ensemble Learning for Model Drift Detection & Adaptive Ensembles

This repository hosts a progressive research program investigating model drift detection, adaptive ensemble architectures, and multi-armed bandit model selection in streaming mobile telemetry networks.

## Repository Layout

```
project_root/
├── README.md                                    # Project overview and instructions
├── requirements.txt                              # Core dependencies
├── experiments/
│   ├── experiment_1/                            # Experiment 1: Drift Comparison Experiment
│   │   ├── run_drift_comparison_experiment.py
│   │   ├── ai4mobile_sample_stream.csv
│   │   ├── ai4mobile_four_metrics_stream.csv
│   │   ├── models/
│   │   ├── plots/
│   │   ├── metrics/
│   │   ├── predictions/
│   │   └── drift_analysis/
│   ├── experiment_2/                            # Experiment 2: Drift-Aware Bandit (DAB) Model Selection
│   │   ├── run_experiment2_bandit.py
│   │   ├── run_experiment2_dab.py
│   │   ├── run_both_experiments.py
│   │   ├── generate_experiment2_figures.py
│   │   ├── experiment2_stationary_dataset.csv
│   │   ├── unicode_chars.txt
│   │   ├── experiment2/
│   │   ├── results/
│   │   ├── plots/
│   │   └── experiment_2_figures/
│   └── experiment_3/                            # Experiment 3: Adaptive Ensembles & Degradation
│       └── experiment_3a/                       # Experiment 3A: Frozen Model Degradation & Retraining Cost
│           ├── experiment_3a.py
│           ├── data/
│           ├── models/
│           ├── results/
│           ├── plots/
│           └── README.md
├── docs/
│   ├── experiment_notes/                        # Detailed architectural and design notes
│   └── verification_report.md                   # Verification and integrity audit
├── papers/                                      # Related literature and manuscript drafts
└── archive/                                     # Archived legacy implementations
    └── implementation/
```

## Research Overview

### Experiment 1 — Baseline Drift Comparison
- **Focus**: Compares static machine learning models (Random Forest, Extra Trees, Decision Trees) and weighted ensembles under binary stationary vs. full drift conditions.
- **Dataset**: AI4Mobile streaming network telemetry (5,000 samples, 12 signals, 74 features).
- **Key finding**: Static models degrade substantially under unmitigated distribution shift; weighted ensembling improves stability.

### Experiment 2 — Drift-Aware Bandit (DAB) Selection
- **Focus**: Dynamic multi-armed bandit (UCB1 and Discounted UCB) selection between frozen and adaptive models under progressive drift levels (0% to 50%).
- **Dataset**: 4-KPI streaming telemetry (speed, distance, delay, throughput) with 24 rolling-window features.
- **Integrity**: Experiment 2 is strictly frozen and preserved with verified SHA-256 integrity hashes.

### Experiment 3A — Frozen Model Degradation & Retraining Cost Under Drift
- **Focus**: Quantitative empirical foundation establishing degradation patterns across 7 drift conditions and benchmarking cumulative computational retraining costs via process-level `psutil` profiling.
- **Dataset**: 10,000 raw samples (2,000 initial training + 8,000 post-deployment across 16 sequential evaluation chunks).
- **Drift Conditions**:
  - `3A-0`: Stationary Baseline (no drift)
  - `3A-1`, `3A-2`, `3A-3`: Covariate Drift (Mild, Moderate, Severe shifts in $P(X)$)
  - `3A-4`, `3A-5`, `3A-6`: Concept Drift (Mild, Moderate, Severe shifts in $P(Y|X)$)
- **Metrics**: Normalized Wasserstein distance (primary), Kolmogorov-Smirnov statistic, Population Stability Index (PSI).
- **Resource Monitoring**: Process-level CPU time, wall-clock duration, average/peak CPU %, and memory RSS footprint.
- **Seeds**: 5 evaluation seeds `[42, 43, 44, 45, 46]` reporting mean $\pm$ standard deviation.

## Quick Start

### Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Running Experiments

#### Running Experiment 3A
```bash
cd experiments/experiment_3/experiment_3a
python experiment_3a.py
```
This produces:
- `results/performance_results.csv`
- `results/resource_results.csv`
- `results/drift_results.csv`
- `results/experiment_config.json`
- 13 publication-quality figures in `plots/`
- Serialized frozen models and scalers in `models/`
- Stream dataset artifacts in `data/`

#### Running Experiment 2
```bash
cd experiments/experiment_2
python run_experiment2_bandit.py
```
