# RAPT-v3 — Low-Cost Champion–Challenger Adaptation Module

This directory contains the implementation, benchmark runner, publication plots, diagnostic trace logs, and final scientific report for **RAPT-v3**.

## Directory Structure

```text
experiments/rapt_v3/
├── configs/
│   └── config.json
├── results/
│   ├── summary.csv
│   ├── per_seed_results.csv
│   ├── per_window_results.csv
│   ├── statistical_tests.csv
│   ├── transition_analysis.csv
│   ├── champion_challenger_trace.csv
│   └── RAPT_V3_REPORT.md
├── plots/
│   ├── fig1_macro_f1_9a.png
│   ├── fig2_macro_f1_9b.png
│   ├── fig3_adapt_cpu_9a.png
│   ├── fig4_adapt_cpu_9b.png
│   ├── fig5_v1_vs_v3_f1.png
│   ├── fig6_streaming_recurrence.png
│   ├── fig7_champion_vs_challenger_scores.png
│   ├── fig8_weight_trajectories.png
│   ├── fig9_cpu_vs_f1_pareto.png
│   └── fig10_selection_counts.png
├── comparison/
│   ├── cross_dataset_comparison.csv
│   └── cross_dataset_comparison.md
├── rapt_v3.py
├── evaluation_v3.py
├── run_v3.py
├── plots_v3.py
└── README.md
```

## How to Run

To run the complete benchmark suite across seeds `42, 43, 44, 45, 46` for both 9A and 9B datasets:

```bash
python experiments/rapt_v3/run_v3.py
```

To regenerate all 10 publication figures:

```bash
python experiments/rapt_v3/plots_v3.py
```
