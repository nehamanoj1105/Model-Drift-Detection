# Experiment 9: RAPT vs Event-Driven Ensemble on 5G Campus Network QoS Dataset

This directory contains the complete, reproducible implementation of **Experiment 1 (RAPT vs Event-Driven Ensemble)** evaluated on the **5G Campus Network QoS Dataset for Open-Source gNB Implementations** (Zenodo 13754300).

---

## 1. Directory Structure

```text
experiments/exp9/
├── configs/
│   └── config.yaml             # Experiment hyperparameters & dataset configuration
├── data/
│   ├── README.txt              # Official dataset documentation
│   ├── ntnu_tput_all_Throughput.csv
│   ├── wue_tput_all_Throughput.csv
│   └── processed_exp9_stream.csv # Aggregated streaming windows with QoS labels
├── results/
│   ├── dataset_profiling_report.md
│   ├── summary.csv             # Summary metrics across random seeds
│   ├── per_seed_results.csv    # Individual seed metric details
│   ├── per_window_results.csv  # Prequential window-by-window telemetry
│   ├── transition_results.csv  # Regime recovery analysis
│   ├── statistical_tests.csv   # Wilcoxon signed-rank statistical test results
│   └── EXP9_REPORT.md          # Comprehensive final research report
├── plots/
│   ├── fig1_f1_streaming.png
│   ├── fig2_cpu_cost_streaming.png
│   ├── fig3_cumulative_cpu.png
│   ├── fig4_f1_around_transitions.png
│   ├── fig5_retraining_events.png
│   ├── fig6_performance_vs_cpu_pareto.png
│   ├── fig7_per_seed_f1.png
│   └── fig8_per_seed_cpu.png
├── preprocessing.py            # Window aggregation & target label generation
├── models.py                   # Heterogeneous Random Forest + Extra Trees ensemble
├── event_driven.py             # Event-Driven Adaptive Ensemble baseline
├── rapt.py                     # RAPT Policy-Transfer system & ablations
├── evaluation.py               # Prequential protocol, recovery metrics & statistical tests
├── profile_dataset.py          # Dataset profiling script
├── load_and_prepare_stream.py  # Packet stream loader and window generator
├── run_exp9.py                 # Main benchmark runner across 5 seeds & 6 methods
└── README.md                   # Setup & reproduction guide
```

---

## 2. Experimental Setup & Protocol

- **Dataset**: 5G New Radio telemetry from OpenAirInterface (OAI) gNB implementations on USRP B200 SDR hardware.
- **Window Size**: 500 packets per window.
- **QoS Target**: 3-class prediction (`GOOD`, `DEGRADED`, `BAD`) derived from next-window latency percentile thresholds ($\tau_1 = 5.916 \text{ s}$, $\tau_2 = 11.443 \text{ s}$) frozen on the initial training prefix.
- **Recurring Regime Stream**: `A -> B -> C -> A -> C -> B -> A -> B -> C` (1,799 windows across 9 blocks).
- **Prequential Protocol**: Strict Test-Then-Train protocol at every streaming window.
- **Random Seeds**: `42`, `43`, `44`, `45`, `46`.
- **Parallelism**: Single-threaded (`n_jobs=1`) for fair CPU timing comparisons.

---

## 3. How to Reproduce

### Step 1: Profile Raw Dataset
```bash
python experiments/exp9/profile_dataset.py
```

### Step 2: Prepare Streaming Windows
```bash
python experiments/exp9/load_and_prepare_stream.py
```

### Step 3: Execute Benchmark & Generate Final Report
```bash
python experiments/exp9/run_exp9.py
```
