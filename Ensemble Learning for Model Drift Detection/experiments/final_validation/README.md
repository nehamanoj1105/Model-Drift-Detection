# Final RAPT-E Validation Module

This module contains the master final validation pipeline, unified timing instrumentation, publication plot generators, and comprehensive markdown report for RAPT-E (Fixed RAPT-v2).

## Module Structure
- `final_validation.py`: Master validation pipeline executing 5-seed benchmarks across Dataset 9A and 9B.
- `timing_utils.py`: High-precision `time.perf_counter()` timing instrumentation and fit trace recorder.
- `generate_final_plots.py`: Generates the 11 publication-quality plots in `plots/`.
- `FINAL_VALIDATION_REPORT.md`: Comprehensive final validation report and Experiment 2 Gate decision.
- `results/`:
  - `final_summary.csv`: Aggregated performance and CPU metrics across 5 seeds.
  - `final_per_seed.csv`: Individual per-seed metric breakdown.
  - `final_per_window.csv`: Prequential window-level evaluation telemetry.
  - `fit_trace.csv`: Audit log of every `.fit()` operation across all methods.
  - `final_statistical_tests.csv`: Wilcoxon signed-rank tests & non-inferiority margin analysis ($\delta=0.005$).
  - `transition_analysis.csv`: Regime transition recovery metrics.
- `plots/`: 11 standalone 300-DPI figures (`fig1_macro_f1_9a.png` through `fig11_cross_summary.png`).
