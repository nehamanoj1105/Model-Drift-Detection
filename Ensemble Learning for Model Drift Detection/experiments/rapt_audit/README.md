# RAPT Audit & Diagnostic Suite

This directory contains the audit scripts, diagnostics, plot visualizers, and reproduction results for the RAPT-v2 audit.

## Structure
- `reproduce_original.py`: Executes original Exp 9A and Exp 9B RAPT pipelines to verify exact reproduction.
- `audit.py`: Runs per-window prediction comparisons, feature preprocessing checks, weight sensitivity tests, and first-divergence isolation.
- `generate_audit_plots.py`: Generates the 8 required audit figures in `plots/`.
- `results/`:
  - `original_reproduction/`: Baseline reproduction CSVs matching original Exp 9A (0.9942) and Exp 9B (0.8894).
  - `protocol_diff.csv` & `PROTOCOL_DIFF.md`: Comprehensive protocol comparison table.
  - `per_window_comparison_seed42.csv`: Per-window prediction log for Seed 42.
  - `preprocessing_comparison.csv`: Tensor shapes, feature statistics, and MD5 hashes.
  - `mechanism_activity.csv`: Mechanism activation counts.
  - `RAPT_AUDIT_REPORT.md`: Master diagnosis report.
- `diagnostics/`:
  - `first_divergence.md`: Diagnostic trace of the first prediction divergence (Window 305).
  - `weight_effect_test.md`: Empirical proof of ensemble weight sensitivity.
- `plots/`: 8 standalone 300-DPI audit figures (`1_streaming_f1_9a.png` through `8_disagreement_rate.png`).
