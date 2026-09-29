# RAPT-v2 — Low-Cost Performance Improvement Module

This module implements **RAPT-v2**, an enhanced Regime-Aware Policy Transfer framework featuring low-cost adaptation mechanisms:

- **Mechanism A**: EWMA Ensemble Model Reweighting
- **Mechanism B**: Momentum-Based Policy Weight Adaptation
- **Mechanism C**: Confidence-Gated Adaptation
- **Mechanism D**: Cheap Champion/Challenger Validation Check
- **Mechanism E**: Bounded Adaptation Buffer
- **Mechanism F**: Performance-Triggered Fallback Escalation
- **Mechanism G**: Compute-Aware Adaptation Hierarchy (Level 0..3)
- **Mechanism H**: Historical Policy Freshness / Trust Score Tracking

## Directory Structure
- `rapt_v2.py`: RAPT-v2 system controller and low-cost adaptation engine.
- `evaluation_v2.py`: Prequential metrics, Wilcoxon tests, and diagnostic efficiency objective.
- `plots_v2.py`: Publication-quality visualization generator.
- `run_v2.py`: Master experiment driver executing evaluations on 9A and 9B datasets.
- `results/`: CSV deliverables and comprehensive markdown report `RAPT_V2_REPORT.md`.
- `plots/`: Standalone 300-DPI figures for 9A and 9B.
- `comparison/`: Cross-dataset comparative plots and summary tables.
