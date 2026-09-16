# Experiment Notes — Overview

## Architecture & Roadmap

This directory contains research design notes and experimental logs across the model drift detection project.

### Experiment Summary
- **Experiment 1**: Initial drift detection and baseline comparison on 12-signal streaming mobile telemetry.
- **Experiment 2**: Multi-Armed Bandit (MAB) adaptive model selection (Frozen Model vs. Adaptive Model) across 6 drift levels (0% - 50%).
- **Experiment 3 Series**:
  - **Experiment 3A**: Frozen Model Degradation & Retraining Cost Under Drift.
    - Establishes empirical baseline of how frozen models fail under Covariate vs. Concept drift.
    - Profiles the exact computational and memory costs of continuous retraining.
  - **Experiment 3B - 3H (Planned)**:
    - Adaptive ensemble strategies, sliding window retraining, drift detection triggering, and hybrid bandit controllers.
