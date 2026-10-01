# RAPT default constants

Problem (k) from `VALIDATION.md`: the manuscript never states RAPT's default
hyperparameters. They are defined in code; this table collects them in one place.
No value was changed.

## Base model (shared by all models)

| Constant | Value | Source |
|----------|-------|--------|
| Ensemble | 1 random forest + 1 extra-trees | `experiments/exp9a/models_9a.py` |
| Trees per forest | 50, depth 7 | `models_9a.py` |
| Adaptation buffer capacity | 1000 | `experiments/exp9a/exp9a_config.py: BUFFER_CAPACITY` |
| Initial training prefix | first 20% of windows (36 cross-dataset, 99 for 5G NR) | `three_dataset_config.py: INITIAL_TRAIN_FRACTION` |
| Event-driven error window | 20, threshold 2.0 | `exp9a_config.py: ERROR_WINDOW_SIZE, ERROR_THRESHOLD_K` |
| Recovery threshold | 0.95 of pre-drift F1, horizon 5 windows | `exp9a_config.py` |

## RAPT controller (Tier-2, Table II "RAPT")

| Constant | Value | Meaning |
|----------|-------|---------|
| Reuse gate | similarity only (no absolute floor) | reuse if fingerprint similarity passes |
| Novelty-refit buffer | 500 samples | refit size on a new/unknown regime |
| Parity refit threshold | 0.5 | refit a reused policy if rolling accuracy < 0.5 |
| Parity window | 3 windows | rolling window for the parity check |

## RAPT-Enhanced

| Constant | Value | Meaning |
|----------|-------|---------|
| Novelty-refit buffer | **1500** samples | larger refit buffer than base RAPT |
| Base similarity tau | 0.455 | dynamic threshold base |
| Lambda tau / tau min | 0.15 / 0.32 | threshold decay and floor |

## Ablation ladder (`Final_Experiments/rapt_ladder.py`)

Defaults are the function signature of `RAPT_Ladder.__init__`:

| Constant | Value |
|----------|-------|
| `novelty_threshold` | 0.65 |
| `refit_n` | 1500 |
| `parity_threshold` / `parity_window` | 0.5 / 3 |
| `refresh_trees` / `refresh_buffer` | 50 / 1000 |
| `refresh_min_acc` (absolute floor) | 0.97 |
| `evidence_window` | 3 |
| `buffer_capacity` | 1000 |
| `gamma` | 1.0 |

`RAPT_CHEAP` overrides the refresh to 20 trees and a 300-sample buffer.

## Why this matters

The manuscript's UGR'16 narrative attributes recovery to the parity refit, but
with `parity_threshold = 0.5` the branch never fires (`parity_refits = 0` on all
seeds; E2/E5). The measurable difference between RAPT and RAPT-Enhanced on
UGR'16 is the novelty-refit buffer: 500 vs 1500. See `PHASE2_FINDINGS.md`.
