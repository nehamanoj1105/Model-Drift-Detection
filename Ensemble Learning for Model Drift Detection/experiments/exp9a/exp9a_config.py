"""
Central configuration for the REVISED Experiment 9A.

All hard-coded experiment values live here so the runner, figures, tables and
report share a single source of truth.

Revised 9A dataset: the INSECTS incremental-reoccurring stream
(river.datasets.Insects, `incremental_reoccurring_balanced`). It is a canonical
recurring-drift benchmark whose repeated blocks share (near-)identical feature
distributions while their class compositions recur, giving the recurring-regime
policy-reuse mechanism of RAPT a genuine opportunity to fire. The stream is
windowed into 500-sample windows and 11 recurring regime visits are extracted.
"""

import os

# ----------------------------------------------------------------------------
# Experiment identity
# ----------------------------------------------------------------------------
EXPERIMENT = "9A"

SEEDS = [42, 43, 44, 45, 46]

# The five final-paper models. Order is used everywhere (figures, tables).
MODELS = [
    "Frozen",
    "Event-Driven",
    "Full_Retraining",
    "RAPT",
    "RAPT-Enhanced",
]

# ----------------------------------------------------------------------------
# Dataset paths (relative to the "Ensemble Learning for Model Drift Detection"
# project directory, which is the working directory for all runners).
# ----------------------------------------------------------------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# INSECTS recurring-drift source cache (downloaded via river).
INSECTS_CACHE_CSV = "/home/openhands/river_data/Insects/incremental_reoccurring_balanced.csv"

EXP9A_DATA_DIR = os.path.join(PROJECT_DIR, "experiments", "exp9a", "data")
REVISED_STREAM_CSV = os.path.join(EXP9A_DATA_DIR, "processed_exp9a_stream.csv")

# Existing 9B artefacts (read-only cross-dataset inputs).
EXP9B_DATA_DIR = os.path.join(PROJECT_DIR, "experiments", "exp9b", "data")
EXP9B_STREAM_CSV = os.path.join(EXP9B_DATA_DIR, "processed_exp9b_stream.csv")
EXP9B_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "experiment_9b")

# ----------------------------------------------------------------------------
# Output directories
# ----------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "experiment_9a")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Cross-dataset figures live next to the 9B drift figures so the whole
# experiment_9a / experiment_9b family is discoverable in one place.
CROSS_FIGURES_DIR = os.path.join(PROJECT_DIR, "results", "experiment_9a", "cross_dataset")

# ----------------------------------------------------------------------------
# Streaming protocol (identical prequential protocol to the 9A/9B harness)
# ----------------------------------------------------------------------------
WINDOW_SIZE = 500                 # samples per prequential window
INITIAL_TRAIN_WINDOWS = 30        # initial training prefix = 20% of the 150-window
                                  # stream (matches the 9B 20% initial fraction);
                                  # leaves several genuine regime recurrences in
                                  # the streaming portion.
BUFFER_CAPACITY = 1000            # adaptation buffer cap (matches baselines)

# Event-driven drift detector
ERROR_WINDOW_SIZE = 20
ERROR_THRESHOLD_K = 2.0

# Recovery metric
RECOVERY_THRESHOLD = 0.95         # 95% of pre-drift F1
RECOVERY_HORIZON = 5              # windows searched for recovery (matches exp9)

# ----------------------------------------------------------------------------
# RAPT-Enhanced hyperparameters (ported from rapt/enhanced_hybrid_rapt.py)
# ----------------------------------------------------------------------------
ENHANCED_BASE_TAU = 0.455
ENHANCED_LAMBDA_TAU = 0.15
ENHANCED_TAU_MIN = 0.32
ENHANCED_NOVELTY_REFIT_N = 1500   # buffer-blended novelty refit size


def ensure_dirs():
    for d in (EXP9A_DATA_DIR, RESULTS_DIR, RAW_DIR, FIGURES_DIR, TABLES_DIR, CROSS_FIGURES_DIR):
        os.makedirs(d, exist_ok=True)
