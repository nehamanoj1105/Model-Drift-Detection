"""
Single Configuration Section for Experiment 9B — Drift Severity & Concept-Drift Evaluation.

All experimental constants live here. Nothing in the drift runners should hard-code
seeds, window sizes, drift levels, model names, or directory paths.

Working directory assumption: the repository sub-root
    "Ensemble Learning for Model Drift Detection"
(the same assumption made by the existing exp9b pipeline).
"""

import os

EXPERIMENT = "9B"

# --- Random seeds (identical to the existing 9B experiment) ---
SEEDS = [42, 43, 44, 45, 46]

# --- Controlled drift severity levels (fraction of target regime) ---
DRIFT_LEVELS = [0.10, 0.20, 0.30, 0.50, 1.00]

# --- Streaming window size (windows are pre-built at 500 packets each) ---
WINDOW_SIZE = 500

# --- Models evaluated across every drift experiment ---
MODELS = [
    "Frozen",
    "Event-Driven",
    "Full_Retraining",
    "RAPT",
    "RAPT-Enhanced",
]

# --- Initial training prefix (fraction of stream), matches existing 9B ---
INITIAL_TRAIN_FRACTION = 0.20

# --- Recovery threshold: fraction of pre-drift Macro-F1 that must be regained ---
RECOVERY_THRESHOLD = 0.95

# --- Concept-drift transformation ---
# Deterministic, seed-fixed affine reversal applied to the most predictive
# features *after* the drift point. Severity = fraction of those features that
# receive the transformation. See drift_construct.build_concept_drift_stream.
CONCEPT_DRIFT_FEATURE_COUNT = 6

# --- Paths (relative to the "Ensemble Learning for Model Drift Detection" root) ---
EXP9B_DIR = os.path.join("experiments", "exp9b")
STREAM_CSV = os.path.join(EXP9B_DIR, "data", "processed_exp9b_stream.csv")
STREAM_JSON = os.path.join(EXP9B_DIR, "results", "stream_definition.json")

RESULTS_ROOT = os.path.join("results", "experiment_9b")
NATURAL_DIR = os.path.join(RESULTS_ROOT, "natural_drift")
COVARIATE_DIR = os.path.join(RESULTS_ROOT, "covariate_drift")
CONCEPT_DIR = os.path.join(RESULTS_ROOT, "concept_drift")
RECURRING_DIR = os.path.join(RESULTS_ROOT, "recurring_concept_drift")
RAW_DIR = os.path.join(RESULTS_ROOT, "raw")
FIGURES_DIR = os.path.join(RESULTS_ROOT, "figures")
TABLES_DIR = os.path.join(RESULTS_ROOT, "tables")

FINAL_REPORT = "EXPERIMENT_9B_FINAL_REPORT.md"


def severity_tag(level):
    """Stable filename/table tag for a drift severity, e.g. 0.1 -> '10'."""
    return f"{int(round(level * 100))}"


def ensure_dirs():
    for d in [NATURAL_DIR, COVARIATE_DIR, CONCEPT_DIR, RECURRING_DIR,
              RAW_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
