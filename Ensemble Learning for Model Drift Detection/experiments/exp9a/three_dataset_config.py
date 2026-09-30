"""
Experiment 9A (three telecom datasets) — central configuration.

Single source of truth for the pilot screening and the full five-seed run.
Datasets: 5G Campus Network QoS, UGR'16, NordicDat.

Run everything from the "Ensemble Learning for Model Drift Detection" directory.
"""

import os

EXPERIMENT = "9A_THREE_DATASETS"

SEEDS_PILOT = [42]
SEEDS_FULL = [42, 43, 44, 45, 46]

WINDOW_SIZE = 500
INITIAL_TRAIN_FRACTION = 0.20
# Pilot caps the chronological stream at this many samples (chronological blocks,
# never random individual records).
PILOT_MAX_SAMPLES = 100_000

MODELS = [
    "Frozen",
    "Event-Driven",
    "Full Retraining",
    "RAPT",
    "RAPT-Enhanced",
]

DRIFT_DETECTORS = ["ADWIN", "EDD", "Page-Hinkley", "EDMA"]

RECOVERY_THRESHOLD = 0.95
RECOVERY_HORIZON = 20

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 5G Campus Network QoS (already windowed inside the repository).
CAMPUS_WINDOW_CSV = os.path.join(PROJECT_DIR, "experiments", "exp9", "data",
                                 "processed_exp9_stream.csv")

# UGR'16 feature data (per-minute netflow aggregates + attack labels).
UGR16_MAT = os.path.join(PROJECT_DIR, "experiments", "exp9a", "data", "ugr16",
                         "UGR16v1.mat")

# NordicDat cross-border predictive QoS traces.
NORDICDAT_CSV = os.path.join(PROJECT_DIR, "experiments", "exp9a", "data",
                             "nordicdat", "nordicdat.csv")

# Existing 9B artefacts (read-only cross-dataset inputs).
EXP9B_NATURAL_SUMMARY = None  # set in tables module

# Output roots.
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments", "exp9a")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "experiment_9a_three")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
SCREENING_DIR = os.path.join(EXPERIMENT_DIR, "dataset_screening")
FIGURES_DIR = os.path.join(EXPERIMENT_DIR, "figures")
TABLES_DIR = os.path.join(EXPERIMENT_DIR, "tables")
REPORTS_DIR = os.path.join(EXPERIMENT_DIR, "reports")

DATASETS = ["5G Campus QoS", "UGR'16", "NordicDat"]

# Short slugs for filenames.
DATASET_SLUG = {
    "5G Campus QoS": "5g_campus",
    "UGR'16": "ugr16",
    "NordicDat": "nordicdat",
}


def ensure_dirs():
    for d in (RESULTS_DIR, RAW_DIR, SCREENING_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
