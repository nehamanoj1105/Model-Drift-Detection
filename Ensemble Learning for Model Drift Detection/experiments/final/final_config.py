"""Central config for the unified final harness.

Everything is a constant here. The frozen values are written to
configs/final.yaml by freeze_config.py after the dev sweep.
"""

import os

EXPERIMENT = "final"

# Seeds
SEEDS_DEV = [42, 43, 44, 45, 46]
SEEDS_FINAL = list(range(42, 72))          # 42..71, 30 seeds

# Protocol
INITIAL_TRAIN_FRACTION = 0.20
BUFFER_CAPACITY = 1000
ERROR_WINDOW_SIZE = 20
ERROR_THRESHOLD_K = 2.0
DEV_WINDOW_FRACTION = 0.60                 # first 60% of post-prefix windows (dev)

# Base architecture
ENSEMBLE = {"type": "RandomForest+ExtraTrees soft vote",
            "n_estimators": 50, "max_depth": 7}

# Streams
STREAMS = ["5G Campus QoS", "UGR'16", "NordicDat", "5G NR Latency"]
TUNE_STREAMS = ["5G Campus QoS", "UGR'16", "NordicDat"]

BASE_MODELS = ["Frozen", "Frozen+Periodic", "Event-Driven", "Event-Driven+Reuse",
               "Full Retraining", "RAPT", "RAPT-Enhanced", "RAPT-Guard",
               "RAPT-Anchor", "ADWIN", "Page-Hinkley", "EDD", "EDMA"]

# ---------------------------------------------------------------------------
# Tunable knobs. These are the values the dev sweep may change. The defaults
# here are the "before tuning" values (old defaults).
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    # RAPT-Guard
    "guard_lookback": 75,          # labelled rows used to score/compare (50-100)
    "guard_margin": 0.0,           # reuse iff stored_acc >= prefix_acc + margin
    "guard_min_rows": 30,
    # RAPT-Anchor ("novelty anchor"): reuse only if the stored policy was
    # trained on the same regime id AND scoring confirms it; otherwise refit
    # from the class-anchored buffer. share_prefix=True evaluates a fraction of
    # windows with the frozen prefix model (extra inference = anchor's cost).
    "anchor_share_prefix": 0.5,
    "anchor_lookback": 75,
    # Event-driven schedule
    "ed_refit_interval": 3,
    # Frozen + periodic refit
    "periodic_interval": 5,
    "periodic_n_trees": 20,
    "periodic_buffer": 300,
    # Cheap refit (used where a cheap refit is configured)
    "cheap_n_trees": 20,
    "cheap_buffer": 300,
    # Historical detectors (set from the dev sweep)
    "adwin_delta": 0.002,
    "ph_min_instances": 30,
    "ph_delta": 0.005,
    "ph_threshold": 50,
    "ph_alpha": 0.9999,
    "edd_warning": 2.0,
    "edd_drift": 3.0,
    "edma_alpha": 0.2,
    "edma_k": 2.0,
    "detector_min_retrain_interval": 3,
    # concept-panel in-regime floor trigger
    "floor_acc_threshold": 0.60,
    "floor_window": 3,
    "floor_min_interval": 2,
    "enable_floor": False,         # ON only for the concept-drift stress panel
}

# Paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_FINAL = os.path.join(PROJECT_DIR, "results", "final")
RAW_FINAL = os.path.join(RESULTS_FINAL, "raw")
TABLES_FINAL = os.path.join(RESULTS_FINAL, "tables")
FIGURES_FINAL = os.path.join(RESULTS_FINAL, "figures")
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")


def ensure_dirs():
    for d in (RESULTS_FINAL, RAW_FINAL, TABLES_FINAL, FIGURES_FINAL, CONFIG_DIR):
        os.makedirs(d, exist_ok=True)
