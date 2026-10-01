"""Capture hardware/software environment and write config_frozen.json.

Run once, before any evaluation run. The SHA-256 of config_frozen.json is
recorded in REPORT.md.
"""
import os
import sys
import json
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, CONFIG_FROZEN, ensure_dirs, sha256


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def lib_versions():
    import numpy, pandas, sklearn, scipy
    v = {"python": sys.version.split()[0], "numpy": numpy.__version__,
         "pandas": pandas.__version__, "sklearn": sklearn.__version__,
         "scipy": scipy.__version__}
    try:
        import river
        v["river"] = river.__version__
    except Exception:
        v["river"] = None
    try:
        import psutil
        v["psutil"] = psutil.__version__
    except Exception:
        v["psutil"] = None
    return v


def main():
    ensure_dirs()
    cfg = {
        "experiment": "revalidation",
        "created": "2026-10-01",
        "seeds": [42, 43, 44, 45, 46],
        "seeds_cheap": list(range(42, 62)),
        "window_prefix_fraction": 0.20,
        "protocol": "chronological, prequential test-then-train, no shuffling",
        "n_jobs": 1,
        "thread_env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "buffer_capacity": 1000,
        "ensemble": {"type": "RandomForest+ExtraTrees soft vote",
                     "n_estimators": 50, "max_depth": 7},
        # --- frozen before evaluation ---
        "cheap_refit": {"n_trees": 20, "buffer": 300},
        "full_refit": {"n_trees": 50, "buffer": 1000},
        "periodic_refresh_windows": [5, 10],
        "tost_margin_macro_f1": 0.01,
        "tost_margins_reported": [0.005, 0.01, 0.02],
        "recovery_threshold": 0.95,
        "concept_gate_similarity_threshold": 0.65,
        "parity_threshold_default": 0.5,
        "parity_threshold_sweep": [0.5, 0.7, 0.8, 0.9, 0.95],
        "rel_drop_sweep": [0.02, 0.03, 0.05, 0.10],
        "adwin_delta_sweep": [0.002, 0.01, 0.05, 0.1, 0.3],
        "ph_threshold_sweep": [0.05, 0.1, 0.5, 1, 5, 50],
        "ph_min_instances_sweep": [5, 10, 30],
        "ewma_k_sweep": [1, 2, 3],
        "ewma_alpha_sweep": [0.05, 0.2, 0.4],
        "bootstrap_n": 2000,
        "bootstrap_seed": 0,
        "selection_note": ("All values above are fixed before the evaluation "
                           "run. Sweeps are labelled sensitivity, not selection."),
        "environment": {
            "cpu": cpu_model(), "cores": os.cpu_count(),
            "platform": platform.platform(),
            "libs": lib_versions(),
        },
    }
    with open(CONFIG_FROZEN, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", CONFIG_FROZEN)
    print("sha256", sha256(CONFIG_FROZEN))


if __name__ == "__main__":
    main()
