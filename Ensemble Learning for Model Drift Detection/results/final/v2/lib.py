"""v2 shared utilities.

Thin wrappers over the existing revalidation harness. Nothing here modifies
existing files; the original modules are imported unchanged.
"""
import os
import sys
import json
import hashlib
import platform

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(FINAL, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
EXP9B = os.path.join(ROOT, "experiments", "exp9b")

V2 = HERE
RAW = os.path.join(V2, "raw")
CONFIG = os.path.join(V2, "config_frozen_v2.json")
GATES = os.path.join(V2, "gates.csv")

sys.path.insert(0, REVAL)
sys.path.insert(0, EXP9A)
sys.path.insert(0, EXP9B)

SEEDS = [42, 43, 44, 45, 46]
SEEDS_CHEAP = list(range(42, 62))          # 20 seeds for cheap configs
DRIFT_LEVELS = [0.10, 0.20, 0.30, 0.50, 1.00]
MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT"]


def ensure_dirs():
    for d in (V2, RAW, os.path.join(V2, "latex_tables"),
              os.path.join(V2, "figures")):
        os.makedirs(d, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    with open(CONFIG) as f:
        return json.load(f)


def write_csv(df, name):
    ensure_dirs()
    p = os.path.join(V2, name)
    df.to_csv(p, index=False)
    return p


def write_raw(df, name):
    ensure_dirs()
    p = os.path.join(RAW, name)
    df.to_csv(p, index=False)
    return p


def gate(step, status, note):
    ensure_dirs()
    import datetime
    row = pd.DataFrame([{
        "gate": step, "step": step, "status": status, "note": note,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"}])
    row.to_csv(GATES, mode="a", header=not os.path.exists(GATES), index=False)
    print(f"[gate {step}] {status}: {note}")


def env_info():
    import sklearn, scipy, numpy
    return {
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def streams():
    import streams as S
    return S


def runner():
    import run_v3 as R
    return R
