"""Revalidation common utilities.

Loads the existing 9A / 9B / Final_Experiments pipelines unchanged and adds
pooled metrics, per-window confusion counts, and a shared RAPT controller that
logs checkpoint provenance.

Nothing here modifies existing files; the original modules are imported.
"""
import os
import sys
import json
import time
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
EXP9B = os.path.join(ROOT, "experiments", "exp9b")
sys.path.insert(0, EXP9A)
sys.path.insert(0, EXP9B)

OUT = os.path.join(ROOT, "results", "revalidation")
RAW = os.path.join(OUT, "raw")
LOGS = os.path.join(OUT, "logs")
CONFIG_FROZEN = os.path.join(OUT, "config_frozen.json")

SEEDS = [42, 43, 44, 45, 46]


def ensure_dirs():
    for d in (OUT, RAW, LOGS):
        os.makedirs(d, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config_frozen():
    with open(CONFIG_FROZEN) as f:
        return json.load(f)


def write_csv(df, name):
    ensure_dirs()
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    return path


def write_raw_csv(df, name):
    ensure_dirs()
    path = os.path.join(RAW, name)
    df.to_csv(path, index=False)
    return path


class Timer:
    """process_time + wall clock around a block, appended to a global log."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.w0 = time.perf_counter()
        self.c0 = time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.w0
        self.cpu = time.process_time() - self.c0
        return False
