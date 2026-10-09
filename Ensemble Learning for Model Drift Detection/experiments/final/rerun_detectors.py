"""
Re-run only the four historical detectors (ADWIN, Page-Hinkley, EDD, EDMA) on
the frozen final config and splice their per-window rows back into the saved
per_window_<slug>_final.csv files. The other nine methods are byte-identical to
the committed run, so this avoids repeating a 51-minute job after a detector
harness fix.

Run from the "Ensemble Learning for Model Drift Detection" directory:
    python experiments/final/rerun_detectors.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp9a"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9b"))

from final_config import SEEDS_FINAL, DEFAULT_CFG, RAW_FINAL, CONFIG_DIR
from final_methods import build_method
from models_9a import make_class_anchor
from run_final import _prepare_stream, _prepare_arrays

DETECTORS = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]
STREAMS = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
           "nordicdat": "NordicDat", "5g_nr": "5G NR Latency"}
BUFFER_CAPACITY = 1000


def _cfg():
    import yaml
    p = os.path.join(CONFIG_DIR, "final.yaml")
    with open(p) as f:
        return {**DEFAULT_CFG, **{k: v for k, v in (yaml.safe_load(f) or {}).items()
                                  if v is not None}}


def run_one(slug, name, seed, cfg):
    from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                                 recall_score, balanced_accuracy_score)
    stream, sd = _prepare_stream(name)
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = \
        _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    rows = []
    for det in DETECTORS:
        dm = build_method(det, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y,
                          cfg=cfg)
        dm.fit_initial(X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            t0 = time.process_time(); yp = dm.predict(X[idx]); pc = time.process_time() - t0
            err = (yp != y_all[idx]).astype(float)
            ac = dm.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err)
            buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            yt = y_all[idx]
            r = dict(seed=seed, method=det, window_id=int(w),
                     regime_id=str(regimes[w]),
                     macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
                     accuracy=accuracy_score(yt, yp),
                     balanced_accuracy=balanced_accuracy_score(yt, yp),
                     precision=precision_score(yt, yp, average="macro", zero_division=0),
                     recall=recall_score(yt, yp, average="macro", zero_division=0),
                     adaptation_cpu_sec=float(ac),
                     prediction_cpu_sec=float(pc),
                     total_cpu_sec=float(ac + pc), n_samples=int(len(yt)),
                     is_retrain=dm.last_flags.get("retrain", 0), is_reuse=0,
                     detected_events=dm.last_flags.get("detected", 0),
                     gate_allow=0, gate_block=0, dataset=slug)
            if len(yt) == 1:
                r["y_true_single"] = int(yt[0]); r["y_pred_single"] = int(yp[0])
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="all")
    a = ap.parse_args()
    slugs = list(STREAMS) if a.streams == "all" else a.streams.split(",")
    cfg = _cfg()
    for slug in slugs:
        name = STREAMS[slug]
        path = os.path.join(RAW_FINAL, f"per_window_{slug}_final.csv")
        old = pd.read_csv(path)
        keep = old[~old["method"].isin(DETECTORS)].copy()
        new = []
        for seed in SEEDS_FINAL:
            new.append(run_one(slug, name, seed, cfg))
            print(f"  {slug} seed={seed} done", flush=True)
        new = pd.concat(new, ignore_index=True)
        merged = pd.concat([keep, new], ignore_index=True)
        merged = merged[old.columns]
        merged.to_csv(path, index=False)
        print(f"{slug}: replaced detectors, {len(merged)} rows", flush=True)


if __name__ == "__main__":
    main()
