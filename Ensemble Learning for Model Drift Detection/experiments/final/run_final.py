"""
Unified final harness for the RAPT paper.

Four streams, one prequential protocol, one base architecture, one shared
class-anchored retraining buffer. Every number in the paper's final results
comes from here.

Run from the "Ensemble Learning for Model Drift Detection" directory:

    python experiments/final/run_final.py <mode> [stream] [--seeds a,b,c]

modes:
    dev    -- 5 seeds (42-46), first 60% of post-prefix windows (config sweep)
    final  -- all 30 seeds (42-71), full stream (frozen config)

Data provenance (third-party files are not committed; see
experiments/exp9a/data/DATASETS.md):
    UGR'16    -> experiments/exp9a/data/ugr16/UGR16v1.mat
    NordicDat -> experiments/exp9a/data/nordicdat/nordicdat.csv
The two small CSV streams (5G Campus, 5G NR latency) are committed.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
EXP9B = os.path.join(ROOT, "experiments", "exp9b")
sys.path.insert(0, HERE)
sys.path.insert(0, EXP9A)
sys.path.insert(0, EXP9B)

from final_config import (SEEDS_DEV, SEEDS_FINAL, BASE_MODELS, DEV_WINDOW_FRACTION,
                          BUFFER_CAPACITY, ERROR_WINDOW_SIZE, ERROR_THRESHOLD_K,
                          TUNE_STREAMS, RESULTS_FINAL, RAW_FINAL, ensure_dirs)
from final_methods import build_method
from three_dataset_load import LOADERS as LOADERS_9A, build_windows as BW_9A
from exp9b_testutil import load_9b_stream
from models_9a import create_base_ensemble, make_class_anchor

DETECTORS = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]
_MODELS = BASE_MODELS


# ---------------------------------------------------------------------------
# Stream adapters -> common (windows_df, sd)
# ---------------------------------------------------------------------------
def _prepare_stream(name):
    if name == "5G NR Latency":
        stream, sd = load_9b_stream()
        sd = dict(sd)
        sd["name"] = name
        sd["slug"] = "5g_nr"
        # 9B evaluates one row per window; the prefix is the first 20% of
        # windows, so the first evaluated window equals initial_train_windows.
        sd["n_windows"] = sd["initial_train_windows"]
        return stream, sd
    data = LOADERS_9A[name]()
    stream, sd = BW_9A(data)
    sd["name"] = name
    sd["n_windows"] = sd["total_windows"]
    return stream, sd


def _prepare_arrays(stream, sd):
    feat = sd["feature_columns"]
    X_all = stream[feat].values.astype(np.float64)
    y_all = stream["y"].values.astype(int)
    n_init = sd["initial_train_windows"]
    n_windows = sd["total_windows"]
    # one regime id per window (sample-level regime_id broadcast to windows)
    win_regime = (stream.groupby("window_id")["regime_id"].first()
                  .reindex(range(n_windows)).ffill().bfill().values)
    init_mask = stream["window_id"].values < n_init
    mu = X_all[init_mask].mean(axis=0)
    sd_ = np.where(X_all[init_mask].std(axis=0) < 1e-9, 1.0,
                   X_all[init_mask].std(axis=0))
    X_scaled = (X_all - mu) / sd_
    X_init = X_scaled[init_mask]
    y_init = y_all[init_mask]
    win_slices = {w: np.where(stream["window_id"].values == w)[0]
                  for w in range(n_init, n_windows)}
    return X_scaled, y_all, win_regime, X_init, y_init, win_slices, n_init, n_windows


# ---------------------------------------------------------------------------
# A controller that also scores itself on the labelled lookback ("guard score")
# ---------------------------------------------------------------------------
class GuardScorer:
    """Wrapper around a base policy that can report rolling accuracy on the
    most recent labelled window rows, mirroring what RAPT-Guard compares."""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict(self, X):
        return self.ensemble.predict(X)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def run_seed_inner(stream, sd, seed, cfg, stop_frac):
    """Run every base method for one (stream, seed). stop_frac caps the number
    of evaluated post-prefix windows (used in dev). Returns per-window rows."""
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = \
        _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream["window_id"].values
    n_eval = max(1, int(round((n_windows - n_init) * stop_frac)))
    stop_w = n_init + n_eval
    rows = []

    def rec(method, w, reg, idx, yp, adapt_cpu, pred_cpu, extra=None):
        yt = y_all[idx]
        from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                                     recall_score, balanced_accuracy_score)
        r = dict(seed=seed, method=method, window_id=int(w), regime_id=str(reg),
                 macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
                 accuracy=accuracy_score(yt, yp),
                 balanced_accuracy=balanced_accuracy_score(yt, yp),
                 precision=precision_score(yt, yp, average="macro", zero_division=0),
                 recall=recall_score(yt, yp, average="macro", zero_division=0),
                 adaptation_cpu_sec=float(adapt_cpu),
                 prediction_cpu_sec=float(pred_cpu),
                 total_cpu_sec=float(adapt_cpu + pred_cpu),
                 n_samples=int(len(yt)))
        # 5G NR is a one-row-per-window stream; keep the single label so the
        # aggregator can reproduce the pooled 9B protocol.
        if len(yt) == 1:
            r["y_true_single"] = int(yt[0])
            r["y_pred_single"] = int(yp[0])
        if extra:
            r.update(extra)
        rows.append(r)

    # ---- Frozen ----
    m = create_base_ensemble(seed=seed)
    m.fit(X_init, y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
        rec("Frozen", w, regimes[w], idx, yp, 0.0, pc, {"is_retrain": 0, "is_reuse": 0})

    # ---- Frozen + periodic refit ----
    fp = build_method("Frozen+Periodic", seed=seed, anchor_X=anchor_X,
                      anchor_y=anchor_y, cfg=cfg)
    fp.fit_initial(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        t0 = time.process_time(); yp = fp.predict(X[idx]); pc = time.process_time() - t0
        ac = fp.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        fl = fp.last_flags
        rec("Frozen+Periodic", w, regimes[w], idx, yp, ac, pc,
            {"is_retrain": fl.get("retrain", 0), "is_reuse": 0})

    # ---- Event-Driven ----
    ed = build_method("Event-Driven", seed=seed, anchor_X=anchor_X,
                      anchor_y=anchor_y, cfg=cfg)
    ed.fit_initial(X_init, y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        t0 = time.process_time(); yp = ed.predict(X[idx]); pc = time.process_time() - t0
        err = 1.0 - np.mean(yp == y_all[idx])
        ac = ed.post_predict(w, idx, X[idx], y_all[idx], None, None, err=err)
        fl = ed.last_flags
        rec("Event-Driven", w, regimes[w], idx, yp, ac, pc,
            {"is_retrain": fl.get("retrain", 0), "is_reuse": 0})

    # ---- Event-Driven + Reuse ----
    edr = build_method("Event-Driven+Reuse", seed=seed, anchor_X=anchor_X,
                       anchor_y=anchor_y, cfg=cfg)
    edr.fit_initial(X_init, y_init, init_regime=regimes[n_init - 1])
    buf_X, buf_y = list(X_init), list(y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        reg = regimes[w]
        t0 = time.process_time(); yp = edr.predict(X[idx]); pc = time.process_time() - t0
        err = 1.0 - np.mean(yp == y_all[idx])
        ac = edr.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y,
                              err=err, regime=reg)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        fl = edr.last_flags
        rec("Event-Driven+Reuse", w, reg, idx, yp, ac, pc,
            {"is_retrain": fl.get("retrain", 0), "is_reuse": fl.get("reuse", 0)})

    # ---- Full Retraining ----
    fr = build_method("Full Retraining", seed=seed, anchor_X=anchor_X,
                      anchor_y=anchor_y, cfg=cfg)
    fr.fit_initial(X_init, y_init, init_regime=regimes[n_init - 1])
    buf_X, buf_y = list(X_init), list(y_init)
    prev = regimes[n_init - 1]
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        reg = regimes[w]
        t0 = time.process_time(); yp = fr.predict(X[idx]); pc = time.process_time() - t0
        ac = fr.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, regime=reg)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        rec("Full Retraining", w, reg, idx, yp, ac, pc,
            {"is_retrain": fr.last_flags.get("retrain", 0), "is_reuse": 0})

    # ---- Historical detectors ----
    for det in DETECTORS:
        dm = build_method(det, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
        dm.fit_initial(X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        for w in range(n_init, stop_w):
            idx = win_slices[w]
            t0 = time.process_time(); yp = dm.predict(X[idx]); pc = time.process_time() - t0
            # historical detectors consume the per-sample 0/1 error stream,
            # which is what ADWIN / Page-Hinkley expect.
            err = (yp != y_all[idx]).astype(float)
            ac = dm.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err)
            buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            fl = dm.last_flags
            rec(det, w, regimes[w], idx, yp, ac, pc,
                {"is_retrain": fl.get("retrain", 0), "is_reuse": 0,
                 "detected_events": fl.get("detected", 0)})

    # ---- RAPT family (regime-boundary keyed, with update hooks) ----
    for method in ["RAPT", "RAPT-Enhanced", "RAPT-Guard", "RAPT-Anchor"]:
        sys_ = build_method(method, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
        sys_.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = regimes[n_init - 1]
        for w in range(n_init, stop_w):
            idx = win_slices[w]
            reg = regimes[w]
            sys_.last_flags = {}
            yp, pc = sys_.predict_window(X[idx], buf_X, buf_y)
            ac = 0.0
            if reg != prev:
                ac = sys_.on_regime_change(reg, w, X[idx], y_all[idx], buf_X, buf_y,
                                           prev_regime=prev)
                prev = reg
            ac += sys_.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y,
                                    prev_regime=prev)
            buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            fl = sys_.last_flags
            rec(method, w, reg, idx, yp, ac, pc,
                {"is_retrain": fl.get("retrain", 0), "is_reuse": fl.get("reuse", 0),
                 "gate_allow": fl.get("gate_allow", 0),
                 "gate_block": fl.get("gate_block", 0)})

    return pd.DataFrame(rows)


def run_seed(stream, sd, seed, cfg, mode):
    stop_frac = DEV_WINDOW_FRACTION if mode == "dev" else 1.0
    return run_seed_inner(stream, sd, seed, cfg, stop_frac)


if __name__ == "__main__":
    print("This module is driven by run_final_driver.py")
