"""
Dev-mode sweeps (STEP 1). All tuning uses 5 seeds (42-46) and the first 60% of
post-prefix windows. Nothing here is used for the final numbers except the
single frozen configuration it produces.

Outputs JSON/CSV into results/final/dev/.

Usage:
    python experiments/final/sweeps.py --what detectors
    python experiments/final/sweeps.py --what guard
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp9a"))

from final_config import SEEDS_DEV, DEFAULT_CFG, DEV_WINDOW_FRACTION, RAW_FINAL
from final_methods import build_method
from models_9a import make_class_anchor
from run_final import _prepare_stream, _prepare_arrays
from sklearn.metrics import f1_score

DEV_DIR = os.path.join(RAW_FINAL, "..", "dev")
os.makedirs(DEV_DIR, exist_ok=True)

STREAMS = ["campus", "ugr16", "nordicdat"]
NAME = {"campus": "5G Campus QoS", "ugr16": "UGR'16", "nordicdat": "NordicDat"}

_CACHE = {}


def _stream(slug):
    if slug not in _CACHE:
        _CACHE[slug] = _prepare_stream(NAME[slug])
    return _CACHE[slug]


def _run_detector(slug, seed, name, override):
    stream, sd = _stream(slug)
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    cfg = dict(DEFAULT_CFG); cfg.update(override)
    dm = build_method(name, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
    dm.fit_initial(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    stop_w = n_init + max(1, int(round((n_windows - n_init) * DEV_WINDOW_FRACTION)))
    events = retrains = 0
    adapt = 0.0
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        yp = dm.predict(X[idx])
        err = (yp != y_all[idx]).astype(float)
        adapt += dm.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err)
        events += dm.last_flags.get("detected", 0)
        retrains += dm.last_flags.get("retrain", 0)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > 1000:
            buf_X, buf_y = buf_X[-1000:], buf_y[-1000:]
    return dict(dataset=slug, seed=seed, detector=name, **override,
                detected_events=events, retrain_events=retrains,
                adaptation_cpu_sec=adapt)


def _run_guard(slug, seed, lb, mg):
    stream, sd = _stream(slug)
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    cfg = dict(DEFAULT_CFG); cfg["guard_lookback"] = lb; cfg["guard_margin"] = mg
    sys_ = build_method("RAPT-Guard", seed=seed, anchor_X=anchor_X,
                        anchor_y=anchor_y, cfg=cfg)
    sys_.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = regimes[n_init - 1]
    stop_w = n_init + max(1, int(round((n_windows - n_init) * DEV_WINDOW_FRACTION)))
    f1s, allow, block = [], 0, 0
    for w in range(n_init, stop_w):
        idx = win_slices[w]; reg = regimes[w]
        sys_.last_flags = {}
        yp, pc = sys_.predict_window(X[idx], buf_X, buf_y)
        ac = 0.0
        if reg != prev:
            ac = sys_.on_regime_change(reg, w, X[idx], y_all[idx], buf_X, buf_y, prev_regime=prev)
            prev = reg
        ac += sys_.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, prev_regime=prev)
        allow += sys_.last_flags.get("gate_allow", 0)
        block += sys_.last_flags.get("gate_block", 0)
        f1s.append(f1_score(y_all[idx], yp, average="macro", zero_division=0))
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > 1000:
            buf_X, buf_y = buf_X[-1000:], buf_y[-1000:]
    return dict(dataset=slug, seed=seed, lookback=lb, margin=mg,
                macro_f1=float(np.mean(f1s)), gate_allow=allow, gate_block=block)


def _parallel(jobs, fn):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=min(4, len(jobs)), verbose=5)(delayed(fn)(*j) for j in jobs)


def sweep_detectors():
    grid = []
    for d in [0.002, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        grid.append(("ADWIN", {"adwin_delta": d}))
    for th in [0.05, 0.5, 5.0, 50.0]:
        for mi in [5, 10, 30]:
            grid.append(("Page-Hinkley",
                         {"ph_threshold": th, "ph_min_instances": mi,
                          "ph_delta": 0.005, "ph_alpha": 0.9999}))
    jobs = [(slug, seed, det, ov) for slug in STREAMS for (det, ov) in grid
            for seed in SEEDS_DEV]
    rows = _parallel(jobs, _run_detector)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DEV_DIR, "detector_sweep.csv"), index=False)
    out = {}
    for det in ["ADWIN", "Page-Hinkley"]:
        sub = df[df["detector"] == det].copy()
        sub["cfg"] = sub.apply(lambda r: json.dumps(
            {k: v for k, v in r.items() if k.startswith(("adwin", "ph_"))}, sort_keys=True), axis=1)
        g = sub.groupby("cfg").agg(median_events=("detected_events", "median"),
                                   retrains=("retrain_events", "mean"),
                                   cpu=("adaptation_cpu_sec", "mean")).reset_index()
        g["score"] = (g["median_events"] - 15).abs()
        best = g.sort_values(["score", "median_events"]).iloc[0]
        out[det] = json.loads(best["cfg"])
        out[det]["_median_events"] = float(best["median_events"])
        out[det]["_mean_retrains"] = float(best["retrains"])
        print(det, "->", out[det], flush=True)
    with open(os.path.join(DEV_DIR, "detector_choice.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def sweep_guard():
    jobs = [(slug, seed, lb, mg) for slug in STREAMS for lb in [50, 75, 100]
            for mg in [-0.05, 0.0, 0.02, 0.05, 0.1] for seed in SEEDS_DEV]
    rows = _parallel(jobs, _run_guard)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DEV_DIR, "guard_sweep.csv"), index=False)
    out = choose_guard(df)
    with open(os.path.join(DEV_DIR, "guard_choice.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("guard ->", out, flush=True)
    return out


def choose_guard(df):
    """Pick (lookback, margin) maximising dev macro-F1 subject to keeping the
    gate a gate: the reuse (allow) rate must be at least 0.30 of gate decisions.
    Without this floor the best-F1 corner is 'block everything', which collapses
    RAPT-Guard onto a plain refit-every-return policy and is not a gate."""
    g = df.groupby(["lookback", "margin"]).agg(
        f1=("macro_f1", "mean"), allow=("gate_allow", "sum"),
        block=("gate_block", "sum")).reset_index()
    g["decisions"] = g["allow"] + g["block"]
    g["allow_rate"] = g["allow"] / g["decisions"].replace(0, np.nan)
    eligible = g[g["allow_rate"] >= 0.30]
    pick = (eligible if len(eligible) else g).sort_values("f1", ascending=False).iloc[0]
    return {"guard_lookback": int(pick["lookback"]),
            "guard_margin": float(pick["margin"]),
            "_dev_f1": float(pick["f1"]), "_allow": int(pick["allow"]),
            "_block": int(pick["block"]), "_allow_rate": float(pick["allow_rate"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["detectors", "guard"])
    a = ap.parse_args()
    t0 = time.time()
    (sweep_detectors if a.what == "detectors" else sweep_guard)()
    print(f"done in {time.time() - t0:.1f}s")
