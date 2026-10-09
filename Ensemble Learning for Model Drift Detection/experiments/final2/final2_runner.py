"""Task A/D runner: evaluate the frozen method set on four streams, all seeds.

Wraps run_final's stream prep + record schema, adds the KPI-operators stream
(Task C) and the RAPT-ED-Guard method (Task A). No base model is changed.

Usage:
    python experiments/final2/final2_runner.py --mode dev  --streams all --tag dev
    python experiments/final2/final2_runner.py --mode final --streams all --tag final
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FINAL = os.path.join(ROOT, "experiments", "final")
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
for p in (HERE, FINAL, EXP9A):
    sys.path.insert(0, p)

from final_config import (SEEDS_DEV, SEEDS_FINAL, DEV_WINDOW_FRACTION,
                          BUFFER_CAPACITY, BASE_MODELS)
from final_methods import build_method
from run_final import _prepare_stream, _prepare_arrays, run_seed_inner
from final2_methods import RAPTEDGuard

RESULTS2 = os.path.join(ROOT, "results", "final2")
RAW2 = os.path.join(RESULTS2, "raw")
os.makedirs(RAW2, exist_ok=True)

METHODS = ["Frozen", "Full Retraining", "Event-Driven", "Event-Driven+Reuse",
           "ADWIN", "Page-Hinkley", "Frozen+Periodic", "RAPT", "RAPT-Guard",
           "RAPT-ED-Guard", "RAPT-Enhanced", "RAPT-Anchor"]

SLUGS = {"campus": "5G Campus QoS", "ugr16": "UGR'16", "5g_nr": "5G NR Latency",
         "kpiops": "Network operator KPIs"}
OUT = {"5G Campus QoS": "5g_campus", "UGR'16": "ugr16",
       "5G NR Latency": "5g_nr", "Network operator KPIs": "kpiops"}


def prepare(name):
    if name == "Network operator KPIs":
        from final2_load import load_kpiops
        stream, sd = load_kpiops()
        sd = dict(sd)
        sd["n_windows"] = sd["total_windows"]
        sd["feature_columns"] = [c for c in sd["feature_columns"]]
        return stream, sd
    return _prepare_stream(name)


def run_seed(stream, sd, seed, cfg, stop_frac):
    """Same protocol as run_seed_inner, but only the frozen method set, plus
    RAPT-ED-Guard."""
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = \
        _prepare_arrays(stream, sd)
    from models_9a import make_class_anchor
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    n_eval = max(1, int(round((n_windows - n_init) * stop_frac)))
    stop_w = n_init + n_eval
    rows = []

    def rec(method, w, reg, idx, yp, adapt_cpu, pred_cpu, extra=None):
        from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                                     recall_score, balanced_accuracy_score)
        yt = y_all[idx]
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
        if len(yt) == 1:
            r["y_true_single"] = int(yt[0]); r["y_pred_single"] = int(yp[0])
        if extra:
            r.update(extra)
        rows.append(r)

    # ---- Frozen ----
    m = build_method("Frozen", seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
    m.fit_initial(X_init, y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
        rec("Frozen", w, regimes[w], idx, yp, 0.0, pc, {"is_retrain": 0, "is_reuse": 0})

    # ---- Frozen + periodic ----
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
        rec("Event-Driven", w, regimes[w], idx, yp, ac, pc,
            {"is_retrain": ed.last_flags.get("retrain", 0), "is_reuse": 0})

    # ---- Event-Driven + Reuse ----
    edr = build_method("Event-Driven+Reuse", seed=seed, anchor_X=anchor_X,
                       anchor_y=anchor_y, cfg=cfg)
    edr.fit_initial(X_init, y_init, init_regime=regimes[n_init - 1])
    buf_X, buf_y = list(X_init), list(y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]; reg = regimes[w]
        t0 = time.process_time(); yp = edr.predict(X[idx]); pc = time.process_time() - t0
        err = 1.0 - np.mean(yp == y_all[idx])
        ac = edr.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err, regime=reg)
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
    for w in range(n_init, stop_w):
        idx = win_slices[w]; reg = regimes[w]
        t0 = time.process_time(); yp = fr.predict(X[idx]); pc = time.process_time() - t0
        ac = fr.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, regime=reg)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        rec("Full Retraining", w, reg, idx, yp, ac, pc,
            {"is_retrain": fr.last_flags.get("retrain", 0), "is_reuse": 0})

    # ---- Page-Hinkley ----
    dm = build_method("Page-Hinkley", seed=seed, anchor_X=anchor_X,
                      anchor_y=anchor_y, cfg=cfg)
    dm.fit_initial(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        t0 = time.process_time(); yp = dm.predict(X[idx]); pc = time.process_time() - t0
        err = (yp != y_all[idx]).astype(float)
        ac = dm.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        rec("Page-Hinkley", w, regimes[w], idx, yp, ac, pc,
            {"is_retrain": dm.last_flags.get("retrain", 0), "is_reuse": 0,
             "detected_events": dm.last_flags.get("detected", 0)})

    # ---- ADWIN ----
    ad = build_method("ADWIN", seed=seed, anchor_X=anchor_X,
                      anchor_y=anchor_y, cfg=cfg)
    ad.fit_initial(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        t0 = time.process_time(); yp = ad.predict(X[idx]); pc = time.process_time() - t0
        err = (yp != y_all[idx]).astype(float)
        ac = ad.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        rec("ADWIN", w, regimes[w], idx, yp, ac, pc,
            {"is_retrain": ad.last_flags.get("retrain", 0), "is_reuse": 0,
             "detected_events": ad.last_flags.get("detected", 0)})

    # ---- RAPT family ----
    for method in ["RAPT", "RAPT-Guard", "RAPT-Enhanced", "RAPT-Anchor"]:
        s = build_method(method, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
        s.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = regimes[n_init - 1]
        for w in range(n_init, stop_w):
            idx = win_slices[w]; reg = regimes[w]
            s.last_flags = {}
            yp, pc = s.predict_window(X[idx], buf_X, buf_y)
            ac = 0.0
            if reg != prev:
                ac = s.on_regime_change(reg, w, X[idx], y_all[idx], buf_X, buf_y,
                                        prev_regime=prev)
                prev = reg
            ac += s.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, prev_regime=prev)
            buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            fl = s.last_flags
            rec(method, w, reg, idx, yp, ac, pc,
                {"is_retrain": fl.get("retrain", 0), "is_reuse": fl.get("reuse", 0),
                 "gate_allow": fl.get("gate_allow", 0), "gate_block": fl.get("gate_block", 0)})

    # ---- RAPT-ED-Guard ----
    g = RAPTEDGuard(seed, anchor_X, anchor_y, cfg)
    g.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = regimes[n_init - 1]
    for w in range(n_init, stop_w):
        idx = win_slices[w]; reg = regimes[w]
        yp, pc = g.predict_window(X[idx], buf_X, buf_y)
        err = 1.0 - np.mean(yp == y_all[idx])
        ac = g.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y,
                            err=err, regime=reg, prev_regime=prev)
        prev = reg
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        fl = g.last_flags
        rec("RAPT-ED-Guard", w, reg, idx, yp, ac, pc,
            {"is_retrain": fl.get("retrain", 0), "is_reuse": fl.get("reuse", 0),
             "gate_allow": fl.get("gate_allow", 0), "gate_block": fl.get("gate_block", 0)})

    df = pd.DataFrame(rows)
    df.attrs["gate_log"] = g.gate_log
    return df


def aggregate(df):
    g = df.groupby("method")
    out = g.agg(
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
        adaptation_cpu_sec=("adaptation_cpu_sec", "mean"),
        prediction_cpu_sec=("prediction_cpu_sec", "mean"),
        total_cpu_sec=("total_cpu_sec", "mean"),
        retrain_events=("is_retrain", "sum"), reuse_events=("is_reuse", "sum"),
        n_windows=("window_id", "count"), n_seeds=("seed", "nunique")).reset_index()
    per_seed = df.groupby(["method", "seed"]).agg(
        r=("is_retrain", "sum"), u=("is_reuse", "sum")).reset_index()
    ev = per_seed.groupby("method").agg(per_seed_retrains=("r", "mean"),
                                        per_seed_reuses=("u", "mean")).reset_index()
    return out.merge(ev, on="method", how="left").sort_values("method").reset_index(drop=True)


def _one(name, seed, cfg, mode):
    stream, sd = prepare(name)
    stop_frac = DEV_WINDOW_FRACTION if mode == "dev" else 1.0
    df = run_seed(stream, sd, seed, cfg, stop_frac)
    df["dataset"] = name
    if df.attrs.get("gate_log"):
        gl = pd.DataFrame(df.attrs["gate_log"])
        gl["dataset"] = name; gl["seed"] = seed
        d = os.path.join(RAW2, "gate_logs"); os.makedirs(d, exist_ok=True)
        gl.to_csv(os.path.join(d, f"gate_{OUT[name]}_{seed}.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "final"])
    ap.add_argument("--streams", default="all")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    streams = ["campus", "ugr16", "5g_nr", "kpiops"] if a.streams == "all" \
        else a.streams.split(",")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else \
        (SEEDS_DEV if a.mode == "dev" else SEEDS_FINAL)
    cfg = {}
    import yaml
    cfgp = a.cfg or os.path.join(ROOT, "configs", "final2.yaml")
    if os.path.exists(cfgp):
        cfg.update(yaml.safe_load(open(cfgp)) or {})
    if a.mode == "dev":
        for k, v in (yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}).items():
            cfg.setdefault(k, v)
    else:
        base = os.path.join(ROOT, "configs", "final.yaml")
        if os.path.exists(base):
            for k, v in (yaml.safe_load(open(base)) or {}).items():
                cfg.setdefault(k, v)
    out_tag = a.tag or a.mode
    from joblib import Parallel, delayed
    jobs = [(SLUGS[s], seed) for s in streams for seed in seeds]
    t0 = time.time()
    res = Parallel(n_jobs=min(a.n_jobs, len(jobs)), verbose=5)(
        delayed(_one)(name, seed, cfg, a.mode) for (name, seed) in jobs)
    allw = pd.concat(res, ignore_index=True)
    for name in set(SLUGS[s] for s in streams):
        sub = allw[allw["dataset"] == name]
        sub.to_csv(os.path.join(RAW2, f"per_window_{OUT[name]}_{out_tag}.csv"), index=False)
        aggregate(sub).to_csv(os.path.join(RAW2, f"summary_{OUT[name]}_{out_tag}.csv"), index=False)
        print(f"[{name}] {len(sub)} rows -> {OUT[name]}_{out_tag}", flush=True)
    allw.to_csv(os.path.join(RAW2, f"per_window_all_{out_tag}.csv"), index=False)
    print(f"TOTAL {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
