"""Cost-reduction final runner.

Nine methods on four streams, one seed per call, batched per-window inference,
n_jobs=1 everywhere. Records adaptation, prediction and total CPU, retrains and
reuses. Mirrors final2_runner's record schema exactly so aggregation is shared.

Usage:
    python experiments/final2/cost_runner.py --mode dev   --tag cost_dev
    python experiments/final2/cost_runner.py --mode final --tag cost_final
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
for p in (HERE, os.path.join(ROOT, "experiments", "final"),
          os.path.join(ROOT, "experiments", "exp9a")):
    sys.path.insert(0, p)

from final_config import SEEDS_DEV, SEEDS_FINAL, DEV_WINDOW_FRACTION, BUFFER_CAPACITY
from final_methods import build_method
from run_final import _prepare_arrays
from final2_runner import prepare, SLUGS, OUT
from cost_methods import build_cost_method, METHODS, RAPTGuardCheap
from models_9a import make_class_anchor

RESULTS2 = os.path.join(ROOT, "results", "final2")
RAW2 = os.path.join(RESULTS2, "raw")
os.makedirs(RAW2, exist_ok=True)


def run_seed(stream, sd, seed, cfg, stop_frac, methods=None):
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = \
        _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    n_eval = max(1, int(round((n_windows - n_init) * stop_frac)))
    stop_w = n_init + n_eval
    rows = []
    methods = methods or METHODS

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
        if extra:
            r.update(extra)
        rows.append(r)

    def err_of(yp, idx):
        return 1.0 - float(np.mean(yp == y_all[idx]))

    for method in methods:
        m = build_cost_method(method, seed=seed, anchor_X=anchor_X,
                              anchor_y=anchor_y, cfg=cfg)
        if method == "RAPT-Guard":
            m.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
        elif method in ("Full Retraining", "Full Retraining-cheap",
                        "Event-Driven", "Event-Driven-cheap", "Event-Driven+Reuse"):
            m.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
        else:
            m.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = regimes[n_init - 1]
        for w in range(n_init, stop_w):
            idx = win_slices[w]; reg = regimes[w]
            m.last_flags = {}
            needs_err = method in ("Event-Driven", "Event-Driven-cheap",
                                   "Event-Driven+Reuse", "Page-Hinkley")
            er = err_of(m.predict(X[idx]), idx) if needs_err else None
            if isinstance(m, RAPTGuardCheap):
                yp, pc = m.predict_window(X[idx], buf_X, buf_y)
                ac = 0.0
                if reg != prev:
                    ac = m.on_regime_change(reg, w, X[idx], y_all[idx],
                                            buf_X, buf_y, prev_regime=prev)
                    prev = reg
                m.record_window(w, X[idx], y_all[idx])
            elif method in ("Full Retraining", "Full Retraining-cheap",
                            "Event-Driven"):
                t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
                ac = m.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y,
                                    err=er, regime=reg)
                prev = reg
            elif method == "Event-Driven-cheap":
                t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
                ac = m.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=er)
                prev = reg
            elif method == "Event-Driven+Reuse":
                t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
                ac = m.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y,
                                    err=er, regime=reg)
                prev = reg
            elif method == "Page-Hinkley":
                t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
                ac = m.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=er)
                prev = reg
            elif method in ("RAPT", "RAPT-Guard-std"):
                yp, pc = m.predict_window(X[idx], buf_X, buf_y)
                ac = 0.0
                if reg != prev:
                    ac = m.on_regime_change(reg, w, X[idx], y_all[idx], buf_X, buf_y,
                                            prev_regime=prev)
                    prev = reg
                ac += m.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y)
            else:  # Frozen
                t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
                ac = m.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y)
                prev = reg
            buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            fl = m.last_flags
            rec(method, w, reg, idx, yp, ac, pc,
                {"is_retrain": fl.get("retrain", 0), "is_reuse": fl.get("reuse", 0),
                 "gate_allow": fl.get("gate_allow", 0),
                 "gate_block": fl.get("gate_block", 0),
                 "gate_skipped": fl.get("gate_skipped", 0)})

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "final"], default="final")
    ap.add_argument("--streams", default="all")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--tag", default="cost_final")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--methods", default="")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    cfg.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})

    slugs = list(SLUGS) if a.streams == "all" else a.streams.split(",")
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds
             else (SEEDS_DEV if a.mode == "dev" else SEEDS_FINAL))
    stop_frac = DEV_WINDOW_FRACTION if a.mode == "dev" else 1.0
    methods = [m for m in a.methods.split(",") if m] or None

    from joblib import Parallel, delayed
    jobs = [(s, seed) for s in slugs for seed in seeds]

    def one(slug, seed):
        stream, sd = prepare(SLUGS[slug])
        df = run_seed(stream, sd, seed, cfg, stop_frac, methods=methods)
        df["dataset"] = SLUGS[slug]
        df["dataset_slug"] = slug
        return df

    t0 = time.time()
    res = Parallel(n_jobs=min(a.n_jobs, len(jobs)), verbose=5)(delayed(one)(*j) for j in jobs)
    allw = pd.concat(res, ignore_index=True)
    out = os.path.join(RAW2, f"per_window_all_{a.tag}.csv")
    allw.to_csv(out, index=False)
    print(f"wrote {out}: {len(allw)} rows, {time.time()-t0:.0f}s")
    print(allw.groupby(["dataset_slug", "method"]).size())


if __name__ == "__main__":
    main()
