"""
Final_Experiments runner.

Prequential Test-Then-Train, sequential chronological, no future leakage.
Reuses the existing exp9a preprocessing (same window size, imputation, target)
and the existing Frozen / Event-Driven / Full-Retraining baselines unchanged.

Adds the RAPT mechanism ladder (T2 -> T1 -> T1+refit -> full).

Usage:  python run_final.py [pilot|full]
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "exp9a"))

from config import (SEEDS, MODELS, RAPT_VARIANTS, INITIAL_FRAC,
                    BUFFER_CAPACITY, RAW_DIR, ensure_dirs)
from rapt_ladder import build_rapt
import three_dataset_load as tdl
from models_9a import create_base_ensemble, make_train_buffer, make_class_anchor
from event_driven_9a import EventDrivenEnsemble

ERROR_WINDOW_SIZE = 20
ERROR_THRESHOLD_K = 2.0


def load_stream(initial_frac=INITIAL_FRAC):
    data = tdl.load_campus()
    stream, sd = tdl.build_windows(data, initial_frac=initial_frac)
    return stream, sd


def _prepare(stream):
    feat_cols = [c for c in stream.columns
                 if c not in ("window_id", "regime_id", "y", "_t0")]
    X = stream[feat_cols].values.astype(np.float64)
    y = stream["y"].values.astype(int)
    regimes = stream["regime_id"].values
    wid = stream["window_id"].values
    n_windows = int(wid.max()) + 1
    win_slices = [np.where(wid == w)[0] for w in range(n_windows)]
    n_init = max(1, int(round(n_windows * INITIAL_FRAC)))
    init_idx = np.where(wid < n_init)[0]
    return (X, y, regimes, X[init_idx], y[init_idx], win_slices, n_init, n_windows)


def _rec(seed, method, w, regime, yt, yp, adapt_cpu, is_retrain, is_reuse, pred_cpu):
    return dict(seed=seed, method=method, window_id=w, regime_id=regime,
                macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
                accuracy=accuracy_score(yt, yp),
                precision=precision_score(yt, yp, average="macro", zero_division=0),
                recall=recall_score(yt, yp, average="macro", zero_division=0),
                adaptation_cpu_sec=adapt_cpu, prediction_cpu_sec=pred_cpu,
                is_retrain=int(is_retrain), is_reuse=int(is_reuse),
                n_samples=len(yt))


def run_seed(stream, seed):
    X, y, regimes, X_init, y_init, win_slices, n_init, n_windows = _prepare(stream)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream["window_id"].values
    records, summaries = [], []

    def agg(method, recs, init_cpu, adapt_cpu, retrains, trained, reused, reuse_ev, wall, extra=None):
        r = pd.DataFrame(recs)
        r = r[r["method"] == method]
        d = dict(seed=seed, method=method, init_cpu_sec=init_cpu,
                 macro_f1=float(r["macro_f1"].mean()),
                 accuracy=float(r["accuracy"].mean()),
                 precision=float(r["precision"].mean()),
                 recall=float(r["recall"].mean()),
                 adaptation_cpu_sec=adapt_cpu, retrains=retrains,
                 trees_trained=trained, trees_reused=reused,
                 reuse_events=reuse_ev, total_runtime_sec=wall)
        if extra:
            d.update(extra)
        summaries.append(d)

    # ---- Frozen ----
    m = create_base_ensemble(seed=seed)
    tw = time.perf_counter(); t0 = time.process_time(); m.fit(X_init, y_init)
    init_cpu = time.process_time() - t0
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
        records.append(_rec(seed, "Frozen", w, regimes[idx[0]], y[idx], yp, 0, 0, 0, pc))
    agg("Frozen", records, init_cpu, 0.0, 0, m.get_num_trees(), 0, 0, time.perf_counter() - tw)

    # ---- Event-Driven ----
    ed = EventDrivenEnsemble(seed=seed, buffer_capacity=BUFFER_CAPACITY,
                             error_window_size=ERROR_WINDOW_SIZE,
                             error_threshold_k=ERROR_THRESHOLD_K,
                             anchor_X=anchor_X, anchor_y=anchor_y)
    tw = time.perf_counter()
    init_cpu, _ = ed.fit_initial(X_init, y_init)
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        t0 = time.process_time(); yp = ed.predict(X[idx]); pc = time.process_time() - t0
        err = 1.0 - np.mean(yp == y[idx])
        trig, adapt_cpu, _ = ed.update_and_adapt(X[idx], y[idx], err)
        records.append(_rec(seed, "Event-Driven", w, regimes[idx[0]], y[idx], yp,
                            adapt_cpu, int(trig), 0, pc))
    agg("Event-Driven", records, init_cpu, ed.cumulative_cpu_time - init_cpu,
        ed.retrain_events, ed.total_trees_trained, 0, 0, time.perf_counter() - tw)

    # ---- Full Retraining (on regime change) ----
    m = create_base_ensemble(seed=seed)
    tw = time.perf_counter(); t0 = time.process_time(); m.fit(X_init, y_init)
    init_cpu = time.process_time() - t0
    buf_X, buf_y = list(X_init), list(y_init)
    adapt_cpu, retrains, trees = 0.0, 0, m.get_num_trees()
    prev = regimes[np.where(wid == n_init - 1)[0][0]]
    for w in range(n_init, n_windows):
        idx = win_slices[w]; reg = regimes[idx[0]]
        is_t = reg != prev; prev = reg; wa = 0.0
        if is_t:
            retrains += 1
            trX, trY = make_train_buffer(np.array(buf_X), np.array(buf_y),
                                         anchor_X, anchor_y, BUFFER_CAPACITY)
            t0 = time.process_time(); m = create_base_ensemble(seed=seed + retrains * 11)
            m.fit(trX, trY); wa = time.process_time() - t0
            adapt_cpu += wa; trees += m.get_num_trees()
        t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
        buf_X.extend(X[idx]); buf_y.extend(y[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        records.append(_rec(seed, "Full Retraining", w, reg, y[idx], yp, wa, int(is_t), 0, pc))
    agg("Full Retraining", records, init_cpu, adapt_cpu, retrains, trees, 0, 0,
        time.perf_counter() - tw)

    # ---- RAPT ladder ----
    for variant in RAPT_VARIANTS:
        sys_ = build_rapt(variant, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y,
                          buffer_capacity=BUFFER_CAPACITY)
        tw = time.perf_counter()
        init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
        init_cpu, _ = sys_.fit_initial(init_regime, X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = init_regime
        for w in range(n_init, n_windows):
            idx = win_slices[w]; reg = regimes[idx[0]]
            wa, reused = 0.0, 0
            if reg != prev:
                reused_i, wa, _ = sys_.handle_regime_transition(
                    reg, X_buffer=buf_X[-BUFFER_CAPACITY:], y_buffer=buf_y[-BUFFER_CAPACITY:])
                reused = int(reused_i); prev = reg
            t0 = time.process_time(); yp = sys_.predict(X[idx]); pc = time.process_time() - t0
            buf_X.extend(X[idx]); buf_y.extend(y[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            wa += sys_.update(X[idx], y[idx], X_buffer=buf_X[-BUFFER_CAPACITY:],
                              y_buffer=buf_y[-BUFFER_CAPACITY:])
            records.append(_rec(seed, variant, w, reg, y[idx], yp, wa, 0, reused, pc))
        reuse_ev = int(pd.DataFrame(records).query("method==@variant")["is_reuse"].sum())
        agg(variant, records, init_cpu, sys_.adaptation_cpu_time,
            sys_.created_policy_count - 1, sys_.trees_trained_count,
            sys_.trees_reused_count, reuse_ev, time.perf_counter() - tw,
            extra={"parity_refits": sys_.parity_refits, "refreshes": sys_.refreshes})

    return pd.DataFrame(records), pd.DataFrame(summaries)


def main(mode="full"):
    ensure_dirs()
    seeds = SEEDS if mode == "full" else SEEDS[:2]
    stream, sd = load_stream()
    print(f"### 5G Campus QoS [{mode}] windows={sd['total_windows']} "
          f"classes={sd['n_classes']} regimes={len(set(sd['regime_sequence']))} "
          f"init={sd['initial_train_windows']}", flush=True)
    with open(os.path.join(RAW_DIR, f"stream_def_{mode}.json"), "w") as f:
        json.dump({k: v for k, v in sd.items() if k != "regime_sequence"}, f, indent=2)

    all_w, all_s = [], []
    for seed in seeds:
        w, s = run_seed(stream, seed)
        all_w.append(w); all_s.append(s)
        print(f"  seed {seed}: " + " | ".join(
            f"{r.method}={r.macro_f1:.4f}" for r in s.itertuples()), flush=True)
    W = pd.concat(all_w, ignore_index=True)
    S = pd.concat(all_s, ignore_index=True)
    W.to_csv(os.path.join(RAW_DIR, f"per_window_{mode}.csv"), index=False)
    S.to_csv(os.path.join(RAW_DIR, f"summary_{mode}.csv"), index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
