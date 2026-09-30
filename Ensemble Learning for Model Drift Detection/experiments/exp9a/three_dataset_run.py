"""
Experiment 9A (three telecom datasets) — prequential runner.

Runs the five primary models (Frozen, Event-Driven, Full Retraining, RAPT,
RAPT-Enhanced) and the four historical drift detectors (ADWIN, EDD,
Page-Hinkley, EDMA) on a dataset stream using the ACTUAL samples of each
window. Protocol:

* initial training prefix = first `initial_train_windows` windows;
* standardisation fit on the training prefix only;
* for every subsequent window: predict all samples (test-then-train), then
  update adaptation state with the same window;
* no future leakage; identical buffer capacity for all adaptive methods.

Outputs per (dataset, seed):
  per_window_<slug>_<mode>.csv   window-level metrics + adaptation flags
  summary_<slug>_<mode>.csv      per-seed aggregate metrics
  stream_<slug>.csv              windowed stream (features aggregated)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp9a_config as base
from three_dataset_config import (
    SEEDS_PILOT, SEEDS_FULL, PILOT_MAX_SAMPLES, DATASETS, DATASET_SLUG,
    RAW_DIR, ensure_dirs,
)
from three_dataset_load import LOADERS, build_windows
from models_9a import create_base_ensemble, make_class_anchor, make_train_buffer
from event_driven_9a import EventDrivenEnsemble
from rapt_9a import RAPTSystem, RAPTEnhancedSystem
from drift_detectors_9a import DetectorAdaptiveModel
from evaluation_9a import window_metrics

BUFFER_CAPACITY = base.BUFFER_CAPACITY
ERROR_WINDOW_SIZE = base.ERROR_WINDOW_SIZE
ERROR_THRESHOLD_K = base.ERROR_THRESHOLD_K
ENHANCED_NOVELTY_REFIT_N = base.ENHANCED_NOVELTY_REFIT_N
DETECTORS = ["ADWIN", "EDD", "Page-Hinkley", "EDMA"]
_PRED = {"macro_f1", "accuracy", "precision", "recall"}


def _wrec(seed, method, w, regime, y_true, y_pred, adapt_cpu, wall_cpu,
          is_retrain, is_reuse, pred_cpu):
    m = window_metrics(y_true, y_pred)
    return {
        "seed": seed, "method": method, "window_id": int(w), "regime_id": str(regime),
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "precision": m["precision"], "recall": m["recall"],
        "adaptation_cpu_sec": float(adapt_cpu),
        "prediction_cpu_sec": float(pred_cpu),
        "wall_sec": float(wall_cpu),
        "is_retrain": int(is_retrain), "is_reuse": int(is_reuse),
        "n_samples": int(len(y_true)),
    }


def _ssum(seed, method, records, init_cpu, adapt_cpu, retrains, trees_trained,
          trees_reused, reuse_events=None, total_wall=None, extra=None):
    df = pd.DataFrame(records)
    df = df[df["method"] == method]
    row = {
        "seed": seed, "method": method,
        "macro_f1": float(df["macro_f1"].mean()),
        "accuracy": float(df["accuracy"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "init_cpu_sec": float(init_cpu),
        "adaptation_cpu_sec": float(adapt_cpu),
        "total_cpu_sec": float(init_cpu + adapt_cpu),
        "prediction_cpu_sec": float(df["prediction_cpu_sec"].sum()),
        "total_runtime_sec": float(total_wall if total_wall is not None
                                   else df["wall_sec"].sum()),
        "retrain_events": int(retrains),
        "reuse_events": int(reuse_events if reuse_events is not None
                            else df["is_reuse"].sum()),
        "trees_trained": int(trees_trained),
        "trees_reused": int(trees_reused),
        "memory_mb": float(psutil.Process().memory_info().rss / (1024 * 1024)),
    }
    if extra:
        row.update(extra)
    return row


def _prepare(stream_df, sd):
    feat = sd["feature_columns"]
    X_all = stream_df[feat].values.astype(np.float64)
    y_all = stream_df["y"].values.astype(int)
    regimes = stream_df["regime_id"].values
    n_init = sd["initial_train_windows"]
    n_windows = sd["total_windows"]
    init_mask = stream_df["window_id"].values < n_init
    X_init_raw = X_all[init_mask]
    y_init = y_all[init_mask]
    mu = X_init_raw.mean(axis=0)
    sd_ = np.where(X_init_raw.std(axis=0) < 1e-9, 1.0, X_init_raw.std(axis=0))
    X_scaled = (X_all - mu) / sd_
    X_init = X_scaled[init_mask]
    win_slices = {w: np.where(stream_df["window_id"].values == w)[0]
                  for w in range(n_init, n_windows)}
    return X_scaled, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows


def run_seed(stream_df, sd, seed):
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = _prepare(stream_df, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    records, summaries = [], []

    # Frozen
    wid = stream_df["window_id"].values
    m = create_base_ensemble(seed=seed)
    t_w0 = time.perf_counter()
    t0 = time.process_time(); m.fit(X_init, y_init); init_cpu = time.process_time() - t0
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
        records.append(_wrec(seed, "Frozen", w, regimes[idx[0]], y_all[idx], yp, 0, 0, 0, 0, pc))
    summaries.append(_ssum(seed, "Frozen", records, init_cpu, 0.0, 0, m.get_num_trees(), 0,
                           total_wall=time.perf_counter() - t_w0))

    # Event-Driven
    ed = EventDrivenEnsemble(seed=seed, buffer_capacity=BUFFER_CAPACITY,
                             error_window_size=ERROR_WINDOW_SIZE,
                             error_threshold_k=ERROR_THRESHOLD_K,
                             anchor_X=anchor_X, anchor_y=anchor_y)
    t_w0 = time.perf_counter()
    init_cpu, _ = ed.fit_initial(X_init, y_init)
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        t0 = time.process_time(); yp = ed.predict(X[idx]); pc = time.process_time() - t0
        err = 1.0 - np.mean(yp == y_all[idx])
        trig, adapt_cpu, _ = ed.update_and_adapt(X[idx], y_all[idx], err)
        records.append(_wrec(seed, "Event-Driven", w, regimes[idx[0]], y_all[idx], yp,
                             adapt_cpu, 0, int(trig), 0, pc))
    summaries.append(_ssum(seed, "Event-Driven", records, init_cpu,
                           ed.cumulative_cpu_time - init_cpu, ed.retrain_events,
                           ed.total_trees_trained, 0,
                           total_wall=time.perf_counter() - t_w0))

    # Full Retraining (on regime change)
    m = create_base_ensemble(seed=seed)
    t_w0 = time.perf_counter()
    t0 = time.process_time(); m.fit(X_init, y_init); init_cpu = time.process_time() - t0
    buf_X, buf_y = list(X_init), list(y_init)
    adapt_cpu, retrains, trees = 0.0, 0, m.get_num_trees()
    prev_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        is_trans = reg != prev_regime
        prev_regime = reg
        wa = 0.0
        if is_trans:
            retrains += 1
            trX, trY = make_train_buffer(np.array(buf_X), np.array(buf_y),
                                         anchor_X, anchor_y, BUFFER_CAPACITY)
            t0 = time.process_time(); m = create_base_ensemble(seed=seed + retrains * 11); m.fit(trX, trY)
            wa = time.process_time() - t0
            adapt_cpu += wa
            trees += m.get_num_trees()
        t0 = time.process_time(); yp = m.predict(X[idx]); pc = time.process_time() - t0
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        records.append(_wrec(seed, "Full Retraining", w, reg, y_all[idx], yp, wa, 0, int(is_trans), 0, pc))
    summaries.append(_ssum(seed, "Full Retraining", records, init_cpu, adapt_cpu, retrains, trees, 0,
                           total_wall=time.perf_counter() - t_w0))

    # Historical drift detectors
    for det in DETECTORS:
        dm = DetectorAdaptiveModel(det, seed=seed, buffer_capacity=BUFFER_CAPACITY,
                                   anchor_X=anchor_X, anchor_y=anchor_y)
        t_w0 = time.perf_counter()
        init_cpu, _ = dm.fit_initial(X_init, y_init)
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            t0 = time.process_time(); yp = dm.predict(X[idx]); pc = time.process_time() - t0
            err = 1.0 - np.mean(yp == y_all[idx])
            trig, did, adapt_cpu, _ = dm.update_and_adapt(w, X[idx], y_all[idx], err)
            records.append(_wrec(seed, det, w, regimes[idx[0]], y_all[idx], yp,
                                 adapt_cpu, 0, int(did), 0, pc))
        summaries.append(_ssum(seed, det, records, init_cpu,
                               dm.cumulative_cpu_time - init_cpu, dm.retrain_events,
                               dm.total_trees_trained, 0,
                               total_wall=time.perf_counter() - t_w0,
                               extra={"detected_events": dm.detected_events}))

    # RAPT / RAPT-Enhanced
    for method, cls, refit_n in (("RAPT", RAPTSystem, 500),
                                 ("RAPT-Enhanced", RAPTEnhancedSystem, ENHANCED_NOVELTY_REFIT_N)):
        kwargs = dict(seed=seed, mode="full", enable_calibration=True,
                      anchor_X=anchor_X, anchor_y=anchor_y,
                      buffer_capacity=BUFFER_CAPACITY)
        if method == "RAPT-Enhanced":
            kwargs["novelty_refit_n"] = refit_n
        sys_ = cls(**kwargs)
        t_w0 = time.perf_counter()
        init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
        init_cpu, _ = sys_.fit_initial(init_regime, X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev_regime = init_regime
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            wa, reused = 0.0, 0
            if reg != prev_regime:
                reused_i, wa, _ = sys_.handle_regime_transition(
                    reg, w, X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
                reused = int(reused_i)
                prev_regime = reg
            t0 = time.process_time(); yp = sys_.predict(X[idx]); pc = time.process_time() - t0
            buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            wa += sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
            records.append(_wrec(seed, method, w, reg, y_all[idx], yp, wa, 0, 0, reused, pc))
        dfm = pd.DataFrame(records)
        dfm = dfm[dfm["method"] == method]
        summaries.append(_ssum(seed, method, records, init_cpu, sys_.adaptation_cpu_time,
                               sys_.created_policy_count - 1, sys_.trees_trained_count,
                               sys_.trees_reused_count,
                               reuse_events=int(dfm["is_reuse"].sum()),
                               total_wall=time.perf_counter() - t_w0,
                               extra={"parity_refits": getattr(sys_, "parity_refits", 0)}))

    return pd.DataFrame(records), pd.DataFrame(summaries)


def stream_df_window_ids(stream_df):
    return stream_df["window_id"].values


def run_dataset(name, mode, seeds):
    data = LOADERS[name]()
    maxs = PILOT_MAX_SAMPLES if mode == "pilot" else None
    stream, sd = build_windows(data, max_samples=maxs)
    slug = DATASET_SLUG[name]
    print(f"\n### {name} [{mode}] windows={sd['total_windows']} "
          f"classes={sd['n_classes']} regimes={len(set(sd['regime_sequence']))} "
          f"init={sd['initial_train_windows']}", flush=True)
    # Stream definitions (small) are saved; the sample-level stream itself is
    # deterministically rebuilt by the loader, so the large CSV is not persisted.
    with open(os.path.join(RAW_DIR, f"stream_def_{slug}_{mode}.json"), "w") as f:
        json.dump({k: v for k, v in sd.items() if k != "regime_sequence"}, f, indent=2)

    wins, sums = [], []
    for seed in seeds:
        dw, ds = run_seed(stream, sd, seed)
        wins.append(dw); sums.append(ds)
        for _, r in ds.iterrows():
            print(f"   [{r['method']:15s}] F1={r['macro_f1']:.4f} "
                  f"cpu={r['adaptation_cpu_sec']:.2f}s retr={r['retrain_events']} "
                  f"reuse={r['reuse_events']}", flush=True)
    df_w = pd.concat(wins, ignore_index=True)
    df_s = pd.concat(sums, ignore_index=True)
    df_w.to_csv(os.path.join(RAW_DIR, f"per_window_{slug}_{mode}.csv"), index=False)
    df_s.to_csv(os.path.join(RAW_DIR, f"summary_{slug}_{mode}.csv"), index=False)
    return df_w, df_s, sd


def main(mode="pilot"):
    ensure_dirs()
    seeds = SEEDS_PILOT if mode == "pilot" else SEEDS_FULL
    for name in DATASETS:
        run_dataset(name, mode, seeds)
    print("\nDone.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    main(mode)
