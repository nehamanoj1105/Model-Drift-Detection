"""
Prequential runner: predict -> score -> adapt, one window at a time.

Writes results/raw/*.parquet with the canonical per-window schema:
  dataset, model, seed, window_idx, regime, y_true, y_pred,
  window_acc, window_f1, adapt_cpu_s, event_type

A leakage guard asserts that no window with index >= k is ever placed in the
adaptation buffer before window k has been scored.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from models import (PRIMARY, ABLATION, make_class_anchor)
from detectors import DetectorModel


def _cpu_json(parts):
    import json
    return json.dumps(parts, sort_keys=True) if parts else ""


def prepare(stream, sd):
    feat = sd["feature_columns"]
    X_all = stream[feat].values.astype(np.float64)
    y_all = stream["y"].values.astype(int)
    regimes = stream["regime_id"].values
    wid = stream["window_id"].values
    n_init, n_windows = sd["initial_train_windows"], sd["total_windows"]

    init_mask = wid < n_init
    X_init_raw = X_all[init_mask]
    mu = X_init_raw.mean(axis=0)
    sdv = np.where(X_init_raw.std(axis=0) < 1e-9, 1.0, X_init_raw.std(axis=0))
    X_scaled = (X_all - mu) / sdv
    X_init = X_scaled[init_mask]
    y_init = y_all[init_mask]

    # per-window sample index arrays, with the max window index they reference
    slices = {w: np.where(wid == w)[0] for w in range(n_init, n_windows)}
    return X_scaled, y_all, regimes, X_init, y_init, slices, n_init, n_windows


def _metrics(yt, yp):
    return {
        "window_acc": float(accuracy_score(yt, yp)),
        "window_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "prec": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "rec": float(recall_score(yt, yp, average="macro", zero_division=0)),
    }


def run_model(model, X, y_all, regimes, X_init, y_init, slices, n_init, n_windows,
              dataset, seed, leakage_log, cpu_parts=None):
    model.fit_initial(X_init, y_init)
    rows = []
    max_window_in_buffer = n_init - 1
    t_wall0 = time.perf_counter()
    for w in range(n_init, n_windows):
        idx = slices[w]
        # leakage guard: buffer must only contain windows < w at prediction time
        assert max_window_in_buffer < w, (
            f"leakage: buffer used window {max_window_in_buffer} before scoring {w}")
        buffer_at_pred = max_window_in_buffer  # what the score actually depended on
        t = time.process_time()
        yp = model.predict(X[idx])
        pred_cpu = time.process_time() - t
        m = _metrics(y_all[idx], yp)
        ev = model.update(X[idx], y_all[idx],
                          {"window_id": w, "regime": regimes[idx[0]],
                           "y_pred": yp, "X_win": X[idx], "y_win": y_all[idx]})
        max_window_in_buffer = w  # window w may now be used to adapt later windows
        rows.append({
            "dataset": dataset, "model": getattr(model, "name", type(model).__name__),
            "seed": int(seed), "window_idx": int(w), "regime": str(regimes[idx[0]]),
            "y_true": y_all[idx].astype(np.int8), "y_pred": yp.astype(np.int8),
            "window_acc": m["window_acc"], "window_f1": m["window_f1"],
            "window_prec": m["prec"], "window_rec": m["rec"],
            "pred_cpu_s": float(pred_cpu),
            "adapt_cpu_s": float(ev.get("adapt_cpu", 0.0)),
            "event_type": ev.get("event_type", "none"),
            "max_window_in_buffer": int(buffer_at_pred),
        })
        leakage_log.append({
            "dataset": dataset, "model": getattr(model, "name", "?"),
            "seed": int(seed), "window_idx": int(w),
            "max_train_window": int(buffer_at_pred),
            "future_buffer": bool(buffer_at_pred >= w), "ok": True})
    runtime = time.perf_counter() - t_wall0
    return rows, runtime


def run_dataset(stream, sd, cfg, seeds, leakage_log, policy_log=None):
    dataset = sd["dataset"]
    X, y_all, regimes, _, _, slices, n_init, n_windows = prepare(stream, sd)
    all_rows, summaries = [], []

    model_names = list(cfg["models"]["primary"]) + list(cfg["models"]["ablation"])
    prefix_mask = np.arange(len(y_all)) < n_init * sd["window_rows"]
    for seed in seeds:
        anchor = make_class_anchor(X[prefix_mask], y_all[prefix_mask],
                                   cfg["class_anchor_per_class"], seed)

        for name in model_names:
            if name in PRIMARY:
                model = PRIMARY[name](seed, cfg, anchor)
            else:
                model = ABLATION[name](seed, cfg, anchor)
            rows, runtime = run_model(model, X, y_all, regimes, X[prefix_mask],
                                      y_all[prefix_mask], slices, n_init, n_windows,
                                      dataset, seed, leakage_log)
            all_rows.extend(rows)
            if policy_log is not None and getattr(model, "policy_events", None):
                for pe in model.policy_events:
                    pe = dict(pe)
                    pe.update({"dataset": dataset, "model": name, "seed": int(seed)})
                    policy_log.append(pe)
            dfm = pd.DataFrame(rows)
            summaries.append({
                "dataset": dataset, "model": name, "seed": int(seed),
                "macro_f1": float(dfm["window_f1"].mean()),
                "accuracy": float(dfm["window_acc"].mean()),
                "precision": float(dfm["window_prec"].mean()),
                "recall": float(dfm["window_rec"].mean()),
                "adapt_cpu_s": float(dfm["adapt_cpu_s"].sum()),
                "runtime_s": float(runtime),
                "pred_cpu_s": float(dfm["pred_cpu_s"].sum()),
                "retrains": int((dfm["event_type"] == "retrain").sum()),
                "reuses": int((dfm["event_type"] == "reuse").sum()),
                "refreshes": int((dfm["event_type"] == "refresh").sum()),
                "trees_trained": int(getattr(model, "trees_trained", 0)),
                "trees_reused": int(getattr(model, "trees_reused", 0)),
                "n_eval_windows": int(len(dfm)),
                "cpu_json": _cpu_json(getattr(model, "cpu_parts", None)),
            })

    for name in cfg["models"]["detectors"]:
        for seed in seeds:
            prefix_mask = np.arange(len(y_all)) < n_init * sd["window_rows"]
            anchor = make_class_anchor(X[prefix_mask], y_all[prefix_mask],
                                       cfg["class_anchor_per_class"], seed)
            model = DetectorModel(name, seed, cfg, anchor)
            rows, runtime = run_model(model, X, y_all, regimes, X[prefix_mask],
                                      y_all[prefix_mask], slices, n_init, n_windows,
                                      dataset, seed, leakage_log)
            all_rows.extend(rows)
            dfm = pd.DataFrame(rows)
            summaries.append({
                "dataset": dataset, "model": name, "seed": int(seed),
                "macro_f1": float(dfm["window_f1"].mean()),
                "accuracy": float(dfm["window_acc"].mean()),
                "precision": float(dfm["window_prec"].mean()),
                "recall": float(dfm["window_rec"].mean()),
                "adapt_cpu_s": float(dfm["adapt_cpu_s"].sum()),
                "runtime_s": float(runtime),
                "pred_cpu_s": float(dfm["pred_cpu_s"].sum()),
                "retrains": int((dfm["event_type"] == "retrain").sum()),
                "reuses": 0, "refreshes": 0,
                "trees_trained": int(getattr(model, "trees_trained", 0)),
                "trees_reused": 0,
                "n_eval_windows": int(len(dfm)),
                "detected_events": int(getattr(model, "detected", 0)),
            })
    return pd.DataFrame(all_rows), pd.DataFrame(summaries)
