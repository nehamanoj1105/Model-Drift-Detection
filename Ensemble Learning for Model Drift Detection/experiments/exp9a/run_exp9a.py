"""
Master runner for the REVISED Experiment 9A.

Strict prequential (Test-Then-Train) evaluation of five models on the
INSECTS recurring-regime stream:
  Frozen, Event-Driven, Full_Retraining, RAPT, RAPT-Enhanced
across seeds [42, 43, 44, 45, 46]. Saves per-window and per-seed raw results.

Protocol notes
--------------
* Features are standardised using statistics fitted ONLY on the initial
  training prefix (no look-ahead).
* Each window is WINDOW_SIZE=500 consecutive samples; all samples in the
  window are predicted before the window is appended to any adaptation buffer
  (strict test-then-train).
* Every adaptive model retrains on a class-anchored buffer (see models_9a) so
  that all classes remain representable; this guard is identical for all
  adaptive models.
* RAPT and RAPT-Enhanced share the same policy-transfer mechanism and differ
  only in the size of the novelty-refit buffer (RAPT: 500 most recent samples;
  RAPT-Enhanced: 1500 most recent samples), matching the Buffer-Blended Novelty
  Refitting enhancement from rapt/enhanced_hybrid_rapt.py.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp9a_config import (
    SEEDS, MODELS, WINDOW_SIZE, BUFFER_CAPACITY, ERROR_WINDOW_SIZE,
    ERROR_THRESHOLD_K, ENHANCED_NOVELTY_REFIT_N, REVISED_STREAM_CSV,
    RAW_DIR, ensure_dirs,
)
from load_and_prepare_stream_9a import build_stream
from models_9a import create_base_ensemble, make_class_anchor, make_train_buffer
from event_driven_9a import EventDrivenEnsemble
from rapt_9a import RAPTSystem, RAPTEnhancedSystem
from evaluation_9a import window_metrics, analyze_transitions, run_wilcoxon_tests

COMPARISONS = [
    ("RAPT", "Event-Driven"),
    ("RAPT", "Frozen"),
    ("RAPT", "Full_Retraining"),
    ("RAPT-Enhanced", "Event-Driven"),
    ("RAPT-Enhanced", "Full_Retraining"),
    ("RAPT-Enhanced", "RAPT"),
]


def _window_record(seed, method, w, regime, y_true, y_pred, adapt_cpu, is_retrain, is_reuse):
    met = window_metrics(y_true, y_pred)
    return {
        "seed": seed, "method": method, "window_id": int(w), "regime_id": regime,
        "macro_f1": met["macro_f1"], "accuracy": met["accuracy"],
        "precision": met["precision"], "recall": met["recall"],
        "adaptation_cpu_sec": float(adapt_cpu),
        "is_retrain": int(is_retrain), "is_reuse": int(is_reuse),
        "is_correct": float(np.mean(y_true == y_pred)),
        "n_samples": int(len(y_true)),
    }


def _seed_summary(seed, method, records, init_cpu, adapt_cpu, retrains,
                  trees_trained, trees_reused, parity_refits=0):
    df = pd.DataFrame(records)
    df = df[df["method"] == method]
    return {
        "seed": seed, "method": method, "window_size": WINDOW_SIZE,
        "macro_f1": float(df["macro_f1"].mean()),
        "accuracy": float(df["accuracy"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "init_cpu_sec": float(init_cpu),
        "adaptation_cpu_sec": float(adapt_cpu),
        "total_cpu_sec": float(init_cpu + adapt_cpu),
        "retrain_events": int(retrains),
        "reuse_events": int(df["is_reuse"].sum()),
        "trees_trained": int(trees_trained),
        "trees_reused": int(trees_reused),
        "parity_refits": int(parity_refits),
        "memory_mb": float(psutil.Process().memory_info().rss / (1024 * 1024)),
    }


def run_seed(stream_df, stream_def, seed):
    feature_cols = stream_def["feature_columns"]
    n_init = stream_def["initial_train_windows"]
    n_windows = stream_def["total_windows"]

    window_ids = stream_df["window_id"].values
    regimes = stream_df["regime_id"].values
    y_all = pd.Categorical(stream_df["species"]).codes
    X_all = stream_df[feature_cols].values.astype(np.float64)

    init_mask = window_ids < n_init
    X_init_raw = X_all[init_mask]
    y_init = y_all[init_mask]

    mu = X_init_raw.mean(axis=0)
    sd = np.where(X_init_raw.std(axis=0) < 1e-9, 1.0, X_init_raw.std(axis=0))
    X_scaled = (X_all - mu) / sd
    X_init = X_scaled[init_mask]

    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    win_slices = {w: np.where(window_ids == w)[0] for w in range(n_init, n_windows)}

    records, summaries = [], []

    # ------------------------------------------------------------------ Frozen
    m = create_base_ensemble(seed=seed)
    t0 = time.process_time(); m.fit(X_init, y_init); init_cpu = time.process_time() - t0
    pred_cpu = 0.0
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        t0 = time.process_time(); yp = m.predict(X_scaled[idx]); pred_cpu += time.process_time() - t0
        records.append(_window_record(seed, "Frozen", w, regimes[idx[0]], y_all[idx], yp, 0.0, 0, 0))
    summaries.append(_seed_summary(seed, "Frozen", records, init_cpu, 0.0, 0, m.get_num_trees(), 0))

    # ------------------------------------------------------------ Event-Driven
    ed = EventDrivenEnsemble(seed=seed, buffer_capacity=BUFFER_CAPACITY,
                             error_window_size=ERROR_WINDOW_SIZE,
                             error_threshold_k=ERROR_THRESHOLD_K,
                             anchor_X=anchor_X, anchor_y=anchor_y)
    init_cpu, _ = ed.fit_initial(X_init, y_init)
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        t0 = time.process_time(); yp = ed.predict(X_scaled[idx]); _ = time.process_time() - t0
        err = 1.0 - np.mean(yp == y_all[idx])
        triggered, adapt_cpu, _ = ed.update_and_adapt(X_scaled[idx], y_all[idx], err)
        records.append(_window_record(seed, "Event-Driven", w, regimes[idx[0]], y_all[idx], yp,
                                      adapt_cpu, int(triggered), 0))
    summaries.append(_seed_summary(seed, "Event-Driven", records, init_cpu,
                                   ed.cumulative_cpu_time - init_cpu, ed.retrain_events,
                                   ed.total_trees_trained, 0))

    # --------------------------------------------------------- Full Retraining
    m = create_base_ensemble(seed=seed)
    t0 = time.process_time(); m.fit(X_init, y_init); init_cpu = time.process_time() - t0
    buf_X, buf_y = list(X_init), list(y_init)
    adapt_cpu, retrains, trees = 0.0, 0, m.get_num_trees()
    prev_regime = regimes[np.where(window_ids == n_init - 1)[0][0]]
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        is_trans = reg != prev_regime
        prev_regime = reg
        win_adapt = 0.0
        if is_trans:
            retrains += 1
            trX, trY = make_train_buffer(np.array(buf_X), np.array(buf_y),
                                         anchor_X, anchor_y, BUFFER_CAPACITY)
            t0 = time.process_time()
            m = create_base_ensemble(seed=seed + retrains * 11)
            m.fit(trX, trY)
            win_adapt = time.process_time() - t0
            adapt_cpu += win_adapt
            trees += m.get_num_trees()
        t0 = time.process_time(); yp = m.predict(X_scaled[idx]); _ = time.process_time() - t0
        buf_X.extend(X_scaled[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        records.append(_window_record(seed, "Full_Retraining", w, reg, y_all[idx], yp,
                                      win_adapt, int(is_trans), 0))
    summaries.append(_seed_summary(seed, "Full_Retraining", records, init_cpu,
                                   adapt_cpu, retrains, trees, 0))

    # ----------------------------------------------------- RAPT / RAPT-Enhanced
    for method, cls, refit_n in (("RAPT", RAPTSystem, 500),
                                 ("RAPT-Enhanced", RAPTEnhancedSystem, ENHANCED_NOVELTY_REFIT_N)):
        kwargs = dict(seed=seed, mode="full", enable_calibration=True,
                      anchor_X=anchor_X, anchor_y=anchor_y,
                      buffer_capacity=BUFFER_CAPACITY)
        if method == "RAPT-Enhanced":
            kwargs["novelty_refit_n"] = refit_n
        sys_ = cls(**kwargs)
        # Regime active at the end of the initial prefix (last initial window).
        init_regime = regimes[np.where(window_ids == n_init - 1)[0][0]]
        init_cpu, _ = sys_.fit_initial(init_regime, X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev_regime = init_regime
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            win_adapt, reused = 0.0, 0
            if reg != prev_regime:
                reused_i, win_adapt, _ = sys_.handle_regime_transition(
                    reg, w, X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
                reused = int(reused_i)
                prev_regime = reg
            t0 = time.process_time(); yp = sys_.predict(X_scaled[idx]); _ = time.process_time() - t0
            buf_X.extend(X_scaled[idx]); buf_y.extend(y_all[idx])
            if len(buf_X) > BUFFER_CAPACITY:
                buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
            # Post-prediction update (RAPT-Enhanced: selective parity refit).
            win_adapt += sys_.update(X_scaled[idx], y_all[idx],
                                     X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
            records.append(_window_record(seed, method, w, reg, y_all[idx], yp,
                                          win_adapt, 0, reused))
        summaries.append(_seed_summary(seed, method, records, init_cpu,
                                       sys_.adaptation_cpu_time,
                                       sys_.created_policy_count - 1,
                                       sys_.trees_trained_count, sys_.trees_reused_count,
                                       parity_refits=getattr(sys_, "parity_refits", 0)))

    return pd.DataFrame(records), pd.DataFrame(summaries)


def main():
    ensure_dirs()
    if os.path.exists(REVISED_STREAM_CSV):
        stream_df = pd.read_csv(REVISED_STREAM_CSV)
        with open(os.path.join(RAW_DIR, "stream_definition_9a.json")) as f:
            stream_def = json.load(f)
    else:
        stream_df, stream_def = build_stream()

    print(f"Revised 9A stream: {stream_def['total_windows']} windows, "
          f"n_init={stream_def['initial_train_windows']}, "
          f"regimes={stream_def['regime_sequence']}")

    all_win, all_seed = [], []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===", flush=True)
        dw, ds = run_seed(stream_df, stream_def, seed)
        all_win.append(dw); all_seed.append(ds)
        for _, r in ds.iterrows():
            print(f"  [{r['method']:15s}] F1={r['macro_f1']:.4f} "
                  f"AdaptCPU={r['adaptation_cpu_sec']:.3f}s "
                  f"retrains={r['retrain_events']} reuse={r['reuse_events']}", flush=True)

    df_win = pd.concat(all_win, ignore_index=True)
    df_seed = pd.concat(all_seed, ignore_index=True)
    df_win.to_csv(os.path.join(RAW_DIR, "per_window_9a.csv"), index=False)
    df_seed.to_csv(os.path.join(RAW_DIR, "per_seed_9a.csv"), index=False)

    summary_rows = []
    for method in MODELS:
        g = df_seed[df_seed["method"] == method]
        row = {"method": method}
        for col in ["macro_f1", "accuracy", "precision", "recall",
                    "adaptation_cpu_sec", "total_cpu_sec", "retrain_events", "reuse_events",
                    "trees_trained", "trees_reused", "memory_mb"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std())
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RAW_DIR, "summary_9a.csv"), index=False)

    trans_frames = []
    for (seed, method), g in df_win.groupby(["seed", "method"]):
        t = analyze_transitions(g.reset_index(drop=True), stream_def)
        t["seed"] = seed; t["method"] = method
        trans_frames.append(t)
    df_trans = pd.concat(trans_frames, ignore_index=True)
    df_trans.to_csv(os.path.join(RAW_DIR, "transitions_9a.csv"), index=False)

    df_stats = run_wilcoxon_tests(df_win, COMPARISONS)
    df_stats.to_csv(os.path.join(RAW_DIR, "statistics_9a.csv"), index=False)

    print("\n=== 9A Summary ===")
    print(df_summary[["method", "macro_f1_mean", "macro_f1_std",
                      "adaptation_cpu_sec_mean", "retrain_events_mean",
                      "reuse_events_mean", "trees_reused_mean"]].to_string(index=False))
    return df_summary, df_stats


if __name__ == "__main__":
    main()
