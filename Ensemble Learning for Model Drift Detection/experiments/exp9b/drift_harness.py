"""
Streaming evaluation harness for Experiment 9B drift studies.

Runs the four existing 9B models — Frozen, Event-Driven, Full Retraining and
RAPT — over an arbitrary chronological stream using the *same* implementations
and protocol as the original run_exp9b.py (initial training on the 20% prefix,
strict Test-Then-Train, no future leakage). Adaptation for Full Retraining and
RAPT is triggered by a `regime_id` change, exactly as in the existing pipeline;
the drift constructions set `regime_id` so that a single change occurs at the
intended drift point.
"""

import time

import numpy as np
import pandas as pd
import psutil

from models_9b import create_base_ensemble
from event_driven_9b import EventDrivenEnsemble
from rapt_9b import RAPTSystem
from rapt_enhanced_9b import RAPTEnhancedSystem
from preprocessing_9b import StreamingPreprocessor, get_feature_names
from evaluation_9b import calculate_window_metrics

METHOD_NAMES = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]


def evaluate_stream(df_stream, n_init, seed, drift_points=None, scenario="", level=None):
    """
    Evaluate all four models on one stream for one seed.

    Returns
    -------
    per_window : DataFrame (one row per model per evaluated window)
    per_seed   : DataFrame (one row per model, aggregate metrics)
    """
    drift_points = set(int(d) for d in (drift_points or []))
    feature_cols = get_feature_names()
    per_window_results = []
    per_seed_results = []

    preproc = StreamingPreprocessor()
    df_init = df_stream.iloc[:n_init]
    preproc.fit_initial(df_init)
    X_init_scaled = preproc.transform(df_init)
    y_init = df_init["qos_target"].values
    initial_reg_id = df_init["regime_id"].values[0]

    common = {"scenario": scenario, "drift_level": level}

    # ------------------------------------------------------------------
    # A. Frozen
    # ------------------------------------------------------------------
    model_frozen = create_base_ensemble(seed=seed)
    t0_cpu = time.process_time()
    model_frozen.fit(X_init_scaled, y_init)
    t_init_cpu = time.process_time() - t0_cpu

    y_true_all, y_pred_all = [], []
    pred_cpu_total = 0.0
    for idx in range(n_init, len(df_stream)):
        df_w = df_stream.iloc[[idx]]
        X_w = preproc.transform(df_w)
        y_w = df_w["qos_target"].values[0]
        w_id = df_w["window_id"].values[0]
        reg_id = df_w["regime_id"].values[0]

        t0 = time.process_time()
        pred = model_frozen.predict(X_w)[0]
        step_cpu = time.process_time() - t0
        pred_cpu_total += step_cpu

        y_true_all.append(y_w)
        y_pred_all.append(pred)
        per_window_results.append({
            **common, "seed": seed, "method": "Frozen", "window_id": w_id,
            "regime_id": reg_id, "y_true": int(y_w), "y_pred": int(pred),
            "is_correct": int(pred == y_w), "step_cpu_sec": step_cpu,
            "is_retrain": 0, "is_checkpoint_reuse": 0,
            "is_drift_point": int(w_id in drift_points),
        })
    m = calculate_window_metrics(y_true_all, y_pred_all)
    m.update({**common, "seed": seed, "method": "Frozen",
              "total_cpu_sec": t_init_cpu + pred_cpu_total,
              "adaptation_cpu_sec": 0.0, "prediction_cpu_sec": pred_cpu_total,
              "retrain_events": 0, "reused_checkpoints": 0,
              "trees_trained": model_frozen.get_num_trees(), "trees_reused": 0,
              "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)})
    per_seed_results.append(m)

    # ------------------------------------------------------------------
    # B. Event-Driven
    # ------------------------------------------------------------------
    ed_system = EventDrivenEnsemble(seed=seed)
    ed_system.fit_initial(X_init_scaled, y_init)

    y_true_all, y_pred_all = [], []
    pred_cpu_total = 0.0
    for idx in range(n_init, len(df_stream)):
        df_w = df_stream.iloc[[idx]]
        X_w = preproc.transform(df_w)
        y_w = df_w["qos_target"].values[0]
        w_id = df_w["window_id"].values[0]
        reg_id = df_w["regime_id"].values[0]

        t0 = time.process_time()
        pred = ed_system.predict(X_w)[0]
        step_cpu = time.process_time() - t0
        pred_cpu_total += step_cpu

        err = 1.0 if pred != y_w else 0.0
        triggered, adapt_cpu, _ = ed_system.update_and_adapt(X_w, [y_w], err)

        y_true_all.append(y_w)
        y_pred_all.append(pred)
        per_window_results.append({
            **common, "seed": seed, "method": "Event-Driven", "window_id": w_id,
            "regime_id": reg_id, "y_true": int(y_w), "y_pred": int(pred),
            "is_correct": int(pred == y_w), "step_cpu_sec": step_cpu + adapt_cpu,
            "is_retrain": int(triggered), "is_checkpoint_reuse": 0,
            "is_drift_point": int(w_id in drift_points),
        })
    m = calculate_window_metrics(y_true_all, y_pred_all)
    m.update({**common, "seed": seed, "method": "Event-Driven",
              "total_cpu_sec": ed_system.cumulative_cpu_time + pred_cpu_total,
              "adaptation_cpu_sec": ed_system.cumulative_cpu_time,
              "prediction_cpu_sec": pred_cpu_total,
              "retrain_events": ed_system.retrain_events, "reused_checkpoints": 0,
              "trees_trained": ed_system.total_trees_trained, "trees_reused": 0,
              "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)})
    per_seed_results.append(m)

    # ------------------------------------------------------------------
    # C. Full Retraining (retrain at every regime boundary)
    # ------------------------------------------------------------------
    model_full = create_base_ensemble(seed=seed)
    model_full.fit(X_init_scaled, y_init)
    hist_X = list(X_init_scaled)
    hist_y = list(y_init)

    y_true_all, y_pred_all = [], []
    pred_cpu_total, adapt_cpu_total = 0.0, 0.0
    retrain_count = 0
    trees_trained = model_full.get_num_trees()
    prev_regime = initial_reg_id
    for idx in range(n_init, len(df_stream)):
        df_w = df_stream.iloc[[idx]]
        X_w = preproc.transform(df_w)
        y_w = df_w["qos_target"].values[0]
        w_id = df_w["window_id"].values[0]
        reg_id = df_w["regime_id"].values[0]

        is_transition = (reg_id != prev_regime)
        prev_regime = reg_id

        t_retrain_cpu = 0.0
        if is_transition:
            retrain_count += 1
            t0 = time.process_time()
            train_X_arr = np.array(hist_X[-1000:])
            train_y_arr = np.array(hist_y[-1000:])
            model_full = create_base_ensemble(seed=seed + retrain_count * 11)
            model_full.fit(train_X_arr, train_y_arr)
            t_retrain_cpu = time.process_time() - t0
            adapt_cpu_total += t_retrain_cpu
            trees_trained += model_full.get_num_trees()

        t0 = time.process_time()
        pred = model_full.predict(X_w)[0]
        step_cpu = time.process_time() - t0
        pred_cpu_total += step_cpu

        hist_X.append(X_w[0])
        hist_y.append(y_w)
        y_true_all.append(y_w)
        y_pred_all.append(pred)
        per_window_results.append({
            **common, "seed": seed, "method": "Full Retraining", "window_id": w_id,
            "regime_id": reg_id, "y_true": int(y_w), "y_pred": int(pred),
            "is_correct": int(pred == y_w), "step_cpu_sec": step_cpu + t_retrain_cpu,
            "is_retrain": int(is_transition), "is_checkpoint_reuse": 0,
            "is_drift_point": int(w_id in drift_points),
        })
    m = calculate_window_metrics(y_true_all, y_pred_all)
    m.update({**common, "seed": seed, "method": "Full Retraining",
              "total_cpu_sec": adapt_cpu_total + pred_cpu_total,
              "adaptation_cpu_sec": adapt_cpu_total,
              "prediction_cpu_sec": pred_cpu_total,
              "retrain_events": retrain_count, "reused_checkpoints": 0,
              "trees_trained": trees_trained, "trees_reused": 0,
              "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)})
    per_seed_results.append(m)

    # ------------------------------------------------------------------
    # D. RAPT (policy transfer at regime boundaries)
    # ------------------------------------------------------------------
    rapt_sys = RAPTSystem(seed=seed, mode="full", enable_calibration=True)
    rapt_sys.fit_initial(initial_reg_id, X_init_scaled, y_init)
    hist_X = list(X_init_scaled)
    hist_y = list(y_init)

    y_true_all, y_pred_all = [], []
    pred_cpu_total = 0.0
    prev_regime = initial_reg_id
    for idx in range(n_init, len(df_stream)):
        df_w = df_stream.iloc[[idx]]
        X_w = preproc.transform(df_w)
        y_w = df_w["qos_target"].values[0]
        w_id = df_w["window_id"].values[0]
        reg_id = df_w["regime_id"].values[0]

        adapt_cpu = 0.0
        reused = False
        if reg_id != prev_regime:
            reused, adapt_cpu, _ = rapt_sys.handle_regime_transition(
                new_regime_id=reg_id, window_id=w_id,
                X_buffer=hist_X[-500:], y_buffer=hist_y[-500:])
            prev_regime = reg_id

        t0 = time.process_time()
        pred = rapt_sys.predict(X_w)[0]
        step_cpu = time.process_time() - t0
        pred_cpu_total += step_cpu

        hist_X.append(X_w[0])
        hist_y.append(y_w)
        y_true_all.append(y_w)
        y_pred_all.append(pred)
        per_window_results.append({
            **common, "seed": seed, "method": "RAPT", "window_id": w_id,
            "regime_id": reg_id, "y_true": int(y_w), "y_pred": int(pred),
            "is_correct": int(pred == y_w), "step_cpu_sec": step_cpu + adapt_cpu,
            "is_retrain": 0, "is_checkpoint_reuse": int(reused),
            "is_drift_point": int(w_id in drift_points),
        })
    m = calculate_window_metrics(y_true_all, y_pred_all)
    m.update({**common, "seed": seed, "method": "RAPT",
              "total_cpu_sec": rapt_sys.cumulative_cpu_time + pred_cpu_total,
              "adaptation_cpu_sec": rapt_sys.cumulative_cpu_time,
              "prediction_cpu_sec": pred_cpu_total,
              "retrain_events": rapt_sys.created_policy_count - 1,
              "reused_checkpoints": rapt_sys.reused_policy_count,
              "trees_trained": rapt_sys.trees_trained_count,
              "trees_reused": rapt_sys.trees_reused_count,
              "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)})
    per_seed_results.append(m)

    # ------------------------------------------------------------------
    # E. RAPT-Enhanced (policy repository + online micro-learner, hybrid blend)
    # ------------------------------------------------------------------
    enh_sys = RAPTEnhancedSystem(seed=seed, mode="full", enable_calibration=True)
    enh_sys.fit_initial(initial_reg_id, X_init_scaled, y_init)
    hist_X = list(X_init_scaled)
    hist_y = list(y_init)

    y_true_all, y_pred_all = [], []
    pred_cpu_total = 0.0
    prev_regime = initial_reg_id
    for idx in range(n_init, len(df_stream)):
        df_w = df_stream.iloc[[idx]]
        X_w = preproc.transform(df_w)
        y_w = df_w["qos_target"].values[0]
        w_id = df_w["window_id"].values[0]
        reg_id = df_w["regime_id"].values[0]

        adapt_cpu = 0.0
        reused = False
        if reg_id != prev_regime:
            reused, adapt_cpu, _ = enh_sys.handle_regime_transition(
                new_regime_id=reg_id, window_id=w_id,
                X_buffer=hist_X[-500:], y_buffer=hist_y[-500:])
            prev_regime = reg_id

        t0 = time.process_time()
        pred = enh_sys.predict(X_w)[0]
        step_cpu = time.process_time() - t0
        pred_cpu_total += step_cpu

        hist_X.append(X_w[0])
        hist_y.append(y_w)
        adapt_cpu += enh_sys.update(X_w, [y_w],
                                    X_buffer=hist_X[-1500:], y_buffer=hist_y[-1500:])

        y_true_all.append(y_w)
        y_pred_all.append(pred)
        per_window_results.append({
            **common, "seed": seed, "method": "RAPT-Enhanced", "window_id": w_id,
            "regime_id": reg_id, "y_true": int(y_w), "y_pred": int(pred),
            "is_correct": int(pred == y_w), "step_cpu_sec": step_cpu + adapt_cpu,
            "is_retrain": 0, "is_checkpoint_reuse": int(reused),
            "is_drift_point": int(w_id in drift_points),
        })
    m = calculate_window_metrics(y_true_all, y_pred_all)
    m.update({**common, "seed": seed, "method": "RAPT-Enhanced",
              "total_cpu_sec": enh_sys.cumulative_cpu_time + pred_cpu_total,
              "adaptation_cpu_sec": enh_sys.cumulative_cpu_time,
              "prediction_cpu_sec": pred_cpu_total,
              "retrain_events": enh_sys.created_policy_count - 1,
              "reused_checkpoints": enh_sys.reused_policy_count,
              "trees_trained": enh_sys.trees_trained_count,
              "trees_reused": enh_sys.trees_reused_count,
              "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024)})
    per_seed_results.append(m)

    return pd.DataFrame(per_window_results), pd.DataFrame(per_seed_results)
