"""
================================================================================
EXPERIMENT 5 — MAIN EXECUTION RUNNER (ToN_IoT DATASET)
================================================================================
Re-runs the complete ensemble drift detection and adaptive model selection
experiment on the official ToN_IoT dataset across 5 deterministic seeds (42-46):
  1. Model 1: Frozen Model (Static baseline trained on initial 20k samples)
  2. Model 2: Continuously Retrained Ensemble (Adaptive baseline)
  3. Model 3: Event-Driven Ensemble (Drift-triggered adaptation)
  4. Individual Adaptive Models: Adaptive RF, Adaptive ET, Adaptive GB

Enforces strict prequential (Test-Then-Train) evaluation, zero data leakage,
and comprehensive process-level psutil and model parameter memory profiling.
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import pandas as pd

from config import (
    SEEDS, N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_STREAM_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, FEATURE_COLS, TARGET_COL,
    RESULTS_DIR, CONFUSION_DIR, PLOTS_DIR, REGIME_BOUNDARIES,
    APPROACH_NAMES
)
from data_loader import load_ton_iot_data, prepare_experiment_split
from preprocessing import Preprocessor
from models import create_candidate_models
from ensemble import HeterogeneousAdaptiveEnsemble
from drift import DualTriggerDriftDetector
from metrics import compute_metrics
from resource_monitor import measure_execution, compute_model_memory


def run_experiment5(seeds=SEEDS, smoke_test=False, max_windows=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFUSION_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 80)
    print("STARTING EXPERIMENT 5: ENSEMBLE DRIFT DETECTION ON ToN_IoT")
    print(f"Total Usable Samples: {N_TOTAL_SAMPLES:,} | Initial Training: {N_INITIAL_TRAINING:,} | Stream: {N_STREAM_SAMPLES:,}")
    print(f"Streaming Windows: {N_WINDOWS} (Window Size: {WINDOW_SIZE}) | Seeds: {seeds}")
    print("=" * 80)

    # 1. Load dataset and prepare splits
    print("\n[Data Loader] Loading official ToN_IoT Weather dataset and verifying timestamps...")
    df_sorted = load_ton_iot_data()
    split_data = prepare_experiment_split(df_sorted)

    X_train_raw = split_data['X_train']
    y_train = split_data['y_train']
    X_stream_raw = split_data['X_stream']
    y_stream = split_data['y_stream']
    window_schedule = split_data['window_schedule']

    total_windows = min(len(window_schedule), max_windows) if max_windows else len(window_schedule)

    # 2. Fit Preprocessor STRICTLY on initial training partition (ZERO data leakage)
    print(f"[Preprocessing] Fitting StandardScaler strictly on initial {len(X_train_raw):,} samples...")
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_stream = preprocessor.transform(X_stream_raw)
    scaler_path = os.path.join(RESULTS_DIR, 'scaler.joblib')
    preprocessor.save(scaler_path)
    print(f"[Preprocessing] Preprocessor fitted and serialized to: {scaler_path}")

    # Trackers for artifacts
    all_window_records = []
    all_regime_records = []
    all_memory_records = []
    all_timing_records = []
    all_drift_records = []

    total_experiment_start = time.time()

    for seed in seeds:
        print(f"\n" + "-" * 75)
        print(f"EXECUTING DETERMINISTIC SEED: {seed}")
        print("-" * 75)

        # ── INITIAL MODEL TRAINING (Samples 0 to 19,999) ──────────────────────
        print(f"[Seed {seed}] Training models on initial {len(X_train):,} samples (Attack rate: {np.mean(y_train):.1%})...")

        # Model 1: Frozen Ensemble
        m1_frozen = HeterogeneousAdaptiveEnsemble(seed, is_frozen=True)
        _, m1_train_res = measure_execution(m1_frozen.fit_initial, X_train, y_train)

        # Model 2: Continuously Retrained Ensemble
        m2_cont = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False)
        _, m2_train_res = measure_execution(m2_cont.fit_initial, X_train, y_train)

        # Model 3: Event-Driven Ensemble
        m3_event = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False)
        _, m3_train_res = measure_execution(m3_event.fit_initial, X_train, y_train)
        drift_detector = DualTriggerDriftDetector(initial_baseline_f1=0.85)

        # Measure Model Parameter Memory & Initial Training RAM
        m1_mem = compute_model_memory(m1_frozen.models['RF'])
        m1_mem['model_size_kb'] += compute_model_memory(m1_frozen.models['ET'])['model_size_kb']
        m1_mem['model_size_kb'] += compute_model_memory(m1_frozen.models['GB'])['model_size_kb']
        m1_mem['model_size_mb'] = m1_mem['model_size_kb'] / 1024.0

        all_memory_records.append({
            'seed': seed,
            'phase': 'initial_training',
            'approach': 'Frozen Model',
            'model_size_kb': m1_mem['model_size_kb'],
            'model_size_mb': m1_mem['model_size_mb'],
            'peak_ram_mb': m1_train_res['peak_ram_mb'],
            'avg_ram_mb': m1_train_res['avg_ram_mb'],
            'cpu_time_s': m1_train_res['total_cpu_time'],
        })
        all_memory_records.append({
            'seed': seed,
            'phase': 'initial_training',
            'approach': 'Continuously Retrained Ensemble',
            'model_size_kb': m1_mem['model_size_kb'],
            'model_size_mb': m1_mem['model_size_mb'],
            'peak_ram_mb': m2_train_res['peak_ram_mb'],
            'avg_ram_mb': m2_train_res['avg_ram_mb'],
            'cpu_time_s': m2_train_res['total_cpu_time'],
        })
        all_memory_records.append({
            'seed': seed,
            'phase': 'initial_training',
            'approach': 'Event-Driven Ensemble',
            'model_size_kb': m1_mem['model_size_kb'],
            'model_size_mb': m1_mem['model_size_mb'],
            'peak_ram_mb': m3_train_res['peak_ram_mb'],
            'avg_ram_mb': m3_train_res['avg_ram_mb'],
            'cpu_time_s': m3_train_res['total_cpu_time'],
        })

        print(f"  [Model 1 - Frozen] Fit in {m1_train_res['wall_clock_time']:.2f}s | Parameter memory: {m1_mem['model_size_mb']:.2f} MB")
        print(f"  [Model 2 - Continuous] Fit in {m2_train_res['wall_clock_time']:.2f}s")
        print(f"  [Model 3 - Event-Driven] Fit in {m3_train_res['wall_clock_time']:.2f}s")

        # ── STREAMING EVALUATION LOOP ─────────────────────────────────────────
        print(f"[Seed {seed}] Streaming prequential evaluation over {total_windows} windows...")

        m3_retrain_count = 0

        for w in range(total_windows):
            w_meta = window_schedule[w]
            regime = w_meta['regime']
            dom_type = w_meta['dominant_type']

            idx_s = w * WINDOW_SIZE
            idx_e = idx_s + WINDOW_SIZE
            X_win = X_stream[idx_s:idx_e]
            y_win = y_stream[idx_s:idx_e]

            # ────────────────────────────────────────────────────────────────
            # PHASE 1: PREQUENTIAL INFERENCE (Out-of-Sample)
            # ────────────────────────────────────────────────────────────────
            # A. Model 1 (Frozen Model)
            (m1_pred, m1_prob, m1_cpreds, m1_cprobs), m1_inf_res = measure_execution(m1_frozen.predict, X_win)
            m1_metrics = compute_metrics(y_win, m1_pred, m1_prob)

            # B. Model 2 (Continuously Retrained Ensemble)
            (m2_pred, m2_prob, m2_cpreds, m2_cprobs), m2_inf_res = measure_execution(m2_cont.predict, X_win)
            m2_metrics = compute_metrics(y_win, m2_pred, m2_prob)

            # C. Model 3 (Event-Driven Ensemble)
            (m3_pred, m3_prob, m3_cpreds, m3_cprobs), m3_inf_res = measure_execution(m3_event.predict, X_win)
            m3_metrics = compute_metrics(y_win, m3_pred, m3_prob)

            # D. Individual Adaptive Models (tracked from m2 component models)
            ind_metrics = {}
            for comp_name, full_name in [('RF', 'Adaptive Random Forest'), ('ET', 'Adaptive Extra Trees'), ('GB', 'Adaptive Gradient Boosting')]:
                ind_pred = m2_cpreds[comp_name]
                ind_prob = m2_cprobs[comp_name]
                ind_metrics[full_name] = compute_metrics(y_win, ind_pred, ind_prob)

            # ────────────────────────────────────────────────────────────────
            # PHASE 2: ONLINE DRIFT DETECTION (For Model 3)
            # ────────────────────────────────────────────────────────────────
            drift_decision = drift_detector.check_drift(X_train, X_win, m3_metrics['f1'])
            drift_detected = drift_decision['drift_detected']

            all_drift_records.append({
                'seed': seed,
                'window_id': w,
                'regime': regime,
                'dominant_type': dom_type,
                'drift_detected': drift_detected,
                'covariate_drift': drift_decision['covariate_drift'],
                'concept_drift': drift_decision['concept_drift'],
                'trigger_reasons': drift_decision['trigger_reasons'],
                'wasserstein_mean': drift_decision['wasserstein_mean'],
                'ks_mean': drift_decision['ks_mean'],
                'psi_mean': drift_decision['psi_mean'],
                'f1_drop': drift_decision['f1_drop'],
            })

            # ────────────────────────────────────────────────────────────────
            # PHASE 3: ADAPTATION & RETRAINING
            # ────────────────────────────────────────────────────────────────
            # A. Model 1 (Frozen): NEVER ADAPTS (Immutability verified)
            m1_retrain_cpu = 0.0
            m1_retrain_wall = 0.0

            # B. Model 2 (Continuous): Retrains at every window
            m2_cont.update_weights(y_win, m2_cpreds, m2_cprobs)
            (_, m2_retrain_res, _), _ = measure_execution(m2_cont.adapt, X_win, y_win)
            m2_retrain_cpu = sum(r['total_cpu_time'] for r in m2_retrain_res.values())
            m2_retrain_wall = sum(r['wall_clock_time'] for r in m2_retrain_res.values())

            # C. Model 3 (Event-Driven): Retrains ONLY when drift detected
            m3_event.update_weights(y_win, m3_cpreds, m3_cprobs)
            if drift_detected:
                (_, m3_retrain_res, _), _ = measure_execution(m3_event.adapt, X_win, y_win)
                m3_retrain_cpu = sum(r['total_cpu_time'] for r in m3_retrain_res.values())
                m3_retrain_wall = sum(r['wall_clock_time'] for r in m3_retrain_res.values())
                m3_retrained_flag = True
                m3_retrain_count += 1
            else:
                m3_event.buffer_X.extend(X_win)
                m3_event.buffer_y.extend(y_win)
                m3_retrain_cpu = 0.0
                m3_retrain_wall = 0.0
                m3_retrained_flag = False

            # ────────────────────────────────────────────────────────────────
            # PHASE 4: RECORD PERFORMANCE & TIMING RECORDS
            # ────────────────────────────────────────────────────────────────
            all_approaches = [
                ('Frozen Model', m1_metrics, m1_inf_res, m1_retrain_cpu, m1_retrain_wall, False),
                ('Continuously Retrained Ensemble', m2_metrics, m2_inf_res, m2_retrain_cpu, m2_retrain_wall, True),
                ('Event-Driven Ensemble', m3_metrics, m3_inf_res, m3_retrain_cpu, m3_retrain_wall, m3_retrained_flag),
                ('Adaptive Random Forest', ind_metrics['Adaptive Random Forest'], m2_inf_res, m2_retrain_cpu / 3.0, m2_retrain_wall / 3.0, True),
                ('Adaptive Extra Trees', ind_metrics['Adaptive Extra Trees'], m2_inf_res, m2_retrain_cpu / 3.0, m2_retrain_wall / 3.0, True),
                ('Adaptive Gradient Boosting', ind_metrics['Adaptive Gradient Boosting'], m2_inf_res, m2_retrain_cpu / 3.0, m2_retrain_wall / 3.0, True),
            ]

            for name, mets, inf_res, retr_cpu, retr_wall, did_retrain in all_approaches:
                tot_cpu = inf_res['total_cpu_time'] + retr_cpu
                tot_wall = inf_res['wall_clock_time'] + retr_wall

                all_window_records.append({
                    'seed': seed,
                    'window_id': w,
                    'approach': name,
                    'regime': regime,
                    'dominant_type': dom_type,
                    'f1': mets['f1'],
                    'macro_f1': mets['macro_f1'],
                    'accuracy': mets['accuracy'],
                    'precision': mets['precision'],
                    'recall': mets['recall'],
                    'balanced_accuracy': mets['balanced_accuracy'],
                    'roc_auc': mets['roc_auc'],
                    'pr_auc': mets['pr_auc'],
                    'cpu_time': tot_cpu,
                    'wall_clock_time': tot_wall,
                    'inference_cpu_time': inf_res['total_cpu_time'],
                    'inference_latency_ms': inf_res['wall_clock_time'] * 1000.0,
                    'retraining_cpu_time': retr_cpu,
                    'retrained': did_retrain,
                    'avg_cpu_percent': inf_res['avg_cpu_percent'],
                    'peak_cpu_percent': inf_res['peak_cpu_percent'],
                    'avg_ram_mb': inf_res['avg_ram_mb'],
                    'peak_ram_mb': inf_res['peak_ram_mb'],
                    'tn': mets['tn'],
                    'fp': mets['fp'],
                    'fn': mets['fn'],
                    'tp': mets['tp'],
                })

                all_timing_records.append({
                    'seed': seed,
                    'window_id': w,
                    'approach': name,
                    'inference_wall_s': inf_res['wall_clock_time'],
                    'retraining_wall_s': retr_wall,
                    'total_wall_s': tot_wall,
                    'inference_cpu_s': inf_res['total_cpu_time'],
                    'retraining_cpu_s': retr_cpu,
                    'total_cpu_s': tot_cpu,
                })

            # Save streaming memory snapshot at final window
            if w == total_windows - 1:
                for app_name, ens_obj, inf_res in [
                    ('Frozen Model', m1_frozen, m1_inf_res),
                    ('Continuously Retrained Ensemble', m2_cont, m2_inf_res),
                    ('Event-Driven Ensemble', m3_event, m3_inf_res)
                ]:
                    m_footprint = compute_model_memory(ens_obj.models['RF'])
                    m_footprint['model_size_kb'] += compute_model_memory(ens_obj.models['ET'])['model_size_kb']
                    m_footprint['model_size_kb'] += compute_model_memory(ens_obj.models['GB'])['model_size_kb']
                    all_memory_records.append({
                        'seed': seed,
                        'phase': 'streaming_end',
                        'approach': app_name,
                        'model_size_kb': m_footprint['model_size_kb'],
                        'model_size_mb': m_footprint['model_size_kb'] / 1024.0,
                        'peak_ram_mb': inf_res['peak_ram_mb'],
                        'avg_ram_mb': inf_res['avg_ram_mb'],
                        'cpu_time_s': inf_res['total_cpu_time'],
                    })

        print(f"[Seed {seed}] Completed stream. Model 3 retrained on {m3_retrain_count}/{total_windows} windows ({(m3_retrain_count/total_windows):.1%}).")

    total_experiment_duration = time.time() - total_experiment_start
    print(f"\n======================================================================")
    print(f"ALL {len(seeds)} SEEDS COMPLETED IN {total_experiment_duration:.2f}s")
    print(f"======================================================================")

    # ────────────────────────────────────────────────────────────────
    # EXPORT CSV ARTIFACTS
    # ────────────────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_window_records)
    metrics_csv = os.path.join(RESULTS_DIR, 'metrics.csv')
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Exported metrics to: {metrics_csv}")

    df_memory = pd.DataFrame(all_memory_records)
    memory_csv = os.path.join(RESULTS_DIR, 'memory.csv')
    df_memory.to_csv(memory_csv, index=False)
    print(f"Exported memory measurements to: {memory_csv}")

    df_timing = pd.DataFrame(all_timing_records)
    timing_csv = os.path.join(RESULTS_DIR, 'timing.csv')
    df_timing.to_csv(timing_csv, index=False)
    print(f"Exported timing measurements to: {timing_csv}")

    # Aggregate per-regime metrics
    regime_summary = df_metrics.groupby(['regime', 'approach']).agg(
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        precision_mean=('precision', 'mean'),
        recall_mean=('recall', 'mean'),
        retrain_events=('retrained', 'sum'),
        total_cpu_time=('cpu_time', 'sum'),
        peak_ram_mb=('peak_ram_mb', 'max'),
    ).reset_index()

    regime_csv = os.path.join(RESULTS_DIR, 'regime_metrics.csv')
    regime_summary.to_csv(regime_csv, index=False)
    print(f"Exported regime metrics to: {regime_csv}")

    # Export confusion matrices summary per approach
    cm_summary = df_metrics.groupby('approach').agg(
        total_tn=('tn', 'sum'),
        total_fp=('fp', 'sum'),
        total_fn=('fn', 'sum'),
        total_tp=('tp', 'sum'),
    ).reset_index()
    cm_csv = os.path.join(CONFUSION_DIR, 'confusion_matrix_summary.csv')
    cm_summary.to_csv(cm_csv, index=False)
    print(f"Exported confusion matrix summary to: {cm_csv}")

    # Export experiment config JSON
    config_dict = {
        'dataset': 'ToN_IoT (Train_Test_IoT_Weather.csv)',
        'total_usable_samples': N_TOTAL_SAMPLES,
        'initial_training_samples': N_INITIAL_TRAINING,
        'streaming_samples': N_STREAM_SAMPLES,
        'window_size': WINDOW_SIZE,
        'n_windows': total_windows,
        'seeds': seeds,
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'regimes': REGIME_BOUNDARIES,
        'approaches': APPROACH_NAMES,
        'total_wall_clock_time_s': total_experiment_duration
    }
    cfg_json = os.path.join(RESULTS_DIR, 'experiment_config.json')
    with open(cfg_json, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Exported experiment config to: {cfg_json}")

    results_dict = {
        'df_metrics': df_metrics,
        'regime_summary': regime_summary,
        'df_memory': df_memory,
        'df_timing': df_timing,
        'df_drift': pd.DataFrame(all_drift_records)
    }

    try:
        from plots import generate_all_plots
        generate_all_plots(results_dict)
    except Exception as e:
        print(f"Plotting encountered an error: {e}")

    return results_dict


if __name__ == '__main__':
    run_experiment5()
