"""
================================================================================
EXPERIMENT 5 — REGIME-AWARE ENSEMBLE WEIGHTING MAIN BENCHMARK
================================================================================
Main execution script for Experiment 5:
  - Dataset: Official ToN_IoT Weather Telemetry (50,000 sequential samples)
  - Seeds: 5 deterministic seeds [42, 43, 44, 45, 46]
  - Methods:
      1. Fixed Ensemble (Baseline: Equal weights 1/3, 1/3, 1/3)
      2. Global Adaptive Ensemble (Global EMA F1 Softmax weights)
      3. Regime-Aware Ensemble (Distributional quantile fingerprinting + regime memory)
      4. Oracle Regime Weights (Offline optimal weight vector upper bound)
  - Ablations: Similarity thresholds tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
  - Prequential evaluation, zero data leakage, process memory profiling,
    strict metric validation: F1 = 2 * Precision * Recall / (Precision + Recall).
================================================================================
"""

import os
import sys
import time
import json
import psutil
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEEDS, N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_STREAM_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, RESULTS_DIR, FIGURES_DIR,
    DEFAULT_SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLDS, METHOD_NAMES
)
from data_loader import load_ton_iot_data, prepare_experiment_split
from preprocessing import Preprocessor
from ensemble_weighting import EnsembleWeightingRunner
from metrics import compute_metrics
from resource_monitor import measure_execution, compute_model_memory
from plots_exp5 import generate_all_experiment5_figures


def run_experiment5(seeds=SEEDS, smoke_test=False, max_windows=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if smoke_test:
        print("!!! RUNNING IN SMOKE TEST MODE (1 seed, 5 windows) !!!")
        seeds = [42]
        max_windows = 5

    print("=" * 85)
    print("EXPERIMENT 5: REGIME-AWARE ENSEMBLE WEIGHTING BENCHMARK")
    print(f"Dataset: ToN_IoT Weather | Total Sequential Samples: {N_TOTAL_SAMPLES:,}")
    print(f"Initial Training: {N_INITIAL_TRAINING:,} | Streaming Partition: {N_STREAM_SAMPLES:,}")
    print(f"Windows: {N_WINDOWS} (Window Size: {WINDOW_SIZE}) | Deterministic Seeds: {seeds}")
    print("=" * 85)

    # 1. Load Data
    print("\n[Data Loader] Loading official ToN_IoT Weather dataset...")
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

    strategies_to_run = [
        ('Fixed Ensemble', 'fixed', DEFAULT_SIMILARITY_THRESHOLD),
        ('Global Adaptive Ensemble', 'global_adaptive', DEFAULT_SIMILARITY_THRESHOLD),
        ('Regime-Aware Ensemble', 'regime_aware', DEFAULT_SIMILARITY_THRESHOLD),
        ('Oracle Regime Weights', 'oracle', DEFAULT_SIMILARITY_THRESHOLD),
    ]

    all_window_records = []
    all_summary_records = []
    all_weights_records = []
    all_regime_records = []
    all_ablation_records = []

    process = psutil.Process(os.getpid())

    # ──────────────────────────────────────────────────────────────────────────
    # CORE EXPERIMENT LOOP ACROSS SEEDS AND METHODS
    # ──────────────────────────────────────────────────────────────────────────
    for seed in seeds:
        print(f"\n" + "=" * 75)
        print(f"EXECUTING DETERMINISTIC SEED: {seed}")
        print("=" * 75)

        for strat_label, strat_type, sim_th in strategies_to_run:
            print(f"\n[Seed {seed}] Strategy: {strat_label} (sim_threshold={sim_th})...")
            start_time = time.time()
            start_cpu = time.process_time()

            runner = EnsembleWeightingRunner(
                seed=seed,
                strategy_type=strat_type,
                similarity_threshold=sim_th
            )
            runner.fit_initial(X_train, y_train)

            accum_adaptation_cpu = 0.0
            accum_retrains = 0

            seed_preds = []
            seed_probs = []
            seed_trues = []

            for w in range(total_windows):
                w_meta = window_schedule[w]
                idx_s = w * WINDOW_SIZE
                idx_e = idx_s + WINDOW_SIZE
                X_win = X_stream[idx_s:idx_e]
                y_win = y_stream[idx_s:idx_e]

                # 1. PREQUENTIAL INFERENCE (Zero Leakage)
                # For Oracle, pass y_win to compute offline optimal weights; for online strategies y_win is unused
                y_oracle_input = y_win if strat_type == 'oracle' else None
                pred_ens, prob_ens, comp_preds, comp_probs, weights = runner.predict(X_win, y_oracle_input)

                # 2. RECORD METRICS
                m = compute_metrics(y_win, pred_ens, prob_ens)

                # Verify F1 mathematical consistency: F1 = 2 * P * R / (P + R)
                p_val, r_val, f1_val = m['precision'], m['recall'], m['f1']
                if p_val + r_val > 1e-8:
                    expected_f1 = 2.0 * p_val * r_val / (p_val + r_val)
                    assert abs(f1_val - expected_f1) < 1e-4, f"F1 consistency assertion failed: {f1_val} vs {expected_f1}"

                seed_preds.extend(pred_ens)
                seed_probs.extend(prob_ens)
                seed_trues.extend(y_win)

                r_info = runner.current_regime_info

                all_window_records.append({
                    'seed': seed,
                    'method': strat_label,
                    'window_id': w,
                    'dominant_regime': w_meta['regime'],
                    'dominant_type': w_meta['dominant_type'],
                    'f1': f1_val,
                    'accuracy': m['accuracy'],
                    'precision': p_val,
                    'recall': r_val,
                    'w_rf': weights['RF'],
                    'w_et': weights['ET'],
                    'w_gb': weights['GB'],
                    'regime_id': r_info['regime_id'],
                    'regime_similarity': r_info['similarity'],
                    'is_retrieved': r_info['is_retrieved']
                })

                all_weights_records.append({
                    'seed': seed,
                    'method': strat_label,
                    'window_id': w,
                    'w_rf': weights['RF'],
                    'w_et': weights['ET'],
                    'w_gb': weights['GB'],
                    'regime_id': r_info['regime_id']
                })

                # 3. POST-PREDICTION UPDATE & DRIFT ADAPTATION
                update_res = runner.update_and_adapt(X_win, y_win, comp_preds, f1_val, X_train)
                accum_adaptation_cpu += update_res['adaptation_time']
                if update_res['drift_detected']:
                    accum_retrains += 1

            total_cpu = time.process_time() - start_cpu
            total_wall = time.time() - start_time
            peak_ram_mb = process.memory_info().rss / (1024.0 * 1024.0)

            overall_metrics = compute_metrics(np.array(seed_trues), np.array(seed_preds), np.array(seed_probs))

            # Regime statistics for Regime-Aware strategy
            if strat_type == 'regime_aware':
                n_unique_regimes = len(runner.regime_memory.regimes)
                retrieval_count = runner.regime_memory.retrieval_count
                avg_similarity = float(np.mean(runner.regime_memory.similarity_history)) if runner.regime_memory.similarity_history else 1.0
            else:
                n_unique_regimes = 1
                retrieval_count = 0
                avg_similarity = 1.0

            all_summary_records.append({
                'seed': seed,
                'method': strat_label,
                'similarity_threshold': sim_th,
                'f1': overall_metrics['f1'],
                'accuracy': overall_metrics['accuracy'],
                'precision': overall_metrics['precision'],
                'recall': overall_metrics['recall'],
                'adaptation_cpu_s': accum_adaptation_cpu,
                'total_cpu_s': total_cpu,
                'wall_clock_s': total_wall,
                'retrain_events': accum_retrains,
                'peak_ram_mb': peak_ram_mb,
                'unique_regimes': n_unique_regimes,
                'regime_retrievals': retrieval_count,
                'avg_regime_similarity': avg_similarity
            })

            print(f"  --> Mean F1: {overall_metrics['f1']:.4f} | Accuracy: {overall_metrics['accuracy']:.4f} | "
                  f"Adapt CPU: {accum_adaptation_cpu:.2f}s | Retrains: {accum_retrains} | Regimes: {n_unique_regimes}")

    # ──────────────────────────────────────────────────────────────────────────
    # THRESHOLD ABLATION STUDY FOR REGIME-AWARE STRATEGY
    # ──────────────────────────────────────────────────────────────────────────
    if not smoke_test:
        print("\n" + "=" * 75)
        print("RUNNING SIMILARITY THRESHOLD ABLATION STUDY FOR REGIME-AWARE ENSEMBLE")
        print("=" * 75)

        for sim_th in SIMILARITY_THRESHOLDS:
            print(f"\n[Ablation] Testing similarity threshold tau = {sim_th:.2f} across all 5 seeds...")
            for seed in seeds:
                runner = EnsembleWeightingRunner(
                    seed=seed,
                    strategy_type='regime_aware',
                    similarity_threshold=sim_th
                )
                runner.fit_initial(X_train, y_train)

                seed_preds = []
                seed_probs = []
                seed_trues = []

                for w in range(total_windows):
                    idx_s = w * WINDOW_SIZE
                    idx_e = idx_s + WINDOW_SIZE
                    X_win = X_stream[idx_s:idx_e]
                    y_win = y_stream[idx_s:idx_e]

                    pred_ens, prob_ens, comp_preds, comp_probs, weights = runner.predict(X_win)
                    m_win = compute_metrics(y_win, pred_ens, prob_ens)
                    f1_val = m_win['f1']

                    seed_preds.extend(pred_ens)
                    seed_probs.extend(prob_ens)
                    seed_trues.extend(y_win)

                    runner.update_and_adapt(X_win, y_win, comp_preds, f1_val, X_train)

                m = compute_metrics(np.array(seed_trues), np.array(seed_preds), np.array(seed_probs))
                n_reg = len(runner.regime_memory.regimes)
                ret_cnt = runner.regime_memory.retrieval_count
                avg_sim = float(np.mean(runner.regime_memory.similarity_history)) if runner.regime_memory.similarity_history else 1.0

                all_ablation_records.append({
                    'seed': seed,
                    'similarity_threshold': sim_th,
                    'f1': m['f1'],
                    'accuracy': m['accuracy'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'unique_regimes': n_reg,
                    'regime_retrievals': ret_cnt,
                    'avg_similarity': avg_sim
                })

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE RESULTS & GENERATE CSV/JSON ARTIFACTS
    # ──────────────────────────────────────────────────────────────────────────
    df_window = pd.DataFrame(all_window_records)
    df_summary = pd.DataFrame(all_summary_records)
    df_weights = pd.DataFrame(all_weights_records)
    df_ablation = pd.DataFrame(all_ablation_records)

    summary_csv = os.path.join(RESULTS_DIR, 'exp5_metrics.csv')
    weights_csv = os.path.join(RESULTS_DIR, 'exp5_weights_over_time.csv')
    windows_csv = os.path.join(RESULTS_DIR, 'exp5_windows.csv')
    ablation_csv = os.path.join(RESULTS_DIR, 'exp5_threshold_ablation.csv')

    df_summary.to_csv(summary_csv, index=False)
    df_weights.to_csv(weights_csv, index=False)
    df_window.to_csv(windows_csv, index=False)
    if not df_ablation.empty:
        df_ablation.to_csv(ablation_csv, index=False)

    # Compute mean +/- std summary table across seeds
    grouped = df_summary.groupby('method').agg({
        'f1': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'adaptation_cpu_s': ['mean', 'std'],
        'total_cpu_s': ['mean', 'std'],
        'retrain_events': ['mean', 'std'],
        'peak_ram_mb': ['mean', 'std'],
        'unique_regimes': ['mean', 'std'],
        'regime_retrievals': ['mean', 'std'],
        'avg_regime_similarity': ['mean', 'std'],
    })

    summary_json = os.path.join(RESULTS_DIR, 'exp5_summary.json')
    summary_dict = {}
    for method_name in METHOD_NAMES:
        if method_name in df_summary['method'].values:
            m_sub = df_summary[df_summary['method'] == method_name]
            summary_dict[method_name] = {
                'f1_mean': float(m_sub['f1'].mean()),
                'f1_std': float(m_sub['f1'].std()),
                'accuracy_mean': float(m_sub['accuracy'].mean()),
                'accuracy_std': float(m_sub['accuracy'].std()),
                'precision_mean': float(m_sub['precision'].mean()),
                'precision_std': float(m_sub['precision'].std()),
                'recall_mean': float(m_sub['recall'].mean()),
                'recall_std': float(m_sub['recall'].std()),
                'adaptation_cpu_mean': float(m_sub['adaptation_cpu_s'].mean()),
                'adaptation_cpu_std': float(m_sub['adaptation_cpu_s'].std()),
                'retrain_events_mean': float(m_sub['retrain_events'].mean()),
                'unique_regimes_mean': float(m_sub['unique_regimes'].mean()),
                'regime_retrievals_mean': float(m_sub['regime_retrievals'].mean()),
                'avg_regime_similarity_mean': float(m_sub['avg_regime_similarity'].mean()),
            }

    with open(summary_json, 'w') as f:
        json.dump(summary_dict, f, indent=4)

    print("\n" + "=" * 85)
    print("EXPERIMENT 5 SUMMARY RESULTS (Mean +/- Std across Seeds)")
    print("=" * 85)
    for method, metrics in summary_dict.items():
        print(f"Method: {method:<28} | F1: {metrics['f1_mean']:.4f} +/- {metrics['f1_std']:.4f} | "
              f"Acc: {metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f} | "
              f"Adapt CPU: {metrics['adaptation_cpu_mean']:.2f}s | Regimes: {metrics['unique_regimes_mean']:.1f}")

    # Generate Figures
    print("\n[Figures] Generating publication-quality plots in results/figures...")
    generate_all_experiment5_figures(df_summary, df_window, df_weights, df_ablation, FIGURES_DIR)
    print(f"[Figures] All figures saved successfully to: {FIGURES_DIR}")

    return summary_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Experiment 5: Regime-Aware Ensemble Weighting")
    parser.add_argument('--smoke-test', action='store_true', help="Run quick smoke test (1 seed, 5 windows)")
    args = parser.parse_args()

    run_experiment5(smoke_test=args.smoke_test)
