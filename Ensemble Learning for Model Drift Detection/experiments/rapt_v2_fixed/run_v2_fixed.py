"""
Master Driver for RAPT-v2 Fixed (Corrected Architecture)
Runs prequential evaluation over 5 seeds (42, 43, 44, 45, 46) on 9A (n_init=200) and 9B (n_init=99).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import psutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rapt_v2_fixed import create_base_ensemble_v2_fixed, RAPTv2SystemFixed
from evaluation_v2_fixed import calculate_window_metrics, compute_efficiency_objective, compute_wilcoxon_tests_v2

# Add original exp9 and exp9b modules for Event-Driven baselines
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "exp9"))
import event_driven as ed_9a

sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "exp9b"))
import event_driven_9b as ed_9b
import preprocessing_9b as preproc_9b

BASE_DIR = os.path.join(ROOT_DIR, "experiments", "rapt_v2_fixed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SEEDS = [42, 43, 44, 45, 46]

FEATURES_9A = [
    'mean_iat', 'std_iat', 'median_iat', 'p90_iat', 'mean_packet_size',
    'std_packet_size', 'mean_delay', 'std_delay', 'median_delay', 'p90_delay',
    'mean_pdist', 'mean_piat', 'mean_psize', 'bandwidth', 'slots', 'ratio', 'packet_count'
]

FEATURES_9B = [
    'mean_latency', 'median_latency', 'std_latency', 'p90_latency',
    'p95_latency', 'max_latency', 'packet_loss_rate', 'delivery_rate',
    'mean_interarrival_time', 'std_interarrival_time', 'packet_count',
    'effective_throughput'
]

def run_dataset_eval(df_stream, feature_cols, target_col, regime_col, dataset_label="9A", n_init=200):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING RAPT-v2 FIXED EVALUATION ON DATASET {dataset_label}", flush=True)
    print(f"=======================================================", flush=True)
    
    per_seed_results = []
    per_window_results = []
    hierarchy_results = []
    
    variants = [
        'Frozen',
        'Event-Driven',
        'Full Retraining',
        'RAPT-v1',
        'RAPT-A',
        'RAPT-B',
        'RAPT-C',
        'RAPT-D',
        'RAPT-E'
    ]
    
    for seed in SEEDS:
        print(f"\n--- [{dataset_label}] Random Seed {seed} ---", flush=True)
        
        df_init = df_stream.iloc[:n_init]
        if dataset_label == "9B":
            preproc = preproc_9b.StreamingPreprocessor()
            preproc.fit_initial(df_init)
            X_init = preproc.transform(df_init)
        else:
            X_init = df_init[feature_cols].values
            
        y_init = df_init[target_col].values
        
        for v in variants:
            print(f"  [{dataset_label} | Seed {seed}] Evaluating variant: {v}...", flush=True)
            y_true_all, y_pred_all = [], []
            pred_cpu_total, adapt_cpu_total = 0.0, 0.0
            retrain_events = 0
            reused_checkpoints = 0
            
            hist_X = list(X_init)
            hist_y = list(y_init)
            
            level_counts = {
                'level_0_pure_reuse': 0,
                'level_1_weight_adapt': 0,
                'level_2_partial_update': 0,
                'level_3_full_retrain': 0
            }
            
            if v == 'Frozen':
                model = create_base_ensemble_v2_fixed(seed=seed)
                t0_c = time.process_time()
                model.fit(X_init, y_init)
                t_init_cpu = time.process_time() - t0_c
                
                for idx in range(n_init, len(df_stream)):
                    df_w = df_stream.iloc[[idx]]
                    X_w = preproc.transform(df_w) if dataset_label == "9B" else df_w[feature_cols].values
                    y_w = df_w[target_col].values[0]
                    w_id = idx
                    reg_id = df_w[regime_col].values[0]
                    
                    t0_p = time.process_time()
                    pred = model.predict(X_w)[0]
                    t_p = time.process_time() - t0_p
                    pred_cpu_total += t_p
                    
                    y_true_all.append(y_w)
                    y_pred_all.append(pred)
                    
                    per_window_results.append({
                        'seed': seed, 'dataset': dataset_label, 'method': v,
                        'window_id': w_id, 'regime_id': reg_id, 'y_true': int(y_w),
                        'y_pred': int(pred), 'is_correct': int(pred == y_w),
                        'step_cpu_sec': t_p, 'level_used': 'level_0_pure_reuse'
                    })
                    
                total_cpu = t_init_cpu + pred_cpu_total
                adapt_cpu = 0.0
                
            elif v == 'Event-Driven':
                if dataset_label == "9B":
                    ed_system = ed_9b.EventDrivenEnsemble(seed=seed)
                else:
                    ed_system = ed_9a.EventDrivenEnsemble(seed=seed)
                    
                t0_c = time.process_time()
                ed_system.fit_initial(X_init, y_init)
                t_init_cpu = time.process_time() - t0_c
                
                for idx in range(n_init, len(df_stream)):
                    df_w = df_stream.iloc[[idx]]
                    X_w = preproc.transform(df_w) if dataset_label == "9B" else df_w[feature_cols].values
                    y_w = df_w[target_col].values[0]
                    w_id = idx
                    reg_id = df_w[regime_col].values[0]
                    
                    t0_p = time.process_time()
                    pred = ed_system.predict(X_w)[0]
                    t_p = time.process_time() - t0_p
                    pred_cpu_total += t_p
                    
                    err = 1.0 if pred != y_w else 0.0
                    triggered, t_adapt, _ = ed_system.update_and_adapt(X_w, [y_w], err)
                    adapt_cpu_total += t_adapt
                    if triggered:
                        retrain_events += 1
                        level_counts['level_3_full_retrain'] += 1
                    else:
                        level_counts['level_0_pure_reuse'] += 1
                        
                    y_true_all.append(y_w)
                    y_pred_all.append(pred)
                    
                    per_window_results.append({
                        'seed': seed, 'dataset': dataset_label, 'method': v,
                        'window_id': w_id, 'regime_id': reg_id, 'y_true': int(y_w),
                        'y_pred': int(pred), 'is_correct': int(pred == y_w),
                        'step_cpu_sec': t_p + t_adapt,
                        'level_used': 'level_3_full_retrain' if triggered else 'level_0_pure_reuse'
                    })
                    
                total_cpu = t_init_cpu + adapt_cpu_total + pred_cpu_total
                adapt_cpu = adapt_cpu_total
                
            elif v == 'Full Retraining':
                model = create_base_ensemble_v2_fixed(seed=seed)
                t0_c = time.process_time()
                model.fit(X_init, y_init)
                t_init_cpu = time.process_time() - t0_c
                
                prev_reg = df_stream.iloc[n_init][regime_col]
                for idx in range(n_init, len(df_stream)):
                    df_w = df_stream.iloc[[idx]]
                    X_w = preproc.transform(df_w) if dataset_label == "9B" else df_w[feature_cols].values
                    y_w = df_w[target_col].values[0]
                    w_id = idx
                    reg_id = df_w[regime_col].values[0]
                    
                    is_trans = (reg_id != prev_reg)
                    prev_reg = reg_id
                    
                    t_adapt = 0.0
                    if is_trans:
                        retrain_events += 1
                        t0_a = time.process_time()
                        model = create_base_ensemble_v2_fixed(seed=seed + retrain_events * 11)
                        model.fit(np.array(hist_X[-1000:]), np.array(hist_y[-1000:]))
                        t_adapt = time.process_time() - t0_a
                        adapt_cpu_total += t_adapt
                        level_counts['level_3_full_retrain'] += 1
                    else:
                        level_counts['level_0_pure_reuse'] += 1
                        
                    t0_p = time.process_time()
                    pred = model.predict(X_w)[0]
                    t_p = time.process_time() - t0_p
                    pred_cpu_total += t_p
                    
                    hist_X.append(X_w[0])
                    hist_y.append(y_w)
                    y_true_all.append(y_w)
                    y_pred_all.append(pred)
                    
                    per_window_results.append({
                        'seed': seed, 'dataset': dataset_label, 'method': v,
                        'window_id': w_id, 'regime_id': reg_id, 'y_true': int(y_w),
                        'y_pred': int(pred), 'is_correct': int(pred == y_w),
                        'step_cpu_sec': t_p + t_adapt,
                        'level_used': 'level_3_full_retrain' if is_trans else 'level_0_pure_reuse'
                    })
                    
                total_cpu = t_init_cpu + adapt_cpu_total + pred_cpu_total
                adapt_cpu = adapt_cpu_total

            else:
                # RAPT Variants: RAPT-v1, RAPT-A, RAPT-B, RAPT-C, RAPT-D, RAPT-E
                variant_code = 'v1' if v == 'RAPT-v1' else v
                rapt = RAPTv2SystemFixed(seed=seed, variant=variant_code, buffer_capacity=50)
                initial_reg = df_init[regime_col].values[0]
                t_init_cpu, _ = rapt.fit_initial(initial_reg, X_init, y_init, window_id=0)
                
                prev_reg = initial_reg
                
                for idx in range(n_init, len(df_stream)):
                    df_w = df_stream.iloc[[idx]]
                    X_w = preproc.transform(df_w) if dataset_label == "9B" else df_w[feature_cols].values
                    y_w = df_w[target_col].values[0]
                    w_id = idx
                    reg_id = df_w[regime_col].values[0]
                    
                    level_used = 'level_0_pure_reuse'
                    t_trans_cpu = 0.0
                    if reg_id != prev_reg:
                        reused_flag, t_trans_cpu, level_used = rapt.handle_regime_transition(
                            new_regime_id=reg_id,
                            window_id=w_id,
                            X_buffer=hist_X[-500:], # Pass full up to 500 samples for new regime training!
                            y_buffer=hist_y[-500:]
                        )
                        prev_reg = reg_id
                        if reused_flag:
                            reused_checkpoints += 1
                        else:
                            retrain_events += 1
                            
                    t0_p = time.process_time()
                    pred = rapt.predict(X_w)[0]
                    t_p = time.process_time() - t0_p
                    pred_cpu_total += t_p
                    
                    rec_err = 1.0 if pred != y_w else 0.0
                    escalated, esc_level, t_esc_cpu = rapt.check_and_escalate_fallback(hist_X[-500:], hist_y[-500:], rec_err)
                    
                    if escalated:
                        level_used = esc_level
                        adapt_cpu_total += t_esc_cpu
                        
                    adapt_cpu_total += t_trans_cpu
                    hist_X.append(X_w[0])
                    hist_y.append(y_w)
                    y_true_all.append(y_w)
                    y_pred_all.append(pred)
                    
                    per_window_results.append({
                        'seed': seed, 'dataset': dataset_label, 'method': v,
                        'window_id': w_id, 'regime_id': reg_id, 'y_true': int(y_w),
                        'y_pred': int(pred), 'is_correct': int(pred == y_w),
                        'step_cpu_sec': t_p + t_trans_cpu + (t_esc_cpu if escalated else 0.0),
                        'level_used': level_used
                    })
                    
                total_cpu = t_init_cpu + adapt_cpu_total + pred_cpu_total
                adapt_cpu = adapt_cpu_total
                
            m_res = calculate_window_metrics(y_true_all, y_pred_all)
            mem_mb = float(psutil.Process().memory_info().rss / (1024 * 1024))
            
            per_seed_results.append({
                'seed': seed, 'dataset': dataset_label, 'method': v,
                'macro_f1': m_res['macro_f1'], 'accuracy': m_res['accuracy'],
                'balanced_accuracy': m_res['balanced_accuracy'],
                'precision': m_res['precision'], 'recall': m_res['recall'],
                'total_cpu_sec': total_cpu, 'adaptation_cpu_sec': adapt_cpu,
                'prediction_cpu_sec': pred_cpu_total, 'retrain_events': retrain_events,
                'reused_checkpoints': reused_checkpoints, 'memory_mb': mem_mb
            })
            
            for lk, lv in rapt.level_counts.items() if v.startswith('RAPT') else level_counts.items():
                hierarchy_results.append({'seed': seed, 'dataset': dataset_label, 'method': v, 'level': lk, 'count': lv})
                
    df_per_seed = pd.DataFrame(per_seed_results)
    df_per_window = pd.DataFrame(per_window_results)
    df_hierarchy = pd.DataFrame(hierarchy_results)
    
    summary_cols = ['macro_f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'total_cpu_sec', 'adaptation_cpu_sec', 'prediction_cpu_sec', 'retrain_events', 'reused_checkpoints', 'memory_mb']
    summary_recs = []
    for v in variants:
        df_v = df_per_seed[df_per_seed['method'] == v]
        rec = {'method': v, 'dataset': dataset_label}
        for col in summary_cols:
            rec[f"{col}_mean"] = float(df_v[col].mean())
            rec[f"{col}_std"] = float(df_v[col].std())
        summary_recs.append(rec)
        
    df_summary = pd.DataFrame(summary_recs)
    df_summary = compute_efficiency_objective(df_summary, lambda_cpu=0.01)
    
    return df_summary, df_per_seed, df_per_window, df_hierarchy

def main():
    print("=" * 80, flush=True)
    print("STARTING RAPT-v2 FIXED EVALUATION SUITE", flush=True)
    print("=" * 80, flush=True)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    stream_9a_path = "experiments/exp9/data/processed_exp9_stream.csv"
    stream_9b_path = "experiments/exp9b/data/processed_exp9b_stream.csv"
    
    df_stream_9a = pd.read_csv(stream_9a_path)
    df_stream_9b = pd.read_csv(stream_9b_path)
    
    # Run Evaluations with correct n_init (9A: 200, 9B: 99)
    sum_9a, seed_9a, win_9a, h_9a = run_dataset_eval(df_stream_9a, FEATURES_9A, 'target', 'regime_label', dataset_label="9A", n_init=200)
    sum_9b, seed_9b, win_9b, h_9b = run_dataset_eval(df_stream_9b, FEATURES_9B, 'qos_target', 'regime_id', dataset_label="9B", n_init=99)
    
    # Save results
    sum_9a.to_csv(os.path.join(RESULTS_DIR, 'summary_9a_fixed.csv'), index=False)
    sum_9b.to_csv(os.path.join(RESULTS_DIR, 'summary_9b_fixed.csv'), index=False)
    
    print("\n" + "=" * 80, flush=True)
    print("RAPT-v2 FIXED SUMMARY RESULTS — EXPERIMENT 9A", flush=True)
    print("=" * 80, flush=True)
    print(sum_9a[['method', 'macro_f1_mean', 'macro_f1_std', 'total_cpu_sec_mean', 'adaptation_cpu_sec_mean', 'retrain_events_mean']].to_string(index=False), flush=True)

    print("\n" + "=" * 80, flush=True)
    print("RAPT-v2 FIXED SUMMARY RESULTS — EXPERIMENT 9B", flush=True)
    print("=" * 80, flush=True)
    print(sum_9b[['method', 'macro_f1_mean', 'macro_f1_std', 'total_cpu_sec_mean', 'adaptation_cpu_sec_mean', 'retrain_events_mean']].to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
