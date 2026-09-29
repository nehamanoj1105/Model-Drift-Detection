"""
Master Runner for RAPT-v3 Benchmark Suite
Executes strict prequential evaluation across 5 random seeds (42, 43, 44, 45, 46) for:
  - Frozen
  - Event-Driven
  - Full Retraining
  - Original RAPT
  - RAPT-E (RAPT-v1 reference)
  - RAPT-v3-50 (Champion-Challenger with buffer=50)
  - RAPT-v3-100 (Champion-Challenger with buffer=100)
  - RAPT-v3-250 (Champion-Challenger with buffer=250)
on Dataset 9A (5G Campus QoS) and Dataset 9B (5G NR End-to-End Latency).
"""

import os
import sys
import copy
import time
import json
import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import original baseline modules
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "exp9"))
import models as models_9a
import event_driven as ed_9a
import rapt as orig_rapt_9a

sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "exp9b"))
import models_9b as models_9b
import event_driven_9b as ed_9b
import rapt_9b as orig_rapt_9b
import preprocessing_9b as preproc_9b

sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "rapt_v2_fixed"))
import rapt_v2_fixed as rapt_v2_fixed

# Import RAPT-v3 modules
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "rapt_v3"))
import rapt_v3 as rapt_v3
from evaluation_v3 import calculate_window_metrics, compute_wilcoxon_and_non_inferiority, analyze_regime_transitions_v3

sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "final_validation"))
from timing_utils import TimingTracker

RAPT_V3_DIR = os.path.join(ROOT_DIR, "experiments", "rapt_v3")
RESULTS_DIR = os.path.join(RAPT_V3_DIR, "results")
PLOTS_DIR = os.path.join(RAPT_V3_DIR, "plots")
COMPARISON_DIR = os.path.join(RAPT_V3_DIR, "comparison")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

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

def run_benchmark_stream(df_stream, feature_cols, target_col, regime_col, dataset_label="9A", n_init=200):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING RAPT-v3 BENCHMARK ON DATASET {dataset_label}", flush=True)
    print(f"=======================================================", flush=True)
    
    per_seed_results = []
    per_window_results = []
    all_trace_records = []
    
    methods = [
        'Frozen',
        'Event-Driven',
        'Full Retraining',
        'Original RAPT',
        'RAPT-E',
        'RAPT-v3-50',
        'RAPT-v3-100',
        'RAPT-v3-250'
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
        initial_regime = df_init[regime_col].values[0]
        
        for m in methods:
            print(f"  [{dataset_label} | Seed {seed}] Evaluating: {m}...", flush=True)
            tracker = TimingTracker(method_name=m, dataset_label=dataset_label, seed=seed)
            
            y_true_all, y_pred_all = [], []
            retrain_events = 0
            reused_checkpoints = 0
            champion_selections = 0
            challenger_selections = 0
            
            hist_X = list(X_init)
            hist_y = list(y_init)
            
            tracker.start_stream()
            
            # INITIAL TRAINING PHASE
            if m == 'Frozen':
                model = models_9b.create_base_ensemble(seed=seed) if dataset_label=="9B" else models_9a.create_base_ensemble(seed=seed)
                t0_fit = time.perf_counter()
                model.fit(X_init, y_init)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                
            elif m == 'Event-Driven':
                ed_sys = ed_9b.EventDrivenEnsemble(seed=seed) if dataset_label=="9B" else ed_9a.EventDrivenEnsemble(seed=seed)
                t0_fit = time.perf_counter()
                ed_sys.fit_initial(X_init, y_init)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                
            elif m == 'Full Retraining':
                model = models_9b.create_base_ensemble(seed=seed) if dataset_label=="9B" else models_9a.create_base_ensemble(seed=seed)
                t0_fit = time.perf_counter()
                model.fit(X_init, y_init)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                
            elif m == 'Original RAPT':
                rapt_sys = orig_rapt_9b.RAPTSystem(seed=seed, mode='full') if dataset_label=="9B" else orig_rapt_9a.RAPTSystem(seed=seed, mode='full')
                t0_fit = time.perf_counter()
                rapt_sys.fit_initial(initial_regime, X_init, y_init, window_id=0)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                
            elif m == 'RAPT-E':
                rapt_sys = rapt_v2_fixed.RAPTv2SystemFixed(seed=seed, variant='RAPT-E', buffer_capacity=50)
                t0_fit = time.perf_counter()
                rapt_sys.fit_initial(initial_regime, X_init, y_init, window_id=0)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                
            elif m.startswith('RAPT-v3'):
                buf_size = int(m.split('-')[-1])
                rapt_sys = rapt_v3.RAPTv3System(seed=seed, variant=m, beta=0.90, alpha=0.10, buffer_size=buf_size, confidence_gate=0.99)
                t0_fit = time.perf_counter()
                rapt_sys.fit_initial(initial_regime, X_init, y_init, window_id=0)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                
            prev_regime = initial_regime
            
            # STREAMING LOOP
            for idx in range(n_init, len(df_stream)):
                df_w = df_stream.iloc[[idx]]
                X_w = preproc.transform(df_w) if dataset_label == "9B" else df_w[feature_cols].values
                y_w = df_w[target_col].values[0]
                w_id = df_w['window_id'].values[0] if 'window_id' in df_w.columns else idx
                reg_id = df_w[regime_col].values[0]
                
                # 1. PREDICTION STEP
                t0_p = time.perf_counter()
                if m == 'Frozen':
                    pred = model.predict(X_w)[0]
                elif m == 'Event-Driven':
                    pred = ed_sys.predict(X_w)[0]
                elif m == 'Full Retraining':
                    pred = model.predict(X_w)[0]
                elif m in ['Original RAPT', 'RAPT-E'] or m.startswith('RAPT-v3'):
                    pred = rapt_sys.predict(X_w)[0]
                t_p = time.perf_counter() - t0_p
                tracker.prediction_time += t_p
                
                # 2. ADAPTATION STEP
                t0_a = time.perf_counter()
                is_retrain = 0
                is_reuse = 0
                sel_pol = 'none'
                
                if m == 'Frozen':
                    pass
                    
                elif m == 'Event-Driven':
                    err = 1.0 if pred != y_w else 0.0
                    t0_ed_fit = time.perf_counter()
                    triggered, _, _ = ed_sys.update_and_adapt(X_w, [y_w], err)
                    t_ed_fit = time.perf_counter() - t0_ed_fit
                    if triggered:
                        retrain_events += 1
                        is_retrain = 1
                        tracker.model_fit_time += t_ed_fit
                        
                elif m == 'Full Retraining':
                    if reg_id != prev_regime:
                        retrain_events += 1
                        is_retrain = 1
                        prev_regime = reg_id
                        model = models_9b.create_base_ensemble(seed=seed + retrain_events * 11) if dataset_label=="9B" else models_9a.create_base_ensemble(seed=seed + retrain_events * 11)
                        t0_full_fit = time.perf_counter()
                        model.fit(np.array(hist_X[-1000:]), np.array(hist_y[-1000:]))
                        t_full_fit = time.perf_counter() - t0_full_fit
                        tracker.model_fit_time += t_full_fit
                        
                elif m == 'Original RAPT':
                    if reg_id != prev_regime:
                        prev_regime = reg_id
                        t0_rapt_trans = time.perf_counter()
                        reused_flag, _, _ = rapt_sys.handle_regime_transition(reg_id, w_id, hist_X[-500:], hist_y[-500:])
                        t_rapt_trans = time.perf_counter() - t0_rapt_trans
                        if reused_flag:
                            reused_checkpoints += 1
                            is_reuse = 1
                            tracker.checkpoint_lookup_time += t_rapt_trans
                        else:
                            retrain_events += 1
                            is_retrain = 1
                            tracker.model_fit_time += t_rapt_trans
                            
                elif m == 'RAPT-E':
                    if reg_id != prev_regime:
                        prev_regime = reg_id
                        t0_rapt_trans = time.perf_counter()
                        reused_flag, _, _ = rapt_sys.handle_regime_transition(reg_id, w_id, hist_X[-500:], hist_y[-500:])
                        t_rapt_trans = time.perf_counter() - t0_rapt_trans
                        if reused_flag:
                            reused_checkpoints += 1
                            is_reuse = 1
                            tracker.checkpoint_lookup_time += t_rapt_trans
                        else:
                            retrain_events += 1
                            is_retrain = 1
                            tracker.model_fit_time += t_rapt_trans
                            
                    rec_err = 1.0 if pred != y_w else 0.0
                    t0_esc = time.perf_counter()
                    escalated, _, _ = rapt_sys.check_and_escalate_fallback(hist_X[-500:], hist_y[-500:], rec_err)
                    t_esc_wall = time.perf_counter() - t0_esc
                    if escalated:
                        tracker.fallback_fit_time += t_esc_wall
                        
                elif m.startswith('RAPT-v3'):
                    if reg_id != prev_regime:
                        prev_regime = reg_id
                        t0_rapt_trans = time.perf_counter()
                        reused_flag, _, sel_pol = rapt_sys.handle_regime_transition(reg_id, w_id, hist_X[-500:], hist_y[-500:], dataset_label=dataset_label, seed=seed)
                        t_rapt_trans = time.perf_counter() - t0_rapt_trans
                        if reused_flag:
                            reused_checkpoints += 1
                            is_reuse = 1
                            if sel_pol == 'champion':
                                champion_selections += 1
                            elif sel_pol == 'challenger':
                                challenger_selections += 1
                        else:
                            retrain_events += 1
                            is_retrain = 1
                            tracker.model_fit_time += t_rapt_trans
                            
                t_a = time.perf_counter() - t0_a
                tracker.adaptation_time += t_a
                
                hist_X.append(X_w[0])
                hist_y.append(y_w)
                y_true_all.append(y_w)
                y_pred_all.append(pred)
                
                per_window_results.append({
                    'seed': seed,
                    'dataset': dataset_label,
                    'method': m,
                    'window_id': w_id,
                    'regime_id': reg_id,
                    'y_true': int(y_w),
                    'y_pred': int(pred),
                    'is_correct': int(pred == y_w),
                    'prediction_time': t_p,
                    'adaptation_time': t_a,
                    'step_total_time': t_p + t_a,
                    'is_retrain': is_retrain,
                    'is_reuse': is_reuse,
                    'selected_policy': sel_pol
                })
                
            tracker.end_stream()
            
            if m.startswith('RAPT-v3'):
                all_trace_records.extend(rapt_sys.trace_records)
                
            m_res = calculate_window_metrics(y_true_all, y_pred_all)
            mem_mb = float(psutil.Process().memory_info().rss / (1024 * 1024))
            
            per_seed_results.append({
                'seed': seed,
                'dataset': dataset_label,
                'method': m,
                'macro_f1': m_res['macro_f1'],
                'accuracy': m_res['accuracy'],
                'balanced_accuracy': m_res['balanced_accuracy'],
                'precision': m_res['precision'],
                'recall': m_res['recall'],
                'total_runtime': tracker.get_total_runtime(),
                'prediction_cpu': tracker.prediction_time,
                'adaptation_cpu': tracker.adaptation_time,
                'fit_cpu': tracker.model_fit_time + tracker.fallback_fit_time,
                'checkpoint_lookup_cpu': tracker.checkpoint_lookup_time + (rapt_sys.checkpoint_lookup_cpu if m.startswith('RAPT-v3') else 0),
                'weight_update_cpu': tracker.weight_update_time + (rapt_sys.weight_calc_cpu if m.startswith('RAPT-v3') else 0),
                'buffer_eval_cpu': (rapt_sys.buffer_eval_cpu if m.startswith('RAPT-v3') else 0),
                'challenger_selection_cpu': (rapt_sys.challenger_selection_cpu if m.startswith('RAPT-v3') else 0),
                'retrain_events': retrain_events,
                'reused_checkpoints': reused_checkpoints,
                'champion_selections': champion_selections,
                'challenger_selections': challenger_selections,
                'memory_mb': mem_mb
            })
            
    df_per_seed = pd.DataFrame(per_seed_results)
    df_per_window = pd.DataFrame(per_window_results)
    df_trace = pd.DataFrame(all_trace_records)
    
    summary_cols = [
        'macro_f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall',
        'total_runtime', 'prediction_cpu', 'adaptation_cpu', 'fit_cpu',
        'checkpoint_lookup_cpu', 'weight_update_cpu', 'buffer_eval_cpu', 'challenger_selection_cpu',
        'retrain_events', 'reused_checkpoints', 'champion_selections', 'challenger_selections', 'memory_mb'
    ]
    
    summary_recs = []
    for m in methods:
        df_m = df_per_seed[df_per_seed['method'] == m]
        rec = {'method': m, 'dataset': dataset_label}
        for col in summary_cols:
            rec[f"{col}_mean"] = float(df_m[col].mean())
            rec[f"{col}_std"] = float(df_m[col].std())
        summary_recs.append(rec)
        
    df_summary = pd.DataFrame(summary_recs)
    return df_summary, df_per_seed, df_per_window, df_trace

def main():
    print("=" * 80, flush=True)
    print("STARTING RAPT-v3 EXPERIMENTAL BENCHMARK SUITE", flush=True)
    print("=" * 80, flush=True)
    
    stream_9a_path = os.path.join(ROOT_DIR, "experiments", "exp9", "data", "processed_exp9_stream.csv")
    stream_9b_path = os.path.join(ROOT_DIR, "experiments", "exp9b", "data", "processed_exp9b_stream.csv")
    
    df_stream_9a = pd.read_csv(stream_9a_path)
    df_stream_9b = pd.read_csv(stream_9b_path)
    
    # 1. Run Benchmarks
    sum_9a, seed_9a, win_9a, trace_9a = run_benchmark_stream(df_stream_9a, FEATURES_9A, 'target', 'regime_label', dataset_label="9A", n_init=200)
    sum_9b, seed_9b, win_9b, trace_9b = run_benchmark_stream(df_stream_9b, FEATURES_9B, 'qos_target', 'regime_id', dataset_label="9B", n_init=99)
    
    # Save Combined Master Tables
    df_final_summary = pd.concat([sum_9a, sum_9b], ignore_index=True)
    df_final_summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    
    pd.concat([seed_9a, seed_9b], ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "per_seed_results.csv"), index=False)
    pd.concat([win_9a, win_9b], ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "per_window_results.csv"), index=False)
    
    # Save Trace CSV
    df_trace_all = pd.concat([trace_9a, trace_9b], ignore_index=True)
    df_trace_all.to_csv(os.path.join(RESULTS_DIR, "champion_challenger_trace.csv"), index=False)
    
    # 2. Wilcoxon & Non-Inferiority Tests
    comparisons = [
        ('RAPT-v3-50', 'Event-Driven'),
        ('RAPT-v3-100', 'Event-Driven'),
        ('RAPT-v3-250', 'Event-Driven'),
        ('RAPT-v3-50', 'RAPT-E'),
        ('RAPT-v3-100', 'RAPT-E'),
        ('RAPT-v3-250', 'RAPT-E'),
        ('RAPT-v3-50', 'Frozen'),
        ('RAPT-v3-100', 'Frozen'),
        ('RAPT-v3-250', 'Frozen')
    ]
    
    stat_9a = compute_wilcoxon_and_non_inferiority(win_9a, comparisons, delta=0.005)
    stat_9a['dataset'] = '9A'
    stat_9b = compute_wilcoxon_and_non_inferiority(win_9b, comparisons, delta=0.005)
    stat_9b['dataset'] = '9B'
    df_stat_all = pd.concat([stat_9a, stat_9b], ignore_index=True)
    df_stat_all.to_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"), index=False)
    
    # 3. Transition Analysis
    trans_9a = analyze_regime_transitions_v3(win_9a, df_stream_9a)
    trans_9a['dataset'] = '9A'
    trans_9b = analyze_regime_transitions_v3(win_9b, df_stream_9b)
    trans_9b['dataset'] = '9B'
    df_trans_all = pd.concat([trans_9a, trans_9b], ignore_index=True)
    df_trans_all.to_csv(os.path.join(RESULTS_DIR, "transition_analysis.csv"), index=False)
    
    # 4. Generate Cross-Dataset Comparison CSV & MD
    comp_records = []
    for ds in ['9A', '9B']:
        ds_sum = df_final_summary[df_final_summary['dataset'] == ds]
        ed_f1 = ds_sum[ds_sum['method'] == 'Event-Driven']['macro_f1_mean'].values[0]
        ed_cpu = ds_sum[ds_sum['method'] == 'Event-Driven']['adaptation_cpu_mean'].values[0]
        rapt_e_f1 = ds_sum[ds_sum['method'] == 'RAPT-E']['macro_f1_mean'].values[0]
        
        for v in ['RAPT-v3-50', 'RAPT-v3-100', 'RAPT-v3-250']:
            v_row = ds_sum[ds_sum['method'] == v]
            if len(v_row) > 0:
                v_f1 = v_row['macro_f1_mean'].values[0]
                v_cpu = v_row['adaptation_cpu_mean'].values[0]
                f1_gain_rapt_e = v_f1 - rapt_e_f1
                f1_diff_ed = v_f1 - ed_f1
                cpu_reduction = (1.0 - v_cpu / ed_cpu) * 100.0 if ed_cpu > 0 else 0.0
                
                comp_records.append({
                    'Dataset': ds,
                    'Variant': v,
                    'Event-Driven F1': float(ed_f1),
                    'RAPT-E F1': float(rapt_e_f1),
                    'RAPT-v3 F1': float(v_f1),
                    'F1 Gain vs RAPT-E': float(f1_gain_rapt_e),
                    'F1 Diff vs Event-Driven': float(f1_diff_ed),
                    'RAPT-v3 Adapt CPU (s)': float(v_cpu),
                    'Event Adapt CPU (s)': float(ed_cpu),
                    'Adaptation CPU Reduction (%)': float(cpu_reduction)
                })
                
    df_comp = pd.DataFrame(comp_records)
    df_comp.to_csv(os.path.join(COMPARISON_DIR, "cross_dataset_comparison.csv"), index=False)
    
    with open(os.path.join(COMPARISON_DIR, "cross_dataset_comparison.md"), "w") as f:
        f.write("# RAPT-v3 Cross-Dataset Comparison Table\n\n")
        f.write(df_comp.to_markdown(index=False))
        
    print("\n" + "=" * 80, flush=True)
    print("RAPT-v3 MASTER SUMMARY TABLE", flush=True)
    print("=" * 80, flush=True)
    cols_disp = ['dataset', 'method', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean', 'total_runtime_mean', 'adaptation_cpu_mean', 'fit_cpu_mean', 'retrain_events_mean', 'champion_selections_mean', 'challenger_selections_mean']
    print(df_final_summary[cols_disp].to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
