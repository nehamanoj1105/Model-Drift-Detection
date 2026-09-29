"""
Final RAPT-E Validation Master Pipeline
Runs fair, unified, high-precision timing evaluation across 5 seeds (42, 43, 44, 45, 46) for:
  - Frozen
  - Event-Driven
  - Full Retraining
  - Original RAPT
  - RAPT-E (Fixed RAPT-v2)
on Dataset 9A (5G Campus QoS) and Dataset 9B (5G NR End-to-End Latency).
"""

import os
import sys
import copy
import time
import psutil
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import original and fixed modules
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

from timing_utils import TimingTracker

VALIDATION_DIR = os.path.join(ROOT_DIR, "experiments", "final_validation")
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
PLOTS_DIR = os.path.join(VALIDATION_DIR, "plots")
TIMING_DIR = os.path.join(VALIDATION_DIR, "timing")

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

def calculate_window_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'macro_f1': float(macro_f1),
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'precision': float(prec),
        'recall': float(rec)
    }

def run_final_validation_stream(df_stream, feature_cols, target_col, regime_col, dataset_label="9A", n_init=200):
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING FINAL VALIDATION BENCHMARK ON DATASET {dataset_label}", flush=True)
    print(f"=======================================================", flush=True)
    
    per_seed_results = []
    per_window_results = []
    fit_trace_all = []
    
    methods = ['Frozen', 'Event-Driven', 'Full Retraining', 'Original RAPT', 'RAPT-E']
    
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
            fallback_events = 0
            
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
                tracker.log_fit_event("BaseEnsemble", initial_regime, len(X_init), t_fit, "initial")
                
            elif m == 'Event-Driven':
                ed_sys = ed_9b.EventDrivenEnsemble(seed=seed) if dataset_label=="9B" else ed_9a.EventDrivenEnsemble(seed=seed)
                t0_fit = time.perf_counter()
                ed_sys.fit_initial(X_init, y_init)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                tracker.log_fit_event("EventDrivenEnsemble", initial_regime, len(X_init), t_fit, "initial")
                
            elif m == 'Full Retraining':
                model = models_9b.create_base_ensemble(seed=seed) if dataset_label=="9B" else models_9a.create_base_ensemble(seed=seed)
                t0_fit = time.perf_counter()
                model.fit(X_init, y_init)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                tracker.log_fit_event("BaseEnsemble", initial_regime, len(X_init), t_fit, "initial")
                
            elif m == 'Original RAPT':
                rapt_sys = orig_rapt_9b.RAPTSystem(seed=seed, mode='full') if dataset_label=="9B" else orig_rapt_9a.RAPTSystem(seed=seed, mode='full')
                t0_fit = time.perf_counter()
                rapt_sys.fit_initial(initial_regime, X_init, y_init, window_id=0)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                tracker.log_fit_event("RAPTSystem", initial_regime, len(X_init), t_fit, "initial")
                
            elif m == 'RAPT-E':
                rapt_sys = rapt_v2_fixed.RAPTv2SystemFixed(seed=seed, variant='RAPT-E', buffer_capacity=50)
                t0_fit = time.perf_counter()
                rapt_sys.fit_initial(initial_regime, X_init, y_init, window_id=0)
                t_fit = time.perf_counter() - t0_fit
                tracker.model_fit_time += t_fit
                tracker.log_fit_event("RAPTv2Fixed", initial_regime, len(X_init), t_fit, "initial")
                
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
                elif m == 'Original RAPT':
                    pred = rapt_sys.predict(X_w)[0]
                elif m == 'RAPT-E':
                    pred = rapt_sys.predict(X_w)[0]
                t_p = time.perf_counter() - t0_p
                tracker.prediction_time += t_p
                
                # 2. ADAPTATION STEP
                t0_a = time.perf_counter()
                t_fit_step = 0.0
                is_retrain = 0
                is_reuse = 0
                
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
                        tracker.log_fit_event("EventDrivenEnsemble", reg_id, len(ed_sys.buffer_X), t_ed_fit, "event_retrain")
                        
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
                        tracker.log_fit_event("BaseEnsemble", reg_id, min(1000, len(hist_X)), t_full_fit, "transition_retrain")
                        
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
                            tracker.log_fit_event("RAPTSystem", reg_id, min(500, len(hist_X)), t_rapt_trans, "new_regime_checkpoint")
                            
                elif m == 'RAPT-E':
                    if reg_id != prev_regime:
                        prev_regime = reg_id
                        t0_rapt_trans = time.perf_counter()
                        reused_flag, _, level_used = rapt_sys.handle_regime_transition(reg_id, w_id, hist_X[-500:], hist_y[-500:])
                        t_rapt_trans = time.perf_counter() - t0_rapt_trans
                        if reused_flag:
                            reused_checkpoints += 1
                            is_reuse = 1
                            tracker.checkpoint_lookup_time += t_rapt_trans
                            tracker.weight_update_time += t_rapt_trans
                        else:
                            retrain_events += 1
                            is_retrain = 1
                            tracker.model_fit_time += t_rapt_trans
                            tracker.log_fit_event("RAPTv2Fixed", reg_id, min(500, len(hist_X)), t_rapt_trans, "new_regime_checkpoint")
                            
                    rec_err = 1.0 if pred != y_w else 0.0
                    t0_esc = time.perf_counter()
                    escalated, esc_level, t_esc_cpu = rapt_sys.check_and_escalate_fallback(hist_X[-500:], hist_y[-500:], rec_err)
                    t_esc_wall = time.perf_counter() - t0_esc
                    if escalated:
                        fallback_events += 1
                        tracker.fallback_fit_time += t_esc_wall
                        tracker.log_fit_event("RAPTv2Fixed", reg_id, min(500, len(hist_X)), t_esc_wall, f"fallback_{esc_level}")
                        
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
                    'is_reuse': is_reuse
                })
                
            tracker.end_stream()
            
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
                'checkpoint_lookup_cpu': tracker.checkpoint_lookup_time,
                'weight_update_cpu': tracker.weight_update_time,
                'retrain_events': retrain_events,
                'reused_checkpoints': reused_checkpoints,
                'fallback_events': fallback_events,
                'memory_mb': mem_mb
            })
            
            fit_trace_all.extend(tracker.fit_trace_records)
            
    df_per_seed = pd.DataFrame(per_seed_results)
    df_per_window = pd.DataFrame(per_window_results)
    df_fit_trace = pd.DataFrame(fit_trace_all)
    
    summary_cols = [
        'macro_f1', 'accuracy', 'balanced_accuracy', 'precision', 'recall',
        'total_runtime', 'prediction_cpu', 'adaptation_cpu', 'fit_cpu',
        'checkpoint_lookup_cpu', 'weight_update_cpu',
        'retrain_events', 'reused_checkpoints', 'fallback_events', 'memory_mb'
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
    return df_summary, df_per_seed, df_per_window, df_fit_trace

def compute_wilcoxon_and_non_inferiority(df_per_window, delta=0.005):
    """
    Computes Wilcoxon signed-rank test and Non-Inferiority analysis (margin delta=0.005).
    """
    seeds = df_per_window['seed'].unique()
    test_records = []
    
    # Primary: RAPT-E vs Event-Driven
    m1, m2 = 'RAPT-E', 'Event-Driven'
    f1_diffs = []
    for seed in seeds:
        df_s = df_per_window[df_per_window['seed'] == seed]
        df_m1 = df_s[df_s['method'] == m1].sort_values('window_id')
        df_m2 = df_s[df_s['method'] == m2].sort_values('window_id')
        
        y1 = df_m1['is_correct'].values.astype(float)
        y2 = df_m2['is_correct'].values.astype(float)
        f1_diffs.extend(y1 - y2)
        
    f1_diffs = np.array(f1_diffs)
    mean_diff = np.mean(f1_diffs)
    std_diff = np.std(f1_diffs) + 1e-9
    
    if np.all(f1_diffs == 0):
        stat, p_val = 0.0, 1.0
    else:
        try:
            stat, p_val = stats.wilcoxon(f1_diffs, zero_method='pratt')
        except Exception:
            stat, p_val = 0.0, 1.0
            
    cohen_d = mean_diff / std_diff
    is_non_inferior = bool(mean_diff >= -delta)
    
    test_records.append({
        'comparison': f"{m1} vs {m2}",
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'wilcoxon_stat': float(stat),
        'p_value': float(p_val),
        'cohen_d': float(cohen_d),
        'significant_p05': bool(p_val < 0.05),
        'margin_delta': delta,
        'non_inferior': is_non_inferior
    })
    
    return pd.DataFrame(test_records)

def analyze_regime_transitions_final(df_per_window, stream_df):
    """
    Measures transition recovery metrics for every detected regime transition.
    """
    records = []
    seeds = df_per_window['seed'].unique()
    methods = ['RAPT-E', 'Event-Driven']
    
    # Identify transition windows
    regimes = stream_df['regime_id' if 'regime_id' in stream_df.columns else 'regime_label'].values
    trans_wins = []
    for w in range(1, len(regimes)):
        if regimes[w] != regimes[w-1]:
            trans_wins.append(w)
            
    for seed in seeds:
        for m in methods:
            df_sm = df_per_window[(df_per_window['seed'] == seed) & (df_per_window['method'] == m)].sort_values('window_id')
            
            for tw in trans_wins:
                df_slice = df_sm[(df_sm['window_id'] >= tw - 5) & (df_sm['window_id'] <= tw + 5)]
                if len(df_slice) < 5:
                    continue
                    
                pre_f1 = df_slice[df_slice['window_id'] < tw]['is_correct'].mean()
                post_f1 = df_slice[df_slice['window_id'] == tw]['is_correct'].mean()
                min_post_f1 = df_slice[df_slice['window_id'] >= tw]['is_correct'].min()
                
                # Windows to recover
                target_acc = 0.95 * pre_f1
                rec_wins = 5
                post_arr = df_slice[df_slice['window_id'] >= tw]['is_correct'].values
                for idx_off, acc_val in enumerate(post_arr):
                    if acc_val >= target_acc:
                        rec_wins = idx_off + 1
                        break
                        
                adapt_cpu = df_slice[df_slice['window_id'] >= tw]['adaptation_time'].sum()
                
                records.append({
                    'seed': seed,
                    'method': m,
                    'transition_window': tw,
                    'pre_event_f1': float(pre_f1),
                    'post_event_f1': float(post_f1),
                    'min_post_f1': float(min_post_f1),
                    'windows_to_recover': rec_wins,
                    'adaptation_cpu': float(adapt_cpu)
                })
                
    return pd.DataFrame(records)

def main():
    print("=" * 80, flush=True)
    print("STARTING FINAL RAPT-E VALIDATION SUITE", flush=True)
    print("=" * 80, flush=True)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TIMING_DIR, exist_ok=True)
    
    stream_9a_path = os.path.join(ROOT_DIR, "experiments", "exp9", "data", "processed_exp9_stream.csv")
    stream_9b_path = os.path.join(ROOT_DIR, "experiments", "exp9b", "data", "processed_exp9b_stream.csv")
    
    df_stream_9a = pd.read_csv(stream_9a_path)
    df_stream_9b = pd.read_csv(stream_9b_path)
    
    # 1. Run Final Benchmarks
    sum_9a, seed_9a, win_9a, fit_9a = run_final_validation_stream(df_stream_9a, FEATURES_9A, 'target', 'regime_label', dataset_label="9A", n_init=200)
    sum_9b, seed_9b, win_9b, fit_9b = run_final_validation_stream(df_stream_9b, FEATURES_9B, 'qos_target', 'regime_id', dataset_label="9B", n_init=99)
    
    # Combine summaries
    df_final_summary = pd.concat([sum_9a, sum_9b], ignore_index=True)
    df_final_summary.to_csv(os.path.join(RESULTS_DIR, "final_summary.csv"), index=False)
    
    # Save per-seed and per-window results
    pd.concat([seed_9a, seed_9b], ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "final_per_seed.csv"), index=False)
    pd.concat([win_9a, win_9b], ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "final_per_window.csv"), index=False)
    
    # Save Fit Trace
    df_fit_all = pd.concat([fit_9a, fit_9b], ignore_index=True)
    df_fit_all.to_csv(os.path.join(RESULTS_DIR, "fit_trace.csv"), index=False)
    
    # 2. Wilcoxon & Non-Inferiority Tests
    stat_9a = compute_wilcoxon_and_non_inferiority(win_9a, delta=0.005)
    stat_9a['dataset'] = '9A'
    stat_9b = compute_wilcoxon_and_non_inferiority(win_9b, delta=0.005)
    stat_9b['dataset'] = '9B'
    df_stat_all = pd.concat([stat_9a, stat_9b], ignore_index=True)
    df_stat_all.to_csv(os.path.join(RESULTS_DIR, "final_statistical_tests.csv"), index=False)
    
    # 3. Transition Analysis
    trans_9a = analyze_regime_transitions_final(win_9a, df_stream_9a)
    trans_9a['dataset'] = '9A'
    trans_9b = analyze_regime_transitions_final(win_9b, df_stream_9b)
    trans_9b['dataset'] = '9B'
    df_trans_all = pd.concat([trans_9a, trans_9b], ignore_index=True)
    df_trans_all.to_csv(os.path.join(RESULTS_DIR, "transition_analysis.csv"), index=False)
    
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY RESULTS TABLE", flush=True)
    print("=" * 80, flush=True)
    cols_disp = ['dataset', 'method', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean', 'total_runtime_mean', 'adaptation_cpu_mean', 'fit_cpu_mean', 'retrain_events_mean']
    print(df_final_summary[cols_disp].to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
