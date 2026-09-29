"""
================================================================================
RERUN COMPONENT-SELECTIVE ENSEMBLE WITH TUNED THRESHOLD (0.12)
================================================================================
Evaluates ComponentSelectiveEnsemble with comp_drop_threshold = 0.12 across
all 5 seeds and 60 windows in the exact single-threaded benchmark harness.
Updates window_results.csv, retraining_results.csv, summary_results.csv,
computational_savings.csv, and master_benchmark_v5.csv.
================================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE, SEEDS,
    KPI_COLS, TARGET_COL, DECISION_THRESHOLD, BUFFER_CAP,
    COMPONENT_DROP_THRESHOLD, RESULTS_DIR, APPROACH_NAMES
)
from data_generation import generate_experiment_dataset
from ensemble import ComponentSelectiveEnsemble
from drift import DualTriggerDriftDetector, tune_detector_hyperparameters
from resource_monitor import measure_execution, measure_model_footprint
from metrics import compute_metrics
from statistics import compute_summary_stats

def main():
    print("=" * 80)
    print(f"EVALUATING TUNED COMPONENT-SELECTIVE ENSEMBLE (THRESHOLD = {COMPONENT_DROP_THRESHOLD})")
    print("=" * 80)
    
    new_window_rows = []
    new_retrain_rows = []
    
    for seed in SEEDS:
        print(f"Running Seed {seed}...")
        df, schedule = generate_experiment_dataset(seed)
        X_all = df[KPI_COLS].values
        y_all = df[TARGET_COL].values
        
        X_init = X_all[:N_INITIAL_TRAINING]
        y_init = y_all[:N_INITIAL_TRAINING]
        X_ref_val = X_init[16000:]
        y_ref_val = y_init[16000:]
        
        cs_ens = ComponentSelectiveEnsemble(
            seed,
            threshold=DECISION_THRESHOLD,
            buffer_cap=BUFFER_CAP,
            comp_drop_threshold=COMPONENT_DROP_THRESHOLD
        )
        cs_ens.fit_initial(X_init, y_init)
        
        # Detector tuning on validation
        tuned_configs = tune_detector_hyperparameters(X_ref_val, y_ref_val, cs_ens, seed=seed)
        custom_params = tuned_configs['Custom Dual-Trigger']
        cs_detector = DualTriggerDriftDetector(
            wasserstein_threshold=custom_params['wasserstein_threshold'],
            perf_drop_threshold=custom_params['perf_drop_threshold'],
            initial_baseline_f1=0.85
        )
        
        for w in range(N_WINDOWS):
            idx_s = N_INITIAL_TRAINING + w * WINDOW_SIZE
            idx_e = idx_s + WINDOW_SIZE
            X_win = X_all[idx_s:idx_e]
            y_win = y_all[idx_s:idx_e]
            
            win_cfg = schedule[w]
            dtype = win_cfg['drift_type']
            sev = win_cfg['severity']
            trans = win_cfg['transition']
            
            (pred, prob, cp, cpr), inf_res = measure_execution(lambda: cs_ens.predict(X_win))
            m = compute_metrics(y_win, pred, prob)
            comp_m, _, _, _ = cs_ens.update_weights(y_win, cp, cpr)
            
            dec = cs_detector.check_drift(X_init[:BUFFER_CAP], X_win, m['f1'])
            cs_fired = dec['drift_detected']
            w_score = dec['wasserstein_score']
            
            if cs_fired:
                _, ret_res, samples, ret_comps = cs_ens.adapt_selective(X_win, y_win, comp_m)
                ret_cpu = sum(r['total_cpu_time'] for r in ret_res.values())
                ret_wall = sum(r['wall_clock_time'] for r in ret_res.values())
                ret_inc_mem = max([r.get('incremental_rss_delta_mb', 0.0) for r in ret_res.values()] or [0.0])
                did_ret = True
            else:
                cs_ens.buffer_X.extend(X_win)
                cs_ens.buffer_y.extend(y_win)
                if len(cs_ens.buffer_X) > BUFFER_CAP:
                    cs_ens.buffer_X = cs_ens.buffer_X[-BUFFER_CAP:]
                    cs_ens.buffer_y = cs_ens.buffer_y[-BUFFER_CAP:]
                ret_cpu, ret_wall, ret_inc_mem = 0.0, 0.0, 0.0
                samples = 0
                ret_comps = []
                did_ret = False
                
            fp = measure_model_footprint(cs_ens)
            tot_cpu = inf_res['total_cpu_time'] + ret_cpu
            tot_wall = inf_res['wall_clock_time'] + ret_wall
            tot_inc_mem = inf_res.get('incremental_rss_delta_mb', 0.0) + ret_inc_mem
            rss = inf_res.get('process_rss_mb', 0.0)
            
            new_window_rows.append({
                'seed': seed,
                'window_id': w,
                'approach': 'Component-Selective Ensemble',
                'f1': round(m['f1'], 4),
                'accuracy': round(m['accuracy'], 4),
                'precision': round(m['precision'], 4),
                'recall': round(m['recall'], 4),
                'auc': round(m['auc'], 4),
                'pr_auc': round(m['pr_auc'], 4),
                'balanced_accuracy': round(m['balanced_accuracy'], 4),
                'cpu_time': round(tot_cpu, 4),
                'wall_time': round(tot_wall, 4),
                'model_footprint_bytes': fp,
                'incremental_rss_delta_mb': round(tot_inc_mem, 4),
                'process_rss_mb': round(rss, 2),
                'drift_type': dtype,
                'severity': sev,
                'transition': trans,
                'drift_detected': bool(cs_fired),
                'wasserstein_mean': round(w_score, 4),
            })
            
            new_retrain_rows.append({
                'seed': seed,
                'window_id': w,
                'approach': 'Component-Selective Ensemble',
                'retrained': bool(did_ret),
                'retraining_cpu_time': round(ret_cpu, 4),
                'retraining_wall_time': round(ret_wall, 4),
                'samples_adapted': samples,
                'model_footprint_bytes': fp,
                'incremental_rss_delta_mb': round(ret_inc_mem, 4),
                'retrained_components': ','.join(ret_comps) if ret_comps else '',
            })
            
    # Merge back into window_results.csv and retraining_results.csv
    df_win_old = pd.read_csv(os.path.join(RESULTS_DIR, 'window_results.csv'))
    df_win_other = df_win_old[df_win_old['approach'] != 'Component-Selective Ensemble']
    df_win_new = pd.concat([df_win_other, pd.DataFrame(new_window_rows)], ignore_index=True)
    df_win_new.to_csv(os.path.join(RESULTS_DIR, 'window_results.csv'), index=False)
    
    df_ret_old = pd.read_csv(os.path.join(RESULTS_DIR, 'retraining_results.csv'))
    df_ret_other = df_ret_old[df_ret_old['approach'] != 'Component-Selective Ensemble']
    df_ret_new = pd.concat([df_ret_other, pd.DataFrame(new_retrain_rows)], ignore_index=True)
    df_ret_new.to_csv(os.path.join(RESULTS_DIR, 'retraining_results.csv'), index=False)
    
    # Recompute summary table
    print("\nRecomputing summary results...")
    summary_list = []
    for app in APPROACH_NAMES:
        sub_w = df_win_new[df_win_new['approach'] == app]
        sub_r = df_ret_new[df_ret_new['approach'] == app]
        
        seed_f1 = sub_w.groupby('seed')['f1'].mean()
        seed_acc = sub_w.groupby('seed')['accuracy'].mean()
        seed_prec = sub_w.groupby('seed')['precision'].mean()
        seed_rec = sub_w.groupby('seed')['recall'].mean()
        seed_cpu = sub_w.groupby('seed')['cpu_time'].sum()
        seed_footprint_kb = sub_w.groupby('seed')['model_footprint_bytes'].mean() / 1024.0
        seed_inc_ram = sub_w.groupby('seed')['incremental_rss_delta_mb'].mean()
        seed_rss = sub_w.groupby('seed')['process_rss_mb'].mean()
        seed_events = sub_r.groupby('seed')['retrained'].sum()
        seed_samples = sub_r.groupby('seed')['samples_adapted'].sum()
        
        f1_st = compute_summary_stats(seed_f1)
        acc_st = compute_summary_stats(seed_acc)
        prec_st = compute_summary_stats(seed_prec)
        rec_st = compute_summary_stats(seed_rec)
        cpu_st = compute_summary_stats(seed_cpu)
        fp_st = compute_summary_stats(seed_footprint_kb)
        inc_st = compute_summary_stats(seed_inc_ram)
        rss_st = compute_summary_stats(seed_rss)
        ev_st = compute_summary_stats(seed_events)
        sm_st = compute_summary_stats(seed_samples)
        
        retrain_freq_pct = (ev_st['mean'] / N_WINDOWS) * 100.0
        
        summary_list.append({
            'Approach': app,
            'Mean F1': f"{f1_st['mean']:.4f} ± {f1_st['std']:.4f}",
            'F1 95% CI': f"[{f1_st['ci_lower']:.4f}, {f1_st['ci_upper']:.4f}]",
            'Mean Accuracy': f"{acc_st['mean']:.4f} ± {acc_st['std']:.4f}",
            'Acc 95% CI': f"[{acc_st['ci_lower']:.4f}, {acc_st['ci_upper']:.4f}]",
            'Mean Precision': f"{prec_st['mean']:.4f} ± {prec_st['std']:.4f}",
            'Mean Recall': f"{rec_st['mean']:.4f} ± {rec_st['std']:.4f}",
            'Cumulative CPU Time (s)': f"{cpu_st['mean']:.2f} ± {cpu_st['std']:.2f}",
            'Model Footprint (KB)': f"{fp_st['mean']:.1f} ± {fp_st['std']:.1f}",
            'Incremental RAM Delta (MB)': f"{inc_st['mean']:.3f} ± {inc_st['std']:.3f}",
            'Process RSS (MB)': f"{rss_st['mean']:.1f} ± {rss_st['std']:.1f}",
            'Retraining Events': f"{ev_st['mean']:.1f} ± {ev_st['std']:.1f}",
            'Retraining Freq (%)': f"{retrain_freq_pct:.1f}%",
            'Total Adapt Samples': f"{int(sm_st['mean']):,}",
            'raw_f1': f1_st['mean'],
            'raw_acc': acc_st['mean'],
            'raw_cpu': cpu_st['mean'],
            'raw_events': ev_st['mean'],
            'raw_footprint_kb': fp_st['mean'],
            'raw_incremental_ram': inc_st['mean'],
            'raw_rss': rss_st['mean'],
        })
        
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'summary_results.csv'), index=False)
    
    # Recompute master benchmark table with exact single "Nx Frozen" ratios
    frozen_row = df_summary[df_summary['Approach'] == 'Frozen Model']
    frozen_mean_cpu = frozen_row['raw_cpu'].values[0]
    frozen_sub = df_win_new[df_win_new['approach'] == 'Frozen Model']
    frozen_seed_cpus = frozen_sub.groupby('seed')['cpu_time'].sum()
    
    c_cpu = df_summary[df_summary['Approach'] == 'Continuously Retrained Ensemble']['raw_cpu'].values[0]
    c_ev = df_summary[df_summary['Approach'] == 'Continuously Retrained Ensemble']['raw_events'].values[0]
    
    master_rows = []
    savings_rows = []
    for _, row in df_summary.iterrows():
        app = row['Approach']
        app_sub = df_win_new[df_win_new['approach'] == app]
        seed_cpus = app_sub.groupby('seed')['cpu_time'].sum()
        
        mean_c = row['raw_cpu']
        cost_ratio = mean_c / frozen_mean_cpu if frozen_mean_cpu > 0 else 1.0
        ratios_per_seed = seed_cpus / frozen_seed_cpus
        ratio_std = float(ratios_per_seed.std())
        
        master_rows.append({
            'Approach': app,
            'Mean F1': row['Mean F1'],
            'Mean Accuracy': row['Mean Accuracy'],
            'Cumulative CPU Time (s)': row['Cumulative CPU Time (s)'],
            'Cost vs. Frozen': f"{cost_ratio:.2f}x ± {ratio_std:.2f}x" if app != 'Frozen Model' else "1.00x",
            'Model Footprint (KB)': row['Model Footprint (KB)'],
            'Retraining Events': row['Retraining Events'],
            'Retraining Freq (%)': row['Retraining Freq (%)'],
            'raw_cost_ratio': cost_ratio,
        })
        
        app_ev = row['raw_events']
        app_f1 = row['raw_f1']
        cpu_sav_pct = 100.0 * (c_cpu - mean_c) / c_cpu if c_cpu > 0 else 0.0
        ev_red_pct = 100.0 * (c_ev - app_ev) / c_ev if c_ev > 0 else 0.0
        eff = app_f1 / mean_c if mean_c > 0 else 0.0
        
        savings_rows.append({
            'Approach': app,
            'Cumulative CPU Time (s)': f"{mean_c:.2f}s",
            'CPU Savings (%) vs. Continuous': f"{cpu_sav_pct:.2f}%",
            'Retraining Events': f"{app_ev:.1f}",
            'Retraining Event Reduction (%)': f"{ev_red_pct:.2f}%",
            'Performance-Cost Efficiency (F1 / CPU s)': f"{eff:.4f}",
        })
        
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(os.path.join(RESULTS_DIR, 'master_benchmark_v5.csv'), index=False)
    
    df_savings = pd.DataFrame(savings_rows)
    df_savings.to_csv(os.path.join(RESULTS_DIR, 'computational_savings.csv'), index=False)
    
    # Recompute drift type breakdown
    type_display_map = {'none': 'Stationary (None)', 'covariate': 'Covariate', 'concept': 'Concept', 'mixed': 'Mixed'}
    drift_type_rows = []
    for dtype_raw, dtype_disp in type_display_map.items():
        row = {'Drift Type': dtype_disp}
        for app in APPROACH_NAMES:
            vals = df_win_new[(df_win_new['drift_type'] == dtype_raw) & (df_win_new['approach'] == app)].groupby('seed')['f1'].mean()
            st = compute_summary_stats(vals)
            row[f'{app} F1'] = f"{st['mean']:.4f} ± {st['std']:.4f}"
        drift_type_rows.append(row)
    df_drift_type = pd.DataFrame(drift_type_rows)
    df_drift_type.to_csv(os.path.join(RESULTS_DIR, 'drift_type_summary.csv'), index=False)
    
    print("\n" + "=" * 95)
    print(f"FINAL TUNED BENCHMARK MASTER TABLE (THRESHOLD = {COMPONENT_DROP_THRESHOLD})")
    print("=" * 95)
    print(df_master[['Approach', 'Mean F1', 'Cumulative CPU Time (s)', 'Cost vs. Frozen', 'Model Footprint (KB)', 'Retraining Events']].to_string(index=False))
    print("=" * 95)

if __name__ == '__main__':
    main()
