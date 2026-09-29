"""
================================================================================
COMPONENT-SELECTIVE RETRAINING HYPERPARAMETER TUNING (Part 4)
================================================================================
Sweeps the per-component degradation threshold (comp_drop_threshold) across:
  [0.05, 0.08, 0.10, 0.12, 0.15]
Strictly evaluated on the 4,000-sample validation split (X_ref_val, y_ref_val)
across all 5 seeds (zero test-stream leakage).
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

from config import SEEDS, KPI_COLS, TARGET_COL, DECISION_THRESHOLD, RESULTS_DIR
from data_generation import generate_experiment_dataset
from ensemble import ComponentSelectiveEnsemble
from drift import DualTriggerDriftDetector, tune_detector_hyperparameters
from resource_monitor import measure_execution
from metrics import compute_metrics

THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]

def evaluate_threshold_on_validation(th_val):
    results_by_seed = []
    
    for seed in SEEDS:
        rng = np.random.RandomState(seed + 100)
        df, _ = generate_experiment_dataset(seed)
        X_all = df[KPI_COLS].values
        y_all = df[TARGET_COL].values
        
        # 16,000 train, 4,000 validation split
        X_train_ref = X_all[:16000]
        y_train_ref = y_all[:16000]
        X_val = X_all[16000:20000].copy()
        y_val = y_all[16000:20000].copy()
        
        # Build 8 validation evaluation windows (500 samples each):
        # Windows 0-3: Stationary
        # Windows 4-5: Covariate shift (+1.5 speed, +30.0 distance)
        # Windows 6-7: Concept drift (flip 25% target labels)
        val_windows = []
        for w in range(4):
            val_windows.append((X_val[w*500:(w+1)*500], y_val[w*500:(w+1)*500]))
        for w in range(2):
            X_s = X_val[(4+w)*500:(5+w)*500].copy()
            X_s[:, 0] += 1.5
            X_s[:, 1] += 30.0
            val_windows.append((X_s, y_val[(4+w)*500:(5+w)*500]))
        for w in range(2):
            y_c = y_val[(6+w)*500:(7+w)*500].copy()
            flip_idx = rng.choice(500, size=125, replace=False)
            y_c[flip_idx] = 1 - y_c[flip_idx]
            val_windows.append((X_val[(6+w)*500:(7+w)*500], y_c))
            
        # Initialize ComponentSelectiveEnsemble with the candidate threshold
        cs_ens = ComponentSelectiveEnsemble(seed, threshold=DECISION_THRESHOLD, comp_drop_threshold=th_val, buffer_cap=5000)
        cs_ens.fit_initial(X_train_ref, y_train_ref)
        
        # Tune base detector on validation
        tuned_configs = tune_detector_hyperparameters(X_val, y_val, cs_ens, seed=seed)
        p = tuned_configs['Custom Dual-Trigger']
        detector = DualTriggerDriftDetector(
            wasserstein_threshold=p['wasserstein_threshold'],
            perf_drop_threshold=p['perf_drop_threshold'],
            initial_baseline_f1=0.85
        )
        
        f1_list = []
        acc_list = []
        retrain_events = 0
        total_cpu = 0.0
        
        for X_win, y_win in val_windows:
            (pred, prob, cp, cpr), inf_res = measure_execution(cs_ens.predict, X_win)
            m = compute_metrics(y_win, pred, prob)
            comp_m, _, _, _ = cs_ens.update_weights(y_win, cp, cpr)
            
            f1_list.append(m['f1'])
            acc_list.append(m['accuracy'])
            total_cpu += inf_res['total_cpu_time']
            
            dec = detector.check_drift(X_train_ref[-5000:], X_win, m['f1'])
            if dec['drift_detected']:
                _, ret_res, _, _ = cs_ens.adapt_selective(X_win, y_win, comp_m)
                r_cpu = sum(r['total_cpu_time'] for r in ret_res.values())
                total_cpu += r_cpu
                retrain_events += 1
            else:
                cs_ens.buffer_X.extend(X_win)
                cs_ens.buffer_y.extend(y_win)
                if len(cs_ens.buffer_X) > 5000:
                    cs_ens.buffer_X = cs_ens.buffer_X[-5000:]
                    cs_ens.buffer_y = cs_ens.buffer_y[-5000:]
                    
        results_by_seed.append({
            'seed': seed,
            'f1': float(np.mean(f1_list)),
            'accuracy': float(np.mean(acc_list)),
            'retrain_events': retrain_events,
            'cpu_time': total_cpu
        })
        
    df_s = pd.DataFrame(results_by_seed)
    return {
        'threshold': th_val,
        'mean_f1': float(df_s['f1'].mean()),
        'std_f1': float(df_s['f1'].std()),
        'mean_acc': float(df_s['accuracy'].mean()),
        'mean_events': float(df_s['retrain_events'].mean()),
        'mean_cpu': float(df_s['cpu_time'].mean()),
        'std_cpu': float(df_s['cpu_time'].std()),
    }

def main():
    print("=" * 80)
    print("SWEEPING COMPONENT DEGRADATION THRESHOLD ON VALIDATION SPLIT")
    print(f"Grid: {THRESHOLDS}")
    print("=" * 80)
    
    sweep_rows = []
    for th in THRESHOLDS:
        print(f"Evaluating comp_drop_threshold = {th:.2f}...")
        res = evaluate_threshold_on_validation(th)
        sweep_rows.append(res)
        print(f"   -> F1: {res['mean_f1']:.4f} ± {res['std_f1']:.4f} | Retrain Events: {res['mean_events']:.1f}/8 | CPU: {res['mean_cpu']:.2f}s ± {res['std_cpu']:.2f}s")
        
    df_sweep = pd.DataFrame(sweep_rows)
    out_path = os.path.join(RESULTS_DIR, 'component_selective_tuning_sweep.csv')
    df_sweep.to_csv(out_path, index=False)
    print("\n" + "=" * 80)
    print("COMPONENT-SELECTIVE TUNING SWEEP RESULTS")
    print("=" * 80)
    print(df_sweep[['threshold', 'mean_f1', 'mean_acc', 'mean_events', 'mean_cpu']].to_string(index=False))
    print("=" * 80)
    print(f"Saved tuning sweep to: {out_path}")

if __name__ == '__main__':
    main()
