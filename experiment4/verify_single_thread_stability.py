"""
Single-threaded stability verification for Frozen Model across 5 passes.
Confirms whether fixing n_jobs=1 and thread environment variables reduces CV below 10%.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE, SEEDS,
    KPI_COLS, TARGET_COL, BASE_MODEL_PARAMS
)
from data_generation import generate_experiment_dataset
from models import create_frozen_rf
from resource_monitor import measure_execution

print(f"RandomForest n_jobs configured as: {BASE_MODEL_PARAMS['RandomForest'].get('n_jobs')}")
assert BASE_MODEL_PARAMS['RandomForest'].get('n_jobs') == 1, "n_jobs must be 1"

pass_results = []

# Pre-generate datasets for all seeds so data generation time is strictly excluded
print("Pre-generating synthetic datasets for all 5 seeds...")
datasets = {}
for seed in SEEDS:
    df, _ = generate_experiment_dataset(seed)
    datasets[seed] = (df[KPI_COLS].values, df[TARGET_COL].values)

print("\nBeginning 5-pass Frozen Model Stability Audit (single-threaded)...")
for pass_idx in range(1, 6):
    pass_cpu = 0.0
    seed_cpus = []
    
    for seed in SEEDS:
        X_all, y_all = datasets[seed]
        X_init, y_init = X_all[:N_INITIAL_TRAINING], y_all[:N_INITIAL_TRAINING]
        
        frozen_rf = create_frozen_rf(seed)
        frozen_rf.fit(X_init, y_init)
        
        seed_cpu = 0.0
        for w in range(N_WINDOWS):
            idx_s = N_INITIAL_TRAINING + w * WINDOW_SIZE
            idx_e = idx_s + WINDOW_SIZE
            X_win = X_all[idx_s:idx_e]
            
            (_, _), inf_res = measure_execution(
                lambda: (frozen_rf.predict(X_win), frozen_rf.predict_proba(X_win)[:, 1])
            )
            seed_cpu += inf_res['total_cpu_time']
            
        seed_cpus.append(seed_cpu)
        pass_cpu += seed_cpu
        
    mean_seed = float(np.mean(seed_cpus))
    per_win_ms = (pass_cpu / (len(SEEDS) * N_WINDOWS)) * 1000.0
    pass_results.append({
        'pass': pass_idx,
        'cumulative_cpu': pass_cpu,
        'mean_seed_cpu': mean_seed,
        'per_win_ms': per_win_ms
    })
    print(f"Pass {pass_idx}: Cumulative CPU = {pass_cpu:.4f}s | Mean per seed = {mean_seed:.4f}s | Per window = {per_win_ms:.2f}ms")

cum_cpus = [p['cumulative_cpu'] for p in pass_results]
mean_cpu = float(np.mean(cum_cpus))
std_cpu = float(np.std(cum_cpus, ddof=1))
cv_pct = (std_cpu / mean_cpu) * 100.0 if mean_cpu > 0 else 0.0

print("\n" + "="*80)
print(f"FROZEN MODEL SINGLE-THREADED STABILITY AUDIT RESULTS (5 PASSES)")
print("="*80)
print(f"Pass 1: {cum_cpus[0]:.4f}s")
print(f"Pass 2: {cum_cpus[1]:.4f}s")
print(f"Pass 3: {cum_cpus[2]:.4f}s")
print(f"Pass 4: {cum_cpus[3]:.4f}s")
print(f"Pass 5: {cum_cpus[4]:.4f}s")
print(f"Mean Cumulative CPU: {mean_cpu:.4f}s +- {std_cpu:.4f}s")
print(f"Coefficient of Variation (CV): {cv_pct:.2f}%")
print(f"Range: {min(cum_cpus):.4f}s - {max(cum_cpus):.4f}s (Ratio: {max(cum_cpus)/min(cum_cpus):.2f}x)")
print("="*80)

if cv_pct < 10.0:
    print(f"SUCCESS: CV ({cv_pct:.2f}%) is below target threshold of 10.0%!")
else:
    print(f"WARNING: CV ({cv_pct:.2f}%) exceeds target threshold of 10.0%.")
