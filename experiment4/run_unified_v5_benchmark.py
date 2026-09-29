"""
================================================================================
EXPERIMENT 4 v5 — UNIFIED SINGLE-THREADED BENCHMARK RUNNER
================================================================================
Executes:
  1. Main 12-approach benchmark across 5 seeds & 60 windows (BUFFER_CAP = 5000)
  2. Controlled buffer cap ablation (Bounded 5000 vs. Unbounded)
  3. Generation of the consolidated 7 paper figures
  4. Synchronization of artifacts to brain directory
  5. Computation of unified master summary with single 'Nx Frozen' cost ratios
All executed in a single uninterrupted single-threaded Python runtime.
================================================================================
"""

import os
# Pin all thread pools
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import shutil
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, BASE_MODEL_PARAMS
from experiment4 import run_experiment4
from run_buffer_ablation import run_ablation
from plots_paper import generate_paper_figures, PAPER_PLOTS_DIR

ARTIFACTS_DIR = r"C:\Users\emhaenn\.gemini\antigravity\brain\e7b78523-74fa-4f1f-9dea-d867ccc55e5b\figures_paper"

def main():
    print("=" * 90)
    print("STARTING EXPERIMENT 4 v5: UNIFIED SINGLE-THREADED BENCHMARK")
    print("=" * 90)
    print(f"RandomForest n_jobs: {BASE_MODEL_PARAMS['RandomForest'].get('n_jobs')}")
    print(f"ExtraTrees n_jobs:   {BASE_MODEL_PARAMS['ExtraTrees'].get('n_jobs')}")
    assert BASE_MODEL_PARAMS['RandomForest'].get('n_jobs') == 1
    assert BASE_MODEL_PARAMS['ExtraTrees'].get('n_jobs') == 1
    
    start_time = time.time()
    
    # 1. Main benchmark run (12 approaches, 5 seeds, 60 windows)
    print("\n>>> PART 2A: RUNNING 12-APPROACH BENCHMARK (BUFFER_CAP = 5000)...")
    run_experiment4()
    
    # 2. Controlled buffer ablation run (Bounded vs Unbounded)
    print("\n>>> PART 2B: RUNNING CONTROLLED BUFFER ABLATION (Bounded vs Unbounded)...")
    run_ablation()
    
    # 3. Generate 7 consolidated paper figures
    print("\n>>> GENERATING 7 CONSOLIDATED MANUSCRIPT FIGURES...")
    df_win = pd.read_csv(os.path.join(RESULTS_DIR, 'window_results.csv'))
    df_ret = pd.read_csv(os.path.join(RESULTS_DIR, 'retraining_results.csv'))
    df_det = pd.read_csv(os.path.join(RESULTS_DIR, 'detector_quality.csv'))
    abl_path = os.path.join(RESULTS_DIR, 'buffer_cap_ablation.csv')
    df_abl = pd.read_csv(abl_path) if os.path.exists(abl_path) else None
    
    generate_paper_figures(df_win, df_ret, df_det, df_abl, output_dir=PAPER_PLOTS_DIR)
    
    # 4. Copy figures to brain artifacts directory
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    for fig_name in os.listdir(PAPER_PLOTS_DIR):
        if fig_name.endswith('.png'):
            src = os.path.join(PAPER_PLOTS_DIR, fig_name)
            dst = os.path.join(ARTIFACTS_DIR, fig_name)
            shutil.copy2(src, dst)
            print(f"   Synced artifact: {fig_name}")
            
    # 5. Build Unified Master Table with exact single "Nx Frozen" cost ratios
    print("\n" + "=" * 95)
    print("UNIFIED MASTER BENCHMARK TABLE (SINGLE-THREADED, ONE SESSION)")
    print("=" * 95)
    
    df_sum = pd.read_csv(os.path.join(RESULTS_DIR, 'summary_results.csv'))
    frozen_row = df_sum[df_sum['Approach'] == 'Frozen Model']
    frozen_mean_cpu = frozen_row['raw_cpu'].values[0]
    
    # Compute per-seed Frozen CPU times to obtain standard deviation of the ratio if needed
    frozen_sub = df_win[df_win['approach'] == 'Frozen Model']
    frozen_seed_cpus = frozen_sub.groupby('seed')['cpu_time'].sum()
    
    master_rows = []
    for _, row in df_sum.iterrows():
        app = row['Approach']
        app_sub = df_win[df_win['approach'] == app]
        seed_cpus = app_sub.groupby('seed')['cpu_time'].sum()
        
        # Ratio of means
        mean_c = row['raw_cpu']
        cost_ratio = mean_c / frozen_mean_cpu if frozen_mean_cpu > 0 else 1.0
        
        # Ratio per seed
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
        
    df_master = pd.DataFrame(master_rows)
    master_csv = os.path.join(RESULTS_DIR, 'master_benchmark_v5.csv')
    df_master.to_csv(master_csv, index=False)
    print(df_master[['Approach', 'Mean F1', 'Cumulative CPU Time (s)', 'Cost vs. Frozen', 'Model Footprint (KB)', 'Retraining Events']].to_string(index=False))
    print("=" * 95)
    
    total_elapsed = time.time() - start_time
    print(f"\nUNIFIED BENCHMARK COMPLETED IN {total_elapsed:.1f}s ({total_elapsed/60.0:.2f} minutes)!")

if __name__ == '__main__':
    main()
