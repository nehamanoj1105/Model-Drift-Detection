"""
================================================================================
REPRODUCIBILITY BENCHMARK VERIFIER (RUN 1 VS RUN 2)
================================================================================
Executes two independent runs of the Stage 3 Master Benchmark with identical
seeds [42, 43, 44, 45, 46], code, and single-threaded environment variables.

Exports raw per-seed per-retrain-event logs and evaluates reproducibility.
================================================================================
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

# Ensure local imports
rapt_dir = os.path.dirname(os.path.abspath(__file__))
if rapt_dir not in sys.path:
    sys.path.insert(0, rapt_dir)

from run_rapt_stage3_benchmark import run_stage3_master_benchmark
from config import RESULTS_DIR

def main():
    print("=" * 85)
    print("EXECUTING RUN 1 OF STAGE 3 MASTER BENCHMARK...")
    print("=" * 85)
    df_win1, df_sum1, df_tests1, df_lat1, df_raw1 = run_stage3_master_benchmark()
    
    # Save Run 1 files explicitly
    path_sum1 = os.path.join(RESULTS_DIR, 'stage3_summary_metrics_run1.csv')
    path_raw1 = os.path.join(RESULTS_DIR, 'stage3_raw_retrain_timings_run1.csv')
    path_win1 = os.path.join(RESULTS_DIR, 'stage3_window_metrics_run1.csv')
    df_sum1.to_csv(path_sum1, index=False)
    df_raw1.to_csv(path_raw1, index=False)
    df_win1.to_csv(path_win1, index=False)
    print(f"\nSaved Run 1 results to:\n  - {path_sum1}\n  - {path_raw1}\n  - {path_win1}")

    print("\n" + "=" * 85)
    print("EXECUTING RUN 2 OF STAGE 3 MASTER BENCHMARK (EXACT SAME SEEDS & CODE)...")
    print("=" * 85)
    df_win2, df_sum2, df_tests2, df_lat2, df_raw2 = run_stage3_master_benchmark()

    # Save Run 2 files explicitly
    path_sum2 = os.path.join(RESULTS_DIR, 'stage3_summary_metrics_run2.csv')
    path_raw2 = os.path.join(RESULTS_DIR, 'stage3_raw_retrain_timings_run2.csv')
    path_win2 = os.path.join(RESULTS_DIR, 'stage3_window_metrics_run2.csv')
    df_sum2.to_csv(path_sum2, index=False)
    df_raw2.to_csv(path_raw2, index=False)
    df_win2.to_csv(path_win2, index=False)
    print(f"\nSaved Run 2 results to:\n  - {path_sum2}\n  - {path_raw2}\n  - {path_win2}")

    # ==============================================================================
    # REPRODUCIBILITY ANALYSIS (RUN 1 VS RUN 2)
    # ==============================================================================
    print("\n" + "=" * 85)
    print("REPRODUCIBILITY EVALUATION: RUN 1 VS RUN 2")
    print("=" * 85)
    
    comp_rows = []
    for app in df_sum1['approach'].unique():
        row1 = df_sum1[df_sum1['approach'] == app].iloc[0]
        row2 = df_sum2[df_sum2['approach'] == app].iloc[0]
        
        cpu1 = row1['mean_adapt_cpu_s']
        cpu2 = row2['mean_adapt_cpu_s']
        delta_cpu = abs(cpu1 - cpu2)
        
        tot_cpu1 = row1['mean_total_cpu_s']
        tot_cpu2 = row2['mean_total_cpu_s']
        delta_tot = abs(tot_cpu1 - tot_cpu2)

        f1_1 = row1['mean_f1']
        f1_2 = row2['mean_f1']
        delta_f1 = abs(f1_1 - f1_2)

        # Tolerance check: Event-Driven ±2.41s, Continuous ±4.31s, others within std_adapt_cpu_s
        if app == 'Event-Driven Baseline':
            tol = 2.41
        elif app == 'Continuous Retraining':
            tol = 4.31
        else:
            tol = max(row1['std_adapt_cpu_s'], 1.0)
            
        reproducible = delta_cpu <= tol

        comp_rows.append({
            'Approach': app,
            'Run1_Adapt_CPU_s': cpu1,
            'Run2_Adapt_CPU_s': cpu2,
            'Delta_Adapt_CPU_s': delta_cpu,
            'Tolerance_s': tol,
            'Reproducible': reproducible,
            'Run1_F1': f1_1,
            'Run2_F1': f1_2,
            'Delta_F1': delta_f1
        })

    df_comp = pd.DataFrame(comp_rows)
    print(df_comp.to_string(index=False))
    print("=" * 85)
    
    all_reproducible = all(df_comp['Reproducible'])
    if all_reproducible:
        print("SUCCESS: All strategies reproduced within specified tolerance!")
    else:
        print("WARNING: Some strategies did not reproduce within specified tolerance.")

if __name__ == '__main__':
    main()
