"""
Reproduce Original RAPT Baseline (Exp 9A & Exp 9B)
Runs original RAPT implementation exactly across seeds [42, 43, 44, 45, 46].
Saves fresh outputs to experiments/rapt_audit/results/original_reproduction/
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import original exp9 modules
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "exp9"))
import run_exp9

# Import original exp9b modules
sys.path.insert(0, os.path.join(ROOT_DIR, "experiments", "exp9b"))
import run_exp9b

OUT_DIR = os.path.join(ROOT_DIR, "experiments", "rapt_audit", "results", "original_reproduction")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print("REPRODUCING ORIGINAL EXP 9A & 9B RAPT BASELINES")
    print("=" * 80)

    # 1. Run Original 9A
    print("\n--- Running Original Exp 9A Pipeline ---")
    sum_9a, stat_9a = run_exp9.run_exp9_pipeline(
        stream_csv=os.path.join(ROOT_DIR, "experiments", "exp9", "data", "processed_exp9_stream.csv")
    )
    
    # Save 9A reproduction results
    sum_9a.to_csv(os.path.join(OUT_DIR, "exp9a_reproduction_summary.csv"), index=False)

    # 2. Run Original 9B
    print("\n--- Running Original Exp 9B Pipeline ---")
    run_exp9b.run_experiment_9b()
    sum_9b = pd.read_csv(os.path.join(ROOT_DIR, "experiments", "exp9b", "results", "summary.csv"))
    sum_9b.to_csv(os.path.join(OUT_DIR, "exp9b_reproduction_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("REPRODUCTION SUMMARY")
    print("=" * 80)
    print("\n[Exp 9A Reproduced Metrics]:")
    print(sum_9a[['method', 'f1_macro_mean', 'f1_macro_std', 'total_cpu_time_mean', 'retrain_events_mean']].to_string(index=False))

    print("\n[Exp 9B Reproduced Metrics]:")
    print(sum_9b[['method', 'macro_f1_mean', 'macro_f1_std', 'total_cpu_sec_mean', 'retrain_events_mean']].to_string(index=False))

if __name__ == '__main__':
    main()
