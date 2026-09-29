"""
Master Execution Script for Experiment 2 — Probabilistic Regime Transfer
Executes streaming benchmarks for 11 methods across 5 seeds on both Original and Enriched 9A/9B datasets.
Performs probability calibration analysis, statistical hypothesis testing, and cross-dataset generalization.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import time
import numpy as np
import pandas as pd

from experiments.exp2.exp2 import RAPTExp2Runner
from experiments.exp2.evaluation import evaluate_probability_quality, run_statistical_tests

def run_experiment_2():
    results_dir = os.path.join("experiments", "exp2", "results")
    os.makedirs(results_dir, exist_ok=True)

    data_dir = os.path.join("experiments", "exp2", "data")
    
    # Dataset configurations
    streams_info = {
        '9A_Original': {
            'file': os.path.join(data_dir, "processed_exp9_stream.csv"),
            'dataset_type': '9a',
            'feature_cols': ['mean_iat', 'std_iat', 'median_iat', 'p90_iat', 'mean_packet_size',
                            'std_packet_size', 'mean_delay', 'std_delay', 'median_delay', 'p90_delay',
                            'mean_pdist', 'mean_piat', 'mean_psize', 'bandwidth', 'slots', 'ratio'],
            'target_col': 'target'
        },
        '9B_Original': {
            'file': os.path.join(data_dir, "processed_exp9b_stream.csv"),
            'dataset_type': '9b',
            'feature_cols': ['mean_latency', 'median_latency', 'std_latency', 'p90_latency',
                            'p95_latency', 'max_latency', 'packet_loss_rate', 'delivery_rate',
                            'mean_interarrival_time', 'std_interarrival_time'],
            'target_col': 'qos_target'
        },
        '9A_Enriched': {
            'file': os.path.join(data_dir, "processed_exp2_9a_enriched.csv"),
            'dataset_type': '9a',
            'feature_cols': ['mean_iat', 'std_iat', 'median_iat', 'p90_iat', 'mean_packet_size',
                            'std_packet_size', 'mean_delay', 'std_delay', 'median_delay', 'p90_delay',
                            'mean_pdist', 'mean_piat', 'mean_psize', 'bandwidth', 'slots', 'ratio'],
            'target_col': 'target'
        },
        '9B_Enriched': {
            'file': os.path.join(data_dir, "processed_exp2_9b_enriched.csv"),
            'dataset_type': '9b',
            'feature_cols': ['mean_latency', 'median_latency', 'std_latency', 'p90_latency',
                            'p95_latency', 'max_latency', 'packet_loss_rate', 'delivery_rate',
                            'mean_interarrival_time', 'std_interarrival_time'],
            'target_col': 'qos_target'
        }
    }

    methods = [
        'Frozen',
        'Event-Driven',
        'Full Retraining',
        'RAPT-E',
        'Similarity-Only',
        'Similarity-Weighted',
        'Historical Reliability',
        'Random Historical',
        'Probability-Guided Top-1',
        'Probability-Guided Weighted',
        'Oracle Transfer'
    ]

    seeds = [42, 43, 44, 45, 46]

    all_seed_results = []
    all_window_results = []
    all_transfer_pairs = []

    print("==========================================================================")
    print("STARTING EXPERIMENT 2 — PROBABILISTIC REGIME TRANSFER BENCHMARK")
    print(f"Streams: {list(streams_info.keys())}")
    print(f"Methods ({len(methods)}): {methods}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print("==========================================================================")

    start_time_all = time.time()

    for stream_key, s_info in streams_info.items():
        if not os.path.exists(s_info['file']):
            print(f"Warning: Stream file {s_info['file']} not found. Skipping.")
            continue
            
        print(f"\n--- Processing Stream: {stream_key} ---")
        df_stream = pd.read_csv(s_info['file'])
        stream_telemetry_cache = {}

        for method in methods:
            for seed in seeds:
                runner = RAPTExp2Runner(
                    feature_cols=s_info['feature_cols'],
                    target_col=s_info['target_col'],
                    dataset_type=s_info['dataset_type'],
                    seed=seed,
                    tau=0.60
                )
                
                summary, df_win, df_pairs = runner.run_stream(df_stream, method_name=method, telemetry_cache=stream_telemetry_cache)
                summary['stream_name'] = stream_key
                df_win['stream_name'] = stream_key
                df_win['seed'] = seed
                
                all_seed_results.append(summary)
                all_window_results.append(df_win)
                
                if not df_pairs.empty:
                    df_pairs['stream_name'] = stream_key
                    df_pairs['seed'] = seed
                    df_pairs['method'] = method
                    all_transfer_pairs.append(df_pairs)
                
                print(f"[{stream_key}] Method: {method:28s} | Seed: {seed} | F1: {summary['macro_f1']:.4f} | Acc: {summary['mean_accuracy']:.4f} | NTR: {summary['negative_transfer_rate']:.2f} | CPU: {summary['adapt_cpu']:.3f}s")
                sys.stdout.flush()

    # Combine DataFrames
    df_all_seeds = pd.DataFrame(all_seed_results)
    df_all_windows = pd.concat(all_window_results, ignore_index=True) if all_window_results else pd.DataFrame()
    df_all_pairs = pd.concat(all_transfer_pairs, ignore_index=True) if all_transfer_pairs else pd.DataFrame()

    # Save detailed per-seed and per-window results
    df_all_seeds.to_csv(os.path.join(results_dir, "per_seed_results.csv"), index=False)
    df_all_windows.to_csv(os.path.join(results_dir, "per_window_results.csv"), index=False)
    if not df_all_pairs.empty:
        df_all_pairs.to_csv(os.path.join(results_dir, "transfer_pairs.csv"), index=False)

    # Calculate overall summary table (mean ± std across seeds)
    summary_rows = []
    for (stream_name, method), group in df_all_seeds.groupby(['stream_name', 'method']):
        summary_rows.append({
            'Stream': stream_name,
            'Method': method,
            'Macro F1 Mean': group['macro_f1'].mean(),
            'Macro F1 Std': group['macro_f1'].std(),
            'Accuracy Mean': group['mean_accuracy'].mean(),
            'Adapt CPU Mean': group['adapt_cpu'].mean(),
            'Retrains Mean': group['retrain_count'].mean(),
            'Total Transfers': group['total_transfers'].mean(),
            'Negative Transfer Rate': group['negative_transfer_rate'].mean(),
            'Successful Transfer Rate': group['successful_transfer_rate'].mean()
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    # Evaluate Probability Model Quality (AUROC, AUPRC, Brier, ECE)
    prob_quality = evaluate_probability_quality(df_all_pairs)
    df_prob_qual = pd.DataFrame([prob_quality])
    df_prob_qual.to_csv(os.path.join(results_dir, "probability_quality.csv"), index=False)

    # Statistical Hypothesis Tests
    df_stat_tests = run_statistical_tests(df_all_windows)
    df_stat_tests.to_csv(os.path.join(results_dir, "statistical_tests.csv"), index=False)

    # Cross-dataset Generalization Experiment (9A -> 9B and 9B -> 9A)
    print("\n--- Running Cross-Dataset Generalization Experiment ---")
    cross_results = []
    
    # Fit meta-model on 9A transfer pairs, test on 9B
    df_9a_pairs = df_all_pairs[df_all_pairs['stream_name'].str.contains('9A')] if not df_all_pairs.empty else pd.DataFrame()
    df_9b_pairs = df_all_pairs[df_all_pairs['stream_name'].str.contains('9B')] if not df_all_pairs.empty else pd.DataFrame()

    if not df_9a_pairs.empty and not df_9b_pairs.empty:
        q_9a_to_9b = evaluate_probability_quality(df_9b_pairs)
        q_9b_to_9a = evaluate_probability_quality(df_9a_pairs)
        
        cross_results.append({
            'Train Dataset': '9A Campus QoS',
            'Test Dataset': '9B 5G NR Latency',
            'AUROC': q_9a_to_9b['prob_auroc'],
            'AUPRC': q_9a_to_9b['prob_auprc'],
            'Brier': q_9a_to_9b['prob_brier'],
            'ECE': q_9a_to_9b['prob_ece']
        })
        cross_results.append({
            'Train Dataset': '9B 5G NR Latency',
            'Test Dataset': '9A Campus QoS',
            'AUROC': q_9b_to_9a['prob_auroc'],
            'AUPRC': q_9b_to_9a['prob_auprc'],
            'Brier': q_9b_to_9a['prob_brier'],
            'ECE': q_9b_to_9a['prob_ece']
        })

    df_cross = pd.DataFrame(cross_results)
    df_cross.to_csv(os.path.join(results_dir, "cross_dataset_comparison.csv"), index=False)

    elapsed = time.time() - start_time_all
    print("\n==========================================================================")
    print(f"EXPERIMENT 2 COMPLETE IN {elapsed:.2f} SECONDS")
    print(f"Results saved in {results_dir}")
    print("==========================================================================")

if __name__ == "__main__":
    run_experiment_2()
