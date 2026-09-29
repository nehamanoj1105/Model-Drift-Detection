"""
Evaluation and Statistical Methodology Module for RAPT-v2
Computes classification metrics, computational profiling, adaptation hierarchy breakdown,
efficiency objective J = F1 - lambda * NormalizedCPU, and paired Wilcoxon signed-rank tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

def calculate_window_metrics(y_true, y_pred):
    """Compute classification metrics."""
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

def compute_efficiency_objective(df_summary, lambda_cpu=0.01):
    """
    Computes diagnostic efficiency objective J = F1 - lambda * NormalizedCPU.
    NormalizedCPU is scaled relative to the maximum Total CPU in the summary.
    """
    max_cpu = df_summary['total_cpu_sec_mean'].max() if df_summary['total_cpu_sec_mean'].max() > 0 else 1.0
    df_summary['normalized_cpu'] = df_summary['total_cpu_sec_mean'] / max_cpu
    df_summary['efficiency_objective_J'] = df_summary['macro_f1_mean'] - lambda_cpu * df_summary['normalized_cpu']
    return df_summary

def compute_wilcoxon_tests_v2(df_per_window, best_variant='RAPT-E'):
    """
    Performs paired window-level Wilcoxon signed-rank hypothesis tests.
    Comparisons:
      - Best RAPT-v2 vs Event-Driven
      - Best RAPT-v2 vs RAPT-v1
      - Best RAPT-v2 vs Frozen
      - Best RAPT-v2 vs Full Retraining
    """
    seeds = df_per_window['seed'].unique()
    test_results = []
    
    comparisons = [
        (best_variant, 'Event-Driven'),
        (best_variant, 'RAPT-v1'),
        (best_variant, 'Frozen'),
        (best_variant, 'Full Retraining')
    ]
    
    for m1, m2 in comparisons:
        f1_diffs = []
        for seed in seeds:
            df_seed = df_per_window[df_per_window['seed'] == seed]
            df_m1 = df_seed[df_seed['method'] == m1].sort_values('window_id')
            df_m2 = df_seed[df_seed['method'] == m2].sort_values('window_id')
            
            if len(df_m1) > 0 and len(df_m2) > 0:
                y1 = df_m1['is_correct'].values.astype(float)
                y2 = df_m2['is_correct'].values.astype(float)
                f1_diffs.extend(y1 - y2)
                
        f1_diffs = np.array(f1_diffs)
        mean_diff = np.mean(f1_diffs) if len(f1_diffs) > 0 else 0.0
        std_diff = np.std(f1_diffs) + 1e-9 if len(f1_diffs) > 0 else 1.0
        
        if len(f1_diffs) == 0 or np.all(f1_diffs == 0):
            stat, p_val = 0.0, 1.0
        else:
            try:
                stat, p_val = stats.wilcoxon(f1_diffs, zero_method='pratt')
            except Exception:
                stat, p_val = 0.0, 1.0
                
        cohen_d = mean_diff / std_diff
        
        test_results.append({
            'comparison': f"{m1} vs {m2}",
            'method1': m1,
            'method2': m2,
            'mean_accuracy_diff': float(mean_diff),
            'std_diff': float(std_diff),
            'wilcoxon_stat': float(stat),
            'p_value': float(p_val),
            'cohen_d': float(cohen_d),
            'significant_p05': bool(p_val < 0.05)
        })
        
    return pd.DataFrame(test_results)
