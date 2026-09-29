"""
Evaluation & Statistical Analysis Module for Exp 9
Strict Prequential Protocol (Test-Then-Train), Window Metrics, Recovery Analysis, and Wilcoxon Tests
"""

import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

warnings.filterwarnings('ignore')

def compute_window_metrics(y_true, y_pred, y_prob=None):
    """Computes F1 macro, accuracy, and balanced accuracy for a single window."""
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return {
        'f1_macro': f1_macro,
        'accuracy': acc,
        'balanced_accuracy': bal_acc
    }

def analyze_regime_transitions(window_metrics_df, block_metadata):
    """
    Measures transition recovery metrics for every detected regime transition:
      - pre-event F1 (mean F1 of 5 windows before transition)
      - post-event F1 (F1 on first window after transition)
      - min F1 post-transition (min F1 in 5 windows after transition)
      - windows to recover (windows until F1 >= 0.95 * pre-event F1)
      - F1 recovered
      - CPU spent recovering
    """
    recovery_records = []
    
    # Map window_id to row index in window_metrics_df
    win_to_idx = {row['window_id']: idx for idx, row in window_metrics_df.iterrows()}
    
    for b_idx in range(1, len(block_metadata)):
        meta = block_metadata[b_idx]
        trans_win = meta['start_window']
        
        if trans_win not in win_to_idx:
            continue
            
        cur_row_idx = win_to_idx[trans_win]
        
        # Pre-event window indices
        pre_start = max(0, cur_row_idx - 5)
        pre_end = cur_row_idx
        
        # Post-event window indices
        post_start = cur_row_idx
        post_end = min(len(window_metrics_df), cur_row_idx + 5)
        
        pre_f1 = float(window_metrics_df.iloc[pre_start:pre_end]['f1_macro'].mean()) if pre_end > pre_start else float(window_metrics_df.iloc[cur_row_idx]['f1_macro'])
        post_f1 = float(window_metrics_df.iloc[cur_row_idx]['f1_macro'])
        
        post_slice = window_metrics_df.iloc[post_start:post_end]['f1_macro'].values
        min_post_f1 = float(np.min(post_slice)) if len(post_slice) > 0 else post_f1
        
        # Recovery window search
        target_f1 = 0.95 * pre_f1
        recovery_windows = 5 # default max if not recovered
        for offset, f1_val in enumerate(post_slice):
            if f1_val >= target_f1:
                recovery_windows = offset + 1
                break
                
        f1_recovered = max(0.0, float(post_slice[-1] - min_post_f1)) if len(post_slice) > 0 else 0.0
        
        # Adaptation CPU spent recovering
        cpu_spent_rec = float(window_metrics_df.iloc[post_start:post_end]['adaptation_cpu_time'].sum())
        
        rec_rec = {
            'transition_id': b_idx,
            'regime': meta['regime'],
            'transition_window': trans_win,
            'pre_event_f1': pre_f1,
            'post_event_f1': post_f1,
            'min_post_f1': min_post_f1,
            'windows_to_recover': recovery_windows,
            'f1_recovered': f1_recovered,
            'cpu_spent_recovering': cpu_spent_rec
        }
        recovery_records.append(rec_rec)
        
    return pd.DataFrame(recovery_records)

def run_wilcoxon_tests(df_window_all):
    """
    Performs paired Wilcoxon signed-rank tests on window-level F1 scores.
    Compares RAPT vs Event-Driven, RAPT vs Frozen, and Event-Driven vs Frozen.
    """
    test_results = []
    
    methods = df_window_all['method'].unique()
    pairs = [
        ('RAPT', 'Event-Driven'),
        ('RAPT', 'Frozen'),
        ('Event-Driven', 'Frozen')
    ]
    
    for m1, m2 in pairs:
        if m1 in methods and m2 in methods:
            f1_m1 = df_window_all[df_window_all['method'] == m1]['f1_macro'].values
            f1_m2 = df_window_all[df_window_all['method'] == m2]['f1_macro'].values
            
            # Ensure equal length paired samples
            min_len = min(len(f1_m1), len(f1_m2))
            f1_m1 = f1_m1[:min_len]
            f1_m2 = f1_m2[:min_len]
            
            diff = f1_m1 - f1_m2
            mean_diff = float(np.mean(diff))
            
            # Wilcoxon signed rank test
            try:
                if np.all(diff == 0):
                    stat, p_val = 0.0, 1.0
                else:
                    stat, p_val = stats.wilcoxon(f1_m1, f1_m2)
            except Exception:
                stat, p_val = 0.0, 1.0
                
            # Effect size r = Z / sqrt(N)
            n_samples = len(diff)
            z_score = stats.norm.ppf(1 - p_val / 2) if (p_val > 0 and p_val < 1) else 0.0
            effect_size = float(z_score / np.sqrt(n_samples)) if n_samples > 0 else 0.0
            
            test_results.append({
                'comparison': f'{m1} vs {m2}',
                'method_1': m1,
                'method_2': m2,
                'f1_difference': mean_diff,
                'p_value': float(p_val),
                'effect_size': effect_size,
                'n_windows': min_len,
                'statistically_significant': bool(p_val < 0.05)
            })
            
    return pd.DataFrame(test_results)
