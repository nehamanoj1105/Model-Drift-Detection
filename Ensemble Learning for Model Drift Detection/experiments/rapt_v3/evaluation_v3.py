"""
Evaluation and Statistical Testing Utilities for RAPT-v3
Computes per-seed and per-window metrics, Wilcoxon signed-rank tests,
practical non-inferiority margins (delta = 0.005), and transition dynamics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

def calculate_window_metrics(y_true, y_pred):
    """Compute standard classification evaluation metrics."""
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

def compute_wilcoxon_and_non_inferiority(df_per_window, comparisons, delta=0.005):
    """
    Computes Wilcoxon signed-rank test and Non-Inferiority analysis (margin delta=0.005).
    """
    seeds = df_per_window['seed'].unique()
    test_records = []
    
    for m1, m2 in comparisons:
        f1_diffs = []
        for seed in seeds:
            df_s = df_per_window[df_per_window['seed'] == seed]
            df_m1 = df_s[df_s['method'] == m1].sort_values('window_id')
            df_m2 = df_s[df_s['method'] == m2].sort_values('window_id')
            
            if len(df_m1) == 0 or len(df_m2) == 0:
                continue
                
            y1 = df_m1['is_correct'].values.astype(float)
            y2 = df_m2['is_correct'].values.astype(float)
            f1_diffs.extend(y1 - y2)
            
        f1_diffs = np.array(f1_diffs)
        if len(f1_diffs) == 0:
            continue
            
        mean_diff = float(np.mean(f1_diffs))
        std_diff = float(np.std(f1_diffs) + 1e-9)
        
        if np.all(f1_diffs == 0):
            stat, p_val = 0.0, 1.0
        else:
            try:
                stat, p_val = stats.wilcoxon(f1_diffs, zero_method='pratt')
            except Exception:
                stat, p_val = 0.0, 1.0
                
        cohen_d = float(mean_diff / std_diff)
        is_non_inferior = bool(mean_diff >= -delta)
        
        test_records.append({
            'comparison': f"{m1} vs {m2}",
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'wilcoxon_stat': float(stat),
            'p_value': float(p_val),
            'cohen_d': cohen_d,
            'significant_p05': bool(p_val < 0.05),
            'margin_delta': delta,
            'non_inferior': is_non_inferior
        })
        
    return pd.DataFrame(test_records)

def analyze_regime_transitions_v3(df_per_window, stream_df):
    """
    Measures transition recovery metrics for every detected regime transition.
    """
    records = []
    seeds = df_per_window['seed'].unique()
    methods = df_per_window['method'].unique()
    
    regime_col = 'regime_id' if 'regime_id' in stream_df.columns else 'regime_label'
    regimes = stream_df[regime_col].values
    
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
                    
                pre_f1 = float(df_slice[df_slice['window_id'] < tw]['is_correct'].mean())
                post_f1 = float(df_slice[df_slice['window_id'] == tw]['is_correct'].mean())
                min_post_f1 = float(df_slice[df_slice['window_id'] >= tw]['is_correct'].min())
                
                target_acc = 0.95 * pre_f1
                rec_wins = 5
                post_arr = df_slice[df_slice['window_id'] >= tw]['is_correct'].values
                for idx_off, acc_val in enumerate(post_arr):
                    if acc_val >= target_acc:
                        rec_wins = idx_off + 1
                        break
                        
                adapt_cpu = float(df_slice[df_slice['window_id'] >= tw]['adaptation_time'].sum())
                
                records.append({
                    'seed': seed,
                    'method': m,
                    'transition_window': tw,
                    'pre_event_f1': pre_f1,
                    'post_event_f1': post_f1,
                    'min_post_f1': min_post_f1,
                    'windows_to_recover': rec_wins,
                    'adaptation_cpu': adapt_cpu
                })
                
    return pd.DataFrame(records)
