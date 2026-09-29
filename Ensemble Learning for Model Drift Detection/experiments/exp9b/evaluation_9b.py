"""
Evaluation and Statistical Methodology Module for Exp 9B
Calculates prequential streaming metrics (Macro F1, Accuracy, Balanced Accuracy, Precision, Recall),
computational profiling (CPU, Memory, Retraining Events, Checkpoint Reuse), transition recovery windows,
and paired Wilcoxon signed-rank statistical hypothesis tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

def calculate_window_metrics(y_true, y_pred):
    """Compute standard classification metrics for a window or sequence of predictions."""
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

def compute_wilcoxon_tests(df_per_window):
    """
    Performs paired window-level Wilcoxon signed-rank hypothesis tests.
    Primary test: RAPT vs Event-Driven.
    Secondary tests: RAPT vs Frozen, RAPT vs Full Retraining, Event-Driven vs Frozen.
    """
    seeds = df_per_window['seed'].unique()
    test_results = []
    
    comparisons = [
        ('RAPT', 'Event-Driven'),
        ('RAPT', 'Frozen'),
        ('RAPT', 'Full Retraining'),
        ('Event-Driven', 'Frozen')
    ]
    
    # Calculate window-level F1 errors or accuracy indicators across all seeds
    for m1, m2 in comparisons:
        f1_diffs = []
        for seed in seeds:
            df_seed = df_per_window[df_per_window['seed'] == seed]
            df_m1 = df_seed[df_seed['method'] == m1].sort_values('window_id')
            df_m2 = df_seed[df_seed['method'] == m2].sort_values('window_id')
            
            if len(df_m1) > 0 and len(df_m2) > 0:
                y1 = df_m1['is_correct'].values.astype(float)
                y2 = df_m2['is_correct'].values.astype(float)
                diff = y1 - y2
                f1_diffs.extend(diff)
                
        f1_diffs = np.array(f1_diffs)
        mean_diff = np.mean(f1_diffs)
        std_diff = np.std(f1_diffs) + 1e-9
        
        # Paired Wilcoxon Signed-Rank Test
        if np.all(f1_diffs == 0):
            stat, p_val = 0.0, 1.0
        else:
            try:
                stat, p_val = stats.wilcoxon(f1_diffs, zero_method='pratt')
            except Exception:
                stat, p_val = 0.0, 1.0
                
        # Cohen's d effect size
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

def analyze_transitions(df_per_window, stream_def):
    """
    Computes recovery analysis for regime transitions.
    Measures Macro F1 immediately before/after transitions, minimum F1, and recovery window count.
    """
    segments = stream_def['segments']
    transition_records = []
    
    # Identify transition boundary windows
    boundaries = [s['start_window'] for s in segments if s['start_window'] > 0]
    
    methods = df_per_window['method'].unique()
    seeds = df_per_window['seed'].unique()
    
    for seed in seeds:
        df_seed = df_per_window[df_per_window['seed'] == seed]
        for b_win in boundaries:
            # Segments info
            seg_curr = [s for s in segments if s['start_window'] == b_win][0]
            regime_curr = seg_curr['regime_id']
            
            for m in methods:
                df_m = df_seed[df_seed['method'] == m].sort_values('window_id')
                
                # Window slices around transition
                pre_wins = df_m[(df_m['window_id'] >= b_win - 5) & (df_m['window_id'] < b_win)]
                post_wins = df_m[(df_m['window_id'] >= b_win) & (df_m['window_id'] < b_win + 5)]
                
                pre_acc = pre_wins['is_correct'].mean() if len(pre_wins) > 0 else 0.0
                post_acc = post_wins['is_correct'].mean() if len(post_wins) > 0 else 0.0
                
                # Minimum accuracy in 10 windows after transition
                post_10 = df_m[(df_m['window_id'] >= b_win) & (df_m['window_id'] < b_win + 10)]
                min_acc = post_10['is_correct'].min() if len(post_10) > 0 else 0.0
                
                # Recovery window count (windows required until accuracy returns to pre_acc level)
                rec_wins = 0
                for idx, row in post_10.iterrows():
                    if row['is_correct'] >= pre_acc:
                        break
                    rec_wins += 1
                    
                transition_records.append({
                    'seed': seed,
                    'method': m,
                    'transition_window': b_win,
                    'new_regime_id': regime_curr,
                    'pre_transition_acc': float(pre_acc),
                    'post_transition_acc': float(post_acc),
                    'min_post_acc': float(min_acc),
                    'recovery_windows': rec_wins
                })
                
    return pd.DataFrame(transition_records)
