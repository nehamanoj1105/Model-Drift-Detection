"""
================================================================================
EXPERIMENT 3 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
================================================================================
Calculates across seeds:
- Mean, standard deviation, 95% Confidence Intervals
- Paired statistical tests (Wilcoxon signed-rank / paired t-test)
- Multiple-comparison correction (Holm-Bonferroni)
- Effect sizes (rank-biserial r, Cohen's d)
Saves results to results/statistical_results.csv.
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_summary_stats(data_series):
    """Compute mean, std, and 95% confidence interval for a series."""
    arr = np.array(data_series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'n': 0}
        
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    
    if n > 1 and std > 1e-9:
        sem = std / np.sqrt(n)
        ci_half = stats.t.ppf(0.975, df=n - 1) * sem
        ci_lower = float(mean - ci_half)
        ci_upper = float(mean + ci_half)
    else:
        ci_lower = mean
        ci_upper = mean
        
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }


def run_paired_tests(df_window):
    """
    Run paired tests comparing strategies across windows/seeds.
    Evaluates:
    - Adaptive Ensemble vs Frozen RF
    - Adaptive Ensemble vs Retrained RF
    - UCB1 vs Frozen RF
    - UCB1 vs Retrained RF
    - UCB1 vs Adaptive Ensemble
    - UCB1 vs Oracle
    """
    comparisons = [
        ('Adaptive Ensemble', 'Frozen RF', 'f1'),
        ('Adaptive Ensemble', 'Frozen RF', 'accuracy'),
        ('Adaptive Ensemble', 'Retrained RF', 'f1'),
        ('Adaptive Ensemble', 'Retrained RF', 'accuracy'),
        ('Adaptive Ensemble', 'Retrained RF', 'cpu_time'),
        ('UCB1', 'Frozen RF', 'f1'),
        ('UCB1', 'Frozen RF', 'accuracy'),
        ('UCB1', 'Retrained RF', 'f1'),
        ('UCB1', 'Retrained RF', 'accuracy'),
        ('UCB1', 'Retrained RF', 'cpu_time'),
        ('UCB1', 'Adaptive Ensemble', 'f1'),
        ('UCB1', 'Oracle', 'f1'),
    ]
    
    results = []
    raw_p_values = []
    
    for strat_a, strat_b, metric in comparisons:
        sub_a = df_window[df_window['strategy'] == strat_a].sort_values(['seed', 'window_id'])
        sub_b = df_window[df_window['strategy'] == strat_b].sort_values(['seed', 'window_id'])
        
        # Merge on seed and window_id to ensure exact pairing
        merged = pd.merge(
            sub_a[['seed', 'window_id', metric]],
            sub_b[['seed', 'window_id', metric]],
            on=['seed', 'window_id'],
            suffixes=('_a', '_b')
        )
        
        if len(merged) == 0:
            continue
            
        vals_a = merged[f'{metric}_a'].values
        vals_b = merged[f'{metric}_b'].values
        diffs = vals_a - vals_b
        n = len(diffs)
        
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        mean_diff = float(np.mean(diffs))
        
        # Check normality of differences
        if n >= 8 and np.std(diffs) > 1e-8:
            _, shapiro_p = stats.shapiro(diffs[:min(n, 5000)])
            is_normal = (shapiro_p >= 0.05)
        else:
            is_normal = False
            
        # Run Wilcoxon signed-rank test (appropriate for drift/non-normal data)
        # Handle zero differences
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) > 5:
            try:
                w_stat, p_val = stats.wilcoxon(vals_a, vals_b, zero_method='wilcox')
                test_type = 'Wilcoxon signed-rank'
                test_stat = float(w_stat)
                
                # Rank-biserial correlation effect size r = 1 - (2W / total_rank_sum)
                n_nonzero = len(nonzero_diffs)
                total_rank = n_nonzero * (n_nonzero + 1) / 2
                effect_size = float(abs(1.0 - (2.0 * w_stat / total_rank))) if total_rank > 0 else 0.0
                effect_type = 'Rank-biserial r'
            except Exception:
                test_type = 'Paired t-test'
                t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
                test_stat = float(t_stat)
                std_diff = np.std(diffs, ddof=1)
                effect_size = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0
                effect_type = "Cohen's d"
        else:
            test_type = 'Paired t-test'
            t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
            test_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
            p_val = float(p_val) if not np.isnan(p_val) else 1.0
            std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 1.0
            effect_size = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0
            effect_type = "Cohen's d"
            
        raw_p_values.append(p_val)
        
        results.append({
            'comparison': f'{strat_a} vs {strat_b}',
            'metric': metric,
            'mean_strategy_a': mean_a,
            'mean_strategy_b': mean_b,
            'mean_diff': mean_diff,
            'test_type': test_type,
            'test_stat': test_stat,
            'p_value': float(p_val),
            'effect_size': float(effect_size),
            'effect_type': effect_type,
        })
        
    # Apply Holm-Bonferroni multiple-comparison correction
    m = len(raw_p_values)
    sorted_indices = np.argsort(raw_p_values)
    adjusted_p_values = np.zeros(m)
    
    current_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = raw_p_values[idx]
        adj_p = p * (m - rank)
        adj_p = min(1.0, max(current_max, adj_p))
        current_max = adj_p
        adjusted_p_values[idx] = adj_p
        
    for idx, res in enumerate(results):
        res['p_value_adjusted'] = float(adjusted_p_values[idx])
        res['is_significant'] = bool(adjusted_p_values[idx] < 0.05)
        
    return pd.DataFrame(results)
