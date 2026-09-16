"""
================================================================================
EXPERIMENT 3 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
================================================================================
Compares the three approaches across 5 seeds:
  1. UCB1 Adaptive Ensemble vs Frozen RF
  2. UCB1 Adaptive Ensemble vs Retrained RF
  3. Retrained RF vs Frozen RF

Methods:
  - Paired Wilcoxon signed-rank test (with paired t-test fallback)
  - Holm-Bonferroni multiple-comparison correction
  - Effect sizes: Rank-biserial correlation r or Cohen's d
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_summary_stats(data_series):
    """Compute mean, standard deviation, and 95% Student-t confidence interval."""
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

    return {'mean': mean, 'std': std, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'n': n}


def run_paired_tests(df_window):
    """
    Run paired statistical hypothesis tests for the 3 key comparisons across windows/seeds.
    Comparisons:
      - UCB1 Adaptive Ensemble vs Frozen RF (F1, Accuracy)
      - UCB1 Adaptive Ensemble vs Retrained RF (F1, Accuracy, CPU Time)
      - Retrained RF vs Frozen RF (F1, Accuracy, CPU Time)
    """
    comparisons = [
        ('UCB1 Adaptive Ensemble', 'Frozen RF', 'f1'),
        ('UCB1 Adaptive Ensemble', 'Frozen RF', 'accuracy'),
        ('UCB1 Adaptive Ensemble', 'Retrained RF', 'f1'),
        ('UCB1 Adaptive Ensemble', 'Retrained RF', 'accuracy'),
        ('UCB1 Adaptive Ensemble', 'Retrained RF', 'cpu_time'),
        ('Retrained RF', 'Frozen RF', 'f1'),
        ('Retrained RF', 'Frozen RF', 'accuracy'),
        ('Retrained RF', 'Frozen RF', 'cpu_time'),
    ]

    results = []
    raw_p_values = []

    for strat_a, strat_b, metric in comparisons:
        sub_a = df_window[df_window['approach'] == strat_a].sort_values(['seed', 'window_id'])
        sub_b = df_window[df_window['approach'] == strat_b].sort_values(['seed', 'window_id'])

        merged = pd.merge(
            sub_a[['seed', 'window_id', metric]],
            sub_b[['seed', 'window_id', metric]],
            on=['seed', 'window_id'],
            suffixes=('_a', '_b'),
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

        # Normality test
        if n >= 8 and np.std(diffs) > 1e-8:
            _, shapiro_p = stats.shapiro(diffs[:min(n, 5000)])
            is_normal = shapiro_p >= 0.05
        else:
            is_normal = False

        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) > 5:
            try:
                w_stat, p_val = stats.wilcoxon(vals_a, vals_b, zero_method='wilcox')
                test_type = 'Wilcoxon signed-rank'
                test_stat = float(w_stat)
                n_nz = len(nonzero_diffs)
                total_rank = n_nz * (n_nz + 1) / 2
                effect_size = float(abs(1.0 - 2.0 * w_stat / total_rank)) if total_rank > 0 else 0.0
                effect_type = 'Rank-biserial r'
            except Exception:
                t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
                test_type = 'Paired t-test'
                test_stat = float(t_stat)
                std_diff = np.std(diffs, ddof=1)
                effect_size = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0
                effect_type = "Cohen's d"
        else:
            t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
            test_type = 'Paired t-test'
            test_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
            p_val = float(p_val) if not np.isnan(p_val) else 1.0
            std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 1.0
            effect_size = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0
            effect_type = "Cohen's d"

        raw_p_values.append(float(p_val))
        results.append({
            'comparison': f'{strat_a} vs {strat_b}',
            'metric': metric,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'test_type': test_type,
            'test_stat': test_stat,
            'p_value': float(p_val),
            'effect_size': float(effect_size),
            'effect_type': effect_type,
        })

    # Holm-Bonferroni correction
    m = len(raw_p_values)
    if m > 0:
        sorted_idx = np.argsort(raw_p_values)
        adjusted = np.zeros(m)
        current_max = 0.0
        for rank, idx in enumerate(sorted_idx):
            adj_p = raw_p_values[idx] * (m - rank)
            adj_p = min(1.0, max(current_max, adj_p))
            current_max = adj_p
            adjusted[idx] = adj_p

        for idx, res in enumerate(results):
            res['p_value_adjusted'] = float(adjusted[idx])
            res['is_significant'] = bool(adjusted[idx] < 0.05)

    return pd.DataFrame(results)
