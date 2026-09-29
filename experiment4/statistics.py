"""
EXPERIMENT 4 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_summary_stats(data_series):
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


def run_paired_tests(df_window, metrics=None):
    if metrics is None:
        metrics = ['f1', 'accuracy', 'cpu_time']

    available_approaches = list(df_window['approach'].unique())
    comparisons = []

    # Priority comparisons:
    # 1. All adaptive/alternative strategies vs. Frozen Model
    # 2. All adaptive/alternative strategies vs. Continuously Retrained Ensemble
    # 3. All new strategies vs. Event-Driven Ensemble (Custom Dual-Trigger)
    # 4. Continuous Retrained Ensemble vs. Frozen Model
    # 5. Pairwise among all adaptive/alternative strategies
    has_frozen = 'Frozen Model' in available_approaches
    has_cont = 'Continuously Retrained Ensemble' in available_approaches
    has_custom = 'Event-Driven Ensemble (Custom Dual-Trigger)' in available_approaches

    adaptive_approaches = [a for a in available_approaches if a != 'Frozen Model']

    for ad in adaptive_approaches:
        if has_frozen and ad != 'Frozen Model':
            for m in metrics:
                comparisons.append((ad, 'Frozen Model', m))
        if has_cont and ad != 'Continuously Retrained Ensemble':
            for m in metrics:
                comparisons.append((ad, 'Continuously Retrained Ensemble', m))
        if has_custom and ad != 'Event-Driven Ensemble (Custom Dual-Trigger)' and ad != 'Continuously Retrained Ensemble':
            for m in metrics:
                comparisons.append((ad, 'Event-Driven Ensemble (Custom Dual-Trigger)', m))

    # Pairwise among all adaptive approaches
    for i in range(len(adaptive_approaches)):
        for j in range(i + 1, len(adaptive_approaches)):
            for m in metrics:
                comparisons.append((adaptive_approaches[i], adaptive_approaches[j], m))

    # Deduplicate while preserving order
    seen = set()
    clean_comparisons = []
    for a, b, m in comparisons:
        pair_key = (min(a, b), max(a, b), m)
        if pair_key not in seen and m in df_window.columns:
            seen.add(pair_key)
            clean_comparisons.append((a, b, m))
    comparisons = clean_comparisons


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

    return results
