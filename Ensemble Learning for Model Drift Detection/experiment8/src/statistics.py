"""
================================================================================
EXPERIMENT 8 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
================================================================================
Performs paired window-level statistical comparisons:
  - Wilcoxon signed-rank tests
  - 95% Confidence Intervals (Bootstrap)
  - Effect size calculation (Wilcoxon r)
  - Holm-Bonferroni correction for multiple comparisons
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_bootstrap_ci(data, n_boot=1000, ci=0.95, seed=42):
    """Computes non-parametric bootstrap confidence interval for mean."""
    rng = np.random.RandomState(seed)
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1.0 - alpha) * 100)
    return float(lower), float(upper)


def paired_statistical_test(vec_a, vec_b, label_a="Baseline", label_b="Proposed", seed=42):
    """
    Performs paired statistical test between vec_a (Baseline) and vec_b (Proposed).
    vec_a and vec_b must be equal length paired observations (e.g. window F1 scores).
    """
    arr_a = np.array(vec_a)
    arr_b = np.array(vec_b)

    assert len(arr_a) == len(arr_b), "Vector lengths must match for paired testing."

    diffs = arr_b - arr_a
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    ci_low, ci_high = compute_bootstrap_ci(diffs, seed=seed)

    # Wilcoxon signed-rank test
    if np.all(diffs == 0):
        stat, p_val = 0.0, 1.0
        effect_size = 0.0
    else:
        try:
            stat, p_val = stats.wilcoxon(arr_a, arr_b)
            # Wilcoxon effect size r = Z / sqrt(N)
            n = len(diffs)
            z_stat = stats.norm.ppf(1.0 - p_val / 2.0) if p_val > 0 else 5.0
            effect_size = float(z_stat / np.sqrt(n))
        except Exception:
            stat, p_val = 0.0, 1.0
            effect_size = 0.0

    return {
        'comparison': f"{label_b} vs {label_a}",
        'mean_a': float(np.mean(arr_a)),
        'mean_b': float(np.mean(arr_b)),
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'ci_95_lower': ci_low,
        'ci_95_upper': ci_high,
        'wilcoxon_stat': float(stat),
        'p_value': float(p_val),
        'effect_size_r': effect_size,
        'n_pairs': len(diffs),
    }


def apply_holm_bonferroni(results_list):
    """
    Applies Holm-Bonferroni correction to a list of statistical result dicts.
    Modifies dicts in-place to add 'p_value_corrected' and 'significant_alpha_005'.
    """
    p_vals = [r['p_value'] for r in results_list]
    m = len(p_vals)

    # Sort p-values with original indices
    sorted_indices = np.argsort(p_vals)
    corrected_p = [0.0] * m

    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        raw_p = p_vals[idx]
        adj_p = min(1.0, raw_p * (m - rank))
        cum_max = max(cum_max, adj_p)
        corrected_p[idx] = float(cum_max)

    for idx, r in enumerate(results_list):
        r['p_value_corrected'] = float(corrected_p[idx])
        r['significant_alpha_005'] = bool(corrected_p[idx] < 0.05)

    return results_list
