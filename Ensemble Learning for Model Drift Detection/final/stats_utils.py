"""
Statistical methodology for the final experiment.

A. Seed-level (PRIMARY): Wilcoxon signed-rank over n=5 paired seed means
   (minimum attainable two-sided p = 0.0625), paired Cohen's d_z, and a 95%
   t-interval on the paired mean difference (t_{0.975,4} = 2.7764).
B. Window-level (SECONDARY): moving-block bootstrap over the evaluated windows
   (block length from config) so autocorrelation is respected; 95% CI of the
   macro-F1 difference and a bootstrap p-value. Plain Wilcoxon on windows is
   NOT used as if windows were independent.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

T_CRIT_975_DF4 = 2.7764451051977987


def seed_level(per_seed_a, per_seed_b):
    """per_seed_*: dict seed -> value (e.g. mean macro-F1). Returns dict."""
    seeds = sorted(set(per_seed_a) & set(per_seed_b))
    a = np.array([per_seed_a[s] for s in seeds], dtype=float)
    b = np.array([per_seed_b[s] for s in seeds], dtype=float)
    d = a - b
    n = len(d)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    dz = float(mean_d / sd_d) if sd_d > 0 else (np.inf if mean_d != 0 else 0.0)
    if n > 1 and sd_d > 0:
        se = sd_d / np.sqrt(n)
        ci = (mean_d - T_CRIT_975_DF4 * se, mean_d + T_CRIT_975_DF4 * se)
    else:
        ci = (mean_d, mean_d)
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        except Exception:
            p = 1.0
    return {"n_seeds": n, "delta": mean_d, "d_z": dz, "p_seed": float(p),
            "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
            "significant": bool(p <= 0.0625)}


def block_bootstrap(per_seed_series_a, per_seed_series_b, block=10, reps=2000, seed=0):
    """per_seed_series_*: dict seed -> ordered array of per-window values."""
    rng = np.random.RandomState(seed)
    seeds = sorted(set(per_seed_series_a) & set(per_seed_series_b))
    series = [np.asarray(per_seed_series_a[s]) - np.asarray(per_seed_series_b[s])
              for s in seeds]
    series = [s for s in series if len(s) > 0]
    if not series:
        return {"delta_window": 0.0, "p_window": 1.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_windows": 0, "block": block}
    obs = float(np.mean(np.concatenate(series)))
    boots = np.empty(reps)
    for r in range(reps):
        parts = []
        for s in series:
            T = len(s)
            nb = int(np.ceil(T / block))
            starts = rng.randint(0, T, size=nb)
            idx = np.concatenate([(np.arange(st, st + block) % T) for st in starts])[:T]
            parts.append(s[idx])
        boots[r] = np.mean(np.concatenate(parts))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    frac_le = float(np.mean(boots <= 0.0))
    frac_ge = float(np.mean(boots >= 0.0))
    p = float(min(1.0, 2 * min(frac_le, frac_ge)))
    return {"delta_window": obs, "p_window": p, "ci_low": float(lo),
            "ci_high": float(hi), "n_windows": int(sum(len(s) for s in series)),
            "block": block}


def cost_reduction_ci(per_seed_cost_a, per_seed_cost_b):
    """% reduction of cost_a relative to cost_b, with a 95% t-interval over seeds."""
    seeds = sorted(set(per_seed_cost_a) & set(per_seed_cost_b))
    a = np.array([per_seed_cost_a[s] for s in seeds], float)
    b = np.array([per_seed_cost_b[s] for s in seeds], float)
    red = (b - a) / np.where(b == 0, np.nan, b) * 100.0
    red = red[~np.isnan(red)]
    if len(red) == 0:
        return {"reduction_pct": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    m = float(np.mean(red))
    if len(red) > 1 and np.std(red, ddof=1) > 0:
        se = np.std(red, ddof=1) / np.sqrt(len(red))
        return {"reduction_pct": m, "ci_low": float(m - T_CRIT_975_DF4 * se),
                "ci_high": float(m + T_CRIT_975_DF4 * se)}
    return {"reduction_pct": m, "ci_low": m, "ci_high": m}
