"""
Recovery / robustness / statistical metrics for Experiment 9B drift studies.

All recovery statistics are computed from the saved per-window predictions so
that every figure and table can be reproduced from raw results.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from exp9b_drift_config import RECOVERY_THRESHOLD


def _window_f1(df_windows):
    if len(df_windows) == 0:
        return np.nan
    return f1_score(df_windows["y_true"], df_windows["y_pred"],
                    average="macro", zero_division=0)


def recovery_analysis(df_window, drift_points, n_pre=30, n_post=40,
                      threshold=RECOVERY_THRESHOLD, smooth=5, pre_reference=None):
    """
    Per (seed, method, drift point) recovery statistics.

    * pre-drift F1  : Macro-F1 over the n_pre windows before the drift point
                      (or over `pre_reference[dp]` window range if supplied)
    * min post F1   : minimum rolling Macro-F1 in the n_post windows after drift
    * recovery F1   : Macro-F1 over the sustained post-recovery block
    * recovery time : number of post-drift windows required for the smoothed
                      correctness rate to reach `threshold * pre-drift correctness`
                      AND remain at/above that level for the rest of the window
                      (a *sustained* recovery requirement). If the level is never
                      sustained, recovery is censored at n_post and
                      `recovered = False`.
    """
    records = []
    pre_reference = pre_reference or {}
    for seed, df_s in df_window.groupby("seed"):
        for method, df_m in df_s.groupby("method"):
            df_m = df_m.sort_values("window_id")
            for dp in drift_points:
                if dp in pre_reference:
                    lo, hi = pre_reference[dp]
                    pre = df_m[(df_m["window_id"] >= lo) & (df_m["window_id"] < hi)]
                else:
                    pre = df_m[(df_m["window_id"] >= dp - n_pre) & (df_m["window_id"] < dp)]
                post = df_m[(df_m["window_id"] >= dp) & (df_m["window_id"] < dp + n_post)]
                if len(pre) == 0 or len(post) == 0:
                    continue

                pre_f1 = _window_f1(pre)
                post_f1 = _window_f1(post)
                post_min = min(
                    _window_f1(post.iloc[i:i + smooth])
                    for i in range(max(1, len(post) - smooth + 1))
                )

                # Sustained-recovery test on the smoothed correctness rate.
                roll = post["is_correct"].rolling(smooth, min_periods=1).mean().values
                pre_correct = pre["is_correct"].mean()
                target_correct = threshold * pre_correct

                rec_windows = n_post
                recovered = False
                rec_f1 = post_f1
                for i in range(len(roll)):
                    if np.all(roll[i:] >= target_correct):
                        rec_windows = i + 1
                        recovered = True
                        rec_f1 = _window_f1(post.iloc[i:])
                        break

                records.append({
                    "seed": int(seed), "method": method, "drift_point": int(dp),
                    "pre_drift_f1": float(pre_f1),
                    "post_drift_f1": float(post_f1),
                    "min_post_f1": float(post_min),
                    "delta_f1": float(post_f1 - pre_f1),
                    "relative_degradation": float((pre_f1 - post_f1) / pre_f1)
                    if pre_f1 > 0 else np.nan,
                    "recovery_f1": float(rec_f1),
                    "recovery_windows": int(rec_windows),
                    "recovered": bool(recovered),
                    "recovery_threshold": float(threshold),
                })
    return pd.DataFrame(records)


def aggregate_recovery(df_rec, by=("method",)):
    """Mean/std recovery metrics grouped by the requested keys."""
    if df_rec.empty:
        return df_rec
    cols = ["pre_drift_f1", "post_drift_f1", "min_post_f1", "delta_f1",
            "relative_degradation", "recovery_f1", "recovery_windows"]
    g = df_rec.groupby(list(by))
    out = g[cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([c for c in col if c]) for col in out.columns]
    return out


def paired_wilcoxon(df_window, m1, m2, metric="is_correct"):
    """
    Paired Wilcoxon signed-rank test of window-level performance between two
    methods across all seeds, plus Cohen's d and a bootstrap 95% CI on the mean
    difference. Pairs are matched on (seed, window_id).
    """
    merged = (df_window[df_window["method"] == m1]
              .merge(df_window[df_window["method"] == m2],
                     on=["seed", "window_id"], suffixes=("_1", "_2")))
    if merged.empty:
        return None
    d = merged[f"{metric}_1"].values.astype(float) - merged[f"{metric}_2"].values.astype(float)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d)) + 1e-12
    if np.allclose(d, 0):
        stat, p = 0.0, 1.0
    else:
        try:
            stat, p = stats.wilcoxon(d, zero_method="pratt")
            stat, p = float(stat), float(p)
        except Exception:
            stat, p = 0.0, 1.0

    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(2000)]
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return {
        "method1": m1, "method2": m2, "n_pairs": int(len(d)),
        "mean_diff": mean_d, "std_diff": std_d,
        "cohen_d": mean_d / std_d,
        "wilcoxon_stat": stat, "p_value": p,
        "ci95_low": ci_lo, "ci95_high": ci_hi,
        "significant_p05": bool(p < 0.05),
    }


def seed_summary(df_per_seed, by=("scenario", "drift_level", "method")):
    """mean ± std of every metric across seeds, grouped by the requested keys."""
    metric_cols = ["macro_f1", "accuracy", "precision", "recall",
                   "balanced_accuracy", "total_cpu_sec", "adaptation_cpu_sec",
                   "retrain_events", "reused_checkpoints", "trees_trained",
                   "trees_reused"]
    g = df_per_seed.groupby(list(by))
    out = g[metric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([c for c in col if c]) for col in out.columns]
    return out


def per_window_f1_series(df_window, smooth=10):
    """Rolling Macro-F1 proxy per (seed, method) — rolling correctness rate."""
    rows = []
    for (seed, method), df_m in df_window.groupby(["seed", "method"]):
        df_m = df_m.sort_values("window_id")
        roll = df_m["is_correct"].rolling(smooth, min_periods=1).mean()
        for wid, val in zip(df_m["window_id"].values, roll.values):
            rows.append({"seed": seed, "method": method,
                         "window_id": int(wid), "roll_correct": float(val)})
    return pd.DataFrame(rows)
