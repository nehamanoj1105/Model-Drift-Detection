"""
Evaluation and statistical methodology for the REVISED Experiment 9A.

Window-level macro-F1 / accuracy / precision / recall, transition recovery
analysis (95% of pre-drift F1), paired Wilcoxon signed-rank tests with Cohen's
d effect size and bootstrap confidence intervals on the mean F1 difference.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from exp9a_config import RECOVERY_THRESHOLD, RECOVERY_HORIZON


def window_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def analyze_transitions(df_window, stream_def):
    """Recovery analysis for every regime transition.

    Uses the transition window list from the stream definition. pre-drift F1 is
    the mean over the 5 windows before the transition; post-drift F1 is the
    first window after; recovery time is the number of windows until F1 returns
    to >= 95% of pre-drift F1 (capped at RECOVERY_HORIZON).
    """
    records = []
    for seg in stream_def["segments"][1:]:
        trans_win = seg["start_window"]
        df_m = df_window[df_window["window_id"] == trans_win]
        if df_m.empty:
            continue
        idx = df_m.index[0]
        pre = df_window.loc[max(df_window.index[0], idx - 5):idx - 1, "macro_f1"]
        pre_f1 = float(pre.mean()) if len(pre) else float(df_window.loc[idx, "macro_f1"])
        post_slice = df_window.loc[idx:idx + RECOVERY_HORIZON - 1, "macro_f1"].values
        if len(post_slice) == 0:
            continue
        post_f1 = float(post_slice[0])
        min_post_f1 = float(np.min(post_slice))
        target = RECOVERY_THRESHOLD * pre_f1
        rec_windows = RECOVERY_HORIZON
        for off, v in enumerate(post_slice):
            if v >= target:
                rec_windows = off + 1
                break
        records.append({
            "regime_id": seg["regime_id"],
            "transition_window": int(trans_win),
            "pre_drift_f1": pre_f1,
            "post_drift_f1": post_f1,
            "min_post_f1": min_post_f1,
            "delta_f1": post_f1 - pre_f1,
            "recovery_windows": rec_windows,
            "recovery_f1": float(post_slice[min(rec_windows - 1, len(post_slice) - 1)]),
        })
    return pd.DataFrame(records)


def _bootstrap_ci(diff, n_boot=2000, seed=0):
    if len(diff) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_wilcoxon_tests(df_window, comparisons):
    """Paired window-level Wilcoxon signed-rank tests over pooled seeds.

    Each window contributes the difference in per-window correctness between
    the two methods for the same seed/window. Reports mean difference, Cohen's
    d, p-value, effect size r, and a bootstrap 95% CI on the mean difference.
    """
    results = []
    seeds = sorted(df_window["seed"].unique())
    for m1, m2 in comparisons:
        diffs = []
        for seed in seeds:
            d = df_window[df_window["seed"] == seed]
            a = d[d["method"] == m1].sort_values("window_id")
            b = d[d["method"] == m2].sort_values("window_id")
            if len(a) and len(b):
                n = min(len(a), len(b))
                diffs.extend((a["is_correct"].values[:n] - b["is_correct"].values[:n]).astype(float))
        diffs = np.array(diffs)
        if len(diffs) == 0:
            continue
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs) + 1e-9)
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            try:
                _, p_val = stats.wilcoxon(diffs, zero_method="pratt")
            except Exception:
                p_val = 1.0
        n = len(diffs)
        z = stats.norm.ppf(1 - p_val / 2) if 0 < p_val < 1 else 0.0
        effect_r = float(z / np.sqrt(n)) if n else 0.0
        ci_lo, ci_hi = _bootstrap_ci(diffs)
        results.append({
            "comparison": f"{m1} vs {m2}",
            "method_1": m1,
            "method_2": m2,
            "mean_diff": mean_diff,
            "cohen_d": float(mean_diff / std_diff),
            "p_value": float(p_val),
            "effect_size_r": effect_r,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "n_windows": n,
            "significant": bool(p_val < 0.05),
        })
    return pd.DataFrame(results)
