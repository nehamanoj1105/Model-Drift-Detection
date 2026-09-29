"""
================================================================================
EXPERIMENT 8 — METRICS & PERFORMANCE EVALUATION
================================================================================
Calculates predictive metrics, post-drift recovery metrics, computational costs,
adaptation efficiency ratios, and Pareto frontier.
================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def evaluate_window_predictions(y_true, y_pred, y_prob=None):
    """Computes window-level predictive metrics."""
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    return {
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
    }


def compute_recovery_metrics(window_f1s, drift_window_indices, baseline_f1=0.90):
    """
    Computes post-drift recovery metrics:
      - F1 at +1, +2, +3, +5 windows
      - Recovery time to 90% of baseline_f1
      - Recovery time to 95% of baseline_f1
    """
    if not drift_window_indices:
        return {
            'f1_plus_1': float(np.mean(window_f1s)),
            'f1_plus_2': float(np.mean(window_f1s)),
            'f1_plus_3': float(np.mean(window_f1s)),
            'f1_plus_5': float(np.mean(window_f1s)),
            'recovery_time_90': 0.0,
            'recovery_time_95': 0.0,
        }

    f1_p1, f1_p2, f1_p3, f1_p5 = [], [], [], []
    rec_times_90, rec_times_95 = [], []

    target_90 = 0.90 * baseline_f1
    target_95 = 0.95 * baseline_f1

    n_windows = len(window_f1s)

    for d_idx in drift_window_indices:
        if d_idx + 1 < n_windows: f1_p1.append(window_f1s[d_idx + 1])
        if d_idx + 2 < n_windows: f1_p2.append(window_f1s[d_idx + 2])
        if d_idx + 3 < n_windows: f1_p3.append(window_f1s[d_idx + 3])
        if d_idx + 5 < n_windows: f1_p5.append(window_f1s[d_idx + 5])

        # Find windows to recover
        t_90 = None
        t_95 = None
        for step in range(1, n_windows - d_idx):
            f_val = window_f1s[d_idx + step]
            if t_90 is None and f_val >= target_90:
                t_90 = step
            if t_95 is None and f_val >= target_95:
                t_95 = step
            if t_90 is not None and t_95 is not None:
                break

        rec_times_90.append(t_90 if t_90 is not None else (n_windows - d_idx))
        rec_times_95.append(t_95 if t_95 is not None else (n_windows - d_idx))

    return {
        'f1_plus_1': float(np.mean(f1_p1)) if f1_p1 else float(np.mean(window_f1s)),
        'f1_plus_2': float(np.mean(f1_p2)) if f1_p2 else float(np.mean(window_f1s)),
        'f1_plus_3': float(np.mean(f1_p3)) if f1_p3 else float(np.mean(window_f1s)),
        'f1_plus_5': float(np.mean(f1_p5)) if f1_p5 else float(np.mean(window_f1s)),
        'recovery_time_90': float(np.mean(rec_times_90)),
        'recovery_time_95': float(np.mean(rec_times_95)),
    }


def compute_adaptation_efficiency(window_f1s, drift_window_indices, total_adaptation_cpu):
    """
    Computes AdaptationEfficiency = RecoveryGain / (AdaptationCPU + 1e-5)
    where RecoveryGain is the cumulative F1 improvement post-drift over post-drift window baseline.
    """
    if not drift_window_indices:
        return 0.0

    recovery_gain = 0.0
    n_windows = len(window_f1s)
    for d_idx in drift_window_indices:
        f1_at_drift = window_f1s[d_idx]
        for step in range(1, min(4, n_windows - d_idx)):
            recovery_gain += max(0.0, window_f1s[d_idx + step] - f1_at_drift)

    return float(recovery_gain / (total_adaptation_cpu + 1e-5))


def compute_pareto_frontier(points):
    """
    Given a list of dicts with 'method', 'f1', 'adaptation_cpu',
    identifies non-dominated points maximizing F1 and minimizing CPU.
    """
    sorted_pts = sorted(points, key=lambda p: p['adaptation_cpu'])
    pareto = []
    max_f1 = -1.0

    for pt in sorted_pts:
        if pt['f1'] > max_f1:
            pareto.append(pt)
            max_f1 = pt['f1']

    return pareto
