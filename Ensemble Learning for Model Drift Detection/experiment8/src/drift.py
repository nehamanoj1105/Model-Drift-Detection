"""
================================================================================
EXPERIMENT 8 — DRIFT DETECTION MODULE
================================================================================
Implements feature drift detection (Normalized Wasserstein Distance) and
performance degradation detection (Rolling Macro F1 drop).
================================================================================
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


def compute_normalized_wasserstein(X_ref, X_cur):
    """
    Computes mean normalized Wasserstein distance across feature columns.
    """
    scores = []
    n_feats = X_ref.shape[1]
    for i in range(n_feats):
        ref_col = X_ref[:, i]
        cur_col = X_cur[:, i]
        
        combined = np.concatenate([ref_col, cur_col])
        f_min = np.min(combined)
        f_max = np.max(combined)
        f_range = f_max - f_min
        
        if f_range > 1e-8:
            r_norm = (ref_col - f_min) / f_range
            c_norm = (cur_col - f_min) / f_range
        else:
            r_norm, c_norm = ref_col, cur_col
            
        w_dist = wasserstein_distance(r_norm, c_norm)
        scores.append(w_dist)
        
    return float(np.mean(scores))


class DriftDetector:
    """
    Event-driven drift detector combining feature drift and performance degradation.
    """
    def __init__(self, wasserstein_threshold=0.12, degradation_threshold=0.05, window_history=3):
        self.w_threshold = wasserstein_threshold
        self.deg_threshold = degradation_threshold
        self.window_history = window_history
        self.f1_history = []
        self.baseline_f1 = None

    def set_baseline(self, baseline_f1):
        self.baseline_f1 = baseline_f1

    def update(self, X_ref, X_cur, current_f1):
        if self.baseline_f1 is None:
            self.baseline_f1 = current_f1

        self.f1_history.append(current_f1)
        recent_f1 = np.mean(self.f1_history[-self.window_history:])
        
        # 1. Feature drift check
        w_dist = compute_normalized_wasserstein(X_ref, X_cur)
        feature_drift = w_dist >= self.w_threshold

        # 2. Performance degradation check
        deg = max(0.0, self.baseline_f1 - current_f1)
        perf_drift = deg >= self.deg_threshold

        is_drift = feature_drift or perf_drift

        return {
            'is_drift': is_drift,
            'wasserstein_distance': w_dist,
            'feature_drift': feature_drift,
            'performance_drift': perf_drift,
            'degradation': deg,
            'rolling_f1': recent_f1,
        }
