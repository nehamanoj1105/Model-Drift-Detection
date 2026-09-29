"""
================================================================================
EXPERIMENT 5 — DRIFT DETECTION & DUAL-TRIGGER DETECTOR
================================================================================
Implements:
  1. Statistical Distributional Drift Metrics:
     - Normalized Wasserstein distance (primary feature drift metric)
     - Kolmogorov-Smirnov (KS) two-sample test
     - Population Stability Index (PSI)
  2. Dual-Trigger Drift Detector for Model 3 (Event-Driven Ensemble):
     - Trigger A (Covariate): Normalized Wasserstein distance > tau_w (0.12)
     - Trigger B (Concept / Performance Drop): Prequential F1 drop > tau_p (0.12)
================================================================================
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance

from config import (
    DRIFT_WASSERSTEIN_THRESHOLD,
    DRIFT_PERF_DROP_THRESHOLD,
    FEATURE_COLS
)


def calculate_psi(reference, comparison, num_buckets=10):
    """
    Compute Population Stability Index (PSI) between reference and comparison arrays.
    """
    ref = reference[~np.isnan(reference)]
    comp = comparison[~np.isnan(comparison)]
    if len(ref) == 0 or len(comp) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(ref, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5

    # Ensure strictly increasing edges
    for b in range(1, len(bucket_edges)):
        if bucket_edges[b] <= bucket_edges[b - 1]:
            bucket_edges[b] = bucket_edges[b - 1] + 1e-5

    ref_counts, _ = np.histogram(ref, bins=bucket_edges)
    comp_counts, _ = np.histogram(comp, bins=bucket_edges)

    ref_pct = np.maximum(ref_counts / len(ref), 1e-4)
    comp_pct = np.maximum(comp_counts / len(comp), 1e-4)

    psi = np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct))
    return float(np.clip(psi, 0.0, 10.0))


def compute_drift_metrics(X_ref, X_current, feature_names=None):
    """
    Compute feature-level drift metrics between reference (initial stationary data)
    and current streaming window.
    """
    if feature_names is None:
        feature_names = FEATURE_COLS

    wasserstein_scores = []
    ks_scores = []
    psi_scores = []
    per_feature = {}

    for i, fname in enumerate(feature_names):
        ref_col = X_ref[:, i]
        cur_col = X_current[:, i]

        combined = np.concatenate([ref_col, cur_col])
        f_min = np.min(combined)
        f_max = np.max(combined)
        f_range = f_max - f_min

        if f_range > 1e-8:
            ref_norm = (ref_col - f_min) / f_range
            cur_norm = (cur_col - f_min) / f_range
        else:
            ref_norm = ref_col
            cur_norm = cur_col

        w_dist = float(wasserstein_distance(ref_norm, cur_norm))
        wasserstein_scores.append(w_dist)

        ks_stat, ks_pval = stats.ks_2samp(ref_col, cur_col)
        ks_scores.append(float(ks_stat))

        psi_val = calculate_psi(ref_col, cur_col)
        psi_scores.append(psi_val)

        per_feature[fname] = {
            'wasserstein': w_dist,
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval),
            'psi': psi_val
        }

    return {
        'wasserstein_mean': float(np.mean(wasserstein_scores)),
        'ks_mean': float(np.mean(ks_scores)),
        'psi_mean': float(np.mean(psi_scores)),
        'per_feature': per_feature,
    }


class DualTriggerDriftDetector:
    """
    Active dual-trigger drift monitor for Model 3 (Event-Driven Ensemble):
      - Covariate Trigger: Normalized Wasserstein distance > tau_w
      - Concept Trigger: Prequential F1 performance degradation > tau_p
    """

    def __init__(self,
                 wasserstein_threshold=DRIFT_WASSERSTEIN_THRESHOLD,
                 perf_drop_threshold=DRIFT_PERF_DROP_THRESHOLD,
                 initial_baseline_f1=0.85):
        self.w_threshold = wasserstein_threshold
        self.p_threshold = perf_drop_threshold
        self.prev_f1 = initial_baseline_f1
        self.detection_history = []

    def check_drift(self, X_ref, X_current, current_f1):
        """
        Evaluate drift on current window.
        Returns a dictionary with detection decision and diagnostic signals.
        """
        drift_stats = compute_drift_metrics(X_ref, X_current)
        w_dist = drift_stats['wasserstein_mean']

        covariate_drift = bool(w_dist > self.w_threshold)
        f1_drop = float(max(0.0, self.prev_f1 - current_f1))
        concept_drift = bool(f1_drop > self.p_threshold)

        drift_detected = covariate_drift or concept_drift

        reasons = []
        if covariate_drift:
            reasons.append(f"Covariate drift (W={w_dist:.4f} > {self.w_threshold})")
        if concept_drift:
            reasons.append(f"Performance drop (Delta_F1={f1_drop:.4f} > {self.p_threshold})")

        decision = {
            'drift_detected': drift_detected,
            'covariate_drift': covariate_drift,
            'concept_drift': concept_drift,
            'trigger_reasons': "; ".join(reasons) if reasons else "None",
            'wasserstein_mean': w_dist,
            'ks_mean': drift_stats['ks_mean'],
            'psi_mean': drift_stats['psi_mean'],
            'f1_drop': f1_drop,
            'prev_f1': self.prev_f1,
            'current_f1': current_f1,
        }

        self.prev_f1 = current_f1
        self.detection_history.append(decision)
        return decision
