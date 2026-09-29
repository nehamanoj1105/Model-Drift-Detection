"""
================================================================================
EXPERIMENT 3 — DRIFT DETECTION & MONITORING
================================================================================
Calculates for every streaming window:
- Normalized Wasserstein distance (primary feature drift metric)
- Kolmogorov-Smirnov (KS) statistic
- Population Stability Index (PSI)

Performance-based drift metrics:
- Rolling F1
- Rolling error rate (1 - Accuracy)
- Performance degradation (Baseline F1 - Current F1)

Analyzes whether:
- Covariate drift is detected by feature metrics
- Concept drift is missed by feature metrics
- Performance monitoring detects concept drift
================================================================================
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


def calculate_psi(reference, comparison, num_buckets=10):
    """
    Compute Population Stability Index (PSI) between two 1D arrays.
    """
    ref = reference[~np.isnan(reference)]
    comp = comparison[~np.isnan(comparison)]
    if len(ref) == 0 or len(comp) == 0:
        return 0.0
        
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(ref, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5
    
    # Ensure strictly increasing bucket edges
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
    Compute feature-level drift metrics between reference (initial training)
    and current streaming window data.
    """
    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(X_ref.shape[1])]
        
    wasserstein_scores = []
    ks_scores = []
    psi_scores = []
    per_feature = {}
    
    for i, fname in enumerate(feature_names):
        ref_col = X_ref[:, i]
        cur_col = X_current[:, i]
        
        # Normalized Wasserstein distance to [0, 1] range
        combined = np.concatenate([ref_col, cur_col])
        feat_min = np.min(combined)
        feat_max = np.max(combined)
        feat_range = feat_max - feat_min
        
        if feat_range > 1e-8:
            ref_norm = (ref_col - feat_min) / feat_range
            cur_norm = (cur_col - feat_min) / feat_range
        else:
            ref_norm = ref_col
            cur_norm = cur_col
            
        w_dist = wasserstein_distance(ref_norm, cur_norm)
        wasserstein_scores.append(w_dist)
        
        # Two-sample KS test
        ks_stat, ks_pval = stats.ks_2samp(ref_col, cur_col)
        ks_scores.append(ks_stat)
        
        # PSI
        psi_val = calculate_psi(ref_col, cur_col)
        psi_scores.append(psi_val)
        
        per_feature[fname] = {
            'wasserstein': float(w_dist),
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval),
            'psi': float(psi_val)
        }
        
    return {
        'wasserstein_mean': float(np.mean(wasserstein_scores)),
        'ks_mean': float(np.mean(ks_scores)),
        'psi_mean': float(np.mean(psi_scores)),
        'wasserstein_per_feature': wasserstein_scores,
        'ks_per_feature': ks_scores,
        'psi_per_feature': psi_scores,
        'per_feature_details': per_feature
    }


class DriftMonitor:
    """
    Tracks rolling performance metrics to detect concept drift and degradation.
    """
    def __init__(self, baseline_f1=0.90, window_history_size=3):
        self.baseline_f1 = baseline_f1
        self.history_size = window_history_size
        self.f1_history = []
        self.error_rate_history = []

    def update(self, current_f1, current_accuracy):
        error_rate = 1.0 - current_accuracy
        self.f1_history.append(current_f1)
        self.error_rate_history.append(error_rate)
        
        recent_f1 = self.f1_history[-self.history_size:]
        recent_err = self.error_rate_history[-self.history_size:]
        
        rolling_f1 = float(np.mean(recent_f1))
        rolling_error_rate = float(np.mean(recent_err))
        performance_degradation = float(max(0.0, self.baseline_f1 - current_f1))
        
        return {
            'rolling_f1': rolling_f1,
            'rolling_error_rate': rolling_error_rate,
            'performance_degradation': performance_degradation
        }
