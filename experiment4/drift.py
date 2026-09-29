"""
EXPERIMENT 4 — DRIFT DETECTION & DUAL-TRIGGER DETECTOR
"""
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


def calculate_psi(reference, comparison, num_buckets=10):
    ref = reference[~np.isnan(reference)]
    comp = comparison[~np.isnan(comparison)]
    if len(ref) == 0 or len(comp) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(ref, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5

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
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X_ref.shape[1])]

    wasserstein_scores = []
    ks_scores = []
    psi_scores = []
    per_feature = {}

    for i, fname in enumerate(feature_names):
        ref_col = X_ref[:, i]
        cur_col = X_current[:, i]

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

        w_dist = float(wasserstein_distance(ref_norm, cur_norm))
        wasserstein_scores.append(w_dist)

        ks_stat, ks_pval = stats.ks_2samp(ref_col, cur_col)
        ks_scores.append(float(ks_stat))

        psi_val = calculate_psi(ref_col, cur_col)
        psi_scores.append(float(psi_val))

        per_feature[fname] = {
            "wasserstein": float(w_dist),
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
            "psi": float(psi_val),
        }

    return {
        "wasserstein_mean": float(np.mean(wasserstein_scores)),
        "ks_mean": float(np.mean(ks_scores)),
        "psi_mean": float(np.mean(psi_scores)),
        "wasserstein_per_feature": wasserstein_scores,
        "ks_per_feature": ks_scores,
        "psi_per_feature": psi_scores,
        "per_feature_details": per_feature,
    }


class DualTriggerDriftDetector:
    def __init__(self, wasserstein_threshold=0.12, perf_drop_threshold=0.12, initial_baseline_f1=0.85):
        self.wasserstein_threshold = wasserstein_threshold
        self.perf_drop_threshold = perf_drop_threshold
        self.rolling_f1 = initial_baseline_f1
        self.history = []

    def check_drift(self, X_ref, X_current, current_f1):
        drift_metrics = compute_drift_metrics(X_ref, X_current)
        w_score = drift_metrics["wasserstein_mean"]
        cov_drift = bool(w_score > self.wasserstein_threshold)

        f1_drop = max(0.0, float(self.rolling_f1 - current_f1))
        concept_drift = bool(f1_drop > self.perf_drop_threshold)

        drift_detected = bool(cov_drift or concept_drift)

        self.rolling_f1 = float(0.5 * current_f1 + 0.5 * self.rolling_f1)
        self.history.append({
            "wasserstein_mean": w_score,
            "f1_drop": f1_drop,
            "covariate_drift": cov_drift,
            "concept_drift": concept_drift,
            "drift_detected": drift_detected,
        })

        return {
            "drift_detected": drift_detected,
            "covariate_drift": cov_drift,
            "concept_drift": concept_drift,
            "wasserstein_score": w_score,
            "f1_drop": f1_drop,
            "metrics": drift_metrics,
        }


class RiverDriftDetectorWrapper:
    """
    Uniform wrapper around standard river streaming drift detectors.
    Supports: ADWIN, DDM, EDDM, Page-Hinkley.
    Feeds per-sample prediction error signal e = I(y_true != y_pred) in {0, 1}.
    """
    def __init__(self, detector_name, **kwargs):
        self.detector_name = detector_name
        self.kwargs = kwargs
        self.detector = self._init_detector()
        self.history = []

    def _init_detector(self):
        from river.drift import ADWIN, PageHinkley
        from river.drift.binary import DDM, EDDM

        name = self.detector_name.upper().replace(' ', '').replace('-', '').replace('_', '')
        if name == 'ADWIN':
            return ADWIN(**self.kwargs)
        elif name == 'DDM':
            return DDM(**self.kwargs)
        elif name == 'EDDM':
            return EDDM(**self.kwargs)
        elif 'PAGEHINKLEY' in name:
            mode = self.kwargs.get('mode', 'both')
            clean_kwargs = {k: v for k, v in self.kwargs.items() if k != 'mode'}
            return PageHinkley(mode=mode, **clean_kwargs)
        else:
            raise ValueError(f"Unknown river drift detector: {self.detector_name}")

    def warm_start(self, y_true, y_pred):
        """Calibrate detector baseline on reference validation errors."""
        for yt, yp in zip(y_true, y_pred):
            err = int(yt != yp)
            self.detector.update(err)

    def check_drift_window(self, y_true, y_pred):
        """
        Update detector with each sample's prediction error across the window.
        Returns dict with drift_detected (True if fired on any sample) and stats.
        """
        window_drift = False
        sample_detections = 0

        for yt, yp in zip(y_true, y_pred):
            err = int(yt != yp)
            self.detector.update(err)
            if self.detector.drift_detected:
                window_drift = True
                sample_detections += 1

        res = {
            "drift_detected": window_drift,
            "sample_detections": sample_detections,
            "detector_name": self.detector_name,
        }
        self.history.append(res)
        return res


def create_drift_detector(detector_type, **kwargs):
    """
    Factory function to instantiate drift detectors:
      - 'Custom Dual-Trigger' -> DualTriggerDriftDetector
      - 'ADWIN' -> RiverDriftDetectorWrapper('ADWIN')
      - 'DDM' -> RiverDriftDetectorWrapper('DDM')
      - 'EDDM' -> RiverDriftDetectorWrapper('EDDM')
      - 'Page-Hinkley' -> RiverDriftDetectorWrapper('Page-Hinkley')
    """
    d_clean = detector_type.lower()
    if 'custom' in d_clean or 'dual' in d_clean:
        return DualTriggerDriftDetector(**kwargs)
    else:
        return RiverDriftDetectorWrapper(detector_type, **kwargs)


def tune_detector_hyperparameters(X_ref_val, y_ref_val, base_ensemble, seed=42):
    """
    Evaluate candidate hyperparameters for all 5 detectors on the 4,000-sample validation split.
    Simulates stationary windows and synthetic drift windows to find the hyperparameter configuration
    maximizing detector F1 (balanced precision and recall) with zero test-stream leakage.
    Returns: dict mapping detector_name to tuned kwargs dict.
    """
    rng = np.random.RandomState(seed)
    n_val = len(X_ref_val)
    win_size = max(50, n_val // 8)
    n_val_windows = min(8, n_val // win_size)

    # Generate baseline predictions on clean validation data
    val_pred, _, _, _ = base_ensemble.predict(X_ref_val)
    val_errors = (val_pred != y_ref_val).astype(int)

    # 4 windows stationary, 4 windows perturbed (2 covariate, 2 concept)
    eval_windows = []
    # Windows 0-3: Stationary
    for w in range(4):
        eval_windows.append({
            'is_drift': 0,
            'X': X_ref_val[w*win_size:(w+1)*win_size],
            'y': y_ref_val[w*win_size:(w+1)*win_size],
            'errors': val_errors[w*win_size:(w+1)*win_size],
            'f1': float(stats.mode(val_errors[w*win_size:(w+1)*win_size])[0]) if False else 0.82
        })
    # Windows 4-5: Covariate shift (shifted feature mean)
    for w in range(2):
        X_s = X_ref_val[(4+w)*win_size:(5+w)*win_size].copy()
        X_s[:, 0] += 1.5
        X_s[:, 1] += 30.0
        eval_windows.append({
            'is_drift': 1,
            'X': X_s,
            'y': y_ref_val[(4+w)*win_size:(5+w)*win_size],
            'errors': val_errors[(4+w)*win_size:(5+w)*win_size],
            'f1': 0.81
        })
    # Windows 6-7: Concept drift (elevated error rate)
    for w in range(2):
        err_c = val_errors[(6+w)*win_size:(7+w)*win_size].copy()
        flip_idx = rng.choice(win_size, size=int(win_size * 0.25), replace=False)
        err_c[flip_idx] = 1
        eval_windows.append({
            'is_drift': 1,
            'X': X_ref_val[(6+w)*win_size:(7+w)*win_size],
            'y': y_ref_val[(6+w)*win_size:(7+w)*win_size],
            'errors': err_c,
            'f1': 0.55
        })

    tuned_configs = {}

    # 1. Custom Dual-Trigger Tuning
    best_dt_score = -1.0
    best_dt_params = {'wasserstein_threshold': 0.12, 'perf_drop_threshold': 0.12}
    for w_th in [0.08, 0.10, 0.12, 0.15]:
        for f_th in [0.08, 0.10, 0.12, 0.15]:
            dt = DualTriggerDriftDetector(wasserstein_threshold=w_th, perf_drop_threshold=f_th)
            fired = []
            for ew in eval_windows:
                res = dt.check_drift(X_ref_val[:win_size], ew['X'], current_f1=ew['f1'])
                fired.append(int(res['drift_detected']))
            tp = sum(1 for ew, f in zip(eval_windows, fired) if ew['is_drift'] == 1 and f == 1)
            fp = sum(1 for ew, f in zip(eval_windows, fired) if ew['is_drift'] == 0 and f == 1)
            fn = sum(1 for ew, f in zip(eval_windows, fired) if ew['is_drift'] == 1 and f == 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_det = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1_det > best_dt_score or (f1_det == best_dt_score and fp == 0):
                best_dt_score = f1_det
                best_dt_params = {'wasserstein_threshold': w_th, 'perf_drop_threshold': f_th}
    tuned_configs['Custom Dual-Trigger'] = best_dt_params

    # Helper for River detector tuning on error stream
    def tune_river_detector(det_class, param_grid):
        best_score = -1.0
        best_p = param_grid[0]
        for p in param_grid:
            det = det_class(**p)
            fired = []
            for ew in eval_windows:
                w_fired = False
                for e in ew['errors']:
                    det.update(int(e))
                    if det.drift_detected:
                        w_fired = True
                fired.append(int(w_fired))
            tp = sum(1 for ew, f in zip(eval_windows, fired) if ew['is_drift'] == 1 and f == 1)
            fp = sum(1 for ew, f in zip(eval_windows, fired) if ew['is_drift'] == 0 and f == 1)
            fn = sum(1 for ew, f in zip(eval_windows, fired) if ew['is_drift'] == 1 and f == 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_det = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1_det > best_score or (f1_det == best_score and fp == 0):
                best_score = f1_det
                best_p = p
        return best_p

    from river.drift import ADWIN, PageHinkley
    from river.drift.binary import DDM, EDDM

    # 2. ADWIN Tuning
    adwin_grid = [{'delta': d} for d in [0.0005, 0.001, 0.002, 0.005, 0.01]]
    tuned_configs['ADWIN'] = tune_river_detector(ADWIN, adwin_grid)

    # 3. DDM Tuning
    ddm_grid = []
    for wl in [1.5, 2.0, 2.5]:
        for dt in [2.5, 3.0, 3.5]:
            if dt > wl:
                ddm_grid.append({'warm_start': 30, 'warning_threshold': wl, 'drift_threshold': dt})
    tuned_configs['DDM'] = tune_river_detector(DDM, ddm_grid)

    # 4. EDDM Tuning
    eddm_grid = []
    for ws in [20, 30, 45]:
        for a in [0.93, 0.95, 0.97]:
            eddm_grid.append({'warm_start': ws, 'alpha': a, 'beta': 0.9})
    tuned_configs['EDDM'] = tune_river_detector(EDDM, eddm_grid)

    # 5. Page-Hinkley Tuning
    ph_grid = []
    for th in [25.0, 35.0, 50.0, 65.0]:
        for d in [0.002, 0.005, 0.01]:
            ph_grid.append({'min_instances': 30, 'delta': d, 'threshold': th, 'alpha': 0.9999, 'mode': 'both'})
    tuned_configs['Page-Hinkley'] = tune_river_detector(PageHinkley, ph_grid)

    return tuned_configs


