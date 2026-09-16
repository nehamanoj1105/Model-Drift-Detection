"""
================================================================================
EXPERIMENT 3A — FROZEN MODEL DEGRADATION & RETRAINING COST UNDER DRIFT
================================================================================
Establishes the empirical foundation for later adaptive ensemble experiments:

1. A frozen Model 1 degrades under distribution shift.
2. Different types/severities of drift produce different degradation patterns.
3. Retraining can recover performance.
4. Retraining introduces measurable computational cost.
5. Repeated retraining accumulates substantial CPU/time/resource cost.

Dataset:     10,000 raw samples → ~2,000 feature windows
Task:        Binary QoS violation classification (same as Experiment 2)
KPIs:        speed, distance, delay, throughput
Drift:       Covariate (P(X)) and Concept (P(Y|X))
Metrics:     Wasserstein distance (primary), KS statistic, PSI
Models:      Frozen RF (Model 1) vs Retrained RF baseline
Monitoring:  psutil process-level CPU/memory measurement

NOTE: This experiment is COMPLETELY INDEPENDENT from Experiments 1 and 2.
No code is imported from other experiments.
================================================================================
"""

import os
import sys
import json
import time
import threading
import platform
import hashlib
import warnings
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import psutil
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Experiment identity
EXPERIMENT_ID = "3A"
EXPERIMENT_NAME = "Frozen Model Degradation & Retraining Cost Under Drift"

# Dataset parameters
N_RAW_SAMPLES = 10000
N_INITIAL_TRAINING = 2000
N_POST_DEPLOYMENT = 8000

# Feature extraction
KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
TARGET_COL = 'qos_violation'
WINDOW_SIZE = 20
STEP_SIZE = 5

# Evaluation window size (number of feature windows per evaluation chunk)
EVAL_WINDOW_SIZE = 100

# Model parameters (same architecture as Experiment 2 for consistency)
MODEL_PARAMS = {
    'n_estimators': 50,
    'max_depth': 7,
}

# Multi-seed evaluation
EVAL_SEEDS = [42, 43, 44, 45, 46]

# Full drift magnitudes (same physical units as Experiment 2)
COVARIATE_DRIFT_FULL = {
    'speed':      +1.0,     # m/s
    'distance':   +30.0,    # m
    'delay':      +8.0,     # ms
    'throughput': -20.0,    # Mbps
}

# Concept drift coefficients modification (changes P(Y|X))
CONCEPT_DRIFT_COEFF_SHIFTS = {
    'mild':     {'dist_w': +0.3, 'delay_w': +0.3, 'tp_w': -0.2, 'speed_w': +0.1, 'intercept': +0.5},
    'moderate': {'dist_w': +0.6, 'delay_w': +0.8, 'tp_w': -0.5, 'speed_w': +0.2, 'intercept': +1.2},
    'severe':   {'dist_w': +1.2, 'delay_w': +1.5, 'tp_w': -1.0, 'speed_w': +0.5, 'intercept': +2.5},
}

# Covariate drift magnitudes (fraction of FULL)
COVARIATE_DRIFT_MAGNITUDES = {
    'mild':     0.15,
    'moderate': 0.35,
    'severe':   0.60,
}

# Drift severity thresholds (Wasserstein distance based, validated empirically)
DRIFT_SEVERITY_THRESHOLDS = {
    'none':     (0.00, 0.05),
    'mild':     (0.05, 0.20),
    'moderate': (0.20, 0.50),
    'severe':   (0.50, float('inf')),
}

# Experiment conditions
EXPERIMENT_CONDITIONS = [
    {'id': '3A-0', 'drift_type': 'none',      'severity': 'none',     'description': 'No drift (stationary baseline)'},
    {'id': '3A-1', 'drift_type': 'covariate', 'severity': 'mild',     'description': 'Mild covariate drift'},
    {'id': '3A-2', 'drift_type': 'covariate', 'severity': 'moderate', 'description': 'Moderate covariate drift'},
    {'id': '3A-3', 'drift_type': 'covariate', 'severity': 'severe',   'description': 'Severe covariate drift'},
    {'id': '3A-4', 'drift_type': 'concept',   'severity': 'mild',     'description': 'Mild concept drift'},
    {'id': '3A-5', 'drift_type': 'concept',   'severity': 'moderate', 'description': 'Moderate concept drift'},
    {'id': '3A-6', 'drift_type': 'concept',   'severity': 'severe',   'description': 'Severe concept drift'},
]

# Output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 1. ENVIRONMENT CAPTURE
# ============================================================

def capture_environment():
    """Record hardware and software environment for reproducibility."""
    env = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'cpu_count_logical': os.cpu_count(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'timestamp': datetime.now().isoformat(),
    }
    try:
        import sklearn
        env['sklearn_version'] = sklearn.__version__
    except ImportError:
        env['sklearn_version'] = 'N/A'
    env['numpy_version'] = np.__version__
    env['pandas_version'] = pd.__version__
    env['psutil_version'] = psutil.__version__ if hasattr(psutil, '__version__') else 'N/A'
    return env


# ============================================================
# 2. DATASET GENERATION — STATIONARY BASELINE
# ============================================================

def generate_stationary_dataset(n_samples, seed):
    """
    Generate a stationary 4-KPI telemetry dataset.
    Same generating process as Experiment 2 to preserve the prediction task.
    Target rule: logistic combination of z-scored KPIs.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples)

    # Speed (m/s): base ~ 1.5, slight route cycle, noise
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, n_samples)
    speed = np.clip(speed, 0.5, 3.5)

    # Distance to gNB (m): triangular warehouse path
    cycle = 800
    tri = 2 * np.abs((t % cycle) / cycle - 0.5)
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, n_samples)
    distance = np.clip(distance, 10.0, 120.0)

    # Delay (ms): correlated with distance + log-normal spikes
    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay_noise = rng.lognormal(mean=0.0, sigma=0.4, size=n_samples)
    delay = delay_base + delay_noise
    delay = np.clip(delay, 2.0, 50.0)

    # Throughput (Mbps): inversely related to distance and delay
    path_loss_factor = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    tp_base = 80.0 * path_loss_factor
    tp_noise = rng.normal(0, 8.0, n_samples)
    throughput = tp_base + tp_noise
    throughput = np.clip(throughput, 5.0, 150.0)

    # TARGET: QoS Violation (binary) — logistic model
    dist_z = (distance - 55.0) / 25.0
    delay_z = (delay - 12.0) / 6.0
    tp_z = (throughput - 55.0) / 20.0
    speed_z = (speed - 1.5) / 0.3

    linear_comb = 0.8 * dist_z + 1.2 * delay_z - 1.0 * tp_z + 0.3 * speed_z
    prob_violation = 1.0 / (1.0 + np.exp(-linear_comb))
    qos_violation = (rng.uniform(0, 1, n_samples) < prob_violation).astype(int)

    df = pd.DataFrame({
        'timestamp': t,
        'speed': np.round(speed, 4),
        'distance': np.round(distance, 2),
        'delay': np.round(delay, 3),
        'throughput': np.round(throughput, 2),
        TARGET_COL: qos_violation
    })
    return df


def generate_dataset_with_drift(n_samples, seed, drift_type, severity,
                                 drift_onset_sample=None):
    """
    Generate dataset with controlled drift starting at drift_onset_sample.

    Covariate drift: shifts P(X) while preserving the target generation rule.
    Concept drift: modifies the target generation coefficients (P(Y|X) changes).
    """
    if drift_onset_sample is None:
        drift_onset_sample = N_INITIAL_TRAINING

    rng = np.random.RandomState(seed)
    t = np.arange(n_samples)

    # Generate base features (same process for all)
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, n_samples)
    speed = np.clip(speed, 0.5, 3.5)

    cycle = 800
    tri = 2 * np.abs((t % cycle) / cycle - 0.5)
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, n_samples)
    distance = np.clip(distance, 10.0, 120.0)

    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay_noise = rng.lognormal(mean=0.0, sigma=0.4, size=n_samples)
    delay = delay_base + delay_noise
    delay = np.clip(delay, 2.0, 50.0)

    path_loss_factor = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    tp_base = 80.0 * path_loss_factor
    tp_noise = rng.normal(0, 8.0, n_samples)
    throughput = tp_base + tp_noise
    throughput = np.clip(throughput, 5.0, 150.0)

    # Create drift mask (which samples are post-onset)
    drift_mask = t >= drift_onset_sample

    if drift_type == 'covariate' and severity != 'none':
        # Apply covariate shift to post-onset features
        magnitude = COVARIATE_DRIFT_MAGNITUDES[severity]
        drift_rng = np.random.RandomState(seed + 1000)

        for kpi, full_shift in COVARIATE_DRIFT_FULL.items():
            actual_shift = magnitude * full_shift
            noise = drift_rng.normal(0, abs(actual_shift) * 0.05, size=n_samples)
            shift_array = np.where(drift_mask, actual_shift + noise, 0.0)

            if kpi == 'speed':
                speed += shift_array
                speed = np.clip(speed, 0.1, 10.0)
            elif kpi == 'distance':
                distance += shift_array
                distance = np.clip(distance, 5.0, 300.0)
            elif kpi == 'delay':
                delay += shift_array
                delay = np.clip(delay, 1.0, 200.0)
            elif kpi == 'throughput':
                throughput += shift_array
                throughput = np.clip(throughput, 1.0, 500.0)

    # Generate targets — base coefficients
    dist_z = (distance - 55.0) / 25.0
    delay_z = (delay - 12.0) / 6.0
    tp_z = (throughput - 55.0) / 20.0
    speed_z = (speed - 1.5) / 0.3

    # Base coefficients (identical to stationary generation)
    base_dist_w = 0.8
    base_delay_w = 1.2
    base_tp_w = -1.0
    base_speed_w = 0.3
    base_intercept = 0.0

    if drift_type == 'concept' and severity != 'none':
        # Concept drift: modify target generation coefficients for post-onset samples
        coeff_shifts = CONCEPT_DRIFT_COEFF_SHIFTS[severity]

        dist_w = np.where(drift_mask, base_dist_w + coeff_shifts['dist_w'], base_dist_w)
        delay_w = np.where(drift_mask, base_delay_w + coeff_shifts['delay_w'], base_delay_w)
        tp_w = np.where(drift_mask, base_tp_w + coeff_shifts['tp_w'], base_tp_w)
        speed_w = np.where(drift_mask, base_speed_w + coeff_shifts['speed_w'], base_speed_w)
        intercept = np.where(drift_mask, base_intercept + coeff_shifts['intercept'], base_intercept)

        linear_comb = dist_w * dist_z + delay_w * delay_z + tp_w * tp_z + speed_w * speed_z + intercept
    else:
        linear_comb = base_dist_w * dist_z + base_delay_w * delay_z + base_tp_w * tp_z + base_speed_w * speed_z

    prob_violation = 1.0 / (1.0 + np.exp(-linear_comb))
    target_rng = np.random.RandomState(seed + 2000)
    qos_violation = (target_rng.uniform(0, 1, n_samples) < prob_violation).astype(int)

    df = pd.DataFrame({
        'timestamp': t,
        'speed': np.round(speed, 4),
        'distance': np.round(distance, 2),
        'delay': np.round(delay, 3),
        'throughput': np.round(throughput, 2),
        TARGET_COL: qos_violation
    })
    return df


# ============================================================
# 3. FEATURE EXTRACTION — ROLLING WINDOW
# ============================================================

def extract_features(df, window_size=None, step_size=None):
    """
    Extract rolling-window statistical features for the 4 core KPIs.
    6 statistics per KPI: mean, std, min, max, q25, q75.
    Total: 4 x 6 = 24 features.
    """
    W = window_size or WINDOW_SIZE
    S = step_size or STEP_SIZE

    stat_names = ['mean', 'std', 'min', 'max', 'q25', 'q75']
    feat_names = [f'{col}_{s}' for col in KPI_COLS for s in stat_names]

    feature_rows = []
    labels = []
    timestamps = []
    n_samples = len(df)

    for start in range(0, n_samples - W + 1, S):
        end = start + W
        window = df.iloc[start:end]
        row_feats = []
        for col in KPI_COLS:
            vals = window[col].values
            row_feats.extend([
                np.mean(vals),
                np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                np.min(vals),
                np.max(vals),
                np.percentile(vals, 25),
                np.percentile(vals, 75),
            ])
        feature_rows.append(row_feats)
        labels.append(df[TARGET_COL].iloc[end - 1])
        timestamps.append(df['timestamp'].iloc[end - 1])

    X = np.array(feature_rows)
    y = np.array(labels)
    ts = np.array(timestamps)
    return X, y, feat_names, ts


# ============================================================
# 4. DRIFT MEASUREMENT
# ============================================================

def calculate_psi(reference, comparison, num_buckets=10):
    """Compute Population Stability Index (PSI) between two 1D arrays."""
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


def compute_drift_metrics(X_ref, X_current, feature_names):
    """
    Compute comprehensive drift metrics between reference and current distributions.

    Returns:
        drift_metrics dict with:
        - wasserstein_mean: mean Wasserstein distance across features (PRIMARY)
        - wasserstein_per_feature: per-feature Wasserstein distances
        - ks_mean: mean KS statistic
        - ks_per_feature: per-feature KS statistics
        - psi_mean: mean PSI
        - psi_per_feature: per-feature PSI values
    """
    wasserstein_scores = []
    ks_scores = []
    psi_scores = []
    per_feature = {}

    for i, fname in enumerate(feature_names):
        ref_feat = X_ref[:, i]
        cur_feat = X_current[:, i]

        # Wasserstein distance (Earth Mover's Distance)
        # Normalize both distributions to [0, 1] range for comparability
        combined = np.concatenate([ref_feat, cur_feat])
        feat_range = np.ptp(combined)
        if feat_range > 1e-10:
            ref_normed = (ref_feat - np.min(combined)) / feat_range
            cur_normed = (cur_feat - np.min(combined)) / feat_range
        else:
            ref_normed = ref_feat
            cur_normed = cur_feat

        w_dist = wasserstein_distance(ref_normed, cur_normed)
        wasserstein_scores.append(w_dist)

        # KS statistic
        ks_stat, ks_pval = stats.ks_2samp(ref_feat, cur_feat)
        ks_scores.append(ks_stat)

        # PSI
        psi_val = calculate_psi(ref_feat, cur_feat)
        psi_scores.append(psi_val)

        per_feature[fname] = {
            'wasserstein': float(w_dist),
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval),
            'psi': float(psi_val),
        }

    return {
        'wasserstein_mean': float(np.mean(wasserstein_scores)),
        'wasserstein_per_feature': wasserstein_scores,
        'ks_mean': float(np.mean(ks_scores)),
        'ks_per_feature': ks_scores,
        'psi_mean': float(np.mean(psi_scores)),
        'psi_per_feature': psi_scores,
        'per_feature_detail': per_feature,
    }


# ============================================================
# 5. RESOURCE MONITOR (PSUTIL PROCESS-LEVEL)
# ============================================================

class ResourceMonitor:
    """
    Monitors CPU and memory usage during a training/retraining operation.
    Uses a background thread to sample process-level metrics at regular intervals.

    Measurements represent PROCESS-LEVEL resources (not system-wide) unless
    system-level metrics are explicitly requested.
    """

    def __init__(self, interval_seconds=0.1):
        self.interval = interval_seconds
        self.process = psutil.Process()
        self._cpu_samples = []
        self._memory_samples = []
        self._running = False
        self._thread = None
        self._start_cpu_times = None
        self._start_time = None
        self._end_time = None

    def _sample_loop(self):
        """Background sampling loop."""
        while self._running:
            try:
                # Process CPU percent (percentage of one CPU core)
                cpu_pct = self.process.cpu_percent(interval=None)
                self._cpu_samples.append(cpu_pct)

                # Process memory (RSS in MB)
                mem_info = self.process.memory_info()
                self._memory_samples.append(mem_info.rss / (1024 ** 2))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval)

    def start(self):
        """Start monitoring. Call immediately before training."""
        self._cpu_samples = []
        self._memory_samples = []
        self._running = True

        # Record initial CPU times for process
        self._start_cpu_times = self.process.cpu_times()
        self._start_time = time.time()

        # Prime the CPU percent measurement
        self.process.cpu_percent(interval=None)

        # Record initial memory sample
        try:
            mem_info = self.process.memory_info()
            self._memory_samples.append(mem_info.rss / (1024 ** 2))
        except Exception:
            pass

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring. Call immediately after training."""
        self._end_time = time.time()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        # Record end memory sample
        try:
            mem_info = self.process.memory_info()
            self._memory_samples.append(mem_info.rss / (1024 ** 2))
        except Exception:
            pass

        # Final CPU times
        end_cpu_times = self.process.cpu_times()

        # Compute CPU time delta
        cpu_time_user = end_cpu_times.user - self._start_cpu_times.user
        cpu_time_system = end_cpu_times.system - self._start_cpu_times.system
        cpu_time_total = cpu_time_user + cpu_time_system

        wall_clock = self._end_time - self._start_time

        # CPU utilization statistics
        if len(self._cpu_samples) > 0:
            avg_cpu = float(np.mean(self._cpu_samples))
            peak_cpu = float(np.max(self._cpu_samples))
        else:
            # Fallback: estimate from CPU time / wall time
            avg_cpu = (cpu_time_total / max(wall_clock, 1e-6)) * 100.0
            peak_cpu = avg_cpu

        # Memory statistics
        if len(self._memory_samples) > 0:
            avg_memory = float(np.mean(self._memory_samples))
            peak_memory = float(np.max(self._memory_samples))
        else:
            mem_info = self.process.memory_info()
            avg_memory = mem_info.rss / (1024 ** 2)
            peak_memory = avg_memory

        return {
            'training_time_seconds': float(wall_clock),
            'cpu_time_seconds': float(cpu_time_total),
            'cpu_time_user_seconds': float(cpu_time_user),
            'cpu_time_system_seconds': float(cpu_time_system),
            'avg_cpu_percent': avg_cpu,
            'peak_cpu_percent': peak_cpu,
            'avg_memory_mb': avg_memory,
            'peak_memory_mb': peak_memory,
            'n_samples_collected': len(self._cpu_samples),
            'measurement_type': 'process_level',
        }


# ============================================================
# 6. MODEL TRAINING WITH RESOURCE MONITORING
# ============================================================

def train_model_with_monitoring(X_train, y_train, seed, monitor_interval=0.05):
    """
    Train a RandomForestClassifier with full resource monitoring.

    Returns:
        model: trained classifier
        resource_metrics: dict of CPU/memory measurements
    """
    monitor = ResourceMonitor(interval_seconds=monitor_interval)

    model = RandomForestClassifier(
        n_estimators=MODEL_PARAMS['n_estimators'],
        max_depth=MODEL_PARAMS['max_depth'],
        random_state=seed,
    )

    monitor.start()
    model.fit(X_train, y_train)
    resource_metrics = monitor.stop()

    resource_metrics['training_samples'] = len(X_train)
    return model, resource_metrics


# ============================================================
# 7. EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate a model on test data and return classification metrics."""
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_prob = _get_pos_probs(model, X_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.5

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc),
    }


def _get_pos_probs(model, X_scaled):
    """Safely retrieve positive-class probabilities."""
    probs = model.predict_proba(X_scaled)
    if probs.shape[1] == 1:
        cls = model.classes_[0]
        return np.ones(len(X_scaled)) if cls == 1 else np.zeros(len(X_scaled))
    cls_idx = np.where(model.classes_ == 1)[0]
    if len(cls_idx) > 0:
        return probs[:, cls_idx[0]]
    return probs[:, 1]


# ============================================================
# 8. MAIN EXPERIMENT PIPELINE
# ============================================================

def run_single_condition(condition, seed, verbose=True):
    """
    Run a single experimental condition for one seed.

    Returns:
        performance_records: list of dicts (per-window performance)
        resource_records: list of dicts (per-retraining resource usage)
        drift_records: list of dicts (per-window drift measurements)
        frozen_model_resource: dict (Model 1 initial training resource)
    """
    cond_id = condition['id']
    drift_type = condition['drift_type']
    severity = condition['severity']

    if verbose:
        print(f"    [{cond_id}] Seed={seed} | {condition['description']}")

    # --- Step 1: Generate dataset ---
    if drift_type == 'none':
        df = generate_stationary_dataset(N_RAW_SAMPLES, seed)
    else:
        df = generate_dataset_with_drift(
            N_RAW_SAMPLES, seed, drift_type, severity,
            drift_onset_sample=N_INITIAL_TRAINING
        )

    # --- Step 2: Extract features ---
    X_all, y_all, feat_names, ts_all = extract_features(df)
    n_total_windows = len(X_all)

    # --- Step 3: Split into training and post-deployment ---
    # Training windows: those derived from the first N_INITIAL_TRAINING raw samples
    # A window at position `start` uses raw samples [start, start+W)
    # Last training window starts at most at N_INITIAL_TRAINING - W
    n_train_windows = 0
    for start in range(0, N_RAW_SAMPLES - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        if end <= N_INITIAL_TRAINING:
            n_train_windows += 1
        else:
            break

    X_train = X_all[:n_train_windows]
    y_train = y_all[:n_train_windows]
    X_post = X_all[n_train_windows:]
    y_post = y_all[n_train_windows:]

    # --- Step 4: Fit scaler on training data ONLY ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Step 5: Train frozen Model 1 with resource monitoring ---
    frozen_model, frozen_resource = train_model_with_monitoring(
        X_train_scaled, y_train, seed, monitor_interval=0.05
    )
    frozen_resource['experiment_id'] = cond_id
    frozen_resource['seed'] = seed
    frozen_resource['model_id'] = 'frozen_model_1'
    frozen_resource['window_id'] = -1  # Pre-deployment initial training window (integer)
    frozen_resource['drift_type'] = drift_type
    frozen_resource['drift_severity'] = severity
    frozen_resource['retraining_triggered'] = False

    # Save frozen model artifact and raw stream data (only for seed 42)
    if seed == 42:
        model_path = os.path.join(MODELS_DIR, f'frozen_model_1_{cond_id}.joblib')
        joblib.dump(frozen_model, model_path)
        scaler_path = os.path.join(MODELS_DIR, f'scaler_{cond_id}.joblib')
        joblib.dump(scaler, scaler_path)
        data_path = os.path.join(DATA_DIR, f'raw_dataset_{cond_id}.csv')
        df.to_csv(data_path, index=False)

    # --- Step 6: Evaluate on sequential windows ---
    n_post = len(X_post)
    n_eval_windows = n_post // EVAL_WINDOW_SIZE

    performance_records = []
    resource_records = [frozen_resource]
    drift_records = []

    # Cumulative data for retraining (starts with initial training data)
    cumulative_X = list(X_train_scaled)
    cumulative_y = list(y_train)

    # In prequential (test-then-train) evaluation, active_retrained_model makes predictions
    # on each incoming window OUT-OF-SAMPLE before the window data is incorporated into training.
    # At window 0, the active model is the initial model trained on X_train.
    active_retrained_model = frozen_model

    for w_idx in range(n_eval_windows):
        w_start = w_idx * EVAL_WINDOW_SIZE
        w_end = min(w_start + EVAL_WINDOW_SIZE, n_post)
        X_window = X_post[w_start:w_end]
        y_window = y_post[w_start:w_end]

        # --- Drift measurement ---
        drift_metrics = compute_drift_metrics(X_train, X_window, feat_names)

        drift_rec = {
            'experiment_id': cond_id,
            'seed': seed,
            'window_id': w_idx,
            'drift_type': drift_type,
            'intended_severity': severity,
            'intended_magnitude': (
                COVARIATE_DRIFT_MAGNITUDES.get(severity, 0.0)
                if drift_type == 'covariate' else 0.0
            ),
            'wasserstein_mean': drift_metrics['wasserstein_mean'],
            'ks_mean': drift_metrics['ks_mean'],
            'psi_mean': drift_metrics['psi_mean'],
            'n_features': len(feat_names),
            'n_samples_reference': len(X_train),
            'n_samples_current': len(X_window),
        }
        drift_records.append(drift_rec)

        # --- Step 6a: Evaluate frozen Model 1 (Out-of-sample) ---
        frozen_perf = evaluate_model(frozen_model, X_window, y_window, scaler)

        performance_records.append({
            'experiment_id': cond_id,
            'seed': seed,
            'window_id': w_idx,
            'model_id': 'frozen_model_1',
            'drift_type': drift_type,
            'drift_severity': severity,
            'drift_score': drift_metrics['wasserstein_mean'],
            **frozen_perf,
            'retraining_triggered': False,
            'performance_loss': 0.0,  # Will be computed later relative to no-drift
            'performance_recovery': 0.0,
        })

        # --- Step 6b: Evaluate active retrained model OUT-OF-SAMPLE (Prequential Test-Then-Train) ---
        # Note: Evaluated on X_window BEFORE incorporating X_window into training data!
        retrained_perf = evaluate_model(active_retrained_model, X_window, y_window, scaler)

        performance_gain = retrained_perf['f1'] - frozen_perf['f1']

        performance_records.append({
            'experiment_id': cond_id,
            'seed': seed,
            'window_id': w_idx,
            'model_id': 'retrained_baseline',
            'drift_type': drift_type,
            'drift_severity': severity,
            'drift_score': drift_metrics['wasserstein_mean'],
            **retrained_perf,
            'retraining_triggered': True,
            'performance_loss': 0.0,
            'performance_recovery': float(performance_gain),
        })

        # --- Step 6c: Retrain model on cumulative observed data (Prequential Train step) ---
        # Add current window to cumulative data
        X_window_scaled = scaler.transform(X_window)
        cumulative_X.extend(X_window_scaled.tolist())
        cumulative_y.extend(y_window.tolist())

        # Retrain on all data observed up through current window
        retrain_X = np.array(cumulative_X)
        retrain_y = np.array(cumulative_y)

        new_retrained_model, retrain_resource = train_model_with_monitoring(
            retrain_X, retrain_y, seed + w_idx + 100, monitor_interval=0.05
        )

        retrain_resource['model_id'] = 'retrained_baseline'
        retrain_resource['seed'] = seed
        retrain_resource['window_id'] = int(w_idx)
        retrain_resource['experiment_id'] = cond_id
        retrain_resource['drift_type'] = drift_type
        retrain_resource['drift_severity'] = severity
        retrain_resource['drift_score'] = drift_metrics['wasserstein_mean']
        retrain_resource['retraining_triggered'] = True
        resource_records.append(retrain_resource)

        # Update active model for subsequent windows
        active_retrained_model = new_retrained_model

    if verbose:
        n_windows = len([r for r in drift_records])
        mean_drift = np.mean([r['wasserstein_mean'] for r in drift_records]) if drift_records else 0.0
        mean_frozen_f1 = np.mean([r['f1'] for r in performance_records if r['model_id'] == 'frozen_model_1'])
        mean_retrained_f1 = np.mean([r['f1'] for r in performance_records if r['model_id'] == 'retrained_baseline'])
        print(f"           Windows={n_windows} | AvgDrift={mean_drift:.4f} | "
              f"FrozenF1={mean_frozen_f1:.4f} | RetrainedF1={mean_retrained_f1:.4f}")

    return performance_records, resource_records, drift_records, frozen_resource


def run_all_experiments(seeds=None, verbose=True):
    """
    Run all 7 experimental conditions across all seeds.

    Returns:
        all_performance: DataFrame
        all_resources: DataFrame
        all_drift: DataFrame
        config: dict
    """
    if seeds is None:
        seeds = EVAL_SEEDS

    env_info = capture_environment()

    print("=" * 80)
    print("  EXPERIMENT 3A — FROZEN MODEL DEGRADATION & RETRAINING COST")
    print(f"  Conditions: {len(EXPERIMENT_CONDITIONS)} | Seeds: {seeds}")
    print(f"  Dataset: {N_RAW_SAMPLES} samples | Training: {N_INITIAL_TRAINING}")
    print("=" * 80)
    print(f"\n  Environment:")
    print(f"    Python: {env_info['python_version'].split()[0]}")
    print(f"    CPU: {env_info['processor']} ({env_info['cpu_count_logical']} logical, "
          f"{env_info['cpu_count_physical']} physical)")
    print(f"    RAM: {env_info['total_ram_gb']} GB")

    all_performance = []
    all_resources = []
    all_drift = []

    total_runs = len(EXPERIMENT_CONDITIONS) * len(seeds)
    run_count = 0
    t_start = time.time()

    for condition in EXPERIMENT_CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  {condition['id']}: {condition['description']}")
        print(f"{'='*60}")

        for seed in seeds:
            run_count += 1
            perf_recs, res_recs, drift_recs, _ = run_single_condition(
                condition, seed, verbose=verbose
            )
            all_performance.extend(perf_recs)
            all_resources.extend(res_recs)
            all_drift.extend(drift_recs)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  Completed {total_runs} runs in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")

    # Convert to DataFrames
    df_perf = pd.DataFrame(all_performance)
    df_res = pd.DataFrame(all_resources)
    df_drift = pd.DataFrame(all_drift)

    # Compute performance loss relative to no-drift baseline
    compute_performance_loss(df_perf)

    # Build config
    config = build_config(seeds, env_info, elapsed)

    return df_perf, df_res, df_drift, config


def compute_performance_loss(df_perf):
    """Compute performance_loss as the drop from the no-drift (3A-0) baseline."""
    for seed in df_perf['seed'].unique():
        for model_id in ['frozen_model_1', 'retrained_baseline']:
            baseline = df_perf[
                (df_perf['experiment_id'] == '3A-0') &
                (df_perf['seed'] == seed) &
                (df_perf['model_id'] == model_id)
            ]
            if len(baseline) == 0:
                continue
            baseline_f1 = baseline['f1'].mean()

            mask = (df_perf['seed'] == seed) & (df_perf['model_id'] == model_id)
            df_perf.loc[mask, 'performance_loss'] = baseline_f1 - df_perf.loc[mask, 'f1']


def build_config(seeds, env_info, elapsed_time):
    """Build comprehensive experiment configuration."""
    return {
        'experiment_id': EXPERIMENT_ID,
        'experiment_name': EXPERIMENT_NAME,
        'dataset': {
            'n_raw_samples': N_RAW_SAMPLES,
            'n_initial_training': N_INITIAL_TRAINING,
            'n_post_deployment': N_POST_DEPLOYMENT,
            'kpi_columns': KPI_COLS,
            'target_column': TARGET_COL,
            'target_rule': 'logistic combination of z-scored KPIs: '
                          '0.8*dist_z + 1.2*delay_z - 1.0*tp_z + 0.3*speed_z',
        },
        'feature_extraction': {
            'window_size': WINDOW_SIZE,
            'step_size': STEP_SIZE,
            'n_features': 24,
            'feature_statistics': ['mean', 'std', 'min', 'max', 'q25', 'q75'],
        },
        'evaluation': {
            'eval_window_size': EVAL_WINDOW_SIZE,
            'seeds': seeds,
            'n_seeds': len(seeds),
        },
        'model': {
            'type': 'RandomForestClassifier',
            'n_estimators': MODEL_PARAMS['n_estimators'],
            'max_depth': MODEL_PARAMS['max_depth'],
            'frozen_after_initial_training': True,
        },
        'drift': {
            'covariate_drift_full_magnitudes': COVARIATE_DRIFT_FULL,
            'covariate_drift_levels': COVARIATE_DRIFT_MAGNITUDES,
            'concept_drift_coefficient_shifts': CONCEPT_DRIFT_COEFF_SHIFTS,
            'primary_drift_metric': 'wasserstein_distance (normalized to [0,1] per feature)',
            'secondary_drift_metrics': ['ks_statistic', 'psi'],
            'severity_thresholds': DRIFT_SEVERITY_THRESHOLDS,
        },
        'resource_monitoring': {
            'method': 'psutil.Process()',
            'measurement_type': 'process_level',
            'sampling_interval_seconds': 0.05,
            'metrics_collected': [
                'wall_clock_time', 'cpu_time_user', 'cpu_time_system',
                'avg_cpu_percent', 'peak_cpu_percent',
                'avg_memory_mb', 'peak_memory_mb',
            ],
        },
        'conditions': EXPERIMENT_CONDITIONS,
        'runtime_seconds': elapsed_time,
        'environment': env_info,
    }


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(df_perf, df_res, df_drift, config):
    """Save all result tables and configuration."""
    # Performance results
    perf_path = os.path.join(RESULTS_DIR, 'performance_results.csv')
    df_perf.to_csv(perf_path, index=False)
    print(f"  Saved: {perf_path}")

    # Resource results
    res_path = os.path.join(RESULTS_DIR, 'resource_results.csv')
    df_res.to_csv(res_path, index=False)
    print(f"  Saved: {res_path}")

    # Drift results
    drift_path = os.path.join(RESULTS_DIR, 'drift_results.csv')
    df_drift.to_csv(drift_path, index=False)
    print(f"  Saved: {drift_path}")

    # Configuration
    config_path = os.path.join(RESULTS_DIR, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    print(f"  Saved: {config_path}")


# ============================================================
# 10. AGGREGATE STATISTICS
# ============================================================

def compute_aggregated_stats(df_perf, df_res, df_drift):
    """Compute mean +/- std across seeds for reporting."""
    print("\n" + "=" * 80)
    print("  AGGREGATED RESULTS (Mean +/- Std across seeds)")
    print("=" * 80)

    # Performance summary
    perf_summary = []
    for cond in EXPERIMENT_CONDITIONS:
        cid = cond['id']
        for model_id in ['frozen_model_1', 'retrained_baseline']:
            subset = df_perf[
                (df_perf['experiment_id'] == cid) &
                (df_perf['model_id'] == model_id)
            ]
            if len(subset) == 0:
                continue

            # Average across windows per seed, then across seeds
            per_seed = subset.groupby('seed').agg({
                'f1': 'mean', 'accuracy': 'mean', 'precision': 'mean',
                'recall': 'mean', 'roc_auc': 'mean',
                'performance_loss': 'mean', 'performance_recovery': 'mean',
            }).reset_index()

            row = {
                'experiment_id': cid,
                'model_id': model_id,
                'drift_type': cond['drift_type'],
                'severity': cond['severity'],
                'f1_mean': per_seed['f1'].mean(),
                'f1_std': per_seed['f1'].std(),
                'accuracy_mean': per_seed['accuracy'].mean(),
                'accuracy_std': per_seed['accuracy'].std(),
                'precision_mean': per_seed['precision'].mean(),
                'recall_mean': per_seed['recall'].mean(),
                'roc_auc_mean': per_seed['roc_auc'].mean(),
                'perf_loss_mean': per_seed['performance_loss'].mean(),
                'perf_recovery_mean': per_seed['performance_recovery'].mean(),
            }
            perf_summary.append(row)

    df_perf_summary = pd.DataFrame(perf_summary)
    print("\n  PERFORMANCE SUMMARY:")
    print(f"  {'Condition':<8s} {'Model':<20s} {'F1':>12s} {'Accuracy':>12s} {'Loss':>10s} {'Recovery':>10s}")
    print("  " + "-" * 75)
    for _, r in df_perf_summary.iterrows():
        f1_str = f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f}"
        acc_str = f"{r['accuracy_mean']:.4f}"
        loss_str = f"{r['perf_loss_mean']:+.4f}"
        rec_str = f"{r['perf_recovery_mean']:+.4f}"
        print(f"  {r['experiment_id']:<8s} {r['model_id']:<20s} {f1_str:>12s} {acc_str:>12s} "
              f"{loss_str:>10s} {rec_str:>10s}")

    # Resource summary
    res_summary = []
    df_res_clean = df_res.copy()
    df_res_clean['window_id'] = pd.to_numeric(df_res_clean['window_id'], errors='coerce')
    retrain_data = df_res_clean[df_res_clean['retraining_triggered'] == True].dropna(subset=['window_id'])
    retrain_data['window_id'] = retrain_data['window_id'].astype(int)
    for cond in EXPERIMENT_CONDITIONS:
        cid = cond['id']
        subset = retrain_data[retrain_data['experiment_id'] == cid] if 'experiment_id' in retrain_data.columns else pd.DataFrame()
        if len(subset) == 0:
            continue
        res_summary.append({
            'experiment_id': cid,
            'n_retrainings': len(subset),
            'total_training_time_mean': subset.groupby('window_id')['training_time_seconds'].mean().sum(),
            'avg_cpu_percent_mean': subset['avg_cpu_percent'].mean(),
            'peak_cpu_percent_mean': subset['peak_cpu_percent'].mean(),
            'avg_cpu_time_mean': subset['cpu_time_seconds'].mean(),
            'avg_memory_mb_mean': subset['avg_memory_mb'].mean(),
            'peak_memory_mb_mean': subset['peak_memory_mb'].mean(),
        })

    if res_summary:
        df_res_summary = pd.DataFrame(res_summary)
        print("\n  RESOURCE SUMMARY (Retraining):")
        print(f"  {'Condition':<8s} {'Retrains':>10s} {'TotalTime':>12s} {'AvgCPU%':>10s} {'PeakCPU%':>10s} {'AvgMem':>10s}")
        print("  " + "-" * 65)
        for _, r in df_res_summary.iterrows():
            print(f"  {r['experiment_id']:<8s} {r['n_retrainings']:10d} "
                  f"{r['total_training_time_mean']:12.3f}s "
                  f"{r['avg_cpu_percent_mean']:10.1f} "
                  f"{r['peak_cpu_percent_mean']:10.1f} "
                  f"{r['avg_memory_mb_mean']:10.1f}MB")

    # Drift summary
    drift_summary = []
    for cond in EXPERIMENT_CONDITIONS:
        cid = cond['id']
        subset = df_drift[df_drift['experiment_id'] == cid]
        if len(subset) == 0:
            continue
        per_seed = subset.groupby('seed')['wasserstein_mean'].mean()
        drift_summary.append({
            'experiment_id': cid,
            'drift_type': cond['drift_type'],
            'severity': cond['severity'],
            'wasserstein_mean': per_seed.mean(),
            'wasserstein_std': per_seed.std(),
            'ks_mean': subset.groupby('seed')['ks_mean'].mean().mean(),
            'psi_mean': subset.groupby('seed')['psi_mean'].mean().mean(),
        })

    df_drift_summary = pd.DataFrame(drift_summary)
    print("\n  DRIFT MEASUREMENT SUMMARY:")
    print(f"  {'Condition':<8s} {'Type':<12s} {'Severity':<10s} {'Wasserstein':>15s} {'KS':>10s} {'PSI':>10s}")
    print("  " + "-" * 70)
    for _, r in df_drift_summary.iterrows():
        w_str = f"{r['wasserstein_mean']:.4f}+/-{r['wasserstein_std']:.4f}"
        print(f"  {r['experiment_id']:<8s} {r['drift_type']:<12s} {r['severity']:<10s} "
              f"{w_str:>15s} {r['ks_mean']:10.4f} {r['psi_mean']:10.4f}")

    return df_perf_summary, df_drift_summary


# ============================================================
# 11. VISUALIZATION — 12 RESEARCH-QUALITY PLOTS
# ============================================================

def generate_all_plots(df_perf, df_res, df_drift, config):
    """Generate all 12+ research-quality diagnostic plots."""
    print("\n" + "=" * 80)
    print("  GENERATING RESEARCH-QUALITY PLOTS")
    print("=" * 80)

    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })

    conditions_ordered = [c['id'] for c in EXPERIMENT_CONDITIONS]

    # ---- Plot 1: Drift Score Over Time (per condition) ----
    _plot_01_drift_over_time(df_drift, conditions_ordered)

    # ---- Plot 2: Frozen Model F1 Over Time ----
    _plot_02_frozen_f1_over_time(df_perf, conditions_ordered)

    # ---- Plot 3: Drift Score vs Frozen Model F1 ----
    _plot_03_drift_vs_f1(df_perf, df_drift)

    # ---- Plot 4: Performance Degradation vs Drift Severity ----
    _plot_04_degradation_vs_severity(df_perf)

    # ---- Plot 5: Frozen vs Retrained F1 (grouped bar) ----
    _plot_05_frozen_vs_retrained(df_perf)

    # ---- Plot 6: Retraining CPU Usage by Drift Severity ----
    _plot_06_cpu_by_severity(df_res)

    # ---- Plot 7: Retraining Time by Drift Severity ----
    _plot_07_time_by_severity(df_res)

    # ---- Plot 8: CPU Usage vs Performance Recovery ----
    _plot_08_cpu_vs_recovery(df_perf, df_res)

    # ---- Plot 9: Memory Usage vs Training Dataset Size ----
    _plot_09_memory_vs_size(df_res)

    # ---- Plot 10: Performance Gain per Unit CPU Cost ----
    _plot_10_efficiency(df_perf, df_res)

    # ---- Plot 11: Cumulative Computational Cost ----
    _plot_11_cumulative_cost(df_res)

    # ---- Plot 12: Experiment Timeline ----
    _plot_12_timeline(df_perf, df_drift, df_res)

    print(f"\n  All 12 plots saved to: {PLOTS_DIR}")


def _plot_01_drift_over_time(df_drift, conditions_ordered):
    """Plot 1: Drift score over time for each condition."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 7))

    for idx, cid in enumerate(conditions_ordered):
        ax = axes[idx]
        subset = df_drift[df_drift['experiment_id'] == cid]
        per_seed = subset.groupby(['window_id', 'seed'])['wasserstein_mean'].first().reset_index()
        means = per_seed.groupby('window_id')['wasserstein_mean'].mean()
        stds = per_seed.groupby('window_id')['wasserstein_mean'].std()

        ax.plot(means.index, means.values, 'o-', color=colors[idx], linewidth=2, markersize=4)
        ax.fill_between(means.index, (means - stds).values, (means + stds).values,
                        color=colors[idx], alpha=0.2)
        ax.set_title(f"{cid}", fontweight='bold')
        ax.set_xlabel("Evaluation Window")
        ax.set_ylabel("Wasserstein Distance")
        ax.set_ylim(bottom=0)

    axes[-1].set_visible(False)
    fig.suptitle("Plot 1: Drift Score Over Time (Wasserstein Distance, Mean +/- Std)",
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_drift_score_over_time.png'))
    plt.close()
    print("  [1/12] Saved: 01_drift_score_over_time.png")


def _plot_02_frozen_f1_over_time(df_perf, conditions_ordered):
    """Plot 2: Frozen Model 1 F1 over time."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 7))

    for idx, cid in enumerate(conditions_ordered):
        ax = axes[idx]
        subset = df_perf[
            (df_perf['experiment_id'] == cid) &
            (df_perf['model_id'] == 'frozen_model_1')
        ]
        per_seed = subset.groupby(['window_id', 'seed'])['f1'].first().reset_index()
        means = per_seed.groupby('window_id')['f1'].mean()
        stds = per_seed.groupby('window_id')['f1'].std()

        ax.plot(means.index, means.values, 'o-', color=colors[idx], linewidth=2, markersize=4)
        ax.fill_between(means.index, (means - stds).values, (means + stds).values,
                        color=colors[idx], alpha=0.2)
        ax.set_title(f"{cid}", fontweight='bold')
        ax.set_xlabel("Evaluation Window")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1)

    axes[-1].set_visible(False)
    fig.suptitle("Plot 2: Frozen Model 1 — F1 Score Over Time (Mean +/- Std)",
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_frozen_model_f1_over_time.png'))
    plt.close()
    print("  [2/12] Saved: 02_frozen_model_f1_over_time.png")


def _plot_03_drift_vs_f1(df_perf, df_drift):
    """Plot 3: Drift score vs frozen Model 1 performance (scatter)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    frozen = df_perf[df_perf['model_id'] == 'frozen_model_1']
    merged = frozen.merge(
        df_drift[['experiment_id', 'seed', 'window_id', 'wasserstein_mean']],
        on=['experiment_id', 'seed', 'window_id'], how='left',
        suffixes=('', '_drift')
    )

    drift_types = {'none': '#2ca02c', 'covariate': '#1f77b4', 'concept': '#d62728'}
    markers = {'none': 'o', 'covariate': 's', 'concept': '^'}
    drift_col = 'wasserstein_mean_drift' if 'wasserstein_mean_drift' in merged.columns else (
        'wasserstein_mean' if 'wasserstein_mean' in merged.columns else 'drift_score'
    )
    for dt, color in drift_types.items():
        sub = merged[merged['drift_type'] == dt]
        ax.scatter(sub[drift_col], sub['f1'],
                   c=color, marker=markers[dt], alpha=0.5, s=40, label=f'{dt.capitalize()} drift')

    ax.set_xlabel("Wasserstein Distance (Drift Score)")
    ax.set_ylabel("Frozen Model 1 — F1 Score")
    ax.set_title("Plot 3: Drift Score vs. Frozen Model Performance", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_drift_vs_frozen_f1.png'))
    plt.close()
    print("  [3/12] Saved: 03_drift_vs_frozen_f1.png")


def _plot_04_degradation_vs_severity(df_perf):
    """Plot 4: Performance degradation vs drift severity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, drift_type, title in [
        (ax1, 'covariate', 'Covariate Drift'),
        (ax2, 'concept', 'Concept Drift')
    ]:
        severities = ['none', 'mild', 'moderate', 'severe']
        x_pos = np.arange(len(severities))

        frozen_means = []
        frozen_stds = []

        for sev in severities:
            if sev == 'none':
                subset = df_perf[
                    (df_perf['experiment_id'] == '3A-0') &
                    (df_perf['model_id'] == 'frozen_model_1')
                ]
            else:
                cid_map = {'covariate': {'mild': '3A-1', 'moderate': '3A-2', 'severe': '3A-3'},
                           'concept': {'mild': '3A-4', 'moderate': '3A-5', 'severe': '3A-6'}}
                subset = df_perf[
                    (df_perf['experiment_id'] == cid_map[drift_type][sev]) &
                    (df_perf['model_id'] == 'frozen_model_1')
                ]

            per_seed = subset.groupby('seed')['f1'].mean()
            frozen_means.append(per_seed.mean())
            frozen_stds.append(per_seed.std())

        ax.bar(x_pos, frozen_means, 0.6, yerr=frozen_stds, capsize=5,
               color=['#2ca02c', '#ffcc00', '#ff7f0e', '#d62728'], alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.capitalize() for s in severities])
        ax.set_xlabel("Drift Severity")
        ax.set_ylabel("Frozen Model F1 Score")
        ax.set_title(f"{title}", fontweight='bold')
        ax.set_ylim(0, 1)

    fig.suptitle("Plot 4: Frozen Model Performance Degradation vs. Drift Severity",
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_degradation_vs_severity.png'))
    plt.close()
    print("  [4/12] Saved: 04_degradation_vs_severity.png")


def _plot_05_frozen_vs_retrained(df_perf):
    """Plot 5: Frozen vs retrained model F1 comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))

    conditions = [c['id'] for c in EXPERIMENT_CONDITIONS]
    x_pos = np.arange(len(conditions))
    width = 0.35

    frozen_means = []
    frozen_stds = []
    retrained_means = []
    retrained_stds = []

    for cid in conditions:
        for model_id, means_list, stds_list in [
            ('frozen_model_1', frozen_means, frozen_stds),
            ('retrained_baseline', retrained_means, retrained_stds),
        ]:
            subset = df_perf[
                (df_perf['experiment_id'] == cid) &
                (df_perf['model_id'] == model_id)
            ]
            per_seed = subset.groupby('seed')['f1'].mean()
            means_list.append(per_seed.mean())
            stds_list.append(per_seed.std())

    ax.bar(x_pos - width/2, frozen_means, width, yerr=frozen_stds, capsize=4,
           label='Frozen Model 1', color='#1f77b4', alpha=0.85)
    ax.bar(x_pos + width/2, retrained_means, width, yerr=retrained_stds, capsize=4,
           label='Retrained Baseline', color='#2ca02c', alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Experiment Condition")
    ax.set_ylabel("F1 Score (Mean +/- Std)")
    ax.set_title("Plot 5: Frozen vs. Retrained Model Performance", fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_frozen_vs_retrained_f1.png'))
    plt.close()
    print("  [5/12] Saved: 05_frozen_vs_retrained_f1.png")


def _plot_06_cpu_by_severity(df_res):
    """Plot 6: Retraining CPU usage by drift severity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    retrain = df_res[df_res['retraining_triggered'] == True].copy()
    if 'experiment_id' not in retrain.columns or len(retrain) == 0:
        ax.text(0.5, 0.5, 'No retraining data available', transform=ax.transAxes, ha='center')
        plt.savefig(os.path.join(PLOTS_DIR, '06_cpu_by_severity.png'))
        plt.close()
        return

    conditions = [c['id'] for c in EXPERIMENT_CONDITIONS]
    x_pos = np.arange(len(conditions))

    cpu_means = []
    cpu_stds = []
    for cid in conditions:
        subset = retrain[retrain['experiment_id'] == cid]
        if len(subset) > 0:
            cpu_means.append(subset['cpu_time_seconds'].mean())
            cpu_stds.append(subset['cpu_time_seconds'].std())
        else:
            cpu_means.append(0)
            cpu_stds.append(0)

    colors = ['#2ca02c'] + ['#1f77b4']*3 + ['#d62728']*3
    ax.bar(x_pos, cpu_means, 0.6, yerr=cpu_stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Experiment Condition")
    ax.set_ylabel("CPU Time per Retraining (seconds)")
    ax.set_title("Plot 6: Retraining CPU Cost by Drift Condition", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_cpu_by_severity.png'))
    plt.close()
    print("  [6/12] Saved: 06_cpu_by_severity.png")


def _plot_07_time_by_severity(df_res):
    """Plot 7: Retraining wall-clock time by drift severity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    retrain = df_res[df_res['retraining_triggered'] == True].copy()
    if 'experiment_id' not in retrain.columns or len(retrain) == 0:
        ax.text(0.5, 0.5, 'No retraining data available', transform=ax.transAxes, ha='center')
        plt.savefig(os.path.join(PLOTS_DIR, '07_retraining_time_by_severity.png'))
        plt.close()
        return

    conditions = [c['id'] for c in EXPERIMENT_CONDITIONS]
    x_pos = np.arange(len(conditions))

    time_means = []
    time_stds = []
    for cid in conditions:
        subset = retrain[retrain['experiment_id'] == cid]
        if len(subset) > 0:
            time_means.append(subset['training_time_seconds'].mean())
            time_stds.append(subset['training_time_seconds'].std())
        else:
            time_means.append(0)
            time_stds.append(0)

    colors = ['#2ca02c'] + ['#1f77b4']*3 + ['#d62728']*3
    ax.bar(x_pos, time_means, 0.6, yerr=time_stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Experiment Condition")
    ax.set_ylabel("Wall-Clock Time per Retraining (seconds)")
    ax.set_title("Plot 7: Retraining Wall-Clock Time by Drift Condition", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_retraining_time_by_severity.png'))
    plt.close()
    print("  [7/12] Saved: 07_retraining_time_by_severity.png")


def _plot_08_cpu_vs_recovery(df_perf, df_res):
    """Plot 8: CPU cost vs performance recovery (scatter)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    retrain = df_res[df_res['retraining_triggered'] == True].copy()
    if 'experiment_id' not in retrain.columns or len(retrain) == 0:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
        plt.savefig(os.path.join(PLOTS_DIR, '08_cpu_vs_recovery.png'))
        plt.close()
        return

    retrain['window_id'] = pd.to_numeric(retrain['window_id'], errors='coerce')
    retrain = retrain.dropna(subset=['window_id'])
    retrain['window_id'] = retrain['window_id'].astype(int)
    retrained_perf = df_perf[df_perf['model_id'] == 'retrained_baseline'].copy()
    retrained_perf['window_id'] = pd.to_numeric(retrained_perf['window_id'], errors='coerce')
    retrained_perf = retrained_perf.dropna(subset=['window_id'])
    retrained_perf['window_id'] = retrained_perf['window_id'].astype(int)

    merged = pd.merge(
        retrained_perf,
        retrain,
        on=['experiment_id', 'seed', 'window_id', 'drift_type'],
        suffixes=('_perf', '_res')
    )

    drift_types = {'covariate': '#1f77b4', 'concept': '#d62728', 'none': '#2ca02c'}
    markers = {'covariate': 's', 'concept': '^', 'none': 'o'}

    for dt, color in drift_types.items():
        sub = merged[merged['drift_type'] == dt]
        if len(sub) > 0:
            ax.scatter(sub['cpu_time_seconds'], sub['performance_recovery'],
                       c=color, marker=markers[dt], alpha=0.5, s=35, label=f'{dt.capitalize()} drift')
        else:
            ax.scatter([], [], c=color, marker=markers[dt], s=60, label=f'{dt.capitalize()} drift')

    ax.set_xlabel("CPU Time per Retraining (seconds)")
    ax.set_ylabel("Performance Recovery (F1 Gain)")
    ax.set_title("Plot 8: Retraining CPU Cost vs. Performance Recovery", fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_cpu_vs_recovery.png'))
    plt.close()
    print("  [8/12] Saved: 08_cpu_vs_recovery.png")


def _plot_09_memory_vs_size(df_res):
    """Plot 9: Memory usage vs training dataset size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    retrain = df_res[df_res['retraining_triggered'] == True].copy()
    if len(retrain) == 0:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
        plt.savefig(os.path.join(PLOTS_DIR, '09_memory_vs_training_size.png'))
        plt.close()
        return

    ax.scatter(retrain['training_samples'], retrain['peak_memory_mb'],
               c='#e74c3c', alpha=0.4, s=30, label='Peak Memory')
    ax.scatter(retrain['training_samples'], retrain['avg_memory_mb'],
               c='#3498db', alpha=0.4, s=30, label='Average Memory')

    ax.set_xlabel("Training Dataset Size (samples)")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Plot 9: Memory Usage vs. Training Dataset Size", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_memory_vs_training_size.png'))
    plt.close()
    print("  [9/12] Saved: 09_memory_vs_training_size.png")


def _plot_10_efficiency(df_perf, df_res):
    """Plot 10: Performance gain per unit CPU cost (efficiency)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    conditions = [c['id'] for c in EXPERIMENT_CONDITIONS]
    x_pos = np.arange(len(conditions))

    efficiencies = []
    for cid in conditions:
        retrained_perf = df_perf[
            (df_perf['experiment_id'] == cid) &
            (df_perf['model_id'] == 'retrained_baseline')
        ]
        retrain_res = df_res[
            (df_res['experiment_id'] == cid) &
            (df_res['retraining_triggered'] == True)
        ] if ('experiment_id' in df_res.columns and 'retraining_triggered' in df_res.columns) else pd.DataFrame()

        if len(retrained_perf) > 0 and len(retrain_res) > 0:
            avg_recovery = retrained_perf['performance_recovery'].mean()
            avg_cpu = retrain_res['cpu_time_seconds'].mean()
            efficiency = avg_recovery / max(avg_cpu, 1e-6)
            efficiencies.append(efficiency)
        else:
            efficiencies.append(0)

    colors = ['#2ca02c'] + ['#1f77b4']*3 + ['#d62728']*3
    ax.bar(x_pos, efficiencies, 0.6, color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Experiment Condition")
    ax.set_ylabel("F1 Gain / CPU Second")
    ax.set_title("Plot 10: Retraining Efficiency — Performance Gain per CPU Second",
                 fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_efficiency_gain_per_cpu.png'))
    plt.close()
    print("  [10/12] Saved: 10_efficiency_gain_per_cpu.png")


def _plot_11_cumulative_cost(df_res):
    """Plot 11: Cumulative computational cost of repeated retraining."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    retrain = df_res[df_res['retraining_triggered'] == True].copy()
    if 'experiment_id' not in retrain.columns or len(retrain) == 0:
        ax1.text(0.5, 0.5, 'No data available', transform=ax1.transAxes, ha='center')
        plt.savefig(os.path.join(PLOTS_DIR, '11_cumulative_retraining_cost.png'))
        plt.close()
        return

    retrain['window_id'] = pd.to_numeric(retrain['window_id'], errors='coerce')
    retrain = retrain.dropna(subset=['window_id'])
    retrain['window_id'] = retrain['window_id'].astype(int)

    colors_map = {
        '3A-0': '#2ca02c', '3A-1': '#aec7e8', '3A-2': '#1f77b4', '3A-3': '#08519c',
        '3A-4': '#fcbba1', '3A-5': '#d62728', '3A-6': '#a50f15',
    }

    for cid in [c['id'] for c in EXPERIMENT_CONDITIONS]:
        subset = retrain[retrain['experiment_id'] == cid]
        if len(subset) == 0:
            continue
        # Average across seeds, then cumsum over windows (sorted numerically)
        per_window = subset.groupby('window_id').agg({
            'training_time_seconds': 'mean',
            'cpu_time_seconds': 'mean',
        }).sort_index()

        cum_time = np.cumsum(per_window['training_time_seconds'].values)
        cum_cpu = np.cumsum(per_window['cpu_time_seconds'].values)

        ax1.plot(per_window.index.astype(int), cum_time, 'o-', color=colors_map.get(cid, 'gray'),
                 label=cid, linewidth=2, markersize=4)
        ax2.plot(per_window.index.astype(int), cum_cpu, 'o-', color=colors_map.get(cid, 'gray'),
                 label=cid, linewidth=2, markersize=4)

    ax1.set_xlabel("Retraining Event (Window)")
    ax1.set_ylabel("Cumulative Wall-Clock Time (seconds)")
    ax1.set_title("Cumulative Retraining Time", fontweight='bold')
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Retraining Event (Window)")
    ax2.set_ylabel("Cumulative CPU Time (seconds)")
    ax2.set_title("Cumulative CPU Cost", fontweight='bold')
    ax2.legend(fontsize=8)

    fig.suptitle("Plot 11: Cumulative Computational Cost of Repeated Retraining",
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '11_cumulative_retraining_cost.png'))
    plt.close()
    print("  [11/12] Saved: 11_cumulative_retraining_cost.png")


def _plot_12_timeline(df_perf, df_drift, df_res):
    """Plot 12: Experiment timeline showing training → deployment → drift → retraining → recovery."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Use 3A-3 (severe covariate drift) as the illustrative example
    cid = '3A-3'
    seed = EVAL_SEEDS[0]

    # Panel 1: Drift score
    ax = axes[0]
    drift_sub = df_drift[(df_drift['experiment_id'] == cid) & (df_drift['seed'] == seed)].copy()
    if len(drift_sub) > 0:
        drift_sub['window_id'] = pd.to_numeric(drift_sub['window_id'], errors='coerce')
        drift_sub = drift_sub.sort_values('window_id')
        ax.fill_between(drift_sub['window_id'], 0, drift_sub['wasserstein_mean'],
                        color='#d62728', alpha=0.3)
        ax.plot(drift_sub['window_id'], drift_sub['wasserstein_mean'], 'o-',
                color='#d62728', linewidth=2, markersize=5)
    ax.set_ylabel("Drift Score\n(Wasserstein)")
    ax.set_title(f"Plot 12: Experiment Timeline — {cid} (Severe Covariate Drift, Seed {seed})",
                 fontweight='bold', fontsize=13)
    ax.axvspan(-1, 0, color='#aec7e8', alpha=0.3, label='Training Phase')
    ax.legend(fontsize=8)

    # Panel 2: Frozen model performance vs Retrained baseline
    ax = axes[1]
    frozen_sub = df_perf[
        (df_perf['experiment_id'] == cid) &
        (df_perf['seed'] == seed) &
        (df_perf['model_id'] == 'frozen_model_1')
    ].copy()
    if len(frozen_sub) > 0:
        frozen_sub['window_id'] = pd.to_numeric(frozen_sub['window_id'], errors='coerce')
        frozen_sub = frozen_sub.sort_values('window_id')
        ax.plot(frozen_sub['window_id'], frozen_sub['f1'], 'o-',
                color='#1f77b4', linewidth=2, markersize=5, label='Frozen Model 1')
    retrained_sub = df_perf[
        (df_perf['experiment_id'] == cid) &
        (df_perf['seed'] == seed) &
        (df_perf['model_id'] == 'retrained_baseline')
    ].copy()
    if len(retrained_sub) > 0:
        retrained_sub['window_id'] = pd.to_numeric(retrained_sub['window_id'], errors='coerce')
        retrained_sub = retrained_sub.sort_values('window_id')
        ax.plot(retrained_sub['window_id'], retrained_sub['f1'], 's-',
                color='#2ca02c', linewidth=2, markersize=5, label='Retrained Baseline')
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Panel 3: Retraining time per window
    ax = axes[2]
    retrain_sub = df_res[
        (df_res['experiment_id'] == cid) &
        (df_res['seed'] == seed) &
        (df_res['retraining_triggered'] == True)
    ].copy() if ('experiment_id' in df_res.columns and 'seed' in df_res.columns and 'retraining_triggered' in df_res.columns) else pd.DataFrame()
    if len(retrain_sub) > 0:
        retrain_sub['window_id'] = pd.to_numeric(retrain_sub['window_id'], errors='coerce').dropna().astype(int)
        retrain_sub = retrain_sub.sort_values('window_id')
        ax.bar(retrain_sub['window_id'].values, retrain_sub['training_time_seconds'].values,
               color='#ff7f0e', alpha=0.8, label='Retraining Time', width=0.6)
    ax.set_ylabel("Retraining\nTime (s)")
    ax.legend(fontsize=8)

    # Panel 4: CPU time per retraining
    ax = axes[3]
    if len(retrain_sub) > 0:
        ax.bar(retrain_sub['window_id'].values, retrain_sub['cpu_time_seconds'].values,
               color='#9467bd', alpha=0.8, label='CPU Time', width=0.6)
    ax.set_xlabel("Evaluation Window Index")
    ax.set_ylabel("CPU Time (s)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '12_experiment_timeline.png'))
    plt.close()
    print("  [12/12] Saved: 12_experiment_timeline.png")


# ============================================================
# 12. RESOURCE COST CURVES
# ============================================================

def generate_resource_cost_curves(df_perf, df_res, df_drift):
    """Generate additional resource cost analysis curves."""
    print("\n  Generating resource cost curves...")

    retrain = df_res[df_res['retraining_triggered'] == True].copy()
    if 'experiment_id' not in retrain.columns or len(retrain) == 0:
        print("    No retraining data available for cost curves.")
        return

    retrain['window_id'] = pd.to_numeric(retrain['window_id'], errors='coerce')
    retrain = retrain.dropna(subset=['window_id'])
    retrain['window_id'] = retrain['window_id'].astype(int)

    df_perf_copy = df_perf.copy()
    df_perf_copy['window_id'] = pd.to_numeric(df_perf_copy['window_id'], errors='coerce')
    df_perf_copy = df_perf_copy.dropna(subset=['window_id'])
    df_perf_copy['window_id'] = df_perf_copy['window_id'].astype(int)

    df_drift_copy = df_drift.copy()
    df_drift_copy['window_id'] = pd.to_numeric(df_drift_copy['window_id'], errors='coerce')
    df_drift_copy = df_drift_copy.dropna(subset=['window_id'])
    df_drift_copy['window_id'] = df_drift_copy['window_id'].astype(int)

    # Merge drift scores with resource data
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    conditions = [c['id'] for c in EXPERIMENT_CONDITIONS]

    # Curve 1: Drift Score → CPU Cost
    ax = axes[0]
    for cid in conditions:
        drift_sub = df_drift_copy[df_drift_copy['experiment_id'] == cid]
        res_sub = retrain[retrain['experiment_id'] == cid]
        if len(drift_sub) > 0 and len(res_sub) > 0:
            mean_drift = drift_sub.groupby('window_id')['wasserstein_mean'].mean()
            mean_cpu = res_sub.groupby('window_id')['cpu_time_seconds'].mean()
            common = mean_drift.index.intersection(mean_cpu.index)
            if len(common) > 0:
                ax.scatter(mean_drift.loc[common], mean_cpu.loc[common], s=30, alpha=0.6, label=cid)
    ax.set_xlabel("Drift Score (Wasserstein)")
    ax.set_ylabel("CPU Time per Retraining (s)")
    ax.set_title("Drift Score -> CPU Cost", fontweight='bold')
    ax.legend(fontsize=7)

    # Curve 2: Drift Score → Performance Recovery
    ax = axes[1]
    retrained_perf = df_perf_copy[df_perf_copy['model_id'] == 'retrained_baseline']
    for cid in conditions:
        drift_sub = df_drift_copy[df_drift_copy['experiment_id'] == cid]
        perf_sub = retrained_perf[retrained_perf['experiment_id'] == cid]
        if len(drift_sub) > 0 and len(perf_sub) > 0:
            mean_drift = drift_sub.groupby('window_id')['wasserstein_mean'].mean()
            mean_rec = perf_sub.groupby('window_id')['performance_recovery'].mean()
            common = mean_drift.index.intersection(mean_rec.index)
            if len(common) > 0:
                ax.scatter(mean_drift.loc[common], mean_rec.loc[common], s=30, alpha=0.6, label=cid)
    ax.set_xlabel("Drift Score (Wasserstein)")
    ax.set_ylabel("Performance Recovery (F1 Gain)")
    ax.set_title("Drift Score -> Performance Recovery", fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7)

    # Curve 3: Drift Score → Efficiency (Recovery / CPU Cost)
    ax = axes[2]
    for cid in conditions:
        drift_sub = df_drift_copy[df_drift_copy['experiment_id'] == cid]
        perf_sub = retrained_perf[retrained_perf['experiment_id'] == cid]
        res_sub = retrain[retrain['experiment_id'] == cid]
        if len(drift_sub) > 0 and len(perf_sub) > 0 and len(res_sub) > 0:
            mean_drift = drift_sub.groupby('window_id')['wasserstein_mean'].mean()
            mean_rec = perf_sub.groupby('window_id')['performance_recovery'].mean()
            mean_cpu = res_sub.groupby('window_id')['cpu_time_seconds'].mean()
            common = mean_drift.index.intersection(mean_rec.index).intersection(mean_cpu.index)
            if len(common) > 0:
                efficiency = mean_rec.loc[common] / np.maximum(mean_cpu.loc[common], 1e-6)
                ax.scatter(mean_drift.loc[common], efficiency, s=30, alpha=0.6, label=cid)
    ax.set_xlabel("Drift Score (Wasserstein)")
    ax.set_ylabel("Efficiency (F1 Gain / CPU Second)")
    ax.set_title("Drift Score -> Efficiency", fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("Resource Cost Curves: Drift vs. Computational Trade-offs",
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '13_resource_cost_curves.png'))
    plt.close()
    print("  Saved: 13_resource_cost_curves.png")


# ============================================================
# 13. MAIN ENTRY POINT
# ============================================================

def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description="Experiment 3A Runner")
    parser.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS,
                        help="Random seeds for evaluation")
    parser.add_argument('--quiet', action='store_true',
                        help="Reduce output verbosity")
    parser.add_argument('--force-rerun', action='store_true',
                        help="Force re-running the entire experiment from scratch")
    args = parser.parse_args()

    t_total_start = time.time()

    perf_path = os.path.join(RESULTS_DIR, 'performance_results.csv')
    res_path = os.path.join(RESULTS_DIR, 'resource_results.csv')
    drift_path = os.path.join(RESULTS_DIR, 'drift_results.csv')
    config_path = os.path.join(RESULTS_DIR, 'experiment_config.json')

    # Check if existing results are valid prequential results (legacy run had string 'initial_training')
    is_valid_prequential = False
    if os.path.exists(res_path):
        try:
            with open(res_path, 'r') as f:
                header_and_first = f.read(500)
                is_valid_prequential = 'initial_training' not in header_and_first
        except Exception:
            is_valid_prequential = False

    can_reuse = (
        not args.force_rerun and
        is_valid_prequential and
        os.path.exists(perf_path) and
        os.path.exists(res_path) and
        os.path.exists(drift_path) and
        os.path.exists(config_path) and
        os.path.getsize(perf_path) > 1000
    )

    if can_reuse:
        print("  Found existing complete results in results/. Reusing results...")
        df_perf = pd.read_csv(perf_path)
        df_res = pd.read_csv(res_path)
        df_drift = pd.read_csv(drift_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Run all experiments
        df_perf, df_res, df_drift, config = run_all_experiments(
            seeds=args.seeds, verbose=not args.quiet
        )

        # Save results
        print("\n" + "=" * 80)
        print("  SAVING RESULTS")
        print("=" * 80)
        save_results(df_perf, df_res, df_drift, config)

    # Ensure consistent numeric integer window_id across all DataFrames
    df_perf['window_id'] = pd.to_numeric(df_perf['window_id'], errors='coerce').fillna(0).astype(int)
    df_res['window_id'] = pd.to_numeric(df_res['window_id'], errors='coerce').fillna(-1).astype(int)
    df_drift['window_id'] = pd.to_numeric(df_drift['window_id'], errors='coerce').fillna(0).astype(int)

    # Compute and print aggregated statistics
    df_perf_summary, df_drift_summary = compute_aggregated_stats(df_perf, df_res, df_drift)

    # Generate plots
    generate_all_plots(df_perf, df_res, df_drift, config)

    # Generate resource cost curves
    generate_resource_cost_curves(df_perf, df_res, df_drift)

    # Final summary
    total_time = time.time() - t_total_start
    print("\n" + "=" * 80)
    print("  EXPERIMENT 3A COMPLETE")
    print("=" * 80)
    print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Plots saved to: {PLOTS_DIR}")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"\n  Output files:")
    print(f"    performance_results.csv  — {len(df_perf)} records")
    print(f"    resource_results.csv     — {len(df_res)} records")
    print(f"    drift_results.csv        — {len(df_drift)} records")
    print(f"    experiment_config.json")
    print(f"    13 diagnostic plots")

    # Key findings preview
    print(f"\n  KEY FINDINGS PREVIEW:")
    frozen_no_drift = df_perf[
        (df_perf['experiment_id'] == '3A-0') &
        (df_perf['model_id'] == 'frozen_model_1')
    ]['f1'].mean()
    frozen_severe_cov = df_perf[
        (df_perf['experiment_id'] == '3A-3') &
        (df_perf['model_id'] == 'frozen_model_1')
    ]['f1'].mean()
    frozen_severe_con = df_perf[
        (df_perf['experiment_id'] == '3A-6') &
        (df_perf['model_id'] == 'frozen_model_1')
    ]['f1'].mean()
    retrained_severe_cov = df_perf[
        (df_perf['experiment_id'] == '3A-3') &
        (df_perf['model_id'] == 'retrained_baseline')
    ]['f1'].mean()
    retrained_severe_con = df_perf[
        (df_perf['experiment_id'] == '3A-6') &
        (df_perf['model_id'] == 'retrained_baseline')
    ]['f1'].mean()

    retrain_total_cpu = df_res[
        (df_res['retraining_triggered'] == True)
    ]['cpu_time_seconds'].sum() if 'retraining_triggered' in df_res.columns else 0

    print(f"    1. Frozen Model F1 (no drift):        {frozen_no_drift:.4f}")
    print(f"    2. Frozen Model F1 (severe covariate): {frozen_severe_cov:.4f} "
          f"(loss: {frozen_no_drift - frozen_severe_cov:+.4f})")
    print(f"    3. Frozen Model F1 (severe concept):   {frozen_severe_con:.4f} "
          f"(loss: {frozen_no_drift - frozen_severe_con:+.4f})")
    print(f"    4. Retrained F1 (severe covariate):    {retrained_severe_cov:.4f} "
          f"(recovery: {retrained_severe_cov - frozen_severe_cov:+.4f})")
    print(f"    5. Retrained F1 (severe concept):      {retrained_severe_con:.4f} "
          f"(recovery: {retrained_severe_con - frozen_severe_con:+.4f})")
    print(f"    6. Total retraining CPU cost:           {retrain_total_cpu:.2f}s")
    print(f"\n  -> These results establish the empirical foundation for Experiment 3B+")


if __name__ == '__main__':
    main()
