"""
================================================================================
EXPERIMENT 2 -- DRIFT-AWARE BANDIT MODEL SELECTION
================================================================================
AI4Mobile Industrial Telemetry -- Four Core KPIs
UCB1 Sliding-Window Bandit for Adaptive Model Selection under Covariate Drift

This experiment is COMPLETELY SEPARATE from Experiment 1.
No Experiment 1 code is imported, modified, or shared.

KPIs used: Speed, Distance, Delay, Throughput
Models:
    Model 1 (Historical)     -- RandomForest, frozen after initial training
    Model 2 (Adaptive)       -- ExtraTrees, sliding-window retraining (window=200)
    Model 3 (Adaptive Ens.)  -- Heterogeneous Ensemble: RF + ET + GradientBoosting (window=50)

Bandit: UCB1 with sliding-window reward estimates
Drift:  Controlled covariate shift at 0%, 10%, 20%, 30%, 40%, 50%

Dataset: A STATIONARY 6000-sample baseline is generated so that controlled
drift is the ONLY source of distribution change. The target rule is fixed;
only features are shifted (pure covariate drift P(X)).
================================================================================
"""

import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import psutil
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
EXP2_SEED = 42
np.random.seed(EXP2_SEED)

EXP2_DATASET_CSV = "experiment2_stationary_dataset.csv"

# The four core KPIs for Experiment 2
EXP2_KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
EXP2_TARGET_COL = 'qos_violation'

# Feature extraction parameters
EXP2_WINDOW_SIZE = 20
EXP2_STEP_SIZE = 5

# Chronological split ratios
EXP2_SPLIT = {
    'train': 0.50,    # Historical training (Model 1)
    'adapt': 0.15,    # Adaptation data (Models 2 & 3 initial)
    'val':   0.10,    # Validation (bandit calibration -- NOT for model training)
    'test':  0.25     # Held-out test (NEVER used for training/adaptation)
}

# Drift levels to test
EXP2_DRIFT_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# Full-drift definition: shift each KPI mean by this many raw-unit amounts
# at 100% drift.  Each level p applies p * FULL_DRIFT.
# Directions are chosen for realistic industrial degradation:
#   speed:      +1.0 m/s    (AGV speeds up in open zone)
#   distance:   +30.0 m     (AGV moves further from base station)
#   delay:      +8.0 ms     (latency increases under NLOS)
#   throughput: -20.0 Mbps  (throughput degrades under interference)
EXP2_FULL_DRIFT = {
    'speed':      +1.0,
    'distance':   +30.0,
    'delay':      +8.0,
    'throughput': -20.0,
}

# Bandit parameters
EXP2_BANDIT_WINDOW = 50
EXP2_UCB_EXPLORATION = 2.0

# Model adaptation parameters
EXP2_MODEL2_ADAPT_WINDOW = 200
EXP2_MODEL3_ADAPT_WINDOW = 50
EXP2_ADAPT_INTERVAL = 20

# Output directories
EXP2_DIRS = {
    'base':        'experiment2',
    'predictions': 'experiment2/predictions',
    'metrics':     'experiment2/metrics',
    'drift':       'experiment2/drift_analysis',
    'plots':       'experiment2/plots',
    'reports':     'experiment2/reports',
}

for d in EXP2_DIRS.values():
    os.makedirs(d, exist_ok=True)


# ============================================================
# 1. HARDWARE / SOFTWARE ENVIRONMENT
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
        'sklearn_version': None,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }
    try:
        import sklearn
        env['sklearn_version'] = sklearn.__version__
    except ImportError:
        pass
    gpu_info = 'N/A -- No GPU detected or no GPU library installed'
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip()
    except Exception:
        pass
    env['gpu_info'] = gpu_info
    return env


# ============================================================
# 2. DATASET GENERATION -- STATIONARY BASELINE
# ============================================================

def exp2_generate_dataset(n_samples=6000, csv_filename=None):
    """
    Generate a stationary AI4Mobile-like industrial telemetry dataset.
    6000 samples chronologically ordered.  No random shuffling.
    Fixed target generation rule.
    """
    if csv_filename and os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if len(df) == n_samples and all(c in df.columns for c in EXP2_KPI_COLS + [EXP2_TARGET_COL]):
            print(f"[EXP2] Loaded existing stationary dataset: {csv_filename} ({len(df)} rows)")
            return df

    rng = np.random.RandomState(EXP2_SEED)
    t = np.arange(n_samples)

    # 1. Speed (m/s): base ~ 1.5, slight route cycle, noise
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, n_samples)
    speed = np.clip(speed, 0.5, 3.5)

    # 2. Distance to gNB (m): triangular warehouse path between 15m and 95m
    cycle = 800
    tri = 2 * np.abs((t % cycle) / cycle - 0.5)  # 0 to 1 to 0
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, n_samples)
    distance = np.clip(distance, 10.0, 120.0)

    # 3. Delay (ms): correlated with distance + log-normal spikes
    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay_noise = rng.lognormal(mean=0.0, sigma=0.4, size=n_samples)
    delay = delay_base + delay_noise
    delay = np.clip(delay, 2.0, 50.0)

    # 4. Throughput (Mbps): inversely related to distance and delay
    path_loss_factor = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    tp_base = 80.0 * path_loss_factor
    tp_noise = rng.normal(0, 8.0, n_samples)
    throughput = tp_base + tp_noise
    throughput = np.clip(throughput, 5.0, 150.0)

    # TARGET: QoS Violation (binary)
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
        'qos_violation': qos_violation
    })

    if csv_filename:
        df.to_csv(csv_filename, index=False)
        print(f"[EXP2] Generated & saved stationary dataset: {csv_filename} ({len(df)} rows)")
    return df


# ============================================================
# 3. FEATURE EXTRACTION -- ROLLING WINDOW
# ============================================================

def exp2_extract_features(df, window_size=None, step_size=None):
    """
    Extract rolling-window statistical features for the 4 core KPIs.
    6 statistics per KPI: mean, std, min, max, q25, q75
    Total input features: 4 * 6 = 24.
    """
    W = window_size or EXP2_WINDOW_SIZE
    S = step_size or EXP2_STEP_SIZE

    feature_rows = []
    labels = []
    timestamps = []

    kpi_cols = EXP2_KPI_COLS
    n_samples = len(df)

    stat_names = ['mean', 'std', 'min', 'max', 'q25', 'q75']
    feat_names = [f'{col}_{s}' for col in kpi_cols for s in stat_names]

    for start in range(0, n_samples - W + 1, S):
        end = start + W
        window = df.iloc[start:end]
        row_feats = []
        for col in kpi_cols:
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
        labels.append(df[EXP2_TARGET_COL].iloc[end - 1])
        timestamps.append(df['timestamp'].iloc[end - 1])

    X = np.array(feature_rows)
    y = np.array(labels)
    ts = np.array(timestamps)
    return X, y, feat_names, ts


# ============================================================
# 4. CONTROLLED DRIFT GENERATION (COVARIATE SHIFT)
# ============================================================

def exp2_apply_drift(X, feature_names, drift_level, seed=42):
    """
    Apply controlled covariate shift P(X) to feature array.
    Shifts location features (mean, min, max, q25, q75) by drift_level * FULL_DRIFT.
    Scale features (std) remain unmodified to preserve natural variance.
    """
    if drift_level == 0.0:
        return X.copy()

    rng = np.random.RandomState(seed)
    X_drifted = X.copy()

    for kpi, full_shift in EXP2_FULL_DRIFT.items():
        actual_shift = drift_level * full_shift
        location_stats = ['mean', 'min', 'max', 'q25', 'q75']
        for s in location_stats:
            col_name = f'{kpi}_{s}'
            if col_name in feature_names:
                idx = feature_names.index(col_name)
                noise = rng.normal(0, abs(actual_shift) * 0.05, size=len(X))
                X_drifted[:, idx] += actual_shift + noise
                if kpi == 'speed':
                    X_drifted[:, idx] = np.clip(X_drifted[:, idx], 0.1, 10.0)
                elif kpi == 'distance':
                    X_drifted[:, idx] = np.clip(X_drifted[:, idx], 5.0, 300.0)
                elif kpi == 'delay':
                    X_drifted[:, idx] = np.clip(X_drifted[:, idx], 1.0, 200.0)
                elif kpi == 'throughput':
                    X_drifted[:, idx] = np.clip(X_drifted[:, idx], 1.0, 500.0)
    return X_drifted


# ============================================================
# 5. CONTINUOUS DRIFT SCORING (KS + PSI)
# ============================================================

def exp2_calculate_psi(reference, comparison, num_buckets=10):
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


def exp2_compute_drift_score(X_ref, X_current, feature_names):
    """
    Compute continuous drift score combining two-sample KS and PSI.
    drift_score = 0.5 * mean(KS) + 0.5 * min(mean(PSI), 2.0)/2.0
    """
    ks_scores, psi_scores = [], []
    for i in range(X_ref.shape[1]):
        ks_stat, _ = stats.ks_2samp(X_ref[:, i], X_current[:, i])
        psi_val = exp2_calculate_psi(X_ref[:, i], X_current[:, i])
        ks_scores.append(ks_stat)
        psi_scores.append(psi_val)

    mean_ks = float(np.mean(ks_scores))
    mean_psi = float(np.mean(psi_scores))
    norm_psi = min(mean_psi, 2.0) / 2.0

    drift_score = 0.5 * mean_ks + 0.5 * norm_psi
    drift_score = float(min(drift_score, 1.0))
    return drift_score, np.array(ks_scores), np.array(psi_scores)


# ============================================================
# 6. MODEL 3: ADAPTIVE HETEROGENEOUS ENSEMBLE CLASSIFIER
# ============================================================

class AdaptiveEnsembleModel3:
    """
    Model 3: Adaptive Heterogeneous Ensemble Model.
    Heterogeneous Base Learners:
      1. RandomForestClassifier (30 estimators, max_depth=5)
      2. ExtraTreesClassifier (30 estimators, max_depth=5)
      3. GradientBoostingClassifier (30 estimators, max_depth=3)
    Decision Rule:
      Soft voting / probability averaging across base models:
        P(y=1) = (P_rf + P_et + P_gb) / 3.0
        pred = 1 if P(y=1) >= 0.5 else 0

    Adaptation:
      Rolling window of 50 most recent samples.
      Retrained every 20 streaming observations on observed samples only.

    Telemetry:
      Records wall-clock training times for each base learner:
        rf_train_time, et_train_time, gb_train_time, total_train_time
      Records wall-clock inference times for each base learner & aggregation:
        rf_infer_time, et_infer_time, gb_infer_time, aggregation_time, total_infer_time
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self.rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=seed)
        self.et = ExtraTreesClassifier(n_estimators=30, max_depth=5, random_state=seed)
        self.gb = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=seed)

        # Training telemetry
        self.last_fit_breakdown = {'rf': 0.0, 'et': 0.0, 'gb': 0.0, 'total': 0.0}
        self.fit_history = []

        # Inference telemetry accumulators
        self.infer_latencies = []
        self.rf_infer_times = []
        self.et_infer_times = []
        self.gb_infer_times = []
        self.agg_times = []

    def fit(self, X, y):
        t0 = time.time()
        t_rf_start = time.time()
        self.rf.fit(X, y)
        t_rf = time.time() - t_rf_start

        t_et_start = time.time()
        self.et.fit(X, y)
        t_et = time.time() - t_et_start

        t_gb_start = time.time()
        self.gb.fit(X, y)
        t_gb = time.time() - t_gb_start

        t_total = time.time() - t0
        self.last_fit_breakdown = {
            'rf': float(t_rf),
            'et': float(t_et),
            'gb': float(t_gb),
            'total': float(t_total)
        }
        self.fit_history.append(self.last_fit_breakdown)
        return self

    def _get_base_prob(self, model, X):
        probs = model.predict_proba(X)
        if probs.shape[1] == 1:
            cls = model.classes_[0]
            return np.ones(len(X)) if cls == 1 else np.zeros(len(X))
        cls_idx = np.where(model.classes_ == 1)[0]
        if len(cls_idx) > 0:
            return probs[:, cls_idx[0]]
        return probs[:, 1]

    def predict_proba(self, X):
        t0 = time.perf_counter()
        t_rf_start = time.perf_counter()
        p_rf = self._get_base_prob(self.rf, X)
        t_rf = time.perf_counter() - t_rf_start

        t_et_start = time.perf_counter()
        p_et = self._get_base_prob(self.et, X)
        t_et = time.perf_counter() - t_et_start

        t_gb_start = time.perf_counter()
        p_gb = self._get_base_prob(self.gb, X)
        t_gb = time.perf_counter() - t_gb_start

        t_agg_start = time.perf_counter()
        p_pos = (p_rf + p_et + p_gb) / 3.0
        p_neg = 1.0 - p_pos
        t_agg = time.perf_counter() - t_agg_start
        t_total = time.perf_counter() - t0

        self.infer_latencies.append(t_total)
        self.rf_infer_times.append(t_rf)
        self.et_infer_times.append(t_et)
        self.gb_infer_times.append(t_gb)
        self.agg_times.append(t_agg)

        return np.column_stack([p_neg, p_pos])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def reset_infer_telemetry(self):
        self.infer_latencies = []
        self.rf_infer_times = []
        self.et_infer_times = []
        self.gb_infer_times = []
        self.agg_times = []


# ============================================================
# 7. UCB1 SLIDING-WINDOW BANDIT
# ============================================================

class UCB1SlidingWindowBandit:
    """
    UCB1 (Upper Confidence Bound) bandit with sliding-window reward estimates.
    Arms:
      Arm 0 -> Model 1 (Frozen RF)
      Arm 1 -> Model 2 (Adaptive ET)
      Arm 2 -> Model 3 (Adaptive Ensemble)
    """

    def __init__(self, n_arms=3, window_size=50, exploration_constant=2.0):
        self.n_arms = n_arms
        self.window_size = window_size
        self.c = exploration_constant
        self.history = []      # (arm, reward) tuples
        self.total_pulls = 0
        self.selection_log = []

    def _get_window_stats(self):
        recent = self.history[-self.window_size:] \
            if len(self.history) > self.window_size else self.history
        counts = np.zeros(self.n_arms)
        sums = np.zeros(self.n_arms)
        for arm, reward in recent:
            counts[arm] += 1
            sums[arm] += reward
        return counts, sums

    def select_arm(self):
        counts, sums = self._get_window_stats()
        N = max(counts.sum(), 1)
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if counts[i] == 0:
                ucb_values[i] = float('inf')
            else:
                mean_reward = sums[i] / counts[i]
                exploration_bonus = np.sqrt(self.c * np.log(N) / counts[i])
                ucb_values[i] = mean_reward + exploration_bonus
        selected = int(np.argmax(ucb_values))
        self.selection_log.append(selected)
        return selected

    def update(self, arm, reward):
        self.history.append((arm, reward))
        self.total_pulls += 1

    def get_selection_counts(self):
        counts = np.zeros(self.n_arms, dtype=int)
        for arm in self.selection_log:
            counts[arm] += 1
        return counts

    def reset(self):
        self.history = []
        self.total_pulls = 0
        self.selection_log = []


# ============================================================
# 8. UTILITY FUNCTIONS
# ============================================================

def exp2_get_pos_probs(model, X_scaled):
    """Safely retrieve positive-class probabilities."""
    probs = model.predict_proba(X_scaled)
    if probs.shape[1] == 1:
        cls = model.classes_[0]
        return np.ones(len(X_scaled)) if cls == 1 else np.zeros(len(X_scaled))
    cls_idx = np.where(model.classes_ == 1)[0]
    if len(cls_idx) > 0:
        return probs[:, cls_idx[0]]
    return probs[:, 1]


def exp2_compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.5
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc}


# ============================================================
# 9. MAIN EXPERIMENT 2 PIPELINE
# ============================================================

def run_experiment2():
    print("=" * 80)
    print("  EXPERIMENT 2 -- DRIFT-AWARE BANDIT MODEL SELECTION")
    print("  AI4Mobile Industrial Telemetry -- Four Core KPIs")
    print("  Arm 0: M1 Frozen RF | Arm 1: M2 Adaptive ET | Arm 2: M3 Adaptive Ensemble")
    print("=" * 80)

    # ----------------------------------------------------------
    # STEP 0: Capture environment
    # ----------------------------------------------------------
    env_info = capture_environment()
    print(f"\n[EXP2] Environment:")
    print(f"    Python: {env_info['python_version'].split()[0]}")
    print(f"    CPU: {env_info['processor']} ({env_info['cpu_count_logical']} logical cores, {env_info['cpu_count_physical']} physical)")
    print(f"    RAM: {env_info['total_ram_gb']} GB")
    print(f"    GPU: {env_info['gpu_info']}")

    # ----------------------------------------------------------
    # STEP 1: Generate stationary dataset
    # ----------------------------------------------------------
    df = exp2_generate_dataset(n_samples=6000, csv_filename=EXP2_DATASET_CSV)
    print(f"[EXP2] KPIs: {EXP2_KPI_COLS}")

    # ----------------------------------------------------------
    # STEP 2: Feature extraction
    # ----------------------------------------------------------
    X_all, y_all, feat_names, ts_all = exp2_extract_features(df)
    n_total = len(X_all)
    print(f"\n[EXP2] Feature extraction: {n_total} windows x {len(feat_names)} features")

    # ----------------------------------------------------------
    # STEP 3: Chronological split
    # ----------------------------------------------------------
    n_train = int(n_total * EXP2_SPLIT['train'])
    n_adapt = int(n_total * EXP2_SPLIT['adapt'])
    n_val   = int(n_total * EXP2_SPLIT['val'])
    n_test  = n_total - n_train - n_adapt - n_val

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_adapt, y_adapt = X_all[n_train:n_train+n_adapt], y_all[n_train:n_train+n_adapt]
    X_val,   y_val   = X_all[n_train+n_adapt:n_train+n_adapt+n_val], \
                       y_all[n_train+n_adapt:n_train+n_adapt+n_val]
    X_test,  y_test  = X_all[n_train+n_adapt+n_val:], y_all[n_train+n_adapt+n_val:]

    print(f"\n[EXP2] Chronological Split (no shuffling):")
    print(f"    Historical Train: {n_train} windows (0 to {n_train-1})")
    print(f"    Adaptation:       {n_adapt} windows ({n_train} to {n_train+n_adapt-1})")
    print(f"    Validation:       {n_val} windows ({n_train+n_adapt} to {n_train+n_adapt+n_val-1})")
    print(f"    Test:             {n_test} windows ({n_train+n_adapt+n_val} to {n_total-1})")
    print(f"    Train target: 0={int((y_train==0).sum())}, 1={int((y_train==1).sum())} "
          f"({y_train.mean()*100:.1f}% positive)")
    print(f"    Test target:  0={int((y_test==0).sum())}, 1={int((y_test==1).sum())} "
          f"({y_test.mean()*100:.1f}% positive)")

    # ----------------------------------------------------------
    # PLOT 01: Chronological Split Timeline
    # ----------------------------------------------------------
    plt.figure(figsize=(14, 3))
    plt.axvspan(0, n_train, color='#aec7e8', alpha=0.6,
                label=f'Historical Train ({n_train})')
    plt.axvspan(n_train, n_train+n_adapt, color='#ffbb78', alpha=0.6,
                label=f'Adaptation ({n_adapt})')
    plt.axvspan(n_train+n_adapt, n_train+n_adapt+n_val, color='#98df8a', alpha=0.6,
                label=f'Validation ({n_val})')
    plt.axvspan(n_train+n_adapt+n_val, n_total, color='#ff9896', alpha=0.6,
                label=f'Test ({n_test})')
    for x_pos in [n_train, n_train+n_adapt, n_train+n_adapt+n_val]:
        plt.axvline(x_pos, color='black', linestyle='--', linewidth=1.5)
    plt.title("Experiment 2 -- Chronological Dataset Split (No Data Leakage)", fontweight='bold')
    plt.xlabel("Window Index")
    plt.yticks([])
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_01_chronological_split.png'), dpi=300)
    plt.close()
    print("[EXP2] Saved Plot 01: Chronological Split")

    # ----------------------------------------------------------
    # STEP 4: Fit scaler on training data
    # ----------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ----------------------------------------------------------
    # STEP 5: Train Model 1 (Historical -- FROZEN)
    # ----------------------------------------------------------
    print("\n" + "-" * 80)
    print("  TRAINING BASELINE MODELS")
    print("-" * 80)

    t0 = time.time()
    model_1 = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=EXP2_SEED)
    model_1.fit(X_train_scaled, y_train)
    model_1_train_time = time.time() - t0
    print(f"[EXP2] Model 1 (Historical RF) trained on {n_train} windows -- FROZEN")
    print(f"    Training time: {model_1_train_time:.4f}s")

    # ----------------------------------------------------------
    # STEP 6: Train Model 2 (Adaptive)
    # ----------------------------------------------------------
    X_adapt_scaled = scaler.transform(X_adapt)
    X_m2_pool = np.vstack([X_train_scaled, X_adapt_scaled])
    y_m2_pool = np.concatenate([y_train, y_adapt])
    X_m2_init = X_m2_pool[-EXP2_MODEL2_ADAPT_WINDOW:]
    y_m2_init = y_m2_pool[-EXP2_MODEL2_ADAPT_WINDOW:]

    t0 = time.time()
    model_2 = ExtraTreesClassifier(n_estimators=50, max_depth=7, random_state=EXP2_SEED)
    model_2.fit(X_m2_init, y_m2_init)
    model_2_train_time = time.time() - t0
    print(f"[EXP2] Model 2 (Adaptive ET) trained on {len(X_m2_init)} recent windows")
    print(f"    Training time: {model_2_train_time:.4f}s")

    # ----------------------------------------------------------
    # STEP 7: Train Model 3 (Adaptive Heterogeneous Ensemble)
    # ----------------------------------------------------------
    X_m3_init = X_m2_pool[-EXP2_MODEL3_ADAPT_WINDOW:]
    y_m3_init = y_m2_pool[-EXP2_MODEL3_ADAPT_WINDOW:]

    model_3 = AdaptiveEnsembleModel3(seed=EXP2_SEED)
    model_3.fit(X_m3_init, y_m3_init)
    model_3_train_time = model_3.last_fit_breakdown['total']
    model_3_rf_train = model_3.last_fit_breakdown['rf']
    model_3_et_train = model_3.last_fit_breakdown['et']
    model_3_gb_train = model_3.last_fit_breakdown['gb']

    print(f"[EXP2] Model 3 (Adaptive Ensemble: RF+ET+GB) trained on {len(X_m3_init)} recent windows")
    print(f"    Total training time: {model_3_train_time:.4f}s "
          f"(RF: {model_3_rf_train:.4f}s, ET: {model_3_et_train:.4f}s, GB: {model_3_gb_train:.4f}s)")

    # ==================================================================
    # STEP 8: RUN BANDIT EVALUATION ACROSS ALL DRIFT LEVELS
    # ==================================================================
    print("\n" + "=" * 80)
    print("  RUNNING BANDIT EVALUATION ACROSS DRIFT LEVELS")
    print("=" * 80)

    all_results = []
    all_drift_scores = []
    all_selection_histories = {}
    all_comp_perf = []
    all_cumulative_regrets = {}

    proc = psutil.Process()

    for drift_pct in EXP2_DRIFT_LEVELS:
        drift_label = f"{int(drift_pct*100)}%"
        print(f"\n" + "-" * 70)
        print(f"  Drift Level: {drift_label}")
        print("-" * 70)

        # Apply controlled drift to test and validation features
        X_test_drifted = exp2_apply_drift(
            X_test, feat_names, drift_pct, seed=EXP2_SEED)
        X_test_d_scaled = scaler.transform(X_test_drifted)

        X_val_drifted = exp2_apply_drift(
            X_val, feat_names, drift_pct, seed=EXP2_SEED + 1)
        X_val_d_scaled = scaler.transform(X_val_drifted)

        # Compute drift score (reference = training set)
        drift_score, ks_arr, psi_arr = exp2_compute_drift_score(
            X_train, X_test_drifted, feat_names)
        all_drift_scores.append({
            'drift_level': drift_pct,
            'drift_score': drift_score,
            'mean_ks': float(np.mean(ks_arr)),
            'mean_psi': float(np.mean(psi_arr))
        })
        print(f"    Drift score: {drift_score:.4f} "
              f"(KS={np.mean(ks_arr):.4f}, PSI={np.mean(psi_arr):.4f})")

        # ---- Fresh copies of adaptive models for this drift level ----
        m2 = ExtraTreesClassifier(n_estimators=50, max_depth=7, random_state=EXP2_SEED)
        m2.fit(X_m2_init, y_m2_init)

        m3 = AdaptiveEnsembleModel3(seed=EXP2_SEED)
        m3.fit(X_m3_init, y_m3_init)
        m3.reset_infer_telemetry()

        # Adaptation buffer (pre-test labelled data, scaled)
        adapt_buf_X = list(X_m2_pool)
        adapt_buf_y = list(y_m2_pool)

        # ---- Bandit calibration on validation set ----
        bandit = UCB1SlidingWindowBandit(
            n_arms=3,
            window_size=EXP2_BANDIT_WINDOW,
            exploration_constant=EXP2_UCB_EXPLORATION
        )
        models = [model_1, m2, m3]

        for vi in range(len(X_val_drifted)):
            x_i = X_val_d_scaled[vi:vi+1]
            preds_i = [m.predict(x_i)[0] for m in models]
            arm = bandit.select_arm()
            reward = 1.0 if preds_i[arm] == y_val[vi] else 0.0
            bandit.update(arm, reward)

        # Clear selection log for test phase
        bandit.selection_log = []

        # ---- Sequential test evaluation & fine-grained latency profiling ----
        mem_before = proc.memory_info().rss / (1024**2)
        cpu_readings = [psutil.cpu_percent(interval=None)]
        t_total_start = time.time()

        m1_preds, m2_preds, m3_preds, bandit_preds = [], [], [], []
        m1_probs, m2_probs, m3_probs, bandit_probs = [], [], [], []
        sel_history = []
        binary_regrets = []

        m1_infer_latencies = []
        m2_infer_latencies = []
        m3_infer_latencies = []

        m2_updates, m3_updates = 0, 0
        m2_adapt_time, m3_adapt_time = 0.0, 0.0
        m3_rf_adapt_time, m3_et_adapt_time, m3_gb_adapt_time = 0.0, 0.0, 0.0

        t_inf_start = time.time()

        for ti in range(n_test):
            x_i = X_test_d_scaled[ti:ti+1]
            y_i = y_test[ti]

            # Model 1 inference
            t_m1 = time.perf_counter()
            p1 = model_1.predict(x_i)[0]
            pr1 = exp2_get_pos_probs(model_1, x_i)[0]
            m1_infer_latencies.append((time.perf_counter() - t_m1) * 1000.0)

            # Model 2 inference
            t_m2 = time.perf_counter()
            p2 = m2.predict(x_i)[0]
            pr2 = exp2_get_pos_probs(m2, x_i)[0]
            m2_infer_latencies.append((time.perf_counter() - t_m2) * 1000.0)

            # Model 3 inference (Adaptive Ensemble)
            t_m3 = time.perf_counter()
            pr3 = exp2_get_pos_probs(m3, x_i)[0]
            p3 = int(pr3 >= 0.5)
            m3_infer_latencies.append((time.perf_counter() - t_m3) * 1000.0)

            m1_preds.append(p1); m1_probs.append(pr1)
            m2_preds.append(p2); m2_probs.append(pr2)
            m3_preds.append(p3); m3_probs.append(pr3)

            # Bandit selection
            arm = bandit.select_arm()
            sel_history.append(arm)

            all_p = [p1, p2, p3]
            all_pr = [pr1, pr2, pr3]
            bandit_preds.append(all_p[arm])
            bandit_probs.append(all_pr[arm])

            reward = 1.0 if all_p[arm] == y_i else 0.0
            bandit.update(arm, reward)

            # Instantaneous regret vs oracle arm
            cand_rewards = [1.0 if p == y_i else 0.0 for p in all_p]
            binary_regrets.append(max(cand_rewards) - reward)

            # Model adaptation (observed sample and label now available)
            adapt_buf_X.append(X_test_d_scaled[ti])
            adapt_buf_y.append(y_i)

            if (ti + 1) % EXP2_ADAPT_INTERVAL == 0 and ti > 0:
                # Model 2: retrain on recent window of 200
                bX = np.array(adapt_buf_X[-EXP2_MODEL2_ADAPT_WINDOW:])
                bY = np.array(adapt_buf_y[-EXP2_MODEL2_ADAPT_WINDOW:])
                if len(np.unique(bY)) > 1:
                    ta = time.time()
                    m2.fit(bX, bY)
                    m2_adapt_time += time.time() - ta
                    m2_updates += 1

                # Model 3: retrain all ensemble members on recent window of 50
                bX3 = np.array(adapt_buf_X[-EXP2_MODEL3_ADAPT_WINDOW:])
                bY3 = np.array(adapt_buf_y[-EXP2_MODEL3_ADAPT_WINDOW:])
                if len(np.unique(bY3)) > 1:
                    ta = time.time()
                    m3.fit(bX3, bY3)
                    m3_adapt_time += time.time() - ta
                    m3_updates += 1
                    m3_rf_adapt_time += m3.last_fit_breakdown['rf']
                    m3_et_adapt_time += m3.last_fit_breakdown['et']
                    m3_gb_adapt_time += m3.last_fit_breakdown['gb']

        t_inf_total = time.time() - t_inf_start
        t_total = time.time() - t_total_start
        cpu_readings.append(psutil.cpu_percent(interval=None))
        mem_after = proc.memory_info().rss / (1024**2)

        # Convert to arrays
        m1_preds = np.array(m1_preds); m1_probs = np.array(m1_probs)
        m2_preds = np.array(m2_preds); m2_probs = np.array(m2_probs)
        m3_preds = np.array(m3_preds); m3_probs = np.array(m3_probs)
        bandit_preds = np.array(bandit_preds); bandit_probs = np.array(bandit_probs)
        cum_regret = np.cumsum(binary_regrets)
        all_cumulative_regrets[drift_label] = cum_regret

        # Classification metrics
        met_m1 = exp2_compute_metrics(y_test, m1_preds, m1_probs)
        met_m2 = exp2_compute_metrics(y_test, m2_preds, m2_probs)
        met_m3 = exp2_compute_metrics(y_test, m3_preds, m3_probs)
        met_bn = exp2_compute_metrics(y_test, bandit_preds, bandit_probs)

        sel_counts = bandit.get_selection_counts()

        print(f"    M1 (Frozen RF):     Acc={met_m1['accuracy']:.4f}  F1={met_m1['f1']:.4f}  AUC={met_m1['roc_auc']:.4f}")
        print(f"    M2 (Adaptive ET):   Acc={met_m2['accuracy']:.4f}  F1={met_m2['f1']:.4f}  AUC={met_m2['roc_auc']:.4f}  updates={m2_updates}")
        print(f"    M3 (Adaptive Ens):  Acc={met_m3['accuracy']:.4f}  F1={met_m3['f1']:.4f}  AUC={met_m3['roc_auc']:.4f}  updates={m3_updates}")
        print(f"    Bandit (UCB1):      Acc={met_bn['accuracy']:.4f}  F1={met_bn['f1']:.4f}  AUC={met_bn['roc_auc']:.4f}  Regret={cum_regret[-1]:.1f}")
        print(f"    Selections: M1={sel_counts[0]} ({sel_counts[0]/n_test*100:.1f}%)  "
              f"M2={sel_counts[1]} ({sel_counts[1]/n_test*100:.1f}%)  "
              f"M3={sel_counts[2]} ({sel_counts[2]/n_test*100:.1f}%)")

        # Store model selection results
        result_row = {
            'drift_level': drift_pct, 'drift_label': drift_label,
            'drift_score': drift_score,
        }
        for prefix, met in [('m1', met_m1), ('m2', met_m2), ('m3', met_m3), ('bandit', met_bn)]:
            for k, v in met.items():
                result_row[f'{prefix}_{k}'] = v
        result_row.update({
            'm1_selections': int(sel_counts[0]),
            'm2_selections': int(sel_counts[1]),
            'm3_selections': int(sel_counts[2]),
            'm1_sel_pct': sel_counts[0] / n_test * 100.0,
            'm2_sel_pct': sel_counts[1] / n_test * 100.0,
            'm3_sel_pct': sel_counts[2] / n_test * 100.0,
            'final_binary_regret': float(cum_regret[-1])
        })
        all_results.append(result_row)
        all_selection_histories[drift_label] = sel_history

        # Latency statistics (ms)
        m1_lat_mean = float(np.mean(m1_infer_latencies))
        m1_lat_med = float(np.median(m1_infer_latencies))
        m1_lat_p95 = float(np.percentile(m1_infer_latencies, 95))

        m2_lat_mean = float(np.mean(m2_infer_latencies))
        m2_lat_med = float(np.median(m2_infer_latencies))
        m2_lat_p95 = float(np.percentile(m2_infer_latencies, 95))

        m3_lat_mean = float(np.mean(m3_infer_latencies))
        m3_lat_med = float(np.median(m3_infer_latencies))
        m3_lat_p95 = float(np.percentile(m3_infer_latencies, 95))

        m3_rf_lat_mean = float(np.mean(m3.rf_infer_times) * 1000.0) if m3.rf_infer_times else 0.0
        m3_et_lat_mean = float(np.mean(m3.et_infer_times) * 1000.0) if m3.et_infer_times else 0.0
        m3_gb_lat_mean = float(np.mean(m3.gb_infer_times) * 1000.0) if m3.gb_infer_times else 0.0
        m3_agg_lat_mean = float(np.mean(m3.agg_times) * 1000.0) if m3.agg_times else 0.0

        cpu_avg = float(np.mean([c for c in cpu_readings if c > 0] or [psutil.cpu_percent(interval=None)]))
        cpu_peak = float(max(cpu_readings)) if cpu_readings else cpu_avg

        all_comp_perf.append({
            'drift_level': drift_pct, 'drift_label': drift_label,
            'm1_train_time': model_1_train_time,
            'm2_train_time': model_2_train_time,
            'm3_train_time': model_3_train_time,
            'm3_rf_train_time': model_3_rf_train,
            'm3_et_train_time': model_3_et_train,
            'm3_gb_train_time': model_3_gb_train,
            'm2_adapt_time': m2_adapt_time,
            'm3_adapt_time': m3_adapt_time,
            'm3_rf_adapt_time': m3_rf_adapt_time,
            'm3_et_adapt_time': m3_et_adapt_time,
            'm3_gb_adapt_time': m3_gb_adapt_time,
            'm2_updates': m2_updates,
            'm3_updates': m3_updates,
            'inference_time': t_inf_total,
            'm1_mean_infer_ms': m1_lat_mean,
            'm1_p95_infer_ms': m1_lat_p95,
            'm2_mean_infer_ms': m2_lat_mean,
            'm2_p95_infer_ms': m2_lat_p95,
            'm3_mean_infer_ms': m3_lat_mean,
            'm3_p95_infer_ms': m3_lat_p95,
            'm3_rf_infer_ms': m3_rf_lat_mean,
            'm3_et_infer_ms': m3_et_lat_mean,
            'm3_gb_infer_ms': m3_gb_lat_mean,
            'm3_agg_infer_ms': m3_agg_lat_mean,
            'total_runtime': t_total,
            'samples_processed': n_test,
            'cpu_avg_pct': cpu_avg,
            'cpu_peak_pct': cpu_peak,
            'ram_usage_mb': mem_after,
            'ram_delta_mb': mem_after - mem_before,
        })

        # Save predictions
        pd.DataFrame({
            'y_true': y_test,
            'pred_m1': m1_preds, 'prob_m1': m1_probs,
            'pred_m2': m2_preds, 'prob_m2': m2_probs,
            'pred_m3': m3_preds, 'prob_m3': m3_probs,
            'pred_bandit': bandit_preds, 'prob_bandit': bandit_probs,
            'bandit_arm': sel_history
        }).to_csv(os.path.join(
            EXP2_DIRS['predictions'],
            f'exp2_predictions_drift{int(drift_pct*100)}.csv'), index=False)

    # ==================================================================
    # STEP 9: RESULT TABLES & EFFICIENCY / COST ANALYSIS
    # ==================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2 -- RESULTS TABLES")
    print("=" * 80)

    df_results = pd.DataFrame(all_results)

    # Table A: Model Selection Analysis
    cols_a = ['drift_label', 'drift_score',
              'm1_f1', 'm2_f1', 'm3_f1', 'bandit_f1',
              'm1_sel_pct', 'm2_sel_pct', 'm3_sel_pct']
    table_a = df_results[cols_a].copy()
    table_a.columns = ['Drift', 'Score', 'M1 F1', 'M2 F1', 'M3 Ensemble F1',
                        'Bandit F1', 'M1%', 'M2%', 'M3%']
    print("\n  TABLE A -- Model Selection Analysis:")
    print("  " + "-" * 95)
    print(table_a.to_string(index=False, float_format='%.4f'))

    # Table B: Full metrics
    table_b_rows = []
    for r in all_results:
        for mk, mn in [('m1', 'M1 Frozen RF'), ('m2', 'M2 Adaptive ET'),
                       ('m3', 'M3 Adaptive Ensemble'), ('bandit', 'Bandit UCB1')]:
            table_b_rows.append({
                'Drift': r['drift_label'], 'Model': mn,
                'Acc': r[f'{mk}_accuracy'], 'Prec': r[f'{mk}_precision'],
                'Rec': r[f'{mk}_recall'], 'F1': r[f'{mk}_f1'],
                'AUC': r[f'{mk}_roc_auc'],
            })
    df_full = pd.DataFrame(table_b_rows)
    print("\n  TABLE B -- Full Metrics:")
    print("  " + "-" * 95)
    print(df_full.to_string(index=False, float_format='%.4f'))

    # Table C: Computational performance
    df_comp = pd.DataFrame(all_comp_perf)
    print("\n  TABLE C -- Computational Performance:")
    print("  " + "-" * 95)
    comp_cols = ['drift_label', 'm1_train_time', 'm2_train_time', 'm3_train_time',
                 'm2_adapt_time', 'm3_adapt_time', 'm1_mean_infer_ms', 'm2_mean_infer_ms',
                 'm3_mean_infer_ms', 'cpu_avg_pct', 'ram_usage_mb']
    print(df_comp[comp_cols].to_string(index=False, float_format='%.4f'))

    # Table D: Performance-vs-Cost & Model Efficiency Analysis
    m1_mean_f1 = float(df_results['m1_f1'].mean())
    m2_mean_f1 = float(df_results['m2_f1'].mean())
    m3_mean_f1 = float(df_results['m3_f1'].mean())

    m1_f1_50 = float(df_results.iloc[-1]['m1_f1'])
    m2_f1_50 = float(df_results.iloc[-1]['m2_f1'])
    m3_f1_50 = float(df_results.iloc[-1]['m3_f1'])

    m1_sel_0 = float(df_results.iloc[0]['m1_sel_pct'])
    m2_sel_0 = float(df_results.iloc[0]['m2_sel_pct'])
    m3_sel_0 = float(df_results.iloc[0]['m3_sel_pct'])

    m1_sel_50 = float(df_results.iloc[-1]['m1_sel_pct'])
    m2_sel_50 = float(df_results.iloc[-1]['m2_sel_pct'])
    m3_sel_50 = float(df_results.iloc[-1]['m3_sel_pct'])

    m1_mean_inf = float(df_comp['m1_mean_infer_ms'].mean())
    m2_mean_inf = float(df_comp['m2_mean_infer_ms'].mean())
    m3_mean_inf = float(df_comp['m3_mean_infer_ms'].mean())

    m1_p95_inf = float(df_comp['m1_p95_infer_ms'].mean())
    m2_p95_inf = float(df_comp['m2_p95_infer_ms'].mean())
    m3_p95_inf = float(df_comp['m3_p95_infer_ms'].mean())

    m2_tot_adapt = float(df_comp['m2_adapt_time'].mean())
    m3_tot_adapt = float(df_comp['m3_adapt_time'].mean())

    eff_rows = [
        {
            'Model': 'M1 (Frozen RF)',
            'Mean_F1': m1_mean_f1,
            'F1_50pct_Drift': m1_f1_50,
            'Initial_Train_s': model_1_train_time,
            'Adaptation_Time_s': 0.0,
            'Mean_Infer_Latency_ms': m1_mean_inf,
            'P95_Infer_Latency_ms': m1_p95_inf,
            'Training_Overhead_vs_M1': 1.0,
            'Inference_Overhead_vs_M1': 1.0,
            'Selection_0pct_Drift': m1_sel_0,
            'Selection_50pct_Drift': m1_sel_50,
        },
        {
            'Model': 'M2 (Adaptive ET)',
            'Mean_F1': m2_mean_f1,
            'F1_50pct_Drift': m2_f1_50,
            'Initial_Train_s': model_2_train_time,
            'Adaptation_Time_s': m2_tot_adapt,
            'Mean_Infer_Latency_ms': m2_mean_inf,
            'P95_Infer_Latency_ms': m2_p95_inf,
            'Training_Overhead_vs_M1': model_2_train_time / model_1_train_time,
            'Inference_Overhead_vs_M1': m2_mean_inf / m1_mean_inf,
            'Selection_0pct_Drift': m2_sel_0,
            'Selection_50pct_Drift': m2_sel_50,
        },
        {
            'Model': 'M3 (Adaptive Ensemble)',
            'Mean_F1': m3_mean_f1,
            'F1_50pct_Drift': m3_f1_50,
            'Initial_Train_s': model_3_train_time,
            'Adaptation_Time_s': m3_tot_adapt,
            'Mean_Infer_Latency_ms': m3_mean_inf,
            'P95_Infer_Latency_ms': m3_p95_inf,
            'Training_Overhead_vs_M1': model_3_train_time / model_1_train_time,
            'Inference_Overhead_vs_M1': m3_mean_inf / m1_mean_inf,
            'Selection_0pct_Drift': m3_sel_0,
            'Selection_50pct_Drift': m3_sel_50,
        },
    ]
    df_efficiency = pd.DataFrame(eff_rows)
    print("\n  TABLE D -- Model Efficiency & Performance vs Cost Summary:")
    print("  " + "-" * 95)
    print(df_efficiency.to_string(index=False, float_format='%.4f'))

    # Save CSVs
    df_results.to_csv(os.path.join(EXP2_DIRS['metrics'], 'exp2_model_selection_analysis.csv'), index=False)
    df_full.to_csv(os.path.join(EXP2_DIRS['metrics'], 'exp2_full_metrics.csv'), index=False)
    df_comp.to_csv(os.path.join(EXP2_DIRS['metrics'], 'exp2_computational_performance.csv'), index=False)
    df_efficiency.to_csv(os.path.join(EXP2_DIRS['metrics'], 'exp2_efficiency_and_cost_analysis.csv'), index=False)
    pd.DataFrame(all_drift_scores).to_csv(os.path.join(EXP2_DIRS['drift'], 'exp2_drift_scores.csv'), index=False)
    print("\n[EXP2] Saved all CSV outputs to experiment2/metrics/ and experiment2/drift_analysis/")

    # ==================================================================
    # STEP 10: PLOTS
    # ==================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2 -- GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)

    # ---- Plot 02: Feature distributions ----
    fig, axes = plt.subplots(len(EXP2_KPI_COLS), len(EXP2_DRIFT_LEVELS),
                             figsize=(4*len(EXP2_DRIFT_LEVELS), 3*len(EXP2_KPI_COLS)))
    for ki, kpi in enumerate(EXP2_KPI_COLS):
        fi = feat_names.index(f'{kpi}_mean')
        for di, dl in enumerate(EXP2_DRIFT_LEVELS):
            ax = axes[ki, di]
            Xd = exp2_apply_drift(X_test, feat_names, dl, seed=EXP2_SEED)
            ax.hist(X_train[:, fi], bins=20, alpha=0.5, label='Train', color='blue', density=True)
            ax.hist(Xd[:, fi], bins=20, alpha=0.5, label=f'Test {int(dl*100)}%', color='red', density=True)
            ax.set_title(f'{kpi}_mean @ {int(dl*100)}%', fontsize=8)
            if ki == 0 and di == 0:
                ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
    plt.suptitle("Experiment 2 -- Feature Distributions by Drift Level", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_02_feature_distributions_by_drift.png'), dpi=200)
    plt.close()

    # ---- Plot 03: Drift score vs level ----
    plt.figure(figsize=(8, 5))
    lvls = [d['drift_level']*100 for d in all_drift_scores]
    plt.plot(lvls, [d['drift_score'] for d in all_drift_scores],
             'o-', label='Combined Score (KS+PSI)', color='#d62728', linewidth=2, markersize=8)
    plt.plot(lvls, [d['mean_ks'] for d in all_drift_scores],
             's--', label='Mean KS', color='#1f77b4', alpha=0.7)
    plt.plot(lvls, [d['mean_psi'] for d in all_drift_scores],
             '^--', label='Mean PSI', color='#2ca02c', alpha=0.7)
    plt.xlabel("Applied Drift Level (%)"); plt.ylabel("Score")
    plt.title("Experiment 2 -- Continuous Drift Score vs Applied Drift Level", fontweight='bold')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_03_drift_score_vs_level.png'), dpi=300)
    plt.close()

    # ---- Plot 04: Model F1 vs Drift ----
    plt.figure(figsize=(10, 6))
    dp = [r['drift_level']*100 for r in all_results]
    plt.plot(dp, [r['m1_f1'] for r in all_results], 'o-',
             label='M1 (Frozen RF)', color='#1f77b4', linewidth=2)
    plt.plot(dp, [r['m2_f1'] for r in all_results], 's-',
             label='M2 (Adaptive ET, W=200)', color='#2ca02c', linewidth=2)
    plt.plot(dp, [r['m3_f1'] for r in all_results], '^-',
             label='M3 (Adaptive Ensemble, W=50)', color='#9467bd', linewidth=2)
    plt.plot(dp, [r['bandit_f1'] for r in all_results], 'D-',
             label='Bandit (UCB1)', color='#ff7f0e', linewidth=2.5, markersize=8)
    plt.xlabel("Applied Drift Level (%)"); plt.ylabel("F1 Score")
    plt.title("Experiment 2 -- Model F1 Scores vs Drift Level (M3 Ensemble)", fontweight='bold')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05); plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_04_model_f1_vs_drift.png'), dpi=300)
    plt.close()

    # ---- Plot 05: Bandit Selection Distribution (Stacked Bar) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    xp = np.arange(len(EXP2_DRIFT_LEVELS))
    s1 = [r['m1_sel_pct'] for r in all_results]
    s2 = [r['m2_sel_pct'] for r in all_results]
    s3 = [r['m3_sel_pct'] for r in all_results]
    ax.bar(xp, s1, 0.5, label='M1 (Frozen RF)', color='#1f77b4')
    ax.bar(xp, s2, 0.5, bottom=s1, label='M2 (Adaptive ET)', color='#2ca02c')
    ax.bar(xp, s3, 0.5, bottom=[a+b for a,b in zip(s1,s2)],
           label='M3 (Adaptive Ensemble)', color='#9467bd')
    ax.set_xticks(xp); ax.set_xticklabels([f'{int(d*100)}%' for d in EXP2_DRIFT_LEVELS])
    ax.set_xlabel("Applied Drift Level"); ax.set_ylabel("Selection %")
    ax.set_title("Experiment 2 -- Bandit Model Selection Distribution", fontweight='bold')
    ax.legend(); ax.set_ylim(0, 105); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_05_bandit_selection_distribution.png'), dpi=300)
    plt.close()

    # ---- Plot 06: Selection Timeline (0%, 30%, 50%) ----
    show_levels = ['0%', '30%', '50%']
    fig, axes = plt.subplots(len(show_levels), 1, figsize=(14, 3*len(show_levels)), sharex=True)
    mcols = {0: '#1f77b4', 1: '#2ca02c', 2: '#9467bd'}
    mlabs = {0: 'M1 (Frozen)', 1: 'M2 (Adaptive ET)', 2: 'M3 (Adaptive Ensemble)'}
    for ax, lstr in zip(axes, show_levels):
        if lstr in all_selection_histories:
            hist = all_selection_histories[lstr]
            ns = min(250, len(hist))
            colors = [mcols[h] for h in hist[:ns]]
            ax.bar(range(ns), [1]*ns, color=colors, width=1.0, edgecolor='none')
            ax.set_ylabel(f"Drift {lstr}", fontsize=9)
            ax.set_yticks([]); ax.set_xlim(0, ns)
            if ax == axes[0]:
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor=mcols[i], label=mlabs[i]) for i in range(3)],
                          loc='upper right', fontsize=8, ncol=3)
    axes[-1].set_xlabel("Test Window Index")
    plt.suptitle("Experiment 2 -- Sequential Model Selection Timeline by Bandit", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_06_bandit_selection_timeline.png'), dpi=300)
    plt.close()

    # ---- Plot 07: Cumulative Regret Across Drift Levels ----
    plt.figure(figsize=(10, 6))
    for dl_label, c_reg in all_cumulative_regrets.items():
        plt.plot(c_reg, label=f"Drift {dl_label}", linewidth=2)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Cumulative Binary Regret")
    plt.title("Experiment 2 -- Cumulative Binary Regret Over Time", fontweight='bold')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_07_cumulative_regret.png'), dpi=300)
    plt.close()

    # ---- Plot 08: Model Selection vs Drift Level (Grouped Bars) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    ax.bar(xp - width, s1, width, label='M1 (Frozen RF)', color='#1f77b4')
    ax.bar(xp, s2, width, label='M2 (Adaptive ET)', color='#2ca02c')
    ax.bar(xp + width, s3, width, label='M3 (Adaptive Ensemble)', color='#9467bd')
    ax.set_xticks(xp); ax.set_xticklabels([f'{int(d*100)}%' for d in EXP2_DRIFT_LEVELS])
    ax.set_xlabel("Drift Level"); ax.set_ylabel("Selection Percentage (%)")
    ax.set_title("Experiment 2 -- Arm Selection Percentage vs Drift Level", fontweight='bold')
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_08_selection_distribution_by_drift.png'), dpi=300)
    plt.close()

    # ---- Plot 09: Training & Adaptation Times Comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(['M1\n(Frozen RF)', 'M2\n(Adaptive ET)', 'M3\n(Adaptive Ens)'],
           [model_1_train_time, model_2_train_time, model_3_train_time],
           color=['#1f77b4', '#2ca02c', '#9467bd'])
    ax.set_ylabel("Seconds"); ax.set_title("Initial Training Wall-Clock Time", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, tv in enumerate([model_1_train_time, model_2_train_time, model_3_train_time]):
        ax.text(i, tv + 0.002, f"{tv:.4f}s", ha='center', fontsize=9)

    ax = axes[1]
    ax.bar(xp - 0.15, [c['m2_adapt_time'] for c in all_comp_perf], 0.3,
           label='M2 (Adaptive ET)', color='#2ca02c')
    ax.bar(xp + 0.15, [c['m3_adapt_time'] for c in all_comp_perf], 0.3,
           label='M3 (Adaptive Ensemble)', color='#9467bd')
    ax.set_xticks(xp); ax.set_xticklabels([f'{int(d*100)}%' for d in EXP2_DRIFT_LEVELS])
    ax.set_xlabel("Drift Level"); ax.set_ylabel("Total Adaptation Time (s)")
    ax.set_title("Cumulative Adaptation Time (15 Retrainings)", fontweight='bold')
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.suptitle("Experiment 2 -- Training & Adaptation Performance", fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_09_training_and_adaptation_times.png'), dpi=300)
    plt.close()

    # ---- Plot 10: Inference Latency by Model (Mean, Median, P95) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x_mod = np.arange(3)
    means = [m1_mean_inf, m2_mean_inf, m3_mean_inf]
    meds = [float(df_comp['m1_mean_infer_ms'].median()), float(df_comp['m2_mean_infer_ms'].median()), float(df_comp['m3_mean_infer_ms'].median())]
    p95s = [m1_p95_inf, m2_p95_inf, m3_p95_inf]

    ax.bar(x_mod - 0.25, means, 0.25, label='Mean Latency', color='#4a90e2')
    ax.bar(x_mod, meds, 0.25, label='Median Latency', color='#50e3c2')
    ax.bar(x_mod + 0.25, p95s, 0.25, label='P95 Latency', color='#f5a623')
    ax.set_xticks(x_mod); ax.set_xticklabels(['M1 (Frozen RF)', 'M2 (Adaptive ET)', 'M3 (Adaptive Ensemble)'])
    ax.set_ylabel("Inference Latency per Sample (ms)")
    ax.set_title("Experiment 2 -- Inference Latency Profile by Model", fontweight='bold')
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_10_inference_latency_by_model.png'), dpi=300)
    plt.close()

    # ---- Plot 11: Hardware Telemetry (RAM & CPU Usage) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(dp, [c['ram_usage_mb'] for c in all_comp_perf], 'o-', color='#e74c3c', linewidth=2)
    ax.set_xlabel("Drift Level (%)"); ax.set_ylabel("Process RAM (MB)")
    ax.set_title("RAM Footprint Across Evaluation", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1]
    ax.plot(dp, [c['cpu_avg_pct'] for c in all_comp_perf], 's-', color='#3498db', linewidth=2, label='Avg CPU %')
    ax.plot(dp, [c['cpu_peak_pct'] for c in all_comp_perf], '^--', color='#2980b9', linewidth=1.5, label='Peak CPU %')
    ax.set_xlabel("Drift Level (%)"); ax.set_ylabel("CPU Utilization (%)")
    ax.set_title("CPU Utilization Across Evaluation", fontweight='bold')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("Experiment 2 -- Hardware Resource Profiling", fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_11_hardware_usage_by_model.png'), dpi=300)
    plt.close()

    # ---- Plot 12: M3 Ensemble Component Breakdown ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(['RF Component', 'ET Component', 'GB Component'],
           [model_3_rf_train, model_3_et_train, model_3_gb_train],
           color=['#3498db', '#2ecc71', '#e67e22'])
    ax.set_ylabel("Seconds"); ax.set_title("M3 Base Learners: Initial Fit Time", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, tv in enumerate([model_3_rf_train, model_3_et_train, model_3_gb_train]):
        ax.text(i, tv + 0.001, f"{tv:.4f}s", ha='center', fontsize=9)

    ax = axes[1]
    ax.bar(['RF Infer', 'ET Infer', 'GB Infer', 'Soft-Voting Agg'],
           [m3_rf_lat_mean, m3_et_lat_mean, m3_gb_lat_mean, m3_agg_lat_mean],
           color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'])
    ax.set_ylabel("Milliseconds per Sample"); ax.set_title("M3 Inference Latency Breakdown", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, tv in enumerate([m3_rf_lat_mean, m3_et_lat_mean, m3_gb_lat_mean, m3_agg_lat_mean]):
        ax.text(i, tv + 0.05, f"{tv:.2f}ms", ha='center', fontsize=9)
    plt.suptitle("Experiment 2 -- M3 Heterogeneous Ensemble Internal Breakdown", fontweight='bold', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_12_m3_ensemble_component_breakdown.png'), dpi=300)
    plt.close()

    # ---- Plot 13: ROC Curves ----
    roc_dl = [0.0, 0.30, 0.50]
    fig, axes = plt.subplots(1, len(roc_dl), figsize=(6*len(roc_dl), 5))
    for ax, dl in zip(axes, roc_dl):
        pf = os.path.join(EXP2_DIRS['predictions'], f'exp2_predictions_drift{int(dl*100)}.csv')
        if not os.path.exists(pf):
            continue
        dp_p = pd.read_csv(pf)
        yt = dp_p['y_true'].values
        if len(np.unique(yt)) > 1:
            for pcol, lab, col in [('prob_m1','M1 (Frozen)','#1f77b4'),
                                   ('prob_m2','M2 (Adaptive ET)','#2ca02c'),
                                   ('prob_m3','M3 (Adaptive Ensemble)','#9467bd'),
                                   ('prob_bandit','Bandit (UCB1)','#ff7f0e')]:
                fpr, tpr, _ = roc_curve(yt, dp_p[pcol].values)
                aval = roc_auc_score(yt, dp_p[pcol].values)
                lw = 2.5 if 'bandit' in pcol else 1.5
                ax.plot(fpr, tpr, label=f'{lab} ({aval:.3f})', color=col, linewidth=lw)
        ax.plot([0,1],[0,1],'k--',alpha=0.4)
        ax.set_title(f'Drift {int(dl*100)}%', fontweight='bold')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(fontsize=7); ax.grid(True, linestyle='--', alpha=0.4)
    plt.suptitle("Experiment 2 -- ROC Curves Across Key Drift Regimes", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EXP2_DIRS['plots'], 'exp2_13_roc_curves.png'), dpi=300)
    plt.close()
    print("[EXP2] Saved all 13 diagnostic figures to experiment2/plots/")

    # ==================================================================
    # STEP 11: SAVE CONFIG
    # ==================================================================
    config = {
        'experiment': 'Experiment 2 -- Drift-Aware Bandit Model Selection',
        'dataset_size': 6000,
        'dataset_file': EXP2_DATASET_CSV,
        'dataset_type': 'STATIONARY baseline (no built-in drift)',
        'chronological_split': EXP2_SPLIT,
        'actual_split_sizes': {'train': n_train, 'adapt': n_adapt, 'val': n_val, 'test': n_test},
        'selected_kpis': EXP2_KPI_COLS,
        'rolling_window_size': EXP2_WINDOW_SIZE,
        'rolling_window_step': EXP2_STEP_SIZE,
        'n_features': len(feat_names),
        'feature_names': feat_names,
        'drift_levels': EXP2_DRIFT_LEVELS,
        'drift_method': 'covariate_shift_fixed_magnitude',
        'drift_full_shifts': EXP2_FULL_DRIFT,
        'drift_definition': 'drift_level p shifts each KPI mean by p * FULL_DRIFT[kpi]. '
                           'Target labels NOT modified (pure covariate drift P(X)).',
        'drift_scoring': 'combined_mean_KS_normalized_PSI',
        'bandit_algorithm': 'UCB1_sliding_window',
        'bandit_window': EXP2_BANDIT_WINDOW,
        'bandit_exploration_constant': EXP2_UCB_EXPLORATION,
        'model1': {
            'type': 'RandomForestClassifier', 'n_estimators': 50, 'max_depth': 7,
            'frozen': True, 'description': 'Historical model. Never updated after initial training.'
        },
        'model2': {
            'type': 'ExtraTreesClassifier', 'n_estimators': 50, 'max_depth': 7,
            'adaptation_window': EXP2_MODEL2_ADAPT_WINDOW,
            'adaptation_interval': EXP2_ADAPT_INTERVAL,
            'description': 'Adaptive model. Retrained on sliding window of 200 recent samples.'
        },
        'model3': {
            'type': 'AdaptiveEnsembleModel3',
            'components': [
                {'name': 'RandomForestClassifier', 'n_estimators': 30, 'max_depth': 5},
                {'name': 'ExtraTreesClassifier', 'n_estimators': 30, 'max_depth': 5},
                {'name': 'GradientBoostingClassifier', 'n_estimators': 30, 'max_depth': 3}
            ],
            'combination_strategy': 'soft_voting_probability_averaging',
            'adaptation_window': EXP2_MODEL3_ADAPT_WINDOW,
            'adaptation_interval': EXP2_ADAPT_INTERVAL,
            'description': 'Adaptive Heterogeneous Ensemble. Fast adaptation (W=50) with model diversity.'
        },
        'seed': EXP2_SEED,
        'environment': env_info,
    }
    cfg_path = os.path.join(EXP2_DIRS['reports'], 'exp2_experiment_configuration.json')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    print(f"\n[EXP2] Config saved: {cfg_path}")

    # ==================================================================
    # STEP 12: INTERPRETATION & PERFORMANCE-VS-COST SYNTHESIS
    # ==================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2 -- PERFORMANCE VS COST SYNTHESIS")
    print("=" * 80)

    print("\n  1. Predictive Performance vs Drift:")
    for r in all_results:
        best_model = 'M1' if r['m1_f1'] >= max(r['m2_f1'], r['m3_f1']) \
            else ('M2' if r['m2_f1'] >= r['m3_f1'] else 'M3 Ensemble')
        print(f"    Drift {r['drift_label']:>3s}: "
              f"M1={r['m1_sel_pct']:5.1f}%  M2={r['m2_sel_pct']:5.1f}%  M3={r['m3_sel_pct']:5.1f}%  |  "
              f"Best: {best_model} (F1={max(r['m1_f1'], r['m2_f1'], r['m3_f1']):.4f})")

    print("\n  2. Computational Overhead of M3 Ensemble:")
    print(f"    Training overhead vs M1: {model_3_train_time / model_1_train_time:.2f}x")
    print(f"    Inference overhead vs M1: {m3_mean_inf / m1_mean_inf:.2f}x")
    print(f"    Adaptation overhead vs M2: {m3_tot_adapt / m2_tot_adapt:.2f}x")
    print(f"    M3 Component training times: RF={model_3_rf_train:.4f}s, ET={model_3_et_train:.4f}s, GB={model_3_gb_train:.4f}s")
    print(f"    M3 Component inference latencies: RF={m3_rf_lat_mean:.2f}ms, ET={m3_et_lat_mean:.2f}ms, GB={m3_gb_lat_mean:.2f}ms, Agg={m3_agg_lat_mean:.2f}ms")

    return df_results, df_full, df_comp, df_efficiency


if __name__ == '__main__':
    run_experiment2()
