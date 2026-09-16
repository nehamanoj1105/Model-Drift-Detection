import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)

DATASET_PATH = "ai4mobile_sample_stream.csv"

# Raw signal columns in the AI4Mobile dataset (12 signals)
RAW_SIGNAL_COLS = [
    'rsrp', 'sinr', 'agv_speed', 'packet_loss',
    'sig5g_1', 'sig5g_2', 'wifi_rssi', 'delay_jitter',
    'agv_pos_x', 'agv_pos_y', 'battery_volt', 'temp_sensor'
]
TARGET_COL = 'target_qos_violation'

# Feature extraction parameters
WINDOW_SIZE = 20
STEP_SIZE = 5

# Output directories
DIRS = {
    'models': 'models',
    'predictions': 'predictions',
    'metrics': 'metrics',
    'plots': 'plots',
    'drift': 'drift_analysis'
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ============================================================
# 1. UTILITY FUNCTIONS
# ============================================================

def get_pos_probs(model, X_scaled):
    """
    Safely retrieves positive class probabilities from a trained sklearn classifier,
    handling single-class edge cases gracefully.
    """
    probs = model.predict_proba(X_scaled)
    if probs.shape[1] == 1:
        cls = model.classes_[0]
        return np.ones(len(X_scaled)) if cls == 1 else np.zeros(len(X_scaled))
    else:
        cls_idx = np.where(model.classes_ == 1)[0]
        if len(cls_idx) > 0:
            return probs[:, cls_idx[0]]
        else:
            return probs[:, 1]


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics for a single model/case."""
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
# 2. DATA LOADING & DRIFT APPLICATION
# ============================================================

def load_ai4mobile_dataset(filepath):
    """
    Loads the AI4Mobile industrial telemetry dataset from CSV.
    No data is generated — uses the existing dataset as-is.
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"[+] Loaded AI4Mobile dataset: {filepath}")
    print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"    Signals: {RAW_SIGNAL_COLS}")
    print(f"    Target: {TARGET_COL}")
    return df


def apply_controlled_drift(df_second_half, seed=42):
    """
    Applies realistic, controlled industrial telemetry feature drift
    (Line-of-Sight -> NLOS Interference Zone) to the second half of the dataset.

    Drift shifts applied (pure covariate drift P(X)):
    - RSRP:         mean shift ↓ 20 dB  (signal strength degrades in NLOS)
    - SINR:         mean shift ↓ 10 dB  (interference increases)
    - wifi_rssi:    mean shift ↓ 15 dB  (WiFi signal degrades)
    - agv_speed:    mean shift ↑ 0.5 m/s (AGV speeds up in open zone)
    - delay_jitter: mean shift ↑ 3 ms   (latency increases in NLOS)
    - sig5g_1/2:    mean shift ↓ 10 dB  (5G signal degrades)
    - packet_loss:  scale ↑ 2x          (more packet drops)

    Target (target_qos_violation) is NOT modified — this is pure feature/covariate drift.
    """
    np.random.seed(seed)
    df_drifted = df_second_half.copy()
    n = len(df_drifted)

    # Apply controlled shifts to feature distributions
    df_drifted['rsrp'] = df_drifted['rsrp'].values - 20.0 + np.random.normal(0, 2.0, n)
    df_drifted['sinr'] = df_drifted['sinr'].values - 10.0 + np.random.normal(0, 1.5, n)
    df_drifted['wifi_rssi'] = df_drifted['wifi_rssi'].values - 15.0 + np.random.normal(0, 2.0, n)
    df_drifted['agv_speed'] = df_drifted['agv_speed'].values + 0.5 + np.random.normal(0, 0.1, n)
    df_drifted['agv_speed'] = np.clip(df_drifted['agv_speed'].values, 0.1, 4.0)
    df_drifted['delay_jitter'] = df_drifted['delay_jitter'].values + 3.0 + np.random.normal(0, 0.5, n)
    df_drifted['delay_jitter'] = np.clip(df_drifted['delay_jitter'].values, 0.5, 20.0)
    df_drifted['sig5g_1'] = df_drifted['sig5g_1'].values - 10.0 + np.random.normal(0, 1.5, n)
    df_drifted['sig5g_2'] = df_drifted['sig5g_2'].values - 10.0 + np.random.normal(0, 1.5, n)
    df_drifted['packet_loss'] = df_drifted['packet_loss'].values * 2.0 + np.random.exponential(0.005, n)

    # Target is explicitly NOT modified — pure covariate drift P(X)
    return df_drifted


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================

def extract_window_features(df, window_size=20, step_size=5):
    """
    Extracts rolling window statistical features from raw telemetry signals.

    Per signal (12 signals):
        6 summary stats: mean, std, min, max, q25, q75  ->  12 x 6 = 72 features

    Cross-metric interaction features:
        sinr_to_jitter_ratio (mean, std)                 ->  2 features

    Total = 74 features per window.
    """
    feature_names = []
    for col in RAW_SIGNAL_COLS:
        for stat in ['mean', 'std', 'min', 'max', 'q25', 'q75']:
            feature_names.append(f"{col}_{stat}")
    feature_names.extend(['sinr_jitter_ratio_mean', 'sinr_jitter_ratio_std'])

    X_list = []
    y_list = []
    timestamps_list = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start: start + window_size]

        row_feat = []
        for col in RAW_SIGNAL_COLS:
            vals = window[col].values
            row_feat.extend([
                np.mean(vals),
                np.std(vals),
                np.min(vals),
                np.max(vals),
                np.percentile(vals, 25),
                np.percentile(vals, 75)
            ])

        # Cross-metric interaction: SINR-to-Jitter ratio
        sinr_vals = window['sinr'].values
        jitter_vals = window['delay_jitter'].values + 1e-5
        ratio = sinr_vals / jitter_vals
        row_feat.extend([np.mean(ratio), np.std(ratio)])

        target = window[TARGET_COL].iloc[-1]
        ts = window['timestamp'].iloc[-1]

        X_list.append(row_feat)
        y_list.append(target)
        timestamps_list.append(ts)

    return np.array(X_list), np.array(y_list), feature_names, pd.to_datetime(timestamps_list)


# ============================================================
# 4. DRIFT QUANTIFICATION METRICS (KS + PSI)
# ============================================================

def calculate_psi(reference, comparison, num_buckets=10):
    """
    Calculates Population Stability Index (PSI) for a single numerical feature.
    PSI < 0.10: No significant drift
    PSI 0.10–0.25: Moderate drift
    PSI > 0.25: Severe drift
    """
    eps = 1e-4
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(reference, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5
    buckets = np.unique(buckets)
    if len(buckets) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=buckets)
    comp_counts, _ = np.histogram(comparison, bins=buckets)

    ref_pct = ref_counts / len(reference) + eps
    comp_pct = comp_counts / len(comparison) + eps

    psi_val = np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct))
    return float(psi_val)


def analyze_drift(X_ref, X_comp, feature_names):
    """
    Computes KS statistics, p-values, PSI, and mean/std shifts between
    reference and comparison feature matrices.
    """
    drift_records = []
    for i, fname in enumerate(feature_names):
        ref_feat = X_ref[:, i]
        comp_feat = X_comp[:, i]

        ks_stat, ks_pval = stats.ks_2samp(ref_feat, comp_feat)
        psi_val = calculate_psi(ref_feat, comp_feat)

        ref_mean, comp_mean = np.mean(ref_feat), np.mean(comp_feat)
        ref_std, comp_std = np.std(ref_feat), np.std(comp_feat)
        mean_diff = comp_mean - ref_mean

        drift_records.append({
            'feature': fname,
            'ref_mean': ref_mean,
            'comp_mean': comp_mean,
            'mean_diff': mean_diff,
            'ref_std': ref_std,
            'comp_std': comp_std,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pval,
            'psi': psi_val,
            'is_drifted': (psi_val > 0.25) or (ks_stat > 0.3 and ks_pval < 0.01)
        })

    return pd.DataFrame(drift_records)


# ============================================================
# 5. WEIGHTED ENSEMBLE LEARNER
# ============================================================

class WeightedEnsemble2:
    """
    Weighted Ensemble combining 2 models (Model 1 + Model 2).
    Weights are optimized on validation set to maximize validation F1 score.
    Used for Case A (No Drift).
    """
    def __init__(self, model1, model2):
        self.m1 = model1
        self.m2 = model2
        self.w1 = 0.5
        self.w2 = 0.5

    def fit_weights(self, X_val, y_val, scaler):
        X_val_scaled = scaler.transform(X_val)
        p1 = get_pos_probs(self.m1, X_val_scaled)
        p2 = get_pos_probs(self.m2, X_val_scaled)

        best_score = -1.0
        best_w1 = 0.5

        # Grid search over W1 from 0.0 to 1.0 in steps of 0.01
        candidate_weights = np.linspace(0.0, 1.0, 101)
        for w1 in candidate_weights:
            w2 = 1.0 - w1
            p_ens = w1 * p1 + w2 * p2
            preds = (p_ens >= 0.5).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_w1 = w1

        self.w1 = float(best_w1)
        self.w2 = float(1.0 - best_w1)
        return self.w1, self.w2, best_score

    def predict_proba(self, X_scaled):
        p1 = get_pos_probs(self.m1, X_scaled)
        p2 = get_pos_probs(self.m2, X_scaled)
        return self.w1 * p1 + self.w2 * p2

    def predict(self, X_scaled):
        p_ens = self.predict_proba(X_scaled)
        return (p_ens >= 0.5).astype(int)


class WeightedEnsemble3:
    """
    Weighted Ensemble combining 3 models (Model 1 + Model 2 + Model 3).
    Weights are optimized on validation set to maximize validation F1 score.
    Used for Case B (With Drift).
    """
    def __init__(self, model1, model2, model3):
        self.m1 = model1
        self.m2 = model2
        self.m3 = model3
        self.w1 = 1.0 / 3.0
        self.w2 = 1.0 / 3.0
        self.w3 = 1.0 / 3.0

    def fit_weights(self, X_val, y_val, scaler):
        X_val_scaled = scaler.transform(X_val)
        p1 = get_pos_probs(self.m1, X_val_scaled)
        p2 = get_pos_probs(self.m2, X_val_scaled)
        p3 = get_pos_probs(self.m3, X_val_scaled)

        best_score = -1.0
        best_w1, best_w2, best_w3 = 1/3, 1/3, 1/3

        # Grid search over (w1, w2, w3) with step 0.05, sum = 1.0
        step = 0.05
        for w1 in np.arange(0.0, 1.0 + step/2, step):
            for w2 in np.arange(0.0, 1.0 - w1 + step/2, step):
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(w3, 0.0)

                p_ens = w1 * p1 + w2 * p2 + w3 * p3
                preds = (p_ens >= 0.5).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_w1, best_w2, best_w3 = w1, w2, w3

        self.w1 = float(round(best_w1, 4))
        self.w2 = float(round(best_w2, 4))
        self.w3 = float(round(best_w3, 4))
        return self.w1, self.w2, self.w3, best_score

    def predict_proba(self, X_scaled):
        p1 = get_pos_probs(self.m1, X_scaled)
        p2 = get_pos_probs(self.m2, X_scaled)
        p3 = get_pos_probs(self.m3, X_scaled)
        return self.w1 * p1 + self.w2 * p2 + self.w3 * p3

    def predict(self, X_scaled):
        p_ens = self.predict_proba(X_scaled)
        return (p_ens >= 0.5).astype(int)


# ============================================================
# 6. MAIN EXPERIMENT PIPELINE
# ============================================================

def run_experiment():
    print("=" * 80)
    print("  ENSEMBLE LEARNING FOR MODEL DRIFT DETECTION")
    print("  AI4Mobile Industrial Telemetry Dataset")
    print("  Pipeline: Model 1 (First Half) -> Model 2/3 (Second Half) -> Weighted Ensemble")
    print("=" * 80)

    # ==================================================================
    # STEP A: Load the existing AI4Mobile dataset 
    # ==================================================================
    df_complete = load_ai4mobile_dataset(DATASET_PATH)

    target_counts = df_complete[TARGET_COL].value_counts()
    print(f"    Target distribution: 0={target_counts.get(0, 0)}, 1={target_counts.get(1, 0)}")

    # ------------------------------------------------------------------
    # PLOT 01: Target Distribution
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    counts_0 = target_counts.get(0, 0)
    counts_1 = target_counts.get(1, 0)
    plt.bar(['Normal (0)', 'QoS Violation (1)'], [counts_0, counts_1],
            color=['#2ca02c', '#d62728'], width=0.5)
    plt.title("Target Distribution (target_qos_violation)")
    plt.ylabel("Sample Count")
    for i, v in enumerate([counts_0, counts_1]):
        plt.text(i, v + 50, f"{v} ({v/len(df_complete)*100:.1f}%)", ha='center', fontweight='bold')
    plt.ylim(0, max(counts_0, counts_1) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '01_target_distribution.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 01: Target Distribution")

    # ------------------------------------------------------------------
    # PLOT 02: Raw Feature Distributions
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
    for ax, col, color in zip(axes.flatten(), RAW_SIGNAL_COLS, colors):
        ax.hist(df_complete[col], bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(f"Distribution: {col}", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle("Raw Signal Feature Distributions (AI4Mobile Dataset)", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(DIRS['plots'], '02_feature_distributions.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 02: Feature Distributions")

    # ------------------------------------------------------------------
    # PLOT 03: Missing Values Summary
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    null_counts = df_complete.isnull().sum()
    plt.bar(null_counts.index, null_counts.values, color='#1f77b4')
    plt.title("Missing Values Summary per Column")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylim(0, max(10, null_counts.max() + 5))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '03_missing_values_summary.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 03: Missing Values Summary")

    # ==================================================================
    # STEP B: Chronological 50/50 Split -> First Half / Second Half
    # ==================================================================
    half_idx = len(df_complete) // 2  # 2500
    df_first_half = df_complete.iloc[:half_idx].copy()
    df_second_half_normal = df_complete.iloc[half_idx:].copy()

    print(f"\n[+] Chronological Split:")
    print(f"    First Half  (Training):  rows 0–{half_idx-1} ({len(df_first_half)} samples)")
    print(f"    Second Half (Runtime):   rows {half_idx}–{len(df_complete)-1} ({len(df_second_half_normal)} samples)")

    # ==================================================================
    # STEP C: Apply controlled drift to second half (Case B)
    # ==================================================================
    df_second_half_drifted = apply_controlled_drift(df_second_half_normal, seed=SEED)
    print("[+] Created drifted version of second half (pure covariate drift P(X), target unchanged).")

    # ------------------------------------------------------------------
    # PLOT 04: Data Split Timeline Diagram
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    total_n = len(df_complete)
    plt.axvspan(0, half_idx, color='#aec7e8', alpha=0.6,
                label=f'First Half — Training Data (0–{half_idx-1})')
    plt.axvspan(half_idx, half_idx + (total_n - half_idx) // 2, color='#ffbb78', alpha=0.6,
                label='Second Half — Validation (Model 2 Train + Weight Learning)')
    plt.axvspan(half_idx + (total_n - half_idx) // 2, total_n, color='#98df8a', alpha=0.6,
                label='Second Half — Test (Final Held-Out Evaluation)')
    plt.axvline(half_idx, color='black', linestyle='--', linewidth=2, label='50/50 Split Point')
    plt.axvline(half_idx + (total_n - half_idx) // 2, color='gray', linestyle=':', linewidth=2,
                label='Val/Test Split Point')
    plt.title("Chronological Dataset Splitting Strategy (Zero Data Leakage)", fontweight='bold')
    plt.xlabel("Sample Index")
    plt.ylabel("Timeline Partition")
    plt.yticks([])
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '04_data_split_timeline.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 04: Data Split Timeline")

    # ==================================================================
    # STEP D: Feature Extraction (windowed)
    # ==================================================================
    X_first_half, y_first_half, feat_names, ts_first = extract_window_features(
        df_first_half, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    X_sec_norm, y_sec_norm, _, ts_sec_norm = extract_window_features(
        df_second_half_normal, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    X_sec_drift, y_sec_drift, _, ts_sec_drift = extract_window_features(
        df_second_half_drifted, window_size=WINDOW_SIZE, step_size=STEP_SIZE)

    print(f"\n[+] Feature Extraction Complete ({len(feat_names)} features per window):")
    print(f"    First Half Normal:   {X_first_half.shape[0]} windows × {X_first_half.shape[1]} features")
    print(f"    Second Half Normal:  {X_sec_norm.shape[0]} windows")
    print(f"    Second Half Drifted: {X_sec_drift.shape[0]} windows")

    # ------------------------------------------------------------------
    # PLOT 05: Temporal Feature & Target Behavior (Normal vs Drifted)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df_complete.index, df_complete['rsrp'], label='RSRP Normal (dBm)', color='#1f77b4', alpha=0.7)
    axes[0].plot(df_second_half_drifted.index, df_second_half_drifted['rsrp'],
                 label='RSRP Drifted (dBm)', color='darkred', alpha=0.7)
    axes[0].axvline(half_idx, color='black', linestyle='--', label='50/50 Split')
    axes[0].set_ylabel("RSRP (dBm)")
    axes[0].set_title("Temporal Feature & Target Behavior (Normal vs Drifted)", fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(df_complete.index, df_complete['sinr'], label='SINR Normal (dB)', color='teal', alpha=0.7)
    axes[1].plot(df_second_half_drifted.index, df_second_half_drifted['sinr'],
                 label='SINR Drifted (dB)', color='darkgreen', alpha=0.7)
    axes[1].axvline(half_idx, color='black', linestyle='--', label='50/50 Split')
    axes[1].set_ylabel("SINR (dB)")
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].plot(df_complete.index, df_complete['delay_jitter'], label='Jitter Normal (ms)', color='crimson', alpha=0.7)
    axes[2].plot(df_second_half_drifted.index, df_second_half_drifted['delay_jitter'],
                 label='Jitter Drifted (ms)', color='purple', alpha=0.7)
    axes[2].axvline(half_idx, color='black', linestyle='--', label='50/50 Split')
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("Delay Jitter (ms)")
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '05_temporal_feature_target_behavior.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 05: Temporal Feature Behavior")

    # ==================================================================
    # STEP E: Drift Statistical Analysis (KS + PSI)
    # ==================================================================
    drift_caseA_df = analyze_drift(X_first_half, X_sec_norm, feat_names)
    drift_caseB_df = analyze_drift(X_first_half, X_sec_drift, feat_names)

    drift_caseA_df.to_csv(os.path.join(DIRS['drift'], 'drift_analysis_caseA_nodrift.csv'), index=False)
    drift_caseB_df.to_csv(os.path.join(DIRS['drift'], 'drift_analysis_caseB_drifted.csv'), index=False)
    print("[+] Saved drift analysis CSVs (KS test & PSI).")

    n_drifted_A = drift_caseA_df['is_drifted'].sum()
    n_drifted_B = drift_caseB_df['is_drifted'].sum()
    print(f"    Case A (No Drift): {n_drifted_A}/{len(feat_names)} features flagged as drifted")
    print(f"    Case B (Drift):    {n_drifted_B}/{len(feat_names)} features flagged as drifted")

    # ------------------------------------------------------------------
    # PLOT 06: Normal vs Drifted Feature Distributions (key features)
    # ------------------------------------------------------------------
    key_feats = ['rsrp_mean', 'sinr_mean', 'delay_jitter_mean', 'agv_speed_mean',
                 'wifi_rssi_mean', 'sinr_jitter_ratio_mean']
    n_key = min(len(key_feats), 6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, fname in zip(axes.flatten()[:n_key], key_feats[:n_key]):
        idx = feat_names.index(fname)
        ax.hist(X_first_half[:, idx], bins=25, alpha=0.5, label='1st Half Normal', color='blue', density=True)
        ax.hist(X_sec_norm[:, idx], bins=25, alpha=0.5, label='2nd Half Normal', color='green', density=True)
        ax.hist(X_sec_drift[:, idx], bins=25, alpha=0.5, label='2nd Half Drifted', color='red', density=True)
        ax.set_title(f"Distribution: {fname}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)
    # Hide any unused axes
    for ax in axes.flatten()[n_key:]:
        ax.set_visible(False)
    plt.suptitle("Feature Distribution Comparison: Normal vs Drifted", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(DIRS['plots'], '06_drift_feature_distribution_comparison.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 06: Normal vs Drifted Feature Distributions")

    # ------------------------------------------------------------------
    # PLOT 07: PSI Drift Magnitude Bar Chart
    # ------------------------------------------------------------------
    top_drifted = drift_caseB_df.sort_values(by='psi', ascending=False).head(12)
    plt.figure(figsize=(10, 6))
    plt.barh(top_drifted['feature'], top_drifted['psi'], color='#d62728')
    plt.axvline(0.25, color='black', linestyle='--', label='PSI = 0.25 (Severe Drift Threshold)')
    plt.axvline(0.10, color='gray', linestyle=':', label='PSI = 0.10 (Moderate Drift Threshold)')
    plt.title("Top Drifted Features by Population Stability Index (PSI)", fontweight='bold')
    plt.xlabel("PSI Value")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '07_drift_magnitude_psi.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 07: PSI Drift Magnitude")

    # ------------------------------------------------------------------
    # PLOT 08: KS-Test Drift Results
    # ------------------------------------------------------------------
    top_ks = drift_caseB_df.sort_values(by='ks_stat', ascending=False).head(12)
    plt.figure(figsize=(10, 6))
    bar_colors = ['#d62728' if p < 0.01 else '#ff7f0e' if p < 0.05 else '#2ca02c'
                  for p in top_ks['ks_pvalue']]
    plt.barh(top_ks['feature'], top_ks['ks_stat'], color=bar_colors)
    plt.axvline(0.3, color='black', linestyle='--', label='KS Stat = 0.3 (High Drift)')
    plt.title("KS-Test Statistics for Top Drifted Features (Case B)", fontweight='bold')
    plt.xlabel("KS Statistic")
    plt.gca().invert_yaxis()
    # Custom legend for p-value colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='p < 0.01 (Highly Significant)'),
        Patch(facecolor='#ff7f0e', label='p < 0.05 (Significant)'),
        Patch(facecolor='#2ca02c', label='p >= 0.05 (Not Significant)'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='KS = 0.3 Threshold')
    ]
    plt.legend(handles=legend_elements, fontsize=8)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '08_ks_test_drift_results.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 08: KS-Test Drift Results")

    # ==================================================================
    # STEP F: Split Second Half into Validation and Test (chronological)
    # ==================================================================
    val_split_idx = len(X_sec_norm) // 2

    # Case A: Second half normal
    X_val_cA, y_val_cA = X_sec_norm[:val_split_idx], y_sec_norm[:val_split_idx]
    X_test_cA, y_test_cA = X_sec_norm[val_split_idx:], y_sec_norm[val_split_idx:]

    # Case B: Second half drifted
    X_val_cB, y_val_cB = X_sec_drift[:val_split_idx], y_sec_drift[:val_split_idx]
    X_test_cB, y_test_cB = X_sec_drift[val_split_idx:], y_sec_drift[val_split_idx:]

    print(f"\n[+] Second Half Val/Test Split (chronological):")
    print(f"    Case A Val:  {X_val_cA.shape[0]} windows | Test: {X_test_cA.shape[0]} windows")
    print(f"    Case B Val:  {X_val_cB.shape[0]} windows | Test: {X_test_cB.shape[0]} windows")

    # ==================================================================
    # STEP G: Fit Scaler & Train Model 1 (First Half)
    # ==================================================================
    scaler = StandardScaler()
    X_first_half_scaled = scaler.fit_transform(X_first_half)

    model_1 = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=SEED)
    model_1.fit(X_first_half_scaled, y_first_half)

    joblib.dump(model_1, os.path.join(DIRS['models'], 'model_1.joblib'))
    joblib.dump(scaler, os.path.join(DIRS['models'], 'scaler.joblib'))
    print(f"\n[+] Model 1 (Random Forest) trained on First Half ({X_first_half.shape[0]} windows)")

    # ==================================================================
    # STEP H: CASE A — No Drift
    # ==================================================================
    print("\n" + "=" * 80)
    print("  CASE A: SECOND HALF — NO DRIFT")
    print("=" * 80)

    X_val_cA_scaled = scaler.transform(X_val_cA)
    model_2_cA = ExtraTreesClassifier(n_estimators=50, max_depth=7, random_state=SEED)
    model_2_cA.fit(X_val_cA_scaled, y_val_cA)
    joblib.dump(model_2_cA, os.path.join(DIRS['models'], 'model_2_caseA.joblib'))
    print(f"[+] Model 2 (Extra Trees) trained on Case A validation data ({X_val_cA.shape[0]} windows)")

    # Learn ensemble weights (2-model)
    ens_cA = WeightedEnsemble2(model_1, model_2_cA)
    w1_cA, w2_cA, val_f1_cA = ens_cA.fit_weights(X_val_cA, y_val_cA, scaler)
    print(f"[+] Case A Ensemble Weights: Model 1 = {w1_cA:.3f}, Model 2 = {w2_cA:.3f}")
    print(f"    Validation F1 = {val_f1_cA:.4f}")

    # Evaluate on held-out test
    X_test_cA_scaled = scaler.transform(X_test_cA)

    pred_m1_cA = model_1.predict(X_test_cA_scaled)
    prob_m1_cA = get_pos_probs(model_1, X_test_cA_scaled)

    pred_m2_cA = model_2_cA.predict(X_test_cA_scaled)
    prob_m2_cA = get_pos_probs(model_2_cA, X_test_cA_scaled)

    pred_ens_cA = ens_cA.predict(X_test_cA_scaled)
    prob_ens_cA = ens_cA.predict_proba(X_test_cA_scaled)

    # Save predictions
    df_preds_cA = pd.DataFrame({
        'y_true': y_test_cA,
        'prediction_model_1': pred_m1_cA,
        'prob_model_1': prob_m1_cA,
        'prediction_model_2': pred_m2_cA,
        'prob_model_2': prob_m2_cA,
        'prediction_ensemble': pred_ens_cA,
        'prob_ensemble': prob_ens_cA
    })
    df_preds_cA.to_csv(os.path.join(DIRS['predictions'], 'caseA_nodrift_predictions.csv'), index=False)

    # ==================================================================
    # STEP I: CASE B — With Drift
    # ==================================================================
    print("\n" + "=" * 80)
    print("  CASE B: SECOND HALF — WITH DRIFT")
    print("=" * 80)

    X_val_cB_scaled = scaler.transform(X_val_cB)

    model_2_cB = ExtraTreesClassifier(n_estimators=50, max_depth=7, random_state=SEED)
    model_2_cB.fit(X_val_cB_scaled, y_val_cB)
    joblib.dump(model_2_cB, os.path.join(DIRS['models'], 'model_2_caseB.joblib'))
    print(f"[+] Model 2 (Extra Trees) trained on Case B drifted validation data ({X_val_cB.shape[0]} windows)")

    model_3_cB = DecisionTreeClassifier(max_depth=5, random_state=SEED)
    model_3_cB.fit(X_val_cB_scaled, y_val_cB)
    joblib.dump(model_3_cB, os.path.join(DIRS['models'], 'model_3_caseB.joblib'))
    print(f"[+] Model 3 (Decision Tree) trained on Case B drifted validation data")
    print(f"    Rationale: lightweight tree adds ensemble diversity under drifted distributions")

    # Learn ensemble weights (3-model)
    ens_cB = WeightedEnsemble3(model_1, model_2_cB, model_3_cB)
    w1_cB, w2_cB, w3_cB, val_f1_cB = ens_cB.fit_weights(X_val_cB, y_val_cB, scaler)
    print(f"[+] Case B Ensemble Weights: Model 1 = {w1_cB:.3f}, Model 2 = {w2_cB:.3f}, Model 3 = {w3_cB:.3f}")
    print(f"    Validation F1 = {val_f1_cB:.4f}")

    # Evaluate on held-out test
    X_test_cB_scaled = scaler.transform(X_test_cB)

    pred_m1_cB = model_1.predict(X_test_cB_scaled)
    prob_m1_cB = get_pos_probs(model_1, X_test_cB_scaled)

    pred_m2_cB = model_2_cB.predict(X_test_cB_scaled)
    prob_m2_cB = get_pos_probs(model_2_cB, X_test_cB_scaled)

    pred_m3_cB = model_3_cB.predict(X_test_cB_scaled)
    prob_m3_cB = get_pos_probs(model_3_cB, X_test_cB_scaled)

    pred_ens_cB = ens_cB.predict(X_test_cB_scaled)
    prob_ens_cB = ens_cB.predict_proba(X_test_cB_scaled)

    # Save predictions
    df_preds_cB = pd.DataFrame({
        'y_true': y_test_cB,
        'prediction_model_1': pred_m1_cB,
        'prob_model_1': prob_m1_cB,
        'prediction_model_2': pred_m2_cB,
        'prob_model_2': prob_m2_cB,
        'prediction_model_3': pred_m3_cB,
        'prob_model_3': prob_m3_cB,
        'prediction_ensemble': pred_ens_cB,
        'prob_ensemble': prob_ens_cB
    })
    df_preds_cB.to_csv(os.path.join(DIRS['predictions'], 'caseB_drifted_predictions.csv'), index=False)

    # ==================================================================
    # STEP J: Compute Evaluation Metrics
    # ==================================================================
    metrics_m1_cA = compute_metrics(y_test_cA, pred_m1_cA, prob_m1_cA)
    metrics_m2_cA = compute_metrics(y_test_cA, pred_m2_cA, prob_m2_cA)
    metrics_ens_cA = compute_metrics(y_test_cA, pred_ens_cA, prob_ens_cA)

    metrics_m1_cB = compute_metrics(y_test_cB, pred_m1_cB, prob_m1_cB)
    metrics_m2_cB = compute_metrics(y_test_cB, pred_m2_cB, prob_m2_cB)
    metrics_m3_cB = compute_metrics(y_test_cB, pred_m3_cB, prob_m3_cB)
    metrics_ens_cB = compute_metrics(y_test_cB, pred_ens_cB, prob_ens_cB)

    summary_rows = [
        {'Case': 'Case A (No Drift)', 'Model': 'Model 1 (RF)', **metrics_m1_cA},
        {'Case': 'Case A (No Drift)', 'Model': 'Model 2 (ET)', **metrics_m2_cA},
        {'Case': 'Case A (No Drift)', 'Model': 'Ensemble', **metrics_ens_cA},
        {'Case': 'Case B (Drift)', 'Model': 'Model 1 (RF)', **metrics_m1_cB},
        {'Case': 'Case B (Drift)', 'Model': 'Model 2 (ET)', **metrics_m2_cB},
        {'Case': 'Case B (Drift)', 'Model': 'Model 3 (DT)', **metrics_m3_cB},
        {'Case': 'Case B (Drift)', 'Model': 'Ensemble', **metrics_ens_cB},
    ]
    df_metrics = pd.DataFrame(summary_rows)
    df_metrics.to_csv(os.path.join(DIRS['metrics'], 'final_model_comparison_metrics.csv'), index=False)

    # Print final performance table
    print("\n" + "=" * 80)
    print("  FINAL MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(df_metrics.to_string(index=False))

    # ==================================================================
    # STEP K: Degradation Analysis (No Drift -> Drift)
    # ==================================================================
    degradation_rows = []
    # Model 1 is in both cases
    for model_label, m_cA, m_cB in [
        ('Model 1 (RF)', metrics_m1_cA, metrics_m1_cB),
        ('Model 2 (ET)', metrics_m2_cA, metrics_m2_cB),
        ('Ensemble', metrics_ens_cA, metrics_ens_cB)
    ]:
        degradation_rows.append({
            'Model': model_label,
            'NoDrift_F1': m_cA['f1'],
            'Drift_F1': m_cB['f1'],
            'F1_Drop': m_cA['f1'] - m_cB['f1'],
            'NoDrift_AUC': m_cA['roc_auc'],
            'Drift_AUC': m_cB['roc_auc'],
            'AUC_Drop': m_cA['roc_auc'] - m_cB['roc_auc']
        })

    df_degradation = pd.DataFrame(degradation_rows)
    df_degradation.to_csv(os.path.join(DIRS['metrics'], 'performance_degradation_summary.csv'), index=False)

    print("\n" + "=" * 80)
    print("  PERFORMANCE DEGRADATION ANALYSIS (No Drift -> Drift)")
    print("=" * 80)
    print(df_degradation.to_string(index=False))

    # ==================================================================
    # STEP L: Save Learned Ensemble Weights
    # ==================================================================
    weights_summary = {
        'Case A (No Drift)': {
            'Model 1 Weight (RF)': w1_cA,
            'Model 2 Weight (ET)': w2_cA,
            'Validation F1': val_f1_cA
        },
        'Case B (Drift)': {
            'Model 1 Weight (RF)': w1_cB,
            'Model 2 Weight (ET)': w2_cB,
            'Model 3 Weight (DT)': w3_cB,
            'Validation F1': val_f1_cB
        }
    }
    with open(os.path.join(DIRS['metrics'], 'learned_ensemble_weights.json'), 'w') as f:
        json.dump(weights_summary, f, indent=4)

    print("\n" + "-" * 80)
    print("  LEARNED ENSEMBLE WEIGHTS")
    print("-" * 80)
    print(f"  Case A (No Drift):")
    print(f"    Model 1 (RF) = {w1_cA:.3f}")
    print(f"    Model 2 (ET) = {w2_cA:.3f}")
    print(f"  Case B (Drift):")
    print(f"    Model 1 (RF) = {w1_cB:.3f}")
    print(f"    Model 2 (ET) = {w2_cB:.3f}")
    print(f"    Model 3 (DT) = {w3_cB:.3f}")

    # ==================================================================
    # STEP M: Generate Remaining Plots (09–17)
    # ==================================================================

    # ------------------------------------------------------------------
    # PLOT 09: Case A Performance Comparison
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics_keys))
    w_bar = 0.25
    plt.bar(x - w_bar, [metrics_m1_cA[k] for k in metrics_keys], width=w_bar,
            label='Model 1 (RF)', color='#1f77b4')
    plt.bar(x, [metrics_m2_cA[k] for k in metrics_keys], width=w_bar,
            label='Model 2 (ET)', color='#2ca02c')
    plt.bar(x + w_bar, [metrics_ens_cA[k] for k in metrics_keys], width=w_bar,
            label='Ensemble', color='#ff7f0e')
    plt.xticks(x, [k.upper().replace('_', ' ') for k in metrics_keys])
    plt.ylim(0, 1.15)
    for i, k in enumerate(metrics_keys):
        for j, (vals, offset) in enumerate([
            ([metrics_m1_cA[k]], -w_bar),
            ([metrics_m2_cA[k]], 0),
            ([metrics_ens_cA[k]], w_bar)
        ]):
            plt.text(i + offset, vals[0] + 0.02, f"{vals[0]:.3f}", ha='center', fontsize=7, rotation=90)
    plt.title("Case A (No Drift): Model Performance Comparison", fontweight='bold')
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '09_caseA_model_performance.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 09: Case A Performance")

    # ------------------------------------------------------------------
    # PLOT 10: Case B Performance Comparison
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    w_bar4 = 0.2
    plt.bar(x - 1.5*w_bar4, [metrics_m1_cB[k] for k in metrics_keys], width=w_bar4,
            label='Model 1 (RF)', color='#1f77b4')
    plt.bar(x - 0.5*w_bar4, [metrics_m2_cB[k] for k in metrics_keys], width=w_bar4,
            label='Model 2 (ET)', color='#2ca02c')
    plt.bar(x + 0.5*w_bar4, [metrics_m3_cB[k] for k in metrics_keys], width=w_bar4,
            label='Model 3 (DT)', color='#9467bd')
    plt.bar(x + 1.5*w_bar4, [metrics_ens_cB[k] for k in metrics_keys], width=w_bar4,
            label='Ensemble', color='#ff7f0e')
    plt.xticks(x, [k.upper().replace('_', ' ') for k in metrics_keys])
    plt.ylim(0, 1.15)
    for i, k in enumerate(metrics_keys):
        for j, (vals, offset) in enumerate([
            ([metrics_m1_cB[k]], -1.5*w_bar4),
            ([metrics_m2_cB[k]], -0.5*w_bar4),
            ([metrics_m3_cB[k]], 0.5*w_bar4),
            ([metrics_ens_cB[k]], 1.5*w_bar4)
        ]):
            plt.text(i + offset, vals[0] + 0.02, f"{vals[0]:.3f}", ha='center', fontsize=6, rotation=90)
    plt.title("Case B (With Drift): Model Performance Comparison", fontweight='bold')
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '10_caseB_model_performance.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 10: Case B Performance")

    # ------------------------------------------------------------------
    # PLOT 11: F1 Degradation (No Drift -> Drift)
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    models_deg = df_degradation['Model']
    x_mod = np.arange(len(models_deg))
    bars1 = plt.bar(x_mod - 0.2, df_degradation['NoDrift_F1'], width=0.4,
                    label='No Drift F1', color='#2ca02c')
    bars2 = plt.bar(x_mod + 0.2, df_degradation['Drift_F1'], width=0.4,
                    label='Drift F1', color='#d62728')
    plt.xticks(x_mod, models_deg)
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.15)
    plt.title("F1 Score Degradation: No Drift -> Drift", fontweight='bold')
    for i in range(len(models_deg)):
        drop = df_degradation['F1_Drop'].iloc[i]
        drift_f1 = df_degradation['Drift_F1'].iloc[i]
        color = 'red' if drop > 0 else 'green'
        plt.text(i + 0.2, drift_f1 + 0.03,
                 f"Drop: {drop:+.3f}", ha='center', fontsize=9, fontweight='bold', color=color)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '11_f1_degradation.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 11: F1 Degradation")

    # ------------------------------------------------------------------
    # PLOT 12: Confusion Matrices (2 rows × 4 columns to include Model 3 in Case B)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cms = [
        # Case A row
        (confusion_matrix(y_test_cA, pred_m1_cA, labels=[0, 1]), "Case A: Model 1 (RF)"),
        (confusion_matrix(y_test_cA, pred_m2_cA, labels=[0, 1]), "Case A: Model 2 (ET)"),
        (confusion_matrix(y_test_cA, pred_ens_cA, labels=[0, 1]), "Case A: Ensemble"),
        None,  # Placeholder
        # Case B row
        (confusion_matrix(y_test_cB, pred_m1_cB, labels=[0, 1]), "Case B: Model 1 (RF)"),
        (confusion_matrix(y_test_cB, pred_m2_cB, labels=[0, 1]), "Case B: Model 2 (ET)"),
        (confusion_matrix(y_test_cB, pred_m3_cB, labels=[0, 1]), "Case B: Model 3 (DT)"),
        (confusion_matrix(y_test_cB, pred_ens_cB, labels=[0, 1]), "Case B: Ensemble"),
    ]
    for ax, item in zip(axes.flatten(), cms):
        if item is None:
            ax.set_visible(False)
            continue
        cm, title = item
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(title, fontsize=9)
    plt.suptitle("Confusion Matrices — All Models × Both Cases", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(DIRS['plots'], '12_confusion_matrices.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 12: Confusion Matrices")

    # ------------------------------------------------------------------
    # PLOT 13: ROC Curves (Both Cases)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Case A ROC
    if len(np.unique(y_test_cA)) > 1:
        fpr_m1a, tpr_m1a, _ = roc_curve(y_test_cA, prob_m1_cA)
        fpr_m2a, tpr_m2a, _ = roc_curve(y_test_cA, prob_m2_cA)
        fpr_ea, tpr_ea, _ = roc_curve(y_test_cA, prob_ens_cA)
        ax1.plot(fpr_m1a, tpr_m1a, label=f"Model 1 (AUC={metrics_m1_cA['roc_auc']:.3f})", color='#1f77b4')
        ax1.plot(fpr_m2a, tpr_m2a, label=f"Model 2 (AUC={metrics_m2_cA['roc_auc']:.3f})", color='#2ca02c')
        ax1.plot(fpr_ea, tpr_ea, label=f"Ensemble (AUC={metrics_ens_cA['roc_auc']:.3f})", color='#ff7f0e', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_title("ROC Curves — Case A (No Drift)", fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Case B ROC
    if len(np.unique(y_test_cB)) > 1:
        fpr_m1b, tpr_m1b, _ = roc_curve(y_test_cB, prob_m1_cB)
        fpr_m2b, tpr_m2b, _ = roc_curve(y_test_cB, prob_m2_cB)
        fpr_m3b, tpr_m3b, _ = roc_curve(y_test_cB, prob_m3_cB)
        fpr_eb, tpr_eb, _ = roc_curve(y_test_cB, prob_ens_cB)
        ax2.plot(fpr_m1b, tpr_m1b, label=f"Model 1 (AUC={metrics_m1_cB['roc_auc']:.3f})", color='#1f77b4')
        ax2.plot(fpr_m2b, tpr_m2b, label=f"Model 2 (AUC={metrics_m2_cB['roc_auc']:.3f})", color='#2ca02c')
        ax2.plot(fpr_m3b, tpr_m3b, label=f"Model 3 (AUC={metrics_m3_cB['roc_auc']:.3f})", color='#9467bd')
        ax2.plot(fpr_eb, tpr_eb, label=f"Ensemble (AUC={metrics_ens_cB['roc_auc']:.3f})", color='#ff7f0e', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_title("ROC Curves — Case B (With Drift)", fontweight='bold')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '13_roc_curves.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 13: ROC Curves")

    # ------------------------------------------------------------------
    # PLOT 14: Precision-Recall Curves (Both Cases)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Case A PR
    if len(np.unique(y_test_cA)) > 1:
        p1a, r1a, _ = precision_recall_curve(y_test_cA, prob_m1_cA)
        p2a, r2a, _ = precision_recall_curve(y_test_cA, prob_m2_cA)
        pea, rea, _ = precision_recall_curve(y_test_cA, prob_ens_cA)
        ax1.plot(r1a, p1a, label='Model 1 (RF)', color='#1f77b4')
        ax1.plot(r2a, p2a, label='Model 2 (ET)', color='#2ca02c')
        ax1.plot(rea, pea, label='Ensemble', color='#ff7f0e', linewidth=2)
    ax1.set_title("Precision-Recall Curves — Case A (No Drift)", fontweight='bold')
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Case B PR
    if len(np.unique(y_test_cB)) > 1:
        p1b, r1b, _ = precision_recall_curve(y_test_cB, prob_m1_cB)
        p2b, r2b, _ = precision_recall_curve(y_test_cB, prob_m2_cB)
        p3b, r3b, _ = precision_recall_curve(y_test_cB, prob_m3_cB)
        peb, reb, _ = precision_recall_curve(y_test_cB, prob_ens_cB)
        ax2.plot(r1b, p1b, label='Model 1 (RF)', color='#1f77b4')
        ax2.plot(r2b, p2b, label='Model 2 (ET)', color='#2ca02c')
        ax2.plot(r3b, p3b, label='Model 3 (DT)', color='#9467bd')
        ax2.plot(reb, peb, label='Ensemble', color='#ff7f0e', linewidth=2)
    ax2.set_title("Precision-Recall Curves — Case B (With Drift)", fontweight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '14_precision_recall_curves.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 14: Precision-Recall Curves")

    # ------------------------------------------------------------------
    # PLOT 15: Model Probability Comparison under Drift (Case B)
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    n_show = min(100, len(y_test_cB))
    sample_idx = np.arange(n_show)
    plt.plot(sample_idx, prob_m1_cB[:n_show], label='Model 1 (RF) Prob', color='#1f77b4', alpha=0.7)
    plt.plot(sample_idx, prob_m2_cB[:n_show], label='Model 2 (ET) Prob', color='#2ca02c', alpha=0.7)
    plt.plot(sample_idx, prob_m3_cB[:n_show], label='Model 3 (DT) Prob', color='#9467bd', alpha=0.7)
    plt.plot(sample_idx, prob_ens_cB[:n_show], label='Ensemble Prob', color='#ff7f0e', linewidth=2)
    plt.axhline(0.5, color='black', linestyle=':', label='Threshold = 0.5')
    plt.scatter(sample_idx, y_test_cB[:n_show], color='red', marker='x', s=20, label='True Label', zorder=5, alpha=0.6)
    plt.title("Model Probability Outputs under Data Drift (Case B — First 100 Test Samples)", fontweight='bold')
    plt.xlabel("Test Sample Index")
    plt.ylabel("Predicted Probability (QoS Violation)")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '15_model_probability_comparison_drift.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 15: Model Probability Comparison under Drift")

    # ------------------------------------------------------------------
    # PLOT 16: Sequential Prediction Timeline (Actual vs Predicted)
    # ------------------------------------------------------------------
    plt.figure(figsize=(14, 5))
    n_seq = min(120, len(y_test_cB))
    seq_idx = np.arange(n_seq)
    plt.step(seq_idx, y_test_cB[:n_seq] + 0.06, label='Actual Target (y_true)',
             color='black', where='post', linewidth=2)
    plt.step(seq_idx, pred_m1_cB[:n_seq] + 0.02, label='Model 1 (RF) Prediction',
             color='#1f77b4', linestyle='--', where='post', alpha=0.8)
    plt.step(seq_idx, pred_m2_cB[:n_seq] - 0.02, label='Model 2 (ET) Prediction',
             color='#2ca02c', linestyle='--', where='post', alpha=0.8)
    plt.step(seq_idx, pred_ens_cB[:n_seq] - 0.06, label='Ensemble Prediction',
             color='#ff7f0e', where='post', linewidth=2)
    plt.title("Sequential Predictions vs Actual Values (Case B — Drift Runtime)", fontweight='bold')
    plt.xlabel("Runtime Step")
    plt.ylabel("QoS Violation Class")
    plt.ylim(-0.2, 1.25)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['plots'], '16_sequential_prediction_timeline.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 16: Sequential Prediction Timeline")

    # ------------------------------------------------------------------
    # PLOT 17: Ensemble Weight Comparison (Case A vs Case B)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Case A weights (2 models)
    ax1.bar(['Model 1\n(RF)', 'Model 2\n(ET)'], [w1_cA, w2_cA],
            color=['#1f77b4', '#2ca02c'], width=0.5)
    for i, w in enumerate([w1_cA, w2_cA]):
        ax1.text(i, w + 0.02, f"{w:.3f}", ha='center', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Case A (No Drift)\nEnsemble Weights", fontweight='bold')
    ax1.set_ylabel("Weight")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Case B weights (3 models)
    ax2.bar(['Model 1\n(RF)', 'Model 2\n(ET)', 'Model 3\n(DT)'], [w1_cB, w2_cB, w3_cB],
            color=['#1f77b4', '#2ca02c', '#9467bd'], width=0.5)
    for i, w in enumerate([w1_cB, w2_cB, w3_cB]):
        ax2.text(i, w + 0.02, f"{w:.3f}", ha='center', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Case B (With Drift)\nEnsemble Weights", fontweight='bold')
    ax2.set_ylabel("Weight")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Learned Ensemble Weights: How Much to Rely on Historical vs Runtime Models",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(DIRS['plots'], '17_ensemble_weight_comparison.png'), dpi=300)
    plt.close()
    print("[+] Saved Plot 17: Ensemble Weight Comparison")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE — ALL OUTPUTS SAVED")
    print("=" * 80)
    print(f"\n  Models:      {DIRS['models']}/")
    print(f"  Predictions: {DIRS['predictions']}/")
    print(f"  Metrics:     {DIRS['metrics']}/")
    print(f"  Drift CSVs:  {DIRS['drift']}/")
    print(f"  Plots:       {DIRS['plots']}/  (17 figures)")
    print()


if __name__ == '__main__':
    run_experiment()
