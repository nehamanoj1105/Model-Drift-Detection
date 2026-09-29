"""
================================================================================
EXPERIMENT 3 — CONFIGURATION
================================================================================
Central configuration for UCB1-Controlled Heterogeneous Adaptive Ensemble
under streaming drift.
All modules import constants from here — no duplicate declarations.
================================================================================
"""

import os

# ── 1. Dataset & Stream Dimensions ──────────────────────────────────────────
N_TOTAL_SAMPLES = 10_000
N_INITIAL_TRAINING = 2_000
N_DEPLOYMENT_SAMPLES = 8_000
N_WINDOWS = 16
WINDOW_SIZE = 500

# ── 2. Experimental Seeds ───────────────────────────────────────────────────
SEEDS = [42, 43, 44, 45, 46]

# ── 3. Telemetry Features & Target ──────────────────────────────────────────
KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
TARGET_COL = 'qos_violation'

# ── 4. Drift Taxonomy (No Prior Drift) ──────────────────────────────────────
DRIFT_TYPES = ['none', 'covariate', 'concept', 'mixed']
DRIFT_SEVERITIES = ['none', 'mild', 'moderate', 'severe']
DRIFT_TRANSITIONS = ['stable', 'gradual', 'sudden', 'recovery']

# Physical covariate shift magnitudes at 100% severity
COVARIATE_DRIFT_FULL = {
    'speed': +1.2,        # m/s
    'distance': +35.0,    # m
    'delay': +10.0,       # ms
    'throughput': -25.0,  # Mbps
}

COVARIATE_MAGNITUDES = {
    'none': 0.0,
    'mild': 0.20,
    'moderate': 0.45,
    'severe': 0.75,
}

# Concept drift coefficient shifts for linear terms AND nonlinear interactions
CONCEPT_DRIFT_SHIFTS = {
    'none': {
        'dist_w': 0.0, 'delay_w': 0.0, 'tp_w': 0.0, 'speed_w': 0.0,
        'i1_w': 0.0, 'i2_w': 0.0, 'intercept': 0.0
    },
    'mild': {
        'dist_w': +0.4, 'delay_w': +0.5, 'tp_w': -0.3, 'speed_w': +0.2,
        'i1_w': +0.5, 'i2_w': -0.4, 'intercept': +0.1
    },
    'moderate': {
        'dist_w': +0.7, 'delay_w': +0.9, 'tp_w': -0.6, 'speed_w': +0.4,
        'i1_w': +1.0, 'i2_w': -0.8, 'intercept': +0.2
    },
    'severe': {
        'dist_w': +1.2, 'delay_w': +1.4, 'tp_w': -1.0, 'speed_w': +0.6,
        'i1_w': +1.8, 'i2_w': -1.4, 'intercept': +0.3
    },
}

# ── 5. Base Model Architectures (Complementary Hyperparameters) ─────────────
BASE_MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': 50,
        'max_depth': 7,
        'min_samples_split': 4,
        'n_jobs': -1,
    },
    'ExtraTrees': {
        'n_estimators': 50,
        'max_depth': 7,
        'min_samples_split': 6,
        'n_jobs': -1,
    },
    'GradientBoosting': {
        'n_estimators': 50,
        'max_depth': 4,
        'learning_rate': 0.08,
        'subsample': 0.85,
    }
}

# ── 6. Multi-Temporal Perspectives (Different Historical Horizons) ──────────
TEMPORAL_HORIZONS = {
    'RF': 3500,   # Long historical buffer (captures broad structural patterns)
    'ET': 2500,   # Medium recent buffer (smooth adaptation to covariate shifts)
    'GB': 1800,   # Short recent buffer (calibrated to fit nonlinear concept shifts)
}

# ── 7. Calibrated Decision Threshold for Imbalanced Stream ─────────────────
DECISION_THRESHOLD = 0.46  # Aligns precision/recall for ~28% QoS violation rate

# ── 8. Diversity-Aware Softmax Weighting ────────────────────────────────────
WEIGHT_ALPHA = 0.7     # EMA smoothing for component F1 tracking
WEIGHT_BETA = 8.0      # Softmax inverse temperature
WEIGHT_GAMMA = 0.15    # Diversity bonus weight (Score = EMA_F1 + gamma * Diversity)

# ── 8. Lightweight Statistical Drift Detector ──────────────────────────────
DRIFT_METRIC = 'wasserstein'
DRIFT_THRESHOLD = 0.12  # Mean normalized Wasserstein distance threshold

# ── 9. UCB1 Bandit Configuration ───────────────────────────────────────────
ARM_NAMES = ['RF', 'ET', 'Ensemble']
N_ARMS = 3
C_EXPLORATION = 1.0        # UCB1 exploration coefficient
LAMBDA_COST = 0.05         # Cost penalty: Reward = F1 - lambda * normalized_cost
REF_CPU_TIME = 0.5         # Reference CPU time for cost normalization (seconds)

# ── 10. Approach Names ─────────────────────────────────────────────────────
APPROACH_NAMES = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']

# ── 11. Directory Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
