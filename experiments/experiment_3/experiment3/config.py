"""
================================================================================
EXPERIMENT 3 — CONFIGURATION PARAMETERS
================================================================================
Configuration settings for Randomized Drift-Aware UCB1 Model Selection.
Evaluates Random Forest, Extra Trees, and Gradient Boosting across 5 fixed seeds.
================================================================================
"""

import os

# 1. Dataset & Stream Dimensions
N_TOTAL_SAMPLES = 10000
N_INITIAL_TRAINING = 2000
N_DEPLOYMENT_SAMPLES = 8000
N_WINDOWS = 16
WINDOW_SIZE = 500

# 2. Experimental Seeds
SEEDS = [42, 43, 44, 45, 46]

# 3. Telemetry Signals
KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
TARGET_COL = 'qos_violation'

# 4. Drift Parameters
DRIFT_TYPES = ['none', 'covariate', 'concept', 'mixed']
DRIFT_SEVERITIES = ['none', 'mild', 'moderate', 'severe']
DRIFT_TRANSITIONS = ['stable', 'gradual', 'sudden', 'recovery']

# Physical drift shift values at 100% magnitude
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

# Concept drift coefficient shifts P(Y|X)
CONCEPT_DRIFT_COEFF_SHIFTS = {
    'none':     {'dist_w': 0.0,  'delay_w': 0.0,  'tp_w': 0.0,  'speed_w': 0.0,  'intercept': 0.0},
    'mild':     {'dist_w': +0.4, 'delay_w': +0.4, 'tp_w': -0.3, 'speed_w': +0.15, 'intercept': +0.6},
    'moderate': {'dist_w': +0.8, 'delay_w': +0.9, 'tp_w': -0.6, 'speed_w': +0.30, 'intercept': +1.4},
    'severe':   {'dist_w': +1.4, 'delay_w': +1.6, 'tp_w': -1.1, 'speed_w': +0.60, 'intercept': +2.6},
}

# Prior drift intercept shifts (for backward compatibility if needed)
PRIOR_DRIFT_INTERCEPT_SHIFTS = {
    'none': 0.0,
    'mild': +1.0,
    'moderate': +2.0,
    'severe': +3.2,
}

# 5. Base Models Architecture
BASE_MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': 50,
        'max_depth': 7,
        'n_jobs': -1,
    },
    'ExtraTrees': {
        'n_estimators': 50,
        'max_depth': 7,
        'n_jobs': -1,
    },
    'GradientBoosting': {
        'n_estimators': 50,
        'max_depth': 4,
    }
}

ARM_NAMES = ['RandomForest', 'ExtraTrees', 'GradientBoosting']

# 6. UCB1 Bandit Parameters
BANDIT_CONFIG = {
    'n_arms': 3,
    'c': 1.0,               # Exploration constant
    'beta': 0.05,            # Cost penalty in reward = F1 - beta * normalized_cost
    'ref_cpu_time': 0.5,     # Reference CPU time for normalization (seconds)
}

# 7. Adaptation & Retraining Settings
ADAPTATION_CONFIG = {
    'drift_detection_threshold': 0.15,  # Wasserstein distance threshold
    'f1_degradation_threshold': 0.08,   # F1 drop threshold from baseline
    'baseline_f1': 0.88,
}

# 8. Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
