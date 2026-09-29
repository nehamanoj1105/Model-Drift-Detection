"""
================================================================================
EXPERIMENT 4 â€” CENTRAL CONFIGURATION
================================================================================
Defines all experimental constants, hyperparameter dictionaries, dataset sizes,
drift taxonomy, and directory paths for the 50,000-sample streaming evaluation:
  Model 1: Frozen Model (Static baseline)
  Model 2: Continuously Retrained Ensemble (Expensive adaptive baseline)
  Model 3: Event-Driven Ensemble (Drift-triggered adaptation)
================================================================================
"""

import os

# Force single-threaded execution across all BLAS / OpenMP / Cython runtimes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# -- 1. Dataset & Stream Dimensions ------------------------------------------
N_TOTAL_SAMPLES = 50_000
N_INITIAL_TRAINING = 20_000
N_DEPLOYMENT_SAMPLES = 30_000
WINDOW_SIZE = 500
N_WINDOWS = N_DEPLOYMENT_SAMPLES // WINDOW_SIZE  # Exactly 60 windows

# -- 2. Experimental Seeds ---------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]

# -- 3. Telemetry KPIs & Target Variable --------------------------------------
KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
TARGET_COL = 'qos_violation'

# -- 4. Drift Taxonomy & Shift Magnitudes -------------------------------------
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

# -- 5. Base Model Architectures ---------------------------------------------
BASE_MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': 50,
        'max_depth': 7,
        'min_samples_split': 4,
        'n_jobs': 1,
    },
    'ExtraTrees': {
        'n_estimators': 50,
        'max_depth': 7,
        'min_samples_split': 6,
        'n_jobs': 1,
    },
    'GradientBoosting': {
        'n_estimators': 50,
        'max_depth': 4,
        'learning_rate': 0.08,
        'subsample': 0.85,
    }
}

# -- 6. Multi-Temporal Perspective Horizons ----------------------------------
# Component buffer capacities for retraining
TEMPORAL_HORIZONS = {
    'RF': 3500,   # Long historical buffer (captures broad structural patterns)
    'ET': 2500,   # Medium recent buffer (smooth adaptation to covariate shifts)
    'GB': 1800,   # Short recent buffer (rapid fit to nonlinear concept shifts)
}

# -- 7. Decision Threshold & Ensemble Weights --------------------------------
DECISION_THRESHOLD = 0.455  # Calibrated for QoS violation imbalance (~25-30%)
RANK_WEIGHTS = [0.18, 0.32, 0.50]  # Dynamic rank weights: [3rd, 2nd, 1st]
WEIGHT_ALPHA = 0.7  # EMA smoothing factor for rolling joint utility

# -- 8. Buffer Cap & New Adaptation Hyperparameters --------------------------
BUFFER_CAP = 5000  # Cap training buffer to most recent 5,000 samples
COMPONENT_DROP_THRESHOLD = 0.12  # Tuned per-component F1 degradation threshold
WARM_START_TREES_PER_EVENT = 20  # Number of new estimators added and oldest pruned per retrain event
ARF_N_MODELS = 15  # Number of streaming trees for Adaptive Random Forest
SRP_N_MODELS = 10  # Number of streaming trees for Streaming Random Patches

# -- 9. Dual-Trigger Drift Detector Thresholds -------------------------------
DRIFT_WASSERSTEIN_THRESHOLD = 0.12  # Normalized Wasserstein feature drift threshold
DRIFT_PERF_DROP_THRESHOLD = 0.12    # Prequential F1 drop degradation threshold

# -- 10. Approach Definitions (13 Approaches in v8) -------------------------
APPROACH_FROZEN = 'Frozen Model'
APPROACH_FROZEN_ENSEMBLE = 'Frozen Ensemble'
APPROACH_CONTINUOUS = 'Continuously Retrained Ensemble'
APPROACH_EVENT_CUSTOM = 'Event-Driven Ensemble (Custom Dual-Trigger)'
APPROACH_EVENT_ADWIN = 'Event-Driven Ensemble (ADWIN)'
APPROACH_EVENT_DDM = 'Event-Driven Ensemble (DDM)'
APPROACH_EVENT_EDDM = 'Event-Driven Ensemble (EDDM)'
APPROACH_EVENT_PAGE_HINKLEY = 'Event-Driven Ensemble (Page-Hinkley)'
APPROACH_WARM_START = 'Warm-Start Ensemble'
APPROACH_COMPONENT_SELECTIVE = 'Component-Selective Ensemble'
APPROACH_ARF = 'Adaptive Random Forest (river)'
APPROACH_SRP = 'Streaming Random Patches (river)'
APPROACH_TWO_TIER = 'Two-Tier Hybrid Ensemble'

APPROACH_NAMES = [
    APPROACH_FROZEN,
    APPROACH_FROZEN_ENSEMBLE,
    APPROACH_CONTINUOUS,
    APPROACH_EVENT_CUSTOM,
    APPROACH_EVENT_ADWIN,
    APPROACH_EVENT_DDM,
    APPROACH_EVENT_EDDM,
    APPROACH_EVENT_PAGE_HINKLEY,
    APPROACH_WARM_START,
    APPROACH_COMPONENT_SELECTIVE,
    APPROACH_ARF,
    APPROACH_SRP,
    APPROACH_TWO_TIER,
]

# Alias for backward compatibility
APPROACH_NAMES_V2 = APPROACH_NAMES

DETECTOR_NAMES = [
    'Custom Dual-Trigger',
    'ADWIN',
    'DDM',
    'EDDM',
    'Page-Hinkley',
]

# -- 11. Paths & Directories -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
PLOTS_DIR = FIGURES_DIR  # Aliased to figures directory to prevent duplication
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
TESTS_DIR = os.path.join(BASE_DIR, 'tests')

# Historical archives (read-only reference)
RESULTS_ARCHIVE_V1 = os.path.join(BASE_DIR, 'results_archive_v1')
FIGURES_ARCHIVE_V1 = os.path.join(BASE_DIR, 'figures_archive_v1')


