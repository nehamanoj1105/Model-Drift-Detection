"""
================================================================================
RAPT (REGIME-AWARE POLICY TRANSFER) — CENTRAL CONFIGURATION
================================================================================
Defines constants, hyperparameter dictionaries, drift regimes, and directories
for the RAPT Go/No-Go evaluation comparing:
  1. Soft Interpolation: Composing stored policies weighted by continuous similarity
  2. Hard Nearest-Regime Retrieval: Selecting the single closest stored regime's policy
================================================================================
"""

import os

# 1. Force single-threaded execution for strict computational reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# 2. Evaluation Seeds (identical to Experiment 4 benchmark standard)
SEEDS = [42, 43, 44, 45, 46]

# 3. Alpha Sweep for Interpolated Regimes D_alpha = (1 - alpha)*R1 + alpha*R2
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 4. Stream Dimensions
N_REGIME_TRAINING_SAMPLES = 3000   # Samples used to fit/learn stored policies for R1 and R2
WINDOW_SIZE = 500                 # Prequential evaluation window size
N_EVAL_WINDOWS = 10               # 10 sequential windows = 5,000 samples per alpha evaluation
N_TOTAL_EVAL_SAMPLES = N_EVAL_WINDOWS * WINDOW_SIZE

# 5. Features & Target
KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
TARGET_COL = 'qos_violation'

# 6. Operational Thresholds
DECISION_THRESHOLD = 0.455        # Calibrated threshold for QoS violation class imbalance (~25-30%)
DRIFT_PERF_DROP_THRESHOLD = 0.12  # Prequential F1 drop threshold to trigger event-driven adaptation
BASELINE_EXPECTED_F1 = 0.85       # Expected prequential F1 baseline under stable conditions
BUFFER_CAP = 5000                 # Sliding buffer cap to prevent memory bloat and buffer lag

# 7. Base Model Hyperparameters (Tri-Model Heterogeneous Ensemble from Experiment 4)
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

# 8. Physical Drift Regimes: R1 (Covariate Dominant) vs R2 (Concept Dominant)
COVARIATE_DRIFT_FULL = {
    'speed': +1.2,        # m/s
    'distance': +35.0,    # m
    'delay': +10.0,       # ms
    'throughput': -25.0,  # Mbps
}

R1_COVARIATE_MAG = 0.75
R1_CONCEPT_SHIFTS = {
    'dist_w': 0.0, 'delay_w': 0.0, 'tp_w': 0.0, 'speed_w': 0.0,
    'i1_w': 0.0, 'i2_w': 0.0, 'intercept': 0.0
}

R2_COVARIATE_MAG = 0.0
R2_CONCEPT_SHIFTS = {
    'dist_w': +1.2, 'delay_w': +1.4, 'tp_w': -1.0, 'speed_w': +0.6,
    'i1_w': +1.8, 'i2_w': -1.4, 'intercept': +0.3
}

# 9. Paths & Directories
RAPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(RAPT_DIR, 'results')
FIGURES_DIR = os.path.join(RAPT_DIR, 'figures')
TESTS_DIR = os.path.join(RAPT_DIR, 'tests')

for d in [RESULTS_DIR, FIGURES_DIR, TESTS_DIR]:
    os.makedirs(d, exist_ok=True)
