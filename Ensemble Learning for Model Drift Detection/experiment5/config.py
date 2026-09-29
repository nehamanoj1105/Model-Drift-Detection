"""
================================================================================
EXPERIMENT 5 — CENTRAL CONFIGURATION (ToN_IoT DATASET)
================================================================================
Defines all experimental constants, hyperparameter dictionaries, dataset sizes,
ToN_IoT regime specifications, and directory paths for the sequential evaluation:
  Model 1: Frozen Model (Static baseline trained on initial 20,000 samples)
  Model 2: Continuously Retrained Ensemble (Expensive adaptive baseline)
  Model 3: Event-Driven Ensemble (Drift-triggered adaptive ensemble)
  Individual Adaptive Models: Adaptive RF, Adaptive ET, Adaptive GB
================================================================================
"""

import os

# -- 1. Paths & Directories (Relative, Machine-Independent) -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

CANDIDATE_DATA_PATHS = [
    os.path.join(PROJECT_ROOT, 'data', 'ton_iot', 'Train_Test_datasets', 'Train_Test_IoT_dataset', 'Train_Test_IoT_Weather.csv'),
    os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'data', 'ton_iot', 'Train_Test_datasets', 'Train_Test_IoT_dataset', 'Train_Test_IoT_Weather.csv')),
]

DATA_FILE = CANDIDATE_DATA_PATHS[0]
for p in CANDIDATE_DATA_PATHS:
    if os.path.exists(p):
        DATA_FILE = p
        break

RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ton_iot')
CONFUSION_DIR = os.path.join(RESULTS_DIR, 'confusion_matrices')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


# -- 2. Dataset & Stream Dimensions ------------------------------------------
# ToN_IoT Weather dataset stream: 50,000 total sequential samples
N_INITIAL_TRAINING = 20_000
N_TOTAL_SAMPLES = 50_000
N_STREAM_SAMPLES = N_TOTAL_SAMPLES - N_INITIAL_TRAINING  # 30,000 samples
WINDOW_SIZE = 500
N_WINDOWS = N_STREAM_SAMPLES // WINDOW_SIZE             # 60 full windows of 500

# -- 3. Experimental Seeds ---------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]

# -- 4. Telemetry Features & Target Column -----------------------------------
FEATURE_COLS = ['temperature', 'pressure', 'humidity']
TARGET_COL = 'label'      # Binary: 0 = Normal, 1 = Attack
DIAGNOSTIC_COL = 'type'   # Multiclass attack taxonomy string
TIMESTAMP_COLS = ['date', 'time']

# -- 5. ToN_IoT Naturally Occurring Temporal Regimes -------------------------
REGIME_BOUNDARIES = [
    {
        'name': 'Initial Baseline',
        'start_idx': 0,
        'end_idx': 20_000,
        'phase': 'training',
        'dominant_type': 'normal + baseline attacks',
        'description': 'Baseline operational regime (75% normal, 25% anomaly)'
    },
    {
        'name': 'Regime 1: DDoS',
        'start_idx': 20_000,
        'end_idx': 25_529,
        'phase': 'stream',
        'dominant_type': 'ddos',
        'description': 'High-frequency DDoS attack wave inducing severe thermal/pressure shifts'
    },
    {
        'name': 'Regime 2: Password',
        'start_idx': 25_529,
        'end_idx': 30_529,
        'phase': 'stream',
        'dominant_type': 'password',
        'description': 'Credential brute-force attack wave causing pressure spikes'
    },
    {
        'name': 'Regime 3: XSS/Ransomware',
        'start_idx': 30_529,
        'end_idx': 34_260,
        'phase': 'stream',
        'dominant_type': 'xss & ransomware',
        'description': 'Multi-stage exploit and encryption wave shifting humidity'
    },
    {
        'name': 'Regime 4: Backdoor',
        'start_idx': 34_260,
        'end_idx': 39_260,
        'phase': 'stream',
        'dominant_type': 'backdoor',
        'description': 'Persistent backdoor intrusion wave with partial environmental stabilization'
    },
]

# -- 6. Base Model Architectures ---------------------------------------------
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

# -- 7. Multi-Temporal Perspective Horizons ----------------------------------
TEMPORAL_HORIZONS = {
    'RF': 3500,   # Long historical buffer (broad structural stability)
    'ET': 2500,   # Medium recent buffer (smooth adaptation to covariate shifts)
    'GB': 1800,   # Short recent buffer (rapid fit to nonlinear concept shifts)
}

# -- 8. Decision Threshold & Ensemble Weights --------------------------------
DECISION_THRESHOLD = 0.50          # Standard classification threshold
WEIGHT_ALPHA = 0.7                 # EMA smoothing factor for rolling joint utility
SOFTMAX_TEMPERATURE = 1.0

# -- 9. Regime-Aware Similarity Parameters ------------------------------------
DEFAULT_SIMILARITY_THRESHOLD = 0.85
SIMILARITY_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# -- 10. Dual-Trigger Drift Detector Thresholds -------------------------------
DRIFT_WASSERSTEIN_THRESHOLD = 0.12  # Normalized Wasserstein feature drift threshold
DRIFT_PERF_DROP_THRESHOLD = 0.12    # Prequential F1 drop degradation threshold

# -- 11. Approach Definitions ------------------------------------------------
METHOD_NAMES = [
    'Fixed Ensemble',
    'Global Adaptive Ensemble',
    'Regime-Aware Ensemble',
    'Oracle Regime Weights'
]
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

