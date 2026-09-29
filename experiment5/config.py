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

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'ton_iot', 'Train_Test_datasets', 'Train_Test_IoT_dataset')
DATA_FILE = os.path.join(DATA_DIR, 'Train_Test_IoT_Weather.csv')

RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ton_iot')
CONFUSION_DIR = os.path.join(RESULTS_DIR, 'confusion_matrices')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# -- 2. Dataset & Stream Dimensions ------------------------------------------
# ToN_IoT Weather dataset contains 39,260 total usable chronologically ordered samples
N_INITIAL_TRAINING = 20_000
N_TOTAL_SAMPLES = 39_260
N_STREAM_SAMPLES = N_TOTAL_SAMPLES - N_INITIAL_TRAINING  # 19,260 samples
WINDOW_SIZE = 500
N_WINDOWS = N_STREAM_SAMPLES // WINDOW_SIZE             # 38 full windows of 500

# -- 3. Experimental Seeds ---------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]

# -- 4. Telemetry Features & Target Column -----------------------------------
FEATURE_COLS = ['temperature', 'pressure', 'humidity']
TARGET_COL = 'label'      # Binary: 0 = Normal, 1 = Attack
DIAGNOSTIC_COL = 'type'   # Multiclass attack taxonomy string
TIMESTAMP_COLS = ['date', 'time']

# -- 5. ToN_IoT Naturally Occurring Temporal Regimes -------------------------
# Documented operational phases based on chronological attack execution in testbed:
# Baseline (Initial Training): Samples 0 - 19,999 (Normal 15,000 + Injection 4,471 + Scanning 529)
# Stream Regime 1: Samples 20,000 - 25,528 (DDoS attack wave, temp jumps to 39°C, pressure drops)
# Stream Regime 2: Samples 25,529 - 30,528 (Password attack wave, temp drops, pressure spikes to 3.1)
# Stream Regime 3: Samples 30,529 - 34,259 (XSS & Ransomware wave, humidity plunges to 19.8%)
# Stream Regime 4: Samples 34,260 - 39,259 (Backdoor intrusion wave, humidity rebounds to 60.1%)
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
# Reusing architecture parameters from the existing codebase for strict continuity
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
# Component buffer capacities for retraining
TEMPORAL_HORIZONS = {
    'RF': 3500,   # Long historical buffer (broad structural stability)
    'ET': 2500,   # Medium recent buffer (smooth adaptation to covariate shifts)
    'GB': 1800,   # Short recent buffer (rapid fit to nonlinear concept shifts)
}

# -- 8. Decision Threshold & Ensemble Weights --------------------------------
DECISION_THRESHOLD = 0.50          # Standard classification threshold
RANK_WEIGHTS = [0.18, 0.32, 0.50]  # Dynamic rank weights: [3rd, 2nd, 1st]
WEIGHT_ALPHA = 0.7                 # EMA smoothing factor for rolling joint utility

# -- 9. Dual-Trigger Drift Detector Thresholds -------------------------------
DRIFT_WASSERSTEIN_THRESHOLD = 0.12  # Normalized Wasserstein feature drift threshold
DRIFT_PERF_DROP_THRESHOLD = 0.12    # Prequential F1 drop degradation threshold

# -- 10. Approach Definitions ------------------------------------------------
APPROACH_NAMES = [
    'Frozen Model',
    'Continuously Retrained Ensemble',
    'Event-Driven Ensemble',
    'Adaptive Random Forest',
    'Adaptive Extra Trees',
    'Adaptive Gradient Boosting'
]
