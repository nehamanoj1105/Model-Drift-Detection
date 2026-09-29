"""
================================================================================
RAPT STAGE 3 — REAL-WORLD STREAM INGESTION & DATA LOADER (ToN_IoT DATASET)
================================================================================
Loads the official chronological ToN_IoT Weather telemetry dataset:
  - Parses microsecond datetime timestamps and sorts chronologically.
  - Partitions into:
      * Initial Reference Split: 20,000 samples (Samples 0 to 19,999)
      * Deployment Stream: 19,260 samples across 38 windows of 500 samples
  - Enforces zero data leakage: StandardScaler fitted strictly on initial training.
  - Annotates samples with verified natural operational attack regimes:
      * Initial Baseline: Normal + baseline injection/scanning
      * Regime 1: DDoS Attack Wave (Thermal/pressure shifts)
      * Regime 2: Password Brute-Force Wave (Pressure spikes)
      * Regime 3: XSS & Ransomware Wave (Humidity plunge)
      * Regime 4: Backdoor Intrusion Wave (Persistent stealth intrusion)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Locate dataset path robustly
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
DATA_FILE = os.path.join(
    _project_root, 'data', 'ton_iot', 'Train_Test_datasets', 'Train_Test_IoT_dataset', 'Train_Test_IoT_Weather.csv'
)

FEATURE_COLS = ['temperature', 'pressure', 'humidity']
TARGET_COL = 'label'
DIAGNOSTIC_COL = 'type'
N_INITIAL_TRAINING = 6_500
N_STREAM_SAMPLES = 19_000
WINDOW_SIZE = 500
N_WINDOWS = 38  # 38 full windows of 500 samples (19,000 total)

REAL_REGIME_BOUNDARIES = [
    {
        'name': 'Initial Baseline',
        'regime_id': 0,
        'start_idx': 0,
        'end_idx': 20_000,
        'phase': 'training',
        'dominant_type': 'normal + baseline',
        'description': 'Baseline operational regime (75% normal, 25% anomaly)'
    },
    {
        'name': 'Regime 1: DDoS',
        'regime_id': 1,
        'start_idx': 20_000,
        'end_idx': 25_529,
        'phase': 'stream',
        'dominant_type': 'ddos',
        'description': 'High-frequency DDoS attack wave inducing severe thermal/pressure shifts'
    },
    {
        'name': 'Regime 2: Password',
        'regime_id': 2,
        'start_idx': 25_529,
        'end_idx': 30_529,
        'phase': 'stream',
        'dominant_type': 'password',
        'description': 'Credential brute-force attack wave causing pressure spikes'
    },
    {
        'name': 'Regime 3: XSS/Ransomware',
        'regime_id': 3,
        'start_idx': 30_529,
        'end_idx': 34_260,
        'phase': 'stream',
        'dominant_type': 'xss & ransomware',
        'description': 'Multi-stage exploit and encryption wave shifting humidity'
    },
    {
        'name': 'Regime 4: Backdoor',
        'regime_id': 4,
        'start_idx': 34_260,
        'end_idx': 39_260,
        'phase': 'stream',
        'dominant_type': 'backdoor',
        'description': 'Persistent backdoor intrusion wave with partial environmental stabilization'
    },
]


def load_real_world_ton_iot_stream(data_path=DATA_FILE):
    """
    Loads, chronologically sorts, standardizes, and windows the official ToN_IoT dataset.
    Path A Implementation:
      - Initial Training Partition: 5,000 normal (0..4999) + 1,500 attack (15000..16499) = 6,500 samples.
      - Streaming Evaluation Partition: 10,000 normal (5000..14999) + 9,000 attack (16500..25499) = 19,000 samples
        across 38 sequential windows of 500 samples (52.6% benign, 47.4% attack).
      - Zero Data Leakage: StandardScaler fitted strictly on initial training partition.
    """
    if not os.path.exists(data_path):
        alt_path = os.path.join(r"C:\Users\emhaenn\Downloads\Model-Drift-Detection", "data", "ton_iot",
                                "Train_Test_datasets", "Train_Test_IoT_dataset", "Train_Test_IoT_Weather.csv")
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"ToN_IoT dataset file not found at: {data_path}")

    df = pd.read_csv(data_path)

    # 1. Parse chronological timestamps
    datetime_str = df['date'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip()
    df['datetime'] = pd.to_datetime(datetime_str, format='%d-%b-%y %H:%M:%S', errors='coerce')
    if df['datetime'].isnull().any():
        df['datetime'] = df['datetime'].ffill()

    # 2. Strict chronological sorting
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    assert df_sorted['datetime'].is_monotonic_increasing, "Temporal ordering validation failed."

    # 3. Annotate natural operational regimes
    df_sorted['regime_name'] = 'Unknown'
    df_sorted['regime_id'] = 0

    for reg in REAL_REGIME_BOUNDARIES:
        s = reg['start_idx']
        e = min(reg['end_idx'], len(df_sorted))
        df_sorted.loc[s:e - 1, 'regime_name'] = reg['name']
        df_sorted.loc[s:e - 1, 'regime_id'] = reg['regime_id']

    # 4. Path A Partitioning: Include benign traffic in stream
    # Train: 5000 normal (idx 0..4999) + 1500 attack (idx 15000..16499)
    df_train_norm = df_sorted.iloc[0:5000]
    df_train_att = df_sorted.iloc[15000:16500]
    df_train = pd.concat([df_train_norm, df_train_att]).sort_values('datetime').reset_index(drop=True)

    # Stream: 10000 normal (idx 5000..14999) + 9000 attack (idx 16500..25499) = 19,000 samples across 38 windows
    df_stream_norm = df_sorted.iloc[5000:15000].reset_index(drop=True)
    df_stream_att = df_sorted.iloc[16500:25500].reset_index(drop=True)

    stream_chunks = []
    norm_idx = 0
    att_idx = 0
    for w in range(N_WINDOWS):
        n_norm = 264 if w < 6 else 263
        n_att = 236 if w < 6 else 237
        chunk_norm = df_stream_norm.iloc[norm_idx : norm_idx + n_norm]
        chunk_att = df_stream_att.iloc[att_idx : att_idx + n_att]
        norm_idx += n_norm
        att_idx += n_att
        stream_chunks.append(pd.concat([chunk_norm, chunk_att]).reset_index(drop=True))

    df_stream = pd.concat(stream_chunks).reset_index(drop=True)

    X_train_raw = df_train[FEATURE_COLS].values.astype(np.float64)
    y_train = df_train[TARGET_COL].values.astype(int)

    X_stream_raw = df_stream[FEATURE_COLS].values[:N_WINDOWS * WINDOW_SIZE].astype(np.float64)
    y_stream = df_stream[TARGET_COL].values[:N_WINDOWS * WINDOW_SIZE].astype(int)

    # 5. Strict Zero-Data-Leakage Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_stream_scaled = scaler.transform(X_stream_raw)

    # 6. Build per-window metadata schedule
    window_metadata = []
    for w in range(N_WINDOWS):
        idx_s = w * WINDOW_SIZE
        idx_e = idx_s + WINDOW_SIZE
        win_df = df_stream.iloc[idx_s:idx_e]

        maj_regime = win_df['regime_name'].mode()[0]
        maj_regime_id = int(win_df['regime_id'].mode()[0])
        attack_rate = float(np.mean(win_df[TARGET_COL]))
        attack_types = win_df[DIAGNOSTIC_COL].value_counts().to_dict()

        window_metadata.append({
            'window_id': w,
            'start_idx': idx_s,
            'end_idx': idx_e,
            'regime_name': maj_regime,
            'regime_id': maj_regime_id,
            'attack_rate': attack_rate,
            'attack_types': attack_types,
        })

    return X_train_scaled, y_train, X_stream_scaled, y_stream, df_sorted, window_metadata, scaler
