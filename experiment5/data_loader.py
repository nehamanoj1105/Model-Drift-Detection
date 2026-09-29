"""
================================================================================
EXPERIMENT 5 — ToN_IoT DATA LOADER & TEMPORAL REGIME BUILDER
================================================================================
Loads the official ToN_IoT Weather telemetry dataset:
  - Parses date and time into microsecond-resolution datetime timestamps.
  - Sorts strictly chronologically (monotonically increasing time).
  - Partitions into:
      * Initial Training: Exactly 20,000 samples (Samples 0 to 19,999)
      * Deployment Stream: Exactly 19,260 samples (Samples 20,000 to 39,259)
        across 38 sequential windows of 500 samples.
  - Annotates samples with verified natural operational attack regimes:
      * Regime 0: Baseline Operations (Normal + Injection/Scanning)
      * Regime 1: DDoS Attack Wave
      * Regime 2: Password Brute-Force Wave
      * Regime 3: XSS & Ransomware Wave
      * Regime 4: Backdoor Intrusion Wave
================================================================================
"""

import os
import pandas as pd
import numpy as np

from config import (
    DATA_FILE, FEATURE_COLS, TARGET_COL, DIAGNOSTIC_COL,
    TIMESTAMP_COLS, N_INITIAL_TRAINING, N_TOTAL_SAMPLES,
    WINDOW_SIZE, N_WINDOWS, REGIME_BOUNDARIES
)


def load_ton_iot_data(data_path=DATA_FILE):
    """
    Loads, parses timestamps, sorts chronologically, and annotates ToN_IoT telemetry.
    Returns:
      df_sorted: pd.DataFrame with all features, targets, timestamps, and regime tags.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"ToN_IoT dataset file not found at: {data_path}")

    df = pd.read_csv(data_path)

    # 1. Parse timestamps into exact datetime objects
    datetime_str = df['date'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip()
    df['datetime'] = pd.to_datetime(datetime_str, format='%d-%b-%y %H:%M:%S', errors='coerce')

    if df['datetime'].isnull().any():
        raise ValueError(f"Found {df['datetime'].isnull().sum()} invalid timestamp entries in ToN_IoT data.")

    # 2. Strict chronological sorting
    df_sorted = df.sort_values('datetime').reset_index(drop=True)

    # Ensure monotonic temporal ordering
    assert df_sorted['datetime'].is_monotonic_increasing, "Temporal ordering validation failed."

    # 3. Annotate naturally occurring operational regimes
    df_sorted['regime_name'] = 'Unknown'
    df_sorted['regime_id'] = 0

    for r_id, reg in enumerate(REGIME_BOUNDARIES):
        s = reg['start_idx']
        e = min(reg['end_idx'], len(df_sorted))
        df_sorted.loc[s:e - 1, 'regime_name'] = reg['name']
        df_sorted.loc[s:e - 1, 'regime_id'] = r_id

    return df_sorted


def prepare_experiment_split(df_sorted):
    """
    Splits the chronologically sorted dataset into:
      - Initial training partition: exactly 20,000 samples
      - Streaming evaluation partition: exactly 19,260 samples (38 windows of 500)
    """
    X_all = df_sorted[FEATURE_COLS].values
    y_all = df_sorted[TARGET_COL].values
    types_all = df_sorted[DIAGNOSTIC_COL].values
    regimes_all = df_sorted['regime_name'].values
    dts_all = df_sorted['datetime'].values

    # Initial training slice (samples 0 to 19,999)
    X_train = X_all[:N_INITIAL_TRAINING]
    y_train = y_all[:N_INITIAL_TRAINING]
    types_train = types_all[:N_INITIAL_TRAINING]

    # Streaming slice (samples 20,000 to end)
    X_stream = X_all[N_INITIAL_TRAINING:]
    y_stream = y_all[N_INITIAL_TRAINING:]
    types_stream = types_all[N_INITIAL_TRAINING:]
    regimes_stream = regimes_all[N_INITIAL_TRAINING:]
    dts_stream = dts_all[N_INITIAL_TRAINING:]

    # Window metadata schedule for streaming phase
    window_schedule = []
    for w in range(N_WINDOWS):
        idx_s = w * WINDOW_SIZE
        idx_e = idx_s + WINDOW_SIZE
        w_regimes = regimes_stream[idx_s:idx_e]
        w_types = types_stream[idx_s:idx_e]
        w_labels = y_stream[idx_s:idx_e]
        w_dts = dts_stream[idx_s:idx_e]

        # Determine dominant regime and attack type in window
        dominant_regime = pd.Series(w_regimes).mode()[0]
        dominant_type = pd.Series(w_types).mode()[0]
        attack_rate = float(np.mean(w_labels))

        window_schedule.append({
            'window_id': w,
            'start_sample': N_INITIAL_TRAINING + idx_s,
            'end_sample': N_INITIAL_TRAINING + idx_e,
            'regime': dominant_regime,
            'dominant_type': dominant_type,
            'attack_rate': attack_rate,
            'start_time': str(w_dts[0]),
            'end_time': str(w_dts[-1])
        })

    return {
        'X_train': X_train,
        'y_train': y_train,
        'types_train': types_train,
        'X_stream': X_stream,
        'y_stream': y_stream,
        'types_stream': types_stream,
        'regimes_stream': regimes_stream,
        'window_schedule': window_schedule,
        'df_full': df_sorted
    }
