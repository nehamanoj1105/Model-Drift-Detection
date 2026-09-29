"""
Preprocessing & Telemetry Windowing Module for 5G Campus Network QoS Dataset
Exp 9 — RAPT vs Event-Driven Ensemble Experiment 1
"""

import os
import pandas as pd
import numpy as np

# Feature list used for prediction (EXCLUDES scenario, rep, direction, Timestamp, target)
PREDICTIVE_FEATURE_NAMES = [
    'mean_iat',
    'std_iat',
    'median_iat',
    'p90_iat',
    'mean_packet_size',
    'std_packet_size',
    'mean_delay',
    'std_delay',
    'median_delay',
    'p90_delay',
    'mean_pdist',
    'mean_piat',
    'mean_psize',
    'bandwidth',
    'slots',
    'ratio',
    'packet_count'
]

def encode_categorical_features(df):
    """Safely encode categorical fields into numerical representations for feature calculation."""
    df_enc = df.copy()
    
    # pdist: det -> 1.0, exp -> 0.0, default -> 0.5
    if 'pdist' in df_enc.columns:
        pdist_map = {'det': 1.0, 'exp': 0.0}
        df_enc['pdist_num'] = df_enc['pdist'].map(lambda x: pdist_map.get(str(x).lower(), 0.5))
    else:
        df_enc['pdist_num'] = 0.5
        
    # psize: small -> 1.0, mix -> 2.0, large -> 3.0, default -> 1.0
    if 'psize' in df_enc.columns:
        psize_map = {'small': 1.0, 'mix': 2.0, 'large': 3.0}
        df_enc['psize_num'] = df_enc['psize'].map(lambda x: psize_map.get(str(x).lower(), 1.0))
    else:
        df_enc['psize_num'] = 1.0
        
    return df_enc

def extract_window_features(window_df):
    """
    Computes summary telemetry features from a single window of packets (e.g. 500 packets).
    Returns a dictionary of numerical features and metadata.
    """
    w = encode_categorical_features(window_df)
    
    # Numerical telemetry extraction
    iat_vals = pd.to_numeric(w['iat'], errors='coerce').fillna(0.0).values if 'iat' in w.columns else np.zeros(len(w))
    psize_vals = pd.to_numeric(w['PacketSize'], errors='coerce').fillna(0.0).values if 'PacketSize' in w.columns else np.zeros(len(w))
    delay_vals = pd.to_numeric(w['trel'], errors='coerce').fillna(0.0).values if 'trel' in w.columns else np.zeros(len(w))
    
    mean_iat = float(np.mean(iat_vals))
    std_iat = float(np.std(iat_vals))
    median_iat = float(np.median(iat_vals))
    p90_iat = float(np.percentile(iat_vals, 90)) if len(iat_vals) > 0 else 0.0
    
    mean_ps = float(np.mean(psize_vals))
    std_ps = float(np.std(psize_vals))
    
    mean_delay = float(np.mean(delay_vals))
    std_delay = float(np.std(delay_vals))
    median_delay = float(np.median(delay_vals))
    p90_delay = float(np.percentile(delay_vals, 90)) if len(delay_vals) > 0 else 0.0
    
    mean_pdist = float(w['pdist_num'].mean()) if 'pdist_num' in w.columns else 0.5
    mean_piat = float(pd.to_numeric(w['piat'], errors='coerce').fillna(1700).mean()) if 'piat' in w.columns else 1700.0
    mean_psize = float(w['psize_num'].mean()) if 'psize_num' in w.columns else 1.0
    
    bw = float(w['bw'].iloc[0]) if 'bw' in w.columns and not w['bw'].empty else 20.0
    slots = float(w['slots'].iloc[0]) if 'slots' in w.columns and not w['slots'].empty else 10.0
    ratio = float(w['ratio'].iloc[0]) if 'ratio' in w.columns and not w['ratio'].empty else 2.0
    
    # Metadata
    ts = float(w['Timestamp'].iloc[0]) if 'Timestamp' in w.columns and not w['Timestamp'].empty else 0.0
    scenario = str(w['scenario'].iloc[0]) if 'scenario' in w.columns and not w['scenario'].empty else "unknown"
    rep = int(w['rep'].iloc[0]) if 'rep' in w.columns and not w['rep'].empty else 1
    direction = str(w['direction'].iloc[0]) if 'direction' in w.columns and not w['direction'].empty else "uplink"
    
    features = {
        'mean_iat': mean_iat,
        'std_iat': std_iat,
        'median_iat': median_iat,
        'p90_iat': p90_iat,
        'mean_packet_size': mean_ps,
        'std_packet_size': std_ps,
        'mean_delay': mean_delay,
        'std_delay': std_delay,
        'median_delay': median_delay,
        'p90_delay': p90_delay,
        'mean_pdist': mean_pdist,
        'mean_piat': mean_piat,
        'mean_psize': mean_psize,
        'bandwidth': bw,
        'slots': slots,
        'ratio': ratio,
        'packet_count': len(window_df),
        # Metadata
        '_timestamp': ts,
        '_scenario': scenario,
        '_rep': rep,
        '_direction': direction
    }
    return features

def aggregate_packet_stream_to_windows(df_packets, window_size=500):
    """
    Aggregates packet-level telemetry dataframe into a sequence of fixed-size windows.
    Returns pandas DataFrame of window features.
    """
    n_packets = len(df_packets)
    n_windows = n_packets // window_size
    
    window_records = []
    for w_idx in range(n_windows):
        s_idx = w_idx * window_size
        e_idx = s_idx + window_size
        w_df = df_packets.iloc[s_idx:e_idx]
        feat = extract_window_features(w_df)
        feat['window_id'] = w_idx
        window_records.append(feat)
        
    return pd.DataFrame(window_records)

def create_qos_targets(windows_df, initial_train_windows=300):
    """
    Constructs next-window QoS classification labels:
      - Target is derived from next-window p90_delay.
      - Thresholds (t1, t2) are computed ONLY on the initial_train_windows prefix.
      - Thresholds are frozen for all streaming windows.
      - 3 QoS Classes:
          0: GOOD     (p90_delay <= t1)
          1: DEGRADED (t1 < p90_delay <= t2)
          2: BAD      (p90_delay > t2)
    """
    df = windows_df.copy()
    
    # Next-window QoS metric: target is the p90_delay of the NEXT window (shift -1)
    df['next_p90_delay'] = df['p90_delay'].shift(-1)
    
    # Drop the very last window because it has no next window
    df = df.dropna(subset=['next_p90_delay']).reset_index(drop=True)
    
    # Determine freezing threshold from initial train portion
    train_subset = df.iloc[:min(initial_train_windows, len(df))]
    delay_train = train_subset['next_p90_delay'].values
    
    t1 = float(np.percentile(delay_train, 33.33))
    t2 = float(np.percentile(delay_train, 66.67))
    
    # Handle edge case where percentiles are identical
    if t2 <= t1:
        t2 = t1 + 1e-5
        
    def classify_qos(val):
        if val <= t1:
            return 0  # GOOD
        elif val <= t2:
            return 1  # DEGRADED
        else:
            return 2  # BAD
            
    df['target'] = df['next_p90_delay'].apply(classify_qos)
    
    target_info = {
        't1_threshold': t1,
        't2_threshold': t2,
        'initial_train_windows': len(train_subset),
        'class_distribution': df['target'].value_counts().to_dict()
    }
    
    return df, target_info

def build_recurring_regime_stream(regime_data_dict, pattern=['A', 'B', 'C', 'A', 'C', 'B', 'A', 'B', 'C'], windows_per_block=200):
    """
    Constructs a deterministic sequential stream of recurring regimes:
      A -> B -> C -> A -> C -> B -> A -> B -> C ...
    Preserves intra-block temporal order.
    
    regime_data_dict: dict of {'A': df_A, 'B': df_B, 'C': df_C} containing window DataFrames.
    """
    stream_blocks = []
    block_metadata = []
    
    regime_block_counts = {'A': 0, 'B': 0, 'C': 0}
    global_win_idx = 0
    
    for block_step, reg_key in enumerate(pattern):
        reg_df = regime_data_dict[reg_key]
        b_idx = regime_block_counts[reg_key]
        
        start_w = b_idx * windows_per_block
        end_w = start_w + windows_per_block
        
        # If dataset segment is smaller, loop or take available slice
        if start_w >= len(reg_df):
            start_w = (start_w % len(reg_df))
            end_w = start_w + windows_per_block
            
        block_slice = reg_df.iloc[start_w:end_w].copy()
        if len(block_slice) < windows_per_block:
            # Wrap around if needed
            deficit = windows_per_block - len(block_slice)
            block_slice = pd.concat([block_slice, reg_df.iloc[:deficit]], ignore_index=True)
            
        block_slice['stream_window_id'] = np.arange(global_win_idx, global_win_idx + len(block_slice))
        block_slice['regime_label'] = reg_key
        block_slice['block_step'] = block_step
        
        stream_blocks.append(block_slice)
        
        block_meta = {
            'block_step': block_step,
            'regime': reg_key,
            'start_window': global_win_idx,
            'end_window': global_win_idx + len(block_slice) - 1,
            'num_windows': len(block_slice),
            'scenario': block_slice['_scenario'].iloc[0]
        }
        block_metadata.append(block_meta)
        
        global_win_idx += len(block_slice)
        regime_block_counts[reg_key] += 1
        
    full_stream_df = pd.concat(stream_blocks, ignore_index=True)
    return full_stream_df, block_metadata
