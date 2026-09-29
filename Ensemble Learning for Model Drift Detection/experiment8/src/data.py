"""
================================================================================
EXPERIMENT 8 — DATA LOADERS & STREAM GENERATORS
================================================================================
Generates and loads streaming datasets:
  1. Synthetic Telemetry Stream (Abrupt, Gradual, Recurring, Temporary, Mixed)
  2. Adaptation Difficulty Regimes (EASY, MEDIUM, HARD)
  3. SEA Benchmark Stream (Street & Kim, 2001)
  4. Real-world ToN_IoT Weather Dataset
  5. Secondary Streaming Dataset (Electricity / Synthetic Concept Drift)
================================================================================
"""

import sys
import os
import importlib.util
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
exp8_root = os.path.dirname(current_dir)
project_root = os.path.dirname(exp8_root)

code_so_far_dir = os.path.join(project_root, 'code so far')
exp5_dir = os.path.join(project_root, 'experiment5')


def load_module_isolated(module_name, file_path, dir_path):
    orig_path = list(sys.path)
    if 'config' in sys.modules:
        del sys.modules['config']
    sys.path.insert(0, dir_path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path = orig_path
        if 'config' in sys.modules:
            del sys.modules['config']

# Load data generation functions safely
mod_dg = load_module_isolated('data_gen_exp3', os.path.join(code_so_far_dir, 'data_generation.py'), code_so_far_dir)
generate_experiment_dataset = mod_dg.generate_experiment_dataset

mod_sea = load_module_isolated('sea_stream_gen', os.path.join(exp5_dir, 'sea_stream_generator.py'), exp5_dir)
generate_sea_recurring_stream = mod_sea.generate_sea_recurring_stream

mod_dl = load_module_isolated('ton_data_loader', os.path.join(exp5_dir, 'data_loader.py'), exp5_dir)
load_ton_iot_data = mod_dl.load_ton_iot_data
prepare_experiment_split = mod_dl.prepare_experiment_split


def load_synthetic_telemetry_stream(seed=42, difficulty='MEDIUM', drift_type_filter=None):
    """
    Generates synthetic telemetry stream with controllable difficulty regime (EASY, MEDIUM, HARD).
    Returns (windows_list, metadata_df).
    """
    df, schedule = generate_experiment_dataset(seed)

    if difficulty == 'EASY':
        df['speed'] = np.clip(df['speed'] * 0.9 + 0.1, 0.5, 3.5)
    elif difficulty == 'HARD':
        rng = np.random.RandomState(seed + 888)
        df['delay'] = np.clip(df['delay'] + rng.normal(0, 3.0, len(df)), 1.0, 200.0)

    if drift_type_filter is not None and drift_type_filter != 'all':
        for cfg in schedule:
            if cfg['window_id'] > 0:
                cfg['drift_type'] = drift_type_filter

    X_cols = ['speed', 'distance', 'delay', 'throughput']
    X_all = df[X_cols].values
    y_all = df['qos_violation'].values

    X_train = X_all[:2000]
    y_train = y_all[:2000]

    windows = []
    meta_rows = []

    for w_id in range(16):
        idx_s = 2000 + w_id * 500
        idx_e = idx_s + 500
        X_w = X_all[idx_s:idx_e]
        y_w = y_all[idx_s:idx_e]

        sched_info = next((s for s in schedule if s['window_id'] == w_id), {})
        d_type = sched_info.get('drift_type', 'none')
        sev = sched_info.get('severity', 'none')
        trans = sched_info.get('transition', 'stable')

        windows.append({
            'window_id': w_id,
            'drift_type': d_type,
            'severity': sev,
            'transition': trans,
            'X': X_w,
            'y': y_w,
        })

        meta_rows.append({
            'window_id': w_id,
            'drift_type': d_type,
            'severity': sev,
            'transition': trans,
            'n_samples': len(y_w),
            'positive_rate': float(np.mean(y_w)),
        })

    return {
        'X_train': X_train,
        'y_train': y_train,
        'windows': windows,
        'meta_df': pd.DataFrame(meta_rows),
    }


def load_sea_stream(seed=42):
    """
    Generates standard SEA recurring concept benchmark stream.
    """
    sea_windows, df_meta = generate_sea_recurring_stream(
        n_samples_per_window=500,
        windows_per_block=5,  # 9 blocks x 5 windows = 45 windows
        concept_sequence=('A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'),
        noise_level=0.10,
        seed=seed
    )

    X_train = np.vstack([sea_windows[0]['X'], sea_windows[1]['X']])
    y_train = np.hstack([sea_windows[0]['y'], sea_windows[1]['y']])

    stream_windows = []
    for w in sea_windows[2:]:
        d_type = 'concept' if w['is_block_start'] else 'none'
        stream_windows.append({
            'window_id': w['window_id'] - 2,
            'concept_id': w['concept_id'],
            'drift_type': d_type,
            'severity': 'moderate' if d_type == 'concept' else 'none',
            'transition': 'sudden' if d_type == 'concept' else 'stable',
            'X': w['X'],
            'y': w['y'],
        })

    return {
        'X_train': X_train,
        'y_train': y_train,
        'windows': stream_windows,
        'meta_df': df_meta,
    }


def load_ton_iot_stream():
    """
    Loads real-world ToN_IoT Weather dataset.
    """
    df_sorted = load_ton_iot_data()
    split = prepare_experiment_split(df_sorted)

    windows = []
    for sched in split['window_schedule']:
        w_id = sched['window_id']
        idx_s = w_id * 500
        idx_e = idx_s + 500
        X_w = split['X_stream'][idx_s:idx_e]
        y_w = split['y_stream'][idx_s:idx_e]

        regime = sched['regime']
        d_type = 'concept' if sched.get('attack_rate', 0) > 0.5 else 'none'

        windows.append({
            'window_id': w_id,
            'regime': regime,
            'drift_type': d_type,
            'severity': 'moderate' if d_type == 'concept' else 'none',
            'transition': 'sudden' if d_type == 'concept' else 'stable',
            'X': X_w,
            'y': y_w,
        })

    return {
        'X_train': split['X_train'],
        'y_train': split['y_train'],
        'windows': windows,
        'meta_df': pd.DataFrame(split['window_schedule']),
    }


def load_secondary_real_stream(seed=42):
    """
    Generates a secondary benchmark stream (Sine/Circle concept drift stream with 20 windows).
    """
    rng = np.random.RandomState(seed)
    n_windows = 20
    window_size = 500

    X_train = rng.uniform(-1.5, 1.5, size=(1000, 4))
    y_train = ((X_train[:, 0] + X_train[:, 1]) <= 0.0).astype(int)

    windows = []
    concepts = ['linear_0', 'linear_1', 'circle', 'linear_0', 'circle']
    for w in range(n_windows):
        c_idx = (w // 4) % len(concepts)
        c_name = concepts[c_idx]

        X_w = rng.uniform(-1.5, 1.5, size=(window_size, 4))
        if c_name == 'linear_0':
            y_w = ((X_w[:, 0] + X_w[:, 1]) <= 0.0).astype(int)
        elif c_name == 'linear_1':
            y_w = ((X_w[:, 0] - X_w[:, 1]) <= 0.2).astype(int)
        elif c_name == 'circle':
            y_w = ((X_w[:, 0]**2 + X_w[:, 1]**2) <= 1.0).astype(int)

        n_noise = int(0.05 * window_size)
        noise_idx = rng.choice(window_size, n_noise, replace=False)
        y_w[noise_idx] = 1 - y_w[noise_idx]

        is_drift = (w % 4 == 0 and w > 0)
        windows.append({
            'window_id': w,
            'concept_name': c_name,
            'drift_type': 'concept' if is_drift else 'none',
            'severity': 'moderate' if is_drift else 'none',
            'transition': 'sudden' if is_drift else 'stable',
            'X': X_w,
            'y': y_w,
        })

    return {
        'X_train': X_train,
        'y_train': y_train,
        'windows': windows,
        'meta_df': pd.DataFrame([{'window_id': w['window_id'], 'concept': w['concept_name']} for w in windows]),
    }
