"""
================================================================================
RAPT STAGE 2 — RECURRING DRIFT STREAM GENERATOR
================================================================================
Generates 50,000 sequential telemetry samples:
  - 20,000 initial stationary reference samples
  - 30,000 streaming deployment samples (60 windows x 500 samples)
  - Alternating physical cycles: Covariate <-> Concept <-> Mixed with
    exact recurrence and partial recurrence (interpolated D_alpha states).
================================================================================
"""

import numpy as np
import pandas as pd

try:
    from .config import (
        KPI_COLS, TARGET_COL,
        COVARIATE_DRIFT_FULL,
        R1_COVARIATE_MAG, R1_CONCEPT_SHIFTS,
        R2_COVARIATE_MAG, R2_CONCEPT_SHIFTS
    )
except ImportError:
    from config import (
        KPI_COLS, TARGET_COL,
        COVARIATE_DRIFT_FULL,
        R1_COVARIATE_MAG, R1_CONCEPT_SHIFTS,
        R2_COVARIATE_MAG, R2_CONCEPT_SHIFTS
    )

# 12 Distinct Operational Episodes (5 windows x 500 = 2,500 samples per episode)
RECURRING_EPISODE_SCHEDULE = [
    {'episode': 1,  'name': 'Stationary_0',        'type': 'stationary', 'start_win': 0,  'end_win': 4,  'cov_mag': 0.00, 'alpha': 0.00, 'concept_mode': 'none'},
    {'episode': 2,  'name': 'Covariate_1',         'type': 'covariate',  'start_win': 5,  'end_win': 9,  'cov_mag': 0.75, 'alpha': 0.00, 'concept_mode': 'none'},
    {'episode': 3,  'name': 'Concept_1',           'type': 'concept',    'start_win': 10, 'end_win': 14, 'cov_mag': 0.00, 'alpha': 1.00, 'concept_mode': 'severe'},
    {'episode': 4,  'name': 'Mixed_1',             'type': 'mixed',      'start_win': 15, 'end_win': 19, 'cov_mag': 0.60, 'alpha': 0.70, 'concept_mode': 'moderate'},
    {'episode': 5,  'name': 'Stationary_Return_1', 'type': 'stationary', 'start_win': 20, 'end_win': 24, 'cov_mag': 0.00, 'alpha': 0.00, 'concept_mode': 'none'},
    {'episode': 6,  'name': 'Covariate_Return_1',  'type': 'covariate',  'start_win': 25, 'end_win': 29, 'cov_mag': 0.75, 'alpha': 0.00, 'concept_mode': 'none'},
    {'episode': 7,  'name': 'Partial_D05',         'type': 'partial',    'start_win': 30, 'end_win': 34, 'cov_mag': 0.38, 'alpha': 0.50, 'concept_mode': 'interpolated'},
    {'episode': 8,  'name': 'Concept_Return_1',    'type': 'concept',    'start_win': 35, 'end_win': 39, 'cov_mag': 0.00, 'alpha': 1.00, 'concept_mode': 'severe'},
    {'episode': 9,  'name': 'Mixed_Return_1',      'type': 'mixed',      'start_win': 40, 'end_win': 44, 'cov_mag': 0.60, 'alpha': 0.70, 'concept_mode': 'moderate'},
    {'episode': 10, 'name': 'Partial_D03',         'type': 'partial',    'start_win': 45, 'end_win': 49, 'cov_mag': 0.52, 'alpha': 0.30, 'concept_mode': 'interpolated'},
    {'episode': 11, 'name': 'Covariate_Return_2',  'type': 'covariate',  'start_win': 50, 'end_win': 54, 'cov_mag': 0.75, 'alpha': 0.00, 'concept_mode': 'none'},
    {'episode': 12, 'name': 'Stationary_Return_2', 'type': 'stationary', 'start_win': 55, 'end_win': 59, 'cov_mag': 0.00, 'alpha': 0.00, 'concept_mode': 'none'},
]


def generate_recurring_dataset(seed, n_initial_training=20_000, n_windows=60, window_size=500):
    """
    Synthesize complete 50,000-sample streaming telemetry dataset with recurring drift cycles.
    """
    n_deployment = n_windows * window_size
    n_total = n_initial_training + n_deployment
    
    rng = np.random.RandomState(seed)
    t = np.arange(n_total)
    
    # 1. Base periodic physical dynamics (matching Experiment 4)
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, n_total)
    speed = np.clip(speed, 0.5, 3.5)
    
    cycle = 800
    tri = 2.0 * np.abs((t % cycle) / cycle - 0.5)
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, n_total)
    distance = np.clip(distance, 10.0, 120.0)
    
    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay = delay_base + rng.lognormal(0.0, 0.4, n_total)
    delay = np.clip(delay, 2.0, 50.0)
    
    path_loss = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    throughput = 80.0 * path_loss + rng.normal(0, 8.0, n_total)
    throughput = np.clip(throughput, 5.0, 150.0)
    
    # 2. Concept coefficients across all samples
    base_dist_w = 0.7
    base_delay_w = 1.1
    base_tp_w = -0.9
    base_speed_w = 0.3
    base_i1_w = 0.6
    base_i2_w = 0.5
    base_intercept = -1.5
    
    dist_w = np.full(n_total, base_dist_w)
    delay_w = np.full(n_total, base_delay_w)
    tp_w = np.full(n_total, base_tp_w)
    speed_w = np.full(n_total, base_speed_w)
    i1_w = np.full(n_total, base_i1_w)
    i2_w = np.full(n_total, base_i2_w)
    intercept = np.full(n_total, base_intercept)
    
    regime_labels = ['stationary'] * n_total
    episode_names = ['Initial_Reference'] * n_total
    
    # 3. Apply recurring drift episodes during deployment phase
    drift_rng = np.random.RandomState(seed + 5000)
    
    for ep in RECURRING_EPISODE_SCHEDULE:
        w_start = ep['start_win']
        if w_start >= n_windows:
            break
        w_end = min(ep['end_win'], n_windows - 1)
        ep_len = (w_end - w_start + 1) * window_size
        idx_s = n_initial_training + w_start * window_size
        idx_e = idx_s + ep_len
        
        regime_labels[idx_s:idx_e] = [ep['type']] * ep_len
        episode_names[idx_s:idx_e] = [ep['name']] * ep_len
        
        cov_mag = ep['cov_mag']
        if cov_mag > 0:
            for kpi, full_shift in COVARIATE_DRIFT_FULL.items():
                shift = cov_mag * full_shift
                noise = drift_rng.normal(0, abs(shift) * 0.05, ep_len)
                total = shift + noise
                if kpi == 'speed': speed[idx_s:idx_e] += total
                elif kpi == 'distance': distance[idx_s:idx_e] += total
                elif kpi == 'delay': delay[idx_s:idx_e] += total
                elif kpi == 'throughput': throughput[idx_s:idx_e] += total
                
        # Concept shifts
        alpha = ep['alpha']
        if ep['concept_mode'] == 'severe':
            c_shifts = R2_CONCEPT_SHIFTS
        elif ep['concept_mode'] == 'none':
            c_shifts = R1_CONCEPT_SHIFTS
        else: # moderate or interpolated
            c_shifts = {k: (1.0 - alpha) * R1_CONCEPT_SHIFTS.get(k, 0.0) + alpha * R2_CONCEPT_SHIFTS.get(k, 0.0) for k in R2_CONCEPT_SHIFTS}
            
        dist_w[idx_s:idx_e] += c_shifts.get('dist_w', 0.0)
        delay_w[idx_s:idx_e] += c_shifts.get('delay_w', 0.0)
        tp_w[idx_s:idx_e] += c_shifts.get('tp_w', 0.0)
        speed_w[idx_s:idx_e] += c_shifts.get('speed_w', 0.0)
        i1_w[idx_s:idx_e] += c_shifts.get('i1_w', 0.0)
        i2_w[idx_s:idx_e] += c_shifts.get('i2_w', 0.0)
        intercept[idx_s:idx_e] += c_shifts.get('intercept', 0.0)
        
    speed = np.clip(speed, 0.1, 10.0)
    distance = np.clip(distance, 5.0, 300.0)
    delay = np.clip(delay, 1.0, 200.0)
    throughput = np.clip(throughput, 1.0, 500.0)
    
    # 4. Standardized features and logistic probability
    dist_z = (distance - 55.0) / 25.0
    delay_z = (delay - 12.0) / 6.0
    tp_z = (throughput - 55.0) / 20.0
    speed_z = (speed - 1.5) / 0.3
    
    int_speed_delay = (speed_z * delay_z) / 3.0
    int_delay_tp = (delay_z * tp_z) / 3.0
    
    linear = (
        dist_w * dist_z +
        delay_w * delay_z +
        tp_w * tp_z +
        speed_w * speed_z +
        i1_w * int_speed_delay +
        i2_w * int_delay_tp +
        intercept
    )
    
    prob_violation = 1.0 / (1.0 + np.exp(-linear))
    target_rng = np.random.RandomState(seed + 9000)
    qos_violation = (target_rng.uniform(0, 1, n_total) < prob_violation).astype(int)
    
    df = pd.DataFrame({
        'timestamp': t,
        'speed': np.round(speed, 4),
        'distance': np.round(distance, 2),
        'delay': np.round(delay, 3),
        'throughput': np.round(throughput, 2),
        TARGET_COL: qos_violation,
        'regime_type': regime_labels,
        'episode_name': episode_names,
    })
    
    X = df[KPI_COLS].values
    y = df[TARGET_COL].values
    
    X_init = X[:n_initial_training]
    y_init = y[:n_initial_training]
    X_stream = X[n_initial_training:]
    y_stream = y[n_initial_training:]
    
    return X_init, y_init, X_stream, y_stream, df
