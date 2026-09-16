"""
================================================================================
EXPERIMENT 3 — DATA GENERATION & RANDOMIZED DRIFT GENERATOR
================================================================================
Generates exactly 10,000 sequential telemetry samples:
- Initial Training: 2,000 samples (Stationary / Reference baseline)
- Deployment Stream: 8,000 samples divided into 16 sequential windows of 500 samples

Features:
- speed (m/s)
- distance (m)
- delay (ms)
- throughput (Mbps)

Target:
- qos_violation (0 / 1) via logistic response

Randomized Drift Generator:
For each of the 16 deployment windows, randomly determines:
- Drift type: 'none', 'covariate', 'concept', 'prior', 'mixed'
- Drift severity: 'none', 'mild', 'moderate', 'severe'
- Drift transition: 'stable', 'gradual', 'sudden', 'recovery'
Deterministic with random seed so exact sequences can be reproduced.
================================================================================
"""

import numpy as np
import pandas as pd

# Constants strictly matching constraint: 10,000 total samples
N_TOTAL_SAMPLES = 10000
N_INITIAL_TRAINING = 2000
N_DEPLOYMENT_SAMPLES = 8000
N_WINDOWS = 16
WINDOW_SIZE = 500

KPI_COLS = ['speed', 'distance', 'delay', 'throughput']
TARGET_COL = 'qos_violation'

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

CONCEPT_DRIFT_COEFF_SHIFTS = {
    'none':     {'dist_w': 0.0, 'delay_w': 0.0, 'tp_w': 0.0, 'speed_w': 0.0, 'intercept': 0.0},
    'mild':     {'dist_w': +0.4, 'delay_w': +0.4, 'tp_w': -0.3, 'speed_w': +0.15, 'intercept': +0.6},
    'moderate': {'dist_w': +0.8, 'delay_w': +0.9, 'tp_w': -0.6, 'speed_w': +0.30, 'intercept': +1.4},
    'severe':   {'dist_w': +1.4, 'delay_w': +1.6, 'tp_w': -1.1, 'speed_w': +0.60, 'intercept': +2.6},
}

PRIOR_DRIFT_INTERCEPT_SHIFTS = {
    'none': 0.0,
    'mild': +1.0,
    'moderate': +2.0,
    'severe': +3.2,
}


def generate_drift_schedule(seed, n_windows=N_WINDOWS):
    """
    Generate randomized drift configuration for each of the 16 deployment windows.
    Ensures all major drift types ('none', 'covariate', 'concept', 'prior', 'mixed'),
    severities ('mild', 'moderate', 'severe'), and transitions ('sudden', 'gradual', 'recovery')
    appear across the experiment deterministically and reproducibly.
    """
    rng = np.random.RandomState(seed + 999)
    
    # Window 0: stationary / baseline window
    initial_windows = [
        {
            'window_id': 0,
            'drift_type': 'none',
            'severity': 'none',
            'transition': 'stable',
            'drift_parameters': 'baseline'
        }
    ]
    
    # Remaining 15 windows: balanced sampling ensuring full coverage
    # Guaranteed types: 3 of each 5 types = 15 windows
    guaranteed_types = (
        ['covariate'] * 3 +
        ['concept'] * 3 +
        ['prior'] * 3 +
        ['mixed'] * 3 +
        ['none'] * 3
    )  # 15 items
    rng.shuffle(guaranteed_types)
    
    # Balanced severities and transitions for the 12 drift windows
    drift_severities_pool = ['mild'] * 4 + ['moderate'] * 4 + ['severe'] * 4
    drift_transitions_pool = ['sudden'] * 4 + ['gradual'] * 4 + ['recovery'] * 4
    rng.shuffle(drift_severities_pool)
    rng.shuffle(drift_transitions_pool)
    
    sev_idx = 0
    trans_idx = 0
    schedule = list(initial_windows)
    
    for i, dtype in enumerate(guaranteed_types):
        w_id = i + 1
        if dtype == 'none':
            sev = 'none'
            trans = 'stable'
            param_str = 'baseline'
        else:
            sev = drift_severities_pool[sev_idx]
            sev_idx += 1
            trans = drift_transitions_pool[trans_idx]
            trans_idx += 1
            
            # Format drift parameters description
            if dtype == 'covariate':
                param_str = f"mag={COVARIATE_MAGNITUDES[sev]}"
            elif dtype == 'concept':
                c = CONCEPT_DRIFT_COEFF_SHIFTS[sev]
                param_str = f"dist={c['dist_w']},delay={c['delay_w']},tp={c['tp_w']},intercept={c['intercept']}"
            elif dtype == 'prior':
                param_str = f"intercept={PRIOR_DRIFT_INTERCEPT_SHIFTS[sev]}"
            elif dtype == 'mixed':
                c = CONCEPT_DRIFT_COEFF_SHIFTS[sev]
                param_str = f"mag={COVARIATE_MAGNITUDES[sev]},dist={c['dist_w']},delay={c['delay_w']},intercept={c['intercept']}"
            else:
                param_str = 'custom'
        
        schedule.append({
            'window_id': w_id,
            'drift_type': dtype,
            'severity': sev,
            'transition': trans,
            'drift_parameters': param_str
        })
        
    return schedule


def generate_experiment_dataset(seed):
    """
    Generate the complete 10,000 sequential telemetry samples dataset with randomized drift.
    Returns:
    - df: DataFrame with 10,000 samples and all annotations
    - schedule: List of drift configurations per window
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_TOTAL_SAMPLES)
    
    # 1. Base Feature Generation (Identical generating physics to Experiment 3A)
    # Speed (m/s)
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, N_TOTAL_SAMPLES)
    speed = np.clip(speed, 0.5, 3.5)
    
    # Distance (m)
    cycle = 800
    tri = 2 * np.abs((t % cycle) / cycle - 0.5)
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, N_TOTAL_SAMPLES)
    distance = np.clip(distance, 10.0, 120.0)
    
    # Delay (ms)
    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay_noise = rng.lognormal(mean=0.0, sigma=0.4, size=N_TOTAL_SAMPLES)
    delay = delay_base + delay_noise
    delay = np.clip(delay, 2.0, 50.0)
    
    # Throughput (Mbps)
    path_loss_factor = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    tp_base = 80.0 * path_loss_factor
    tp_noise = rng.normal(0, 8.0, N_TOTAL_SAMPLES)
    throughput = tp_base + tp_noise
    throughput = np.clip(throughput, 5.0, 150.0)
    
    # Tracking annotations
    drift_types = ['none'] * N_INITIAL_TRAINING
    drift_severities = ['none'] * N_INITIAL_TRAINING
    drift_transitions = ['stable'] * N_INITIAL_TRAINING
    drift_params = ['baseline'] * N_INITIAL_TRAINING
    window_ids = [-1] * N_INITIAL_TRAINING
    
    # 2. Apply Windowed Randomized Drift
    schedule = generate_drift_schedule(seed)
    
    # Base coefficients
    base_dist_w = 0.8
    base_delay_w = 1.2
    base_tp_w = -1.0
    base_speed_w = 0.3
    base_intercept = 0.0
    
    dist_w = np.full(N_TOTAL_SAMPLES, base_dist_w)
    delay_w = np.full(N_TOTAL_SAMPLES, base_delay_w)
    tp_w = np.full(N_TOTAL_SAMPLES, base_tp_w)
    speed_w = np.full(N_TOTAL_SAMPLES, base_speed_w)
    intercept = np.full(N_TOTAL_SAMPLES, base_intercept)
    
    drift_rng = np.random.RandomState(seed + 1000)
    
    for cfg in schedule:
        w_id = cfg['window_id']
        dtype = cfg['drift_type']
        sev = cfg['severity']
        trans = cfg['transition']
        
        idx_start = N_INITIAL_TRAINING + w_id * WINDOW_SIZE
        idx_end = idx_start + WINDOW_SIZE
        
        drift_types.extend([dtype] * WINDOW_SIZE)
        drift_severities.extend([sev] * WINDOW_SIZE)
        drift_transitions.extend([trans] * WINDOW_SIZE)
        drift_params.extend([cfg.get('drift_parameters', 'none')] * WINDOW_SIZE)
        window_ids.extend([w_id] * WINDOW_SIZE)
        
        if dtype == 'none' or sev == 'none':
            continue
            
        # Transition ramp profile
        if trans == 'sudden':
            profile = np.ones(WINDOW_SIZE)
        elif trans == 'gradual':
            profile = np.linspace(0.1, 1.0, WINDOW_SIZE)
        elif trans == 'recovery':
            profile = np.linspace(1.0, 0.05, WINDOW_SIZE)
        else:  # stable
            profile = np.ones(WINDOW_SIZE)
            
        # A. Covariate drift (P(X))
        if dtype in ['covariate', 'mixed']:
            mag = COVARIATE_MAGNITUDES[sev]
            for kpi, full_shift in COVARIATE_DRIFT_FULL.items():
                shift_mag = mag * full_shift * profile
                noise = drift_rng.normal(0, abs(mag * full_shift) * 0.05, size=WINDOW_SIZE)
                total_shift = shift_mag + noise
                
                if kpi == 'speed':
                    speed[idx_start:idx_end] += total_shift
                elif kpi == 'distance':
                    distance[idx_start:idx_end] += total_shift
                elif kpi == 'delay':
                    delay[idx_start:idx_end] += total_shift
                elif kpi == 'throughput':
                    throughput[idx_start:idx_end] += total_shift
                    
        # B. Concept drift (P(Y|X))
        if dtype in ['concept', 'mixed']:
            c_shifts = CONCEPT_DRIFT_COEFF_SHIFTS[sev]
            dist_w[idx_start:idx_end] += c_shifts['dist_w'] * profile
            delay_w[idx_start:idx_end] += c_shifts['delay_w'] * profile
            tp_w[idx_start:idx_end] += c_shifts['tp_w'] * profile
            speed_w[idx_start:idx_end] += c_shifts['speed_w'] * profile
            intercept[idx_start:idx_end] += c_shifts['intercept'] * profile
            
        # C. Prior drift (P(Y))
        if dtype == 'prior':
            p_shift = PRIOR_DRIFT_INTERCEPT_SHIFTS[sev]
            intercept[idx_start:idx_end] += p_shift * profile
            
    # Clip features to reasonable physical boundaries
    speed = np.clip(speed, 0.1, 10.0)
    distance = np.clip(distance, 5.0, 300.0)
    delay = np.clip(delay, 1.0, 200.0)
    throughput = np.clip(throughput, 1.0, 500.0)
    
    # 3. Target Calculation: P(violation)
    dist_z = (distance - 55.0) / 25.0
    delay_z = (delay - 12.0) / 6.0
    tp_z = (throughput - 55.0) / 20.0
    speed_z = (speed - 1.5) / 0.3
    
    linear_comb = (
        dist_w * dist_z +
        delay_w * delay_z +
        tp_w * tp_z +
        speed_w * speed_z +
        intercept
    )
    
    prob_violation = 1.0 / (1.0 + np.exp(-linear_comb))
    target_rng = np.random.RandomState(seed + 2000)
    qos_violation = (target_rng.uniform(0, 1, N_TOTAL_SAMPLES) < prob_violation).astype(int)
    
    df = pd.DataFrame({
        'timestamp': t,
        'speed': np.round(speed, 4),
        'distance': np.round(distance, 2),
        'delay': np.round(delay, 3),
        'throughput': np.round(throughput, 2),
        TARGET_COL: qos_violation,
        'drift_type': drift_types,
        'drift_severity': drift_severities,
        'drift_transition': drift_transitions,
        'drift_parameters': drift_params,
        'window_id': window_ids,
    })
    
    return df, schedule
