"""
================================================================================
EXPERIMENT 3 — DATA GENERATION & HETEROGENEOUS DRIFT GENERATOR
================================================================================
Generates 10,000 sequential telemetry samples:
  - Initial training : 2,000 samples (stationary baseline)
  - Deployment stream: 8,000 samples -> 16 windows x 500 samples

Features: speed (m/s), distance (m), delay (ms), throughput (Mbps)
Nonlinear Interactions: speed x delay, delay x throughput, speed x throughput
Target: qos_violation (0/1) via logistic model with realistic ~25-32% violation rate

Randomized, balanced drift schedule per seed:
  - Drift Types: none (4), covariate (4), concept (4), mixed (4)
  - Severities: mild (4), moderate (4), severe (4)
  - Transitions: sudden (4), gradual (4), recovery (4)
  - NO Prior Drift.
================================================================================
"""

import numpy as np
import pandas as pd

from config import (
    N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE,
    KPI_COLS, TARGET_COL, SEEDS,
    COVARIATE_DRIFT_FULL, COVARIATE_MAGNITUDES,
    CONCEPT_DRIFT_SHIFTS
)


def generate_drift_schedule(seed, n_windows=N_WINDOWS):
    """
    Generate a reproducible, balanced drift configuration across 16 windows.
    Window 0 is always stationary baseline.
    The 15 deployment windows balance:
      - 3 'none', 4 'covariate', 4 'concept', 4 'mixed' (total: 4 of each type across 16 windows)
      - 4 'mild', 4 'moderate', 4 'severe' across the 12 drift windows
      - 4 'sudden', 4 'gradual', 4 'recovery' across the 12 drift windows
    """
    rng = np.random.RandomState(seed + 999)

    schedule = [{
        'window_id': 0,
        'drift_type': 'none',
        'severity': 'none',
        'transition': 'stable',
        'drift_parameters': 'baseline',
    }]

    # 15 remaining windows: balanced 3 none, 4 covariate, 4 concept, 4 mixed
    types_pool = ['none'] * 3 + ['covariate'] * 4 + ['concept'] * 4 + ['mixed'] * 4
    rng.shuffle(types_pool)

    sev_pool = ['mild'] * 4 + ['moderate'] * 4 + ['severe'] * 4
    trans_pool = ['sudden'] * 4 + ['gradual'] * 4 + ['recovery'] * 4
    rng.shuffle(sev_pool)
    rng.shuffle(trans_pool)

    si, ti = 0, 0
    for i, dtype in enumerate(types_pool):
        w_id = i + 1
        if dtype == 'none':
            schedule.append({
                'window_id': w_id,
                'drift_type': 'none',
                'severity': 'none',
                'transition': 'stable',
                'drift_parameters': 'baseline',
            })
            continue

        sev = sev_pool[si]; si += 1
        trans = trans_pool[ti]; ti += 1

        if dtype == 'covariate':
            param_str = f"mag={COVARIATE_MAGNITUDES[sev]}"
        elif dtype == 'concept':
            c = CONCEPT_DRIFT_SHIFTS[sev]
            param_str = (f"dist={c['dist_w']},delay={c['delay_w']},tp={c['tp_w']},"
                         f"speed={c['speed_w']},i1={c['i1_w']},i2={c['i2_w']}")
        elif dtype == 'mixed':
            c = CONCEPT_DRIFT_SHIFTS[sev]
            param_str = (f"mag={COVARIATE_MAGNITUDES[sev]},delay={c['delay_w']},"
                         f"i1={c['i1_w']},i2={c['i2_w']}")
        else:
            param_str = 'custom'

        schedule.append({
            'window_id': w_id,
            'drift_type': dtype,
            'severity': sev,
            'transition': trans,
            'drift_parameters': param_str,
        })

    return schedule


def generate_experiment_dataset(seed):
    """
    Generate 10,000 sequential telemetry samples with realistic nonlinear interactions
    and heterogeneous drift.
    Returns (df, schedule).
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_TOTAL_SAMPLES)

    # 1. Base telemetry signals
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, N_TOTAL_SAMPLES)
    speed = np.clip(speed, 0.5, 3.5)

    cycle = 800
    tri = 2 * np.abs((t % cycle) / cycle - 0.5)
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, N_TOTAL_SAMPLES)
    distance = np.clip(distance, 10.0, 120.0)

    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay = delay_base + rng.lognormal(0.0, 0.4, N_TOTAL_SAMPLES)
    delay = np.clip(delay, 2.0, 50.0)

    path_loss = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    throughput = 80.0 * path_loss + rng.normal(0, 8.0, N_TOTAL_SAMPLES)
    throughput = np.clip(throughput, 5.0, 150.0)

    # Tracking metadata
    drift_types = ['none'] * N_INITIAL_TRAINING
    drift_severities = ['none'] * N_INITIAL_TRAINING
    drift_transitions = ['stable'] * N_INITIAL_TRAINING
    drift_params = ['baseline'] * N_INITIAL_TRAINING
    window_ids = [-1] * N_INITIAL_TRAINING

    # Drift schedule
    schedule = generate_drift_schedule(seed)

    # Base coefficients (linear + nonlinear interactions)
    base_dist_w = 0.7
    base_delay_w = 1.1
    base_tp_w = -0.9
    base_speed_w = 0.3
    base_i1_w = 0.6   # speed x delay interaction
    base_i2_w = 0.5   # delay x throughput interaction
    base_intercept = -1.5  # calibrated to yield ~25-30% QoS violation rate

    dist_w = np.full(N_TOTAL_SAMPLES, base_dist_w)
    delay_w = np.full(N_TOTAL_SAMPLES, base_delay_w)
    tp_w = np.full(N_TOTAL_SAMPLES, base_tp_w)
    speed_w = np.full(N_TOTAL_SAMPLES, base_speed_w)
    i1_w = np.full(N_TOTAL_SAMPLES, base_i1_w)
    i2_w = np.full(N_TOTAL_SAMPLES, base_i2_w)
    intercept = np.full(N_TOTAL_SAMPLES, base_intercept)

    drift_rng = np.random.RandomState(seed + 1000)

    for cfg in schedule:
        w_id = cfg['window_id']
        dtype = cfg['drift_type']
        sev = cfg['severity']
        trans = cfg['transition']

        idx_s = N_INITIAL_TRAINING + w_id * WINDOW_SIZE
        idx_e = idx_s + WINDOW_SIZE

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
        else:
            profile = np.ones(WINDOW_SIZE)

        # A. Covariate Drift P(X)
        if dtype in ('covariate', 'mixed'):
            mag = COVARIATE_MAGNITUDES[sev]
            for kpi, full_shift in COVARIATE_DRIFT_FULL.items():
                shift = mag * full_shift * profile
                noise = drift_rng.normal(0, abs(mag * full_shift) * 0.05, WINDOW_SIZE)
                total = shift + noise
                if kpi == 'speed':      speed[idx_s:idx_e] += total
                elif kpi == 'distance': distance[idx_s:idx_e] += total
                elif kpi == 'delay':    delay[idx_s:idx_e] += total
                elif kpi == 'throughput': throughput[idx_s:idx_e] += total

        # B. Concept Drift P(Y|X) - alters linear weights AND nonlinear feature interactions
        if dtype in ('concept', 'mixed'):
            cs = CONCEPT_DRIFT_SHIFTS[sev]
            dist_w[idx_s:idx_e] += cs['dist_w'] * profile
            delay_w[idx_s:idx_e] += cs['delay_w'] * profile
            tp_w[idx_s:idx_e] += cs['tp_w'] * profile
            speed_w[idx_s:idx_e] += cs['speed_w'] * profile
            i1_w[idx_s:idx_e] += cs['i1_w'] * profile
            i2_w[idx_s:idx_e] += cs['i2_w'] * profile
            intercept[idx_s:idx_e] += cs['intercept'] * profile

    # Physical clamping
    speed = np.clip(speed, 0.1, 10.0)
    distance = np.clip(distance, 5.0, 300.0)
    delay = np.clip(delay, 1.0, 200.0)
    throughput = np.clip(throughput, 1.0, 500.0)

    # Standardized features for logistic response
    dist_z = (distance - 55.0) / 25.0
    delay_z = (delay - 12.0) / 6.0
    tp_z = (throughput - 55.0) / 20.0
    speed_z = (speed - 1.5) / 0.3

    # Nonlinear interaction terms (normalized to prevent explosive scaling under shift)
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


def export_drift_configuration(seeds, output_path):
    """
    Export the exact drift configuration schedule across all seeds to CSV.
    """
    rows = []
    for seed in seeds:
        schedule = generate_drift_schedule(seed)
        for cfg in schedule:
            dtype = cfg['drift_type']
            if dtype == 'covariate':
                affected = 'speed,distance,delay,throughput'
            elif dtype == 'concept':
                affected = 'P(Y|X) linear coefficients + interactions'
            elif dtype == 'mixed':
                affected = 'Features + P(Y|X) linear coefficients + interactions'
            else:
                affected = 'none'

            rows.append({
                'seed': seed,
                'window': cfg['window_id'],
                'drift_type': dtype,
                'severity': cfg['severity'],
                'transition': cfg['transition'],
                'affected_features': affected,
                'drift_parameters': cfg['drift_parameters'],
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)
