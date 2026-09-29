"""
================================================================================
RAPT DATA GENERATION — SYNTHETIC D_ALPHA REGIMES
================================================================================
Constructs synthetic D_alpha telemetry streams that interpolate between two
known, previously-seen drift regimes at varying alpha in [0.0, 1.0]:
  - Regime 1 (alpha = 0.0): Severe Covariate Shift (mag = 0.75, baseline concept)
  - Regime 2 (alpha = 1.0): Severe Concept Drift (mag = 0.0, severe concept shift)
  - D_alpha (0.0 < alpha < 1.0): Partial recurrence via convex parameter mixture:
      theta(alpha) = (1 - alpha)*theta_1 + alpha*theta_2
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


def get_interpolated_regime_parameters(alpha):
    """
    Interpolate physical drift parameters between Regime 1 and Regime 2.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    cov_mag = (1.0 - alpha) * R1_COVARIATE_MAG + alpha * R2_COVARIATE_MAG
    
    concept_shifts = {}
    for k in R2_CONCEPT_SHIFTS:
        shift_1 = R1_CONCEPT_SHIFTS.get(k, 0.0)
        shift_2 = R2_CONCEPT_SHIFTS.get(k, 0.0)
        concept_shifts[k] = (1.0 - alpha) * shift_1 + alpha * shift_2
        
    return cov_mag, concept_shifts


def generate_regime_telemetry(n_samples, seed, alpha=0.0):
    """
    Generate synthetic mobile telemetry stream for interpolated regime D_alpha:
      - 4-KPI physical telemetry: speed, distance, delay, throughput
      - Target: qos_violation (imbalanced binary label generated from physical logistic model)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples)
    
    # 1. Base periodic physical dynamics (matching Experiment 4)
    speed = 1.5 + 0.3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 0.15, n_samples)
    speed = np.clip(speed, 0.5, 3.5)
    
    cycle = 800
    tri = 2.0 * np.abs((t % cycle) / cycle - 0.5)
    distance = 15.0 + 80.0 * tri + rng.normal(0, 2.0, n_samples)
    distance = np.clip(distance, 10.0, 120.0)
    
    dist_norm = (distance - 15.0) / 80.0
    delay_base = 5.0 + 12.0 * dist_norm
    delay = delay_base + rng.lognormal(0.0, 0.4, n_samples)
    delay = np.clip(delay, 2.0, 50.0)
    
    path_loss = np.clip(1.0 - 0.6 * dist_norm, 0.2, 1.0)
    throughput = 80.0 * path_loss + rng.normal(0, 8.0, n_samples)
    throughput = np.clip(throughput, 5.0, 150.0)
    
    # 2. Apply interpolated covariate shift P(X)
    cov_mag, concept_shifts = get_interpolated_regime_parameters(alpha)
    if cov_mag > 0:
        drift_rng = np.random.RandomState(seed + 1000)
        for kpi, full_shift in COVARIATE_DRIFT_FULL.items():
            shift = cov_mag * full_shift
            noise = drift_rng.normal(0, abs(cov_mag * full_shift) * 0.05, n_samples)
            total = shift + noise
            if kpi == 'speed': speed += total
            elif kpi == 'distance': distance += total
            elif kpi == 'delay': delay += total
            elif kpi == 'throughput': throughput += total
            
    speed = np.clip(speed, 0.1, 10.0)
    distance = np.clip(distance, 5.0, 300.0)
    delay = np.clip(delay, 1.0, 200.0)
    throughput = np.clip(throughput, 1.0, 500.0)
    
    # 3. Apply interpolated concept drift P(Y|X)
    base_dist_w = 0.7 + concept_shifts.get('dist_w', 0.0)
    base_delay_w = 1.1 + concept_shifts.get('delay_w', 0.0)
    base_tp_w = -0.9 + concept_shifts.get('tp_w', 0.0)
    base_speed_w = 0.3 + concept_shifts.get('speed_w', 0.0)
    base_i1_w = 0.6 + concept_shifts.get('i1_w', 0.0)
    base_i2_w = 0.5 + concept_shifts.get('i2_w', 0.0)
    base_intercept = -1.5 + concept_shifts.get('intercept', 0.0)
    
    # Standardized features for non-linear logit
    dist_z = (distance - 55.0) / 25.0
    delay_z = (delay - 12.0) / 6.0
    tp_z = (throughput - 55.0) / 20.0
    speed_z = (speed - 1.5) / 0.3
    
    int_speed_delay = (speed_z * delay_z) / 3.0
    int_delay_tp = (delay_z * tp_z) / 3.0
    
    linear = (
        base_dist_w * dist_z +
        base_delay_w * delay_z +
        base_tp_w * tp_z +
        base_speed_w * speed_z +
        base_i1_w * int_speed_delay +
        base_i2_w * int_delay_tp +
        base_intercept
    )
    
    prob_violation = 1.0 / (1.0 + np.exp(-linear))
    target_rng = np.random.RandomState(seed + 2000)
    qos_violation = (target_rng.uniform(0, 1, n_samples) < prob_violation).astype(int)
    
    df = pd.DataFrame({
        'timestamp': t,
        'speed': np.round(speed, 4),
        'distance': np.round(distance, 2),
        'delay': np.round(delay, 3),
        'throughput': np.round(throughput, 2),
        TARGET_COL: qos_violation,
        'alpha': alpha,
        'covariate_mag': cov_mag,
    })
    
    X = df[KPI_COLS].values
    y = df[TARGET_COL].values
    return X, y, df
