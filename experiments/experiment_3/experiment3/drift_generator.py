"""
================================================================================
EXPERIMENT 3 — DRIFT GENERATOR MODULE
================================================================================
Generates reproducible, randomized distribution drift configurations and datasets:
- Drift Types: 'none', 'covariate', 'concept', 'mixed', 'prior'
- Severities: 'none', 'mild', 'moderate', 'severe'
- Transitions: 'stable', 'gradual', 'sudden', 'recovery'
Strictly reproducible from experimental seeds.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from data_generation import (
    generate_drift_schedule,
    generate_experiment_dataset,
    N_TOTAL_SAMPLES,
    N_INITIAL_TRAINING,
    N_DEPLOYMENT_SAMPLES,
    N_WINDOWS,
    WINDOW_SIZE,
    KPI_COLS,
    TARGET_COL,
    COVARIATE_DRIFT_FULL,
    COVARIATE_MAGNITUDES,
    CONCEPT_DRIFT_COEFF_SHIFTS,
    PRIOR_DRIFT_INTERCEPT_SHIFTS
)


def export_drift_configuration(seeds, output_csv_path):
    """
    Generate and persist the exact drift configuration for all seeds and windows.
    Schema: seed, window, drift_type, severity, transition, affected_features, drift_parameters
    """
    records = []
    for seed in seeds:
        schedule = generate_drift_schedule(seed)
        for cfg in schedule:
            w_id = cfg['window_id']
            dtype = cfg['drift_type']
            sev = cfg['severity']
            trans = cfg['transition']
            param_str = cfg.get('drift_parameters', 'baseline')
            
            # Identify affected features
            if dtype == 'none':
                affected = 'none'
            elif dtype == 'covariate':
                affected = 'speed,distance,delay,throughput'
            elif dtype == 'concept':
                affected = 'speed,distance,delay,throughput,qos_violation'
            elif dtype == 'mixed':
                affected = 'speed,distance,delay,throughput,qos_violation'
            elif dtype == 'prior':
                affected = 'qos_violation'
            else:
                affected = 'all'

            records.append({
                'seed': seed,
                'window': w_id,
                'drift_type': dtype,
                'severity': sev,
                'transition': trans,
                'affected_features': affected,
                'drift_parameters': param_str
            })

    df_cfg = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df_cfg.to_csv(output_csv_path, index=False)
    return df_cfg
