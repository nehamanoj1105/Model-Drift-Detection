"""
================================================================================
EXPERIMENT 8 — ADAPTATION PROBES & GAIN ESTIMATION
================================================================================
Executes bounded adaptation probes on current streaming data to estimate marginal
predictive gain per unit computational cost.
Also fits candidate gain models:
  1. Linear model
  2. Exponential diminishing-return model: G(c) = alpha * (1 - exp(-beta * c))
  3. Empirical piecewise interpolation
================================================================================
"""

import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

PROBE_SPECS = {
    'RF': {'class': RandomForestClassifier, 'n_estimators': 5, 'max_depth': 4},
    'ET': {'class': ExtraTreesClassifier, 'n_estimators': 5, 'max_depth': 4},
    'GB': {'class': GradientBoostingClassifier, 'n_estimators': 5, 'max_depth': 3, 'learning_rate': 0.1},
}


def run_adaptation_probe(model_key, current_model, X_cur, y_cur, seed, probe_sample_size=50):
    """
    Runs a lightweight adaptation probe on recent streaming window data X_cur, y_cur.
    Measures:
      - Loss/F1 before probe adaptation
      - Probe adaptation execution CPU time
      - Loss/F1 after probe adaptation
      - Probe predictive gain: delta_f1 = F1_probe - F1_before
    """
    # 1. Performance before probe
    p_before = current_model.predict(X_cur)
    f1_before = f1_score(y_cur, p_before, average='macro', zero_division=0)

    # 2. Extract probe sample, ensuring at least 2 classes if possible
    n_sub = min(probe_sample_size, len(X_cur))
    sub_X = X_cur[-n_sub:]
    sub_y = y_cur[-n_sub:]

    if len(np.unique(sub_y)) < 2 and len(X_cur) > n_sub:
        # Expand sample until 2 classes are included or full window is reached
        for extra in range(n_sub + 10, len(X_cur) + 1, 10):
            sub_X = X_cur[-extra:]
            sub_y = y_cur[-extra:]
            if len(np.unique(sub_y)) >= 2:
                break

    # If still single-class window, return 0 probe gain
    if len(np.unique(sub_y)) < 2:
        return {
            'model_key': model_key,
            'f1_before': float(f1_before),
            'f1_probe': float(f1_before),
            'delta_f1': 0.0,
            'probe_cpu_time': 1e-4,
            'gain_per_cost': 0.0,
        }

    probe_spec = PROBE_SPECS[model_key]
    cls = probe_spec['class']
    kwargs = {k: v for k, v in probe_spec.items() if k != 'class'}
    kwargs['random_state'] = seed

    t_start = time.perf_counter()
    try:
        probe_model = cls(**kwargs)
        probe_model.fit(sub_X, sub_y)
        t_end = time.perf_counter()
        probe_cpu_time = max(1e-4, t_end - t_start)

        p_probe = probe_model.predict(X_cur)
        f1_probe = f1_score(y_cur, p_probe, average='macro', zero_division=0)
    except Exception:
        f1_probe = f1_before
        probe_cpu_time = 1e-4

    delta_f1 = f1_probe - f1_before
    gain_per_cost = delta_f1 / (probe_cpu_time + 1e-5)

    return {
        'model_key': model_key,
        'f1_before': float(f1_before),
        'f1_probe': float(f1_probe),
        'delta_f1': float(delta_f1),
        'probe_cpu_time': float(probe_cpu_time),
        'gain_per_cost': float(gain_per_cost),
    }


def estimate_gain_linear(probe_res, target_cost):
    k = probe_res['gain_per_cost']
    return max(0.0, float(k * target_cost))


def estimate_gain_diminishing(probe_res, target_cost, full_cost_estimate=1.0):
    c_p = probe_res['probe_cpu_time']
    delta_p = max(1e-4, probe_res['delta_f1'])

    alpha = max(delta_p * 1.5, 0.25)
    ratio = max(1e-4, 1.0 - (delta_p / alpha))
    beta = -np.log(ratio) / (c_p + 1e-5)
    beta = np.clip(beta, 0.01, 100.0)

    g_est = alpha * (1.0 - np.exp(-beta * target_cost))
    return float(max(0.0, g_est))


def estimate_gain_piecewise(probe_res, target_cost, partial_cost=0.35, full_cost=1.0):
    c_p = probe_res['probe_cpu_time']
    delta_p = probe_res['delta_f1']

    if target_cost <= 0:
        return 0.0
    elif target_cost <= c_p:
        return float((target_cost / c_p) * delta_p)
    elif target_cost <= partial_cost:
        partial_gain = delta_p * 1.8
        slope = (partial_gain - delta_p) / (partial_cost - c_p + 1e-5)
        return float(delta_p + slope * (target_cost - c_p))
    else:
        partial_gain = delta_p * 1.8
        full_gain = delta_p * 2.5
        slope = (full_gain - partial_gain) / (full_cost - partial_cost + 1e-5)
        return float(partial_gain + slope * (target_cost - partial_cost))
