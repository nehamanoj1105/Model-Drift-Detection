"""
================================================================================
EXPERIMENT 8 — MASTER PIPELINE & STATE MACHINE EXECUTOR
================================================================================
Executes States 0 to 13 of Experiment 8 with automated validation gates:
  State 0:  Repository Audit
  State 1:  Baseline Reproduction
  State 2:  Synthetic Stream Generator
  State 3:  Adaptation Actions Execution
  State 4:  Probe Mechanism Evaluation
  State 5:  Oracle Allocation Execution
  State 6:  Proposed Value-Based Allocation Engine
  State 7:  Synthetic Stream Benchmark (5 seeds x 4 drift types x 3 difficulty levels)
  State 8:  SEA Benchmark Experiments (5 seeds)
  State 9:  Real-World Stream Experiments (ToN_IoT + Secondary)
  State 10: Statistical Testing & Holm-Bonferroni Correction
  State 11: Theoretical Derivation Verification
  State 12: Figure & Table Generation (15 figures, 12 tables)
  State 13: Final Reports & Executive Markdown Output
================================================================================
"""

import sys
import os
import json
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Local imports
from data import (
    load_synthetic_telemetry_stream,
    load_sea_stream,
    load_ton_iot_stream,
    load_secondary_real_stream,
)
from models import create_ensemble_models, fit_model_with_action
from drift import DriftDetector
from probes import run_adaptation_probe, estimate_gain_diminishing, estimate_gain_linear, estimate_gain_piecewise
from allocation import (
    select_proposed_value_based,
    select_oracle_allocation,
    select_random_selective,
    select_weakest_selective,
    select_equal_budget_selective,
    ACTION_COSTS,
)
from adaptation import SelectiveAdaptiveEnsemble
from metrics import (
    evaluate_window_predictions,
    compute_recovery_metrics,
    compute_adaptation_efficiency,
    compute_pareto_frontier,
)
from statistics import paired_statistical_test, apply_holm_bonferroni
from validation import (
    validate_data_integrity,
    validate_prequential_integrity,
    validate_budget_integrity,
    validate_probe_sanity,
    validate_oracle_sanity,
    validate_metric_integrity,
)


def run_single_stream_experiment(method_name, stream_data, seed=42, budget=1.35, gain_model='diminishing', lagrangian_lambda=0.05):
    """
    Runs a single streaming experiment for a specific method and seed under prequential protocol.
    """
    X_train = stream_data['X_train']
    y_train = stream_data['y_train']
    windows = stream_data['windows']

    ensemble = SelectiveAdaptiveEnsemble(seed=seed)
    ensemble.fit_initial(X_train, y_train)

    drift_detector = DriftDetector(wasserstein_threshold=0.12, degradation_threshold=0.05)
    
    pred_init, _, _, _ = ensemble.predict(X_train)
    init_m = evaluate_window_predictions(y_train, pred_init)
    drift_detector.set_baseline(init_m['f1'])

    window_results = []
    drift_window_indices = []

    total_adaptation_cpu = 0.0
    models_retrained_count = 0
    samples_consumed = 0

    is_budget_constrained_method = method_name in [
        'random_selective', 'weakest_selective', 'equal_budget',
        'proposed_value_based', 'oracle_allocation'
    ]

    for w_idx, win in enumerate(windows):
        X_cur = win['X']
        y_cur = win['y']

        # Step 1: Prequential prediction
        pred_ens, prob_ens, comp_preds, comp_probs = ensemble.predict(X_cur)

        # Step 2: Receive labels y_cur and compute evaluation metrics
        m_eval = evaluate_window_predictions(y_cur, pred_ens, prob_ens)
        f1_cur = m_eval['f1']

        # Step 3: Drift Detection
        X_ref = X_train
        drift_res = drift_detector.update(X_ref, X_cur, f1_cur)
        is_drift = drift_res['is_drift']

        actions = {'RF': 'KEEP', 'ET': 'KEEP', 'GB': 'KEEP'}
        action_cost = 0.0
        cached_oracle_models = {}

        if is_drift:
            drift_window_indices.append(w_idx)

        # Step 4: Action Selection
        if method_name == 'baseline_frozen':
            actions = {'RF': 'KEEP', 'ET': 'KEEP', 'GB': 'KEEP'}

        elif method_name == 'baseline_continuous':
            actions = {'RF': 'FULL', 'ET': 'FULL', 'GB': 'FULL'}

        elif method_name == 'baseline_event_driven':
            if is_drift:
                actions = {'RF': 'FULL', 'ET': 'FULL', 'GB': 'FULL'}
            else:
                actions = {'RF': 'KEEP', 'ET': 'KEEP', 'GB': 'KEEP'}

        elif method_name == 'random_selective':
            if is_drift:
                res_alloc = select_random_selective(seed + w_idx * 10, budget=budget)
                actions = {'RF': res_alloc['RF'], 'ET': res_alloc['ET'], 'GB': res_alloc['GB']}

        elif method_name == 'weakest_selective':
            if is_drift:
                recent_f1s = {c: f1_score(y_cur, comp_preds[c], average='macro', zero_division=0) for c in ensemble.component_names}
                res_alloc = select_weakest_selective(recent_f1s, budget=budget)
                actions = {'RF': res_alloc['RF'], 'ET': res_alloc['ET'], 'GB': res_alloc['GB']}

        elif method_name == 'equal_budget':
            if is_drift:
                res_alloc = select_equal_budget_selective(budget=budget)
                actions = {'RF': res_alloc['RF'], 'ET': res_alloc['ET'], 'GB': res_alloc['GB']}

        elif method_name == 'proposed_value_based':
            if is_drift:
                gain_estimates = {}
                for m_key in ensemble.component_names:
                    probe_res = run_adaptation_probe(m_key, ensemble.models[m_key], X_cur, y_cur, seed + w_idx)
                    
                    if gain_model == 'diminishing':
                        g_part = estimate_gain_diminishing(probe_res, ACTION_COSTS['PARTIAL'])
                        g_full = estimate_gain_diminishing(probe_res, ACTION_COSTS['FULL'])
                    elif gain_model == 'linear':
                        g_part = estimate_gain_linear(probe_res, ACTION_COSTS['PARTIAL'])
                        g_full = estimate_gain_linear(probe_res, ACTION_COSTS['FULL'])
                    else:
                        g_part = estimate_gain_piecewise(probe_res, ACTION_COSTS['PARTIAL'])
                        g_full = estimate_gain_piecewise(probe_res, ACTION_COSTS['FULL'])

                    gain_estimates[m_key] = {
                        'KEEP': 0.0,
                        'PARTIAL': g_part,
                        'FULL': g_full,
                    }

                res_alloc = select_proposed_value_based(gain_estimates, budget=budget, mode='budget_constrained')
                actions = {'RF': res_alloc['RF'], 'ET': res_alloc['ET'], 'GB': res_alloc['GB']}

        elif method_name == 'oracle_allocation':
            if is_drift:
                realized_gains = {}
                for m_key in ensemble.component_names:
                    m_keep_f1 = f1_score(y_cur, comp_preds[m_key], average='macro', zero_division=0)
                    
                    m_part, n_p, _ = fit_model_with_action(m_key, ensemble.models[m_key], 'PARTIAL', ensemble.buffer_X + list(X_cur), ensemble.buffer_y + list(y_cur), seed)
                    m_part_f1 = f1_score(y_cur, m_part.predict(X_cur), average='macro', zero_division=0)
                    
                    m_full, n_f, _ = fit_model_with_action(m_key, ensemble.models[m_key], 'FULL', ensemble.buffer_X + list(X_cur), ensemble.buffer_y + list(y_cur), seed)
                    m_full_f1 = f1_score(y_cur, m_full.predict(X_cur), average='macro', zero_division=0)

                    g_part = max(0.0, m_part_f1 - m_keep_f1)
                    g_full = max(0.0, m_full_f1 - m_keep_f1)

                    realized_gains[m_key] = {'KEEP': 0.0, 'PARTIAL': g_part, 'FULL': g_full}
                    cached_oracle_models[m_key] = {'PARTIAL': m_part, 'FULL': m_full}

                res_alloc = select_oracle_allocation(realized_gains, budget=budget)
                actions = {'RF': res_alloc['RF'], 'ET': res_alloc['ET'], 'GB': res_alloc['GB']}

        # Step 5: Perform Adaptation & Save State
        if method_name == 'oracle_allocation' and is_drift:
            # Re-use cached dry-run models for oracle
            ensemble.buffer_X.extend(X_cur)
            ensemble.buffer_y.extend(y_cur)
            cpu_cost = 0.05  # minimal overhead
            adapt_res = {'total_adaptation_cpu': cpu_cost, 'samples_used': {}, 'estimators_used': {}}
            for name in ensemble.component_names:
                act = actions[name]
                if act != 'KEEP':
                    ensemble.models[name] = cached_oracle_models[name][act]
                    ensemble.retrain_counts[name][act] += 1
                    models_retrained_count += 1
                adapt_res['samples_used'][name] = len(X_cur) if act != 'KEEP' else 0
        else:
            adapt_res = ensemble.adapt_selective(X_cur, y_cur, actions)

        # Update ensemble weights post-labels
        ensemble.update_weights(y_cur, comp_preds)

        # Track costs & validate budget for constrained methods
        action_cost = sum(ACTION_COSTS[a] for a in actions.values())
        if is_drift and is_budget_constrained_method:
            validate_budget_integrity(actions, action_cost, budget)

        cpu_cost = adapt_res['total_adaptation_cpu']
        total_adaptation_cpu += cpu_cost

        if method_name != 'oracle_allocation':
            for a in actions.values():
                if a != 'KEEP':
                    models_retrained_count += 1

        samples_consumed += sum(adapt_res['samples_used'].values())

        window_results.append({
            'window_id': w_idx,
            'method': method_name,
            'seed': seed,
            'budget': budget,
            'is_drift': is_drift,
            'f1': m_eval['f1'],
            'accuracy': m_eval['accuracy'],
            'precision': m_eval['precision'],
            'recall': m_eval['recall'],
            'action_rf': actions['RF'],
            'action_et': actions['ET'],
            'action_gb': actions['GB'],
            'action_cost': action_cost,
            'adaptation_cpu': cpu_cost,
        })

    # Summary metrics across stream
    f1_list = [w['f1'] for w in window_results]
    mean_f1 = float(np.mean(f1_list))
    mean_acc = float(np.mean([w['accuracy'] for w in window_results]))
    mean_prec = float(np.mean([w['precision'] for w in window_results]))
    mean_rec = float(np.mean([w['recall'] for w in window_results]))

    rec_m = compute_recovery_metrics(f1_list, drift_window_indices, baseline_f1=init_m['f1'])
    eff = compute_adaptation_efficiency(f1_list, drift_window_indices, total_adaptation_cpu)

    summary = {
        'method': method_name,
        'seed': seed,
        'budget': budget,
        'mean_f1': mean_f1,
        'mean_accuracy': mean_acc,
        'mean_precision': mean_prec,
        'mean_recall': mean_rec,
        'total_adaptation_cpu': float(total_adaptation_cpu),
        'models_retrained': int(models_retrained_count),
        'samples_consumed': int(samples_consumed),
        'efficiency': float(eff),
        'f1_plus_1': rec_m['f1_plus_1'],
        'f1_plus_2': rec_m['f1_plus_2'],
        'f1_plus_3': rec_m['f1_plus_3'],
        'f1_plus_5': rec_m['f1_plus_5'],
        'recovery_time_90': rec_m['recovery_time_90'],
        'recovery_time_95': rec_m['recovery_time_95'],
    }

    return summary, window_results
