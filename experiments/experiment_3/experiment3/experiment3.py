"""
================================================================================
EXPERIMENT 3 — MAIN EXECUTION & ORCHESTRATION SCRIPT
================================================================================
Adaptive Ensemble + UCB1 Under Randomized Drift
Industrial Telemetry QoS Violation Classification

Compares 3 high-level strategies under UCB1 Bandit selection:
- Strategy 1: Frozen Random Forest (50 trees, max depth 7)
- Strategy 2: Continuously Retrained Random Forest (50 trees, max depth 7)
- Strategy 3: Adaptive Ensemble (Random Forest + ExtraTrees + Gradient Boosting)
And Oracle selector reference.

Evaluated across 5 deterministic seeds: [42, 43, 44, 45, 46]
Dataset: 10,000 samples (2,000 initial training, 8,000 streaming over 16 windows)
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_generation import (
    generate_experiment_dataset,
    KPI_COLS,
    TARGET_COL,
    N_INITIAL_TRAINING,
    WINDOW_SIZE,
    N_WINDOWS
)
from models import create_frozen_rf, create_retrained_rf
from ensemble import AdaptiveEnsemble
from bandit import UCB1Bandit, RandomSelector
from drift import compute_drift_metrics, DriftMonitor
from metrics import evaluate_predictions, evaluate_model
from resource_monitor import measure_execution, ResourceMonitor
from statistics import run_paired_tests, compute_summary_stats
from plots import generate_all_figures


def run_experiment(seeds=None, lambda_val=0.05):
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    figures_dir = os.path.join(script_dir, 'figures')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Data storage across seeds
    window_records = []
    model_metric_records = []
    resource_records = []
    ensemble_weight_records = []
    bandit_records = []
    drift_records = []
    
    # Section 13 required specific tables
    drift_config_records = []
    predictions_records = []
    performance_records = []
    bandit_clean_records = []
    ensemble_weights_clean_records = []
    resources_clean_records = []
    retraining_records = []
    
    print(f"=== Starting Experiment 3 across {len(seeds)} seeds: {seeds} ===")
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        # 1. Dataset Generation
        df, schedule = generate_experiment_dataset(seed)
        
        # 2. Partitioning: Initial Training vs Deployment Stream
        X_train_raw = df.iloc[:N_INITIAL_TRAINING][KPI_COLS].values
        y_train = df.iloc[:N_INITIAL_TRAINING][TARGET_COL].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        
        # 3. Model Initialization
        print("Initializing and fitting baseline models on 2,000 initial training samples...")
        frozen_rf = create_frozen_rf(seed)
        _, init_res_frozen = measure_execution(frozen_rf.fit, X_train_scaled, y_train)
        
        retrained_rf = create_retrained_rf(seed)
        _, init_res_retr = measure_execution(retrained_rf.fit, X_train_scaled, y_train)
        cumulative_X = list(X_train_scaled)
        cumulative_y = list(y_train)
        
        ensemble = AdaptiveEnsemble(seed=seed, temperature=5.0)
        init_res_ens = ensemble.fit_initial(X_train_scaled, y_train)
        
        bandit = UCB1Bandit(n_arms=3, c=1.0, ref_cpu_time=0.5)
        random_selector = RandomSelector(seed=seed, n_arms=3)
        drift_monitor = DriftMonitor(baseline_f1=0.90)
        
        retrain_count_rf = 0
        
        # 4. Streaming Loop over 32 Windows
        for w_idx, cfg in enumerate(schedule):
            start_idx = N_INITIAL_TRAINING + w_idx * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            
            X_win_raw = df.iloc[start_idx:end_idx][KPI_COLS].values
            y_win = df.iloc[start_idx:end_idx][TARGET_COL].values
            X_win_scaled = scaler.transform(X_win_raw)
            
            # A. Calculate Drift (Unsupervised, no labels)
            drift_res = compute_drift_metrics(X_train_raw, X_win_raw, KPI_COLS)
            
            # B. UCB1 selects strategy BEFORE predicting
            arm_chosen = bandit.select_arm()
            arm_random = random_selector.select_arm()
            arm_names = ['Frozen RF', 'Retrained RF', 'Adaptive Ensemble']
            
            # C. Strategy Predictions (Strict Test-Then-Train)
            
            # 1. Strategy 1: Frozen RF
            frozen_prob, res_inf_frozen = measure_execution(lambda: frozen_rf.predict_proba(X_win_scaled)[:, 1])
            frozen_pred = (frozen_prob >= 0.5).astype(int)
            frozen_m = evaluate_predictions(y_win, frozen_pred, frozen_prob)
            res_train_frozen = {
                'total_cpu_time': 0.0, 'wall_clock_time': 0.0, 'cpu_user_time': 0.0,
                'cpu_system_time': 0.0, 'avg_cpu_percent': 0.0, 'peak_cpu_percent': 0.0,
                'avg_ram_mb': res_inf_frozen['avg_ram_mb'], 'peak_ram_mb': res_inf_frozen['peak_ram_mb']
            }
            
            # 2. Strategy 2: Retrained RF
            retr_prob, res_inf_retr = measure_execution(lambda: retrained_rf.predict_proba(X_win_scaled)[:, 1])
            retr_pred = (retr_prob >= 0.5).astype(int)
            retr_m = evaluate_predictions(y_win, retr_pred, retr_prob)
            # Incorporate window into buffer and retrain cumulatively
            cumulative_X.extend(X_win_scaled)
            cumulative_y.extend(y_win)
            _, res_train_retr = measure_execution(retrained_rf.fit, cumulative_X, cumulative_y)
            retrain_count_rf += 1
            
            # 3. Strategy 3: Adaptive Ensemble
            # Inference & Component Predictions
            pred_ens, prob_ens, comp_preds, comp_probs, comp_res_inf, res_inf_ens = ensemble.predict(X_win_scaled)
            ens_m = evaluate_predictions(y_win, pred_ens, prob_ens)
            
            # Component evaluation & adaptive weight update
            comp_metrics, current_weights, weight_diff = ensemble.update_weights(
                y_win, comp_preds, comp_probs=comp_probs, lambda_cost=lambda_val
            )
            
            # Section 9: Adaptive Retraining Decision
            adaptation_occurred, adapted_components, comp_res_train, res_adapt_ens = ensemble.adapt_if_needed(
                X_win_scaled, y_win, drift_res, ens_m['f1']
            )
            
            # D. Oracle Selector (Upper-Bound Reference)
            f1_candidates = {
                'Frozen RF': frozen_m['f1'],
                'Retrained RF': retr_m['f1'],
                'Adaptive Ensemble': ens_m['f1']
            }
            oracle_best_strat = max(f1_candidates, key=f1_candidates.get)
            oracle_f1 = f1_candidates[oracle_best_strat]
            
            strat_metrics = {
                'Frozen RF': frozen_m,
                'Retrained RF': retr_m,
                'Adaptive Ensemble': ens_m
            }
            strat_res_inf = {
                'Frozen RF': res_inf_frozen,
                'Retrained RF': res_inf_retr,
                'Adaptive Ensemble': res_inf_ens
            }
            strat_res_train = {
                'Frozen RF': res_train_frozen,
                'Retrained RF': res_train_retr,
                'Adaptive Ensemble': res_adapt_ens
            }
            strat_retrained = {
                'Frozen RF': False,
                'Retrained RF': True,
                'Adaptive Ensemble': adaptation_occurred
            }
            
            # E. UCB1 Evaluation & Reward Update
            chosen_strat_name = arm_names[arm_chosen]
            chosen_m = strat_metrics[chosen_strat_name]
            chosen_inf_res = strat_res_inf[chosen_strat_name]
            chosen_train_res = strat_res_train[chosen_strat_name]
            chosen_total_cpu = chosen_inf_res['total_cpu_time'] + chosen_train_res['total_cpu_time']
            chosen_total_wall = chosen_inf_res['wall_clock_time'] + chosen_train_res['wall_clock_time']
            
            reward, cost_norm = bandit.compute_reward(
                f1=chosen_m['f1'],
                cpu_time=chosen_total_cpu,
                wall_time=chosen_total_wall,
                retrained=strat_retrained[chosen_strat_name],
                lambda_val=lambda_val
            )
            
            # Oracle Reward
            oracle_total_cpu = strat_res_inf[oracle_best_strat]['total_cpu_time'] + strat_res_train[oracle_best_strat]['total_cpu_time']
            oracle_total_wall = strat_res_inf[oracle_best_strat]['wall_clock_time'] + strat_res_train[oracle_best_strat]['wall_clock_time']
            oracle_reward, _ = bandit.compute_reward(
                f1=oracle_f1,
                cpu_time=oracle_total_cpu,
                wall_time=oracle_total_wall,
                retrained=strat_retrained[oracle_best_strat],
                lambda_val=lambda_val
            )
            
            # Random Selector Reward
            random_strat_name = arm_names[arm_random]
            random_m = strat_metrics[random_strat_name]
            random_total_cpu = strat_res_inf[random_strat_name]['total_cpu_time'] + strat_res_train[random_strat_name]['total_cpu_time']
            random_total_wall = strat_res_inf[random_strat_name]['wall_clock_time'] + strat_res_train[random_strat_name]['wall_clock_time']
            random_reward, _ = bandit.compute_reward(
                f1=random_m['f1'],
                cpu_time=random_total_cpu,
                wall_time=random_total_wall,
                retrained=strat_retrained[random_strat_name],
                lambda_val=lambda_val
            )
            
            # Update UCB1 state
            bandit.update(arm_chosen, reward)
            
            # F. Drift Monitoring Update
            perf_drift = drift_monitor.update(ens_m['f1'], ens_m['accuracy'])
            
            # --- Record Data ---
            # 1. Window Results (All 5 strategies: Frozen RF, Retrained RF, Adaptive Ensemble, UCB1, Oracle)
            for s_name in ['Frozen RF', 'Retrained RF', 'Adaptive Ensemble', 'UCB1', 'Oracle']:
                if s_name == 'UCB1':
                    m = chosen_m
                    inf_t = chosen_inf_res['wall_clock_time']
                    train_t = chosen_train_res['wall_clock_time']
                    train_cpu = chosen_train_res['total_cpu_time']
                    tot_cpu = chosen_total_cpu
                    tot_wall = chosen_total_wall
                    avg_cpu = max(chosen_inf_res['avg_cpu_percent'], chosen_train_res['avg_cpu_percent'])
                    inf_user = chosen_inf_res['cpu_user_time']
                    inf_sys = chosen_inf_res['cpu_system_time']
                    train_user = chosen_train_res['cpu_user_time']
                    train_sys = chosen_train_res['cpu_system_time']
                    avg_ram = max(chosen_inf_res['avg_ram_mb'], chosen_train_res['avg_ram_mb'])
                    peak_ram = max(chosen_inf_res['peak_ram_mb'], chosen_train_res['peak_ram_mb'])
                    was_retr = strat_retrained[chosen_strat_name]
                    was_adapt = (chosen_strat_name == 'Adaptive Ensemble' and adaptation_occurred)
                elif s_name == 'Oracle':
                    m = strat_metrics[oracle_best_strat]
                    inf_t = strat_res_inf[oracle_best_strat]['wall_clock_time']
                    train_t = strat_res_train[oracle_best_strat]['wall_clock_time']
                    train_cpu = strat_res_train[oracle_best_strat]['total_cpu_time']
                    tot_cpu = strat_res_inf[oracle_best_strat]['total_cpu_time'] + strat_res_train[oracle_best_strat]['total_cpu_time']
                    tot_wall = strat_res_inf[oracle_best_strat]['wall_clock_time'] + strat_res_train[oracle_best_strat]['wall_clock_time']
                    avg_cpu = max(strat_res_inf[oracle_best_strat]['avg_cpu_percent'], strat_res_train[oracle_best_strat]['avg_cpu_percent'])
                    inf_user = strat_res_inf[oracle_best_strat]['cpu_user_time']
                    inf_sys = strat_res_inf[oracle_best_strat]['cpu_system_time']
                    train_user = strat_res_train[oracle_best_strat]['cpu_user_time']
                    train_sys = strat_res_train[oracle_best_strat]['cpu_system_time']
                    avg_ram = max(strat_res_inf[oracle_best_strat]['avg_ram_mb'], strat_res_train[oracle_best_strat]['avg_ram_mb'])
                    peak_ram = max(strat_res_inf[oracle_best_strat]['peak_ram_mb'], strat_res_train[oracle_best_strat]['peak_ram_mb'])
                    was_retr = strat_retrained[oracle_best_strat]
                    was_adapt = (oracle_best_strat == 'Adaptive Ensemble' and adaptation_occurred)
                else:
                    m = strat_metrics[s_name]
                    inf_t = strat_res_inf[s_name]['wall_clock_time']
                    train_t = strat_res_train[s_name]['wall_clock_time']
                    train_cpu = strat_res_train[s_name]['total_cpu_time']
                    tot_cpu = strat_res_inf[s_name]['total_cpu_time'] + strat_res_train[s_name]['total_cpu_time']
                    tot_wall = strat_res_inf[s_name]['wall_clock_time'] + strat_res_train[s_name]['wall_clock_time']
                    avg_cpu = max(strat_res_inf[s_name]['avg_cpu_percent'], strat_res_train[s_name]['avg_cpu_percent'])
                    inf_user = strat_res_inf[s_name]['cpu_user_time']
                    inf_sys = strat_res_inf[s_name]['cpu_system_time']
                    train_user = strat_res_train[s_name]['cpu_user_time']
                    train_sys = strat_res_train[s_name]['cpu_system_time']
                    avg_ram = max(strat_res_inf[s_name]['avg_ram_mb'], strat_res_train[s_name]['avg_ram_mb'])
                    peak_ram = max(strat_res_inf[s_name]['peak_ram_mb'], strat_res_train[s_name]['peak_ram_mb'])
                    was_retr = strat_retrained[s_name]
                    was_adapt = (s_name == 'Adaptive Ensemble' and adaptation_occurred)
                    
                window_records.append({
                    'seed': seed,
                    'window_id': w_idx,
                    'drift_type': cfg['drift_type'],
                    'drift_severity': cfg['severity'],
                    'drift_transition': cfg['transition'],
                    'strategy': s_name,
                    'f1': m['f1'],
                    'accuracy': m['accuracy'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'auc': m['auc'],
                    'inference_time': inf_t,
                    'training_time': train_t,
                    'retraining_time': train_t,
                    'training_cpu_time': train_cpu,
                    'retraining_cpu_time': train_cpu,
                    'wall_time': tot_wall,
                    'cpu_time': tot_cpu,
                    'cpu_utilization': avg_cpu,
                    'cpu_user_time': inf_user + train_user,
                    'cpu_system_time': inf_sys + train_sys,
                    'avg_ram_mb': avg_ram,
                    'peak_ram_mb': peak_ram,
                    'retrained': was_retr,
                    'adapted': was_adapt
                })
                
            # 2. Model Metrics Record (Individual candidate models & strategies with complete metrics)
            model_resource_map = {
                'Frozen RF': (res_inf_frozen, res_train_frozen),
                'Retrained RF': (res_inf_retr, res_train_retr),
                'Adaptive Ensemble': (res_inf_ens, res_adapt_ens),
                'RandomForest_comp': (comp_res_inf['RandomForest'], comp_res_train['RandomForest']),
                'ExtraTrees_comp': (comp_res_inf['ExtraTrees'], comp_res_train['ExtraTrees']),
                'GradientBoosting_comp': (comp_res_inf['GradientBoosting'], comp_res_train['GradientBoosting']),
                'UCB1': (chosen_inf_res, chosen_train_res),
                'Oracle': (strat_res_inf[oracle_best_strat], strat_res_train[oracle_best_strat])
            }
            for m_name, m_dict in {
                'Frozen RF': frozen_m,
                'Retrained RF': retr_m,
                'Adaptive Ensemble': ens_m,
                'RandomForest_comp': comp_metrics['RandomForest'],
                'ExtraTrees_comp': comp_metrics['ExtraTrees'],
                'GradientBoosting_comp': comp_metrics['GradientBoosting'],
                'UCB1': chosen_m,
                'Oracle': strat_metrics[oracle_best_strat]
            }.items():
                m_inf, m_tr = model_resource_map[m_name]
                model_metric_records.append({
                    'seed': seed,
                    'window_id': w_idx,
                    'model_name': m_name,
                    'f1': m_dict['f1'],
                    'accuracy': m_dict['accuracy'],
                    'precision': m_dict.get('precision', 0.0),
                    'recall': m_dict.get('recall', 0.0),
                    'auc': m_dict.get('auc', 0.5),
                    'inference_time': m_inf['wall_clock_time'],
                    'training_time': m_tr['wall_clock_time'],
                    'retraining_time': m_tr['wall_clock_time'],
                    'wall_time': m_inf['wall_clock_time'] + m_tr['wall_clock_time'],
                    'cpu_time': m_inf['total_cpu_time'] + m_tr['total_cpu_time'],
                    'cpu_utilization': max(m_inf['avg_cpu_percent'], m_tr['avg_cpu_percent']),
                    'cpu_user_time': m_inf['cpu_user_time'] + m_tr['cpu_user_time'],
                    'cpu_system_time': m_inf['cpu_system_time'] + m_tr['cpu_system_time'],
                    'avg_ram_mb': max(m_inf['avg_ram_mb'], m_tr['avg_ram_mb']),
                    'peak_ram_mb': max(m_inf['peak_ram_mb'], m_tr['peak_ram_mb'])
                })
                
            # 3. Resource Metrics Record
            for m_name, inf_r, tr_r, r_flag, a_flag, n_retr in [
                ('Frozen RF', res_inf_frozen, res_train_frozen, False, False, 0),
                ('Retrained RF', res_inf_retr, res_train_retr, True, False, retrain_count_rf),
                ('Adaptive Ensemble', res_inf_ens, res_adapt_ens, adaptation_occurred, adaptation_occurred, ensemble.total_retrain_events),
                ('RandomForest_comp', comp_res_inf['RandomForest'], comp_res_train['RandomForest'], 'RandomForest' in adapted_components, False, ensemble.retrain_counts['RandomForest']),
                ('ExtraTrees_comp', comp_res_inf['ExtraTrees'], comp_res_train['ExtraTrees'], 'ExtraTrees' in adapted_components, False, ensemble.retrain_counts['ExtraTrees']),
                ('GradientBoosting_comp', comp_res_inf['GradientBoosting'], comp_res_train['GradientBoosting'], 'GradientBoosting' in adapted_components, False, ensemble.retrain_counts['GradientBoosting']),
                ('UCB1', chosen_inf_res, chosen_train_res, strat_retrained[chosen_strat_name], chosen_strat_name == 'Adaptive Ensemble' and adaptation_occurred, 1 if strat_retrained[chosen_strat_name] else 0),
            ]:
                tot_cpu = inf_r['total_cpu_time'] + tr_r['total_cpu_time']
                tot_wall = inf_r['wall_clock_time'] + tr_r['wall_clock_time']
                resource_records.append({
                    'seed': seed,
                    'window_id': w_idx,
                    'model_name': m_name,
                    'avg_cpu_pct': max(inf_r['avg_cpu_percent'], tr_r['avg_cpu_percent']),
                    'avg_cpu_percent': max(inf_r['avg_cpu_percent'], tr_r['avg_cpu_percent']),
                    'peak_cpu_pct': max(inf_r['peak_cpu_percent'], tr_r['peak_cpu_percent']),
                    'peak_cpu_percent': max(inf_r['peak_cpu_percent'], tr_r['peak_cpu_percent']),
                    'cpu_user_time': inf_r['cpu_user_time'] + tr_r['cpu_user_time'],
                    'cpu_sys_time': inf_r['cpu_system_time'] + tr_r['cpu_system_time'],
                    'total_cpu_time': tot_cpu,
                    'wall_time': tot_wall,
                    'training_time': tr_r['wall_clock_time'],
                    'retraining_time': tr_r['wall_clock_time'],
                    'inference_time': inf_r['wall_clock_time'],
                    'avg_ram_mb': max(inf_r['avg_ram_mb'], tr_r['avg_ram_mb']),
                    'peak_ram_mb': max(inf_r['peak_ram_mb'], tr_r['peak_ram_mb']),
                    'retraining_occurred': r_flag,
                    'adaptation_occurred': a_flag,
                    'n_retrain_events': n_retr
                })
                
            # 4. Ensemble Weights Record
            w_vals_rec = list(current_weights.values())
            ensemble_weight_records.append({
                'seed': seed,
                'window_id': w_idx,
                'rf_weight': current_weights['RandomForest'],
                'et_weight': current_weights['ExtraTrees'],
                'gb_weight': current_weights['GradientBoosting'],
                'weight_diff': weight_diff,
                'weight_spread': weight_diff,
                'max_weight': float(np.max(w_vals_rec)),
                'min_weight': float(np.min(w_vals_rec)),
                'rf_f1': comp_metrics['RandomForest']['f1'],
                'rf_acc': comp_metrics['RandomForest']['accuracy'],
                'rf_reward': comp_metrics['RandomForest']['reward'],
                'et_f1': comp_metrics['ExtraTrees']['f1'],
                'et_acc': comp_metrics['ExtraTrees']['accuracy'],
                'et_reward': comp_metrics['ExtraTrees']['reward'],
                'gb_f1': comp_metrics['GradientBoosting']['f1'],
                'gb_acc': comp_metrics['GradientBoosting']['accuracy'],
                'gb_reward': comp_metrics['GradientBoosting']['reward'],
                'adaptation_occurred': adaptation_occurred,
                'adapted_components': adapted_components
            })
            
            # 5. Bandit Results Record
            bandit_records.append({
                'seed': seed,
                'window_id': w_idx,
                'selected_arm': arm_chosen,
                'arm_name': chosen_strat_name,
                'reward': reward,
                'oracle_reward': oracle_reward,
                'random_reward': random_reward,
                'ucb_score': bandit.ucb_scores[arm_chosen],
                'arm_selection_count': bandit.counts[arm_chosen],
                'ucb_score_arm1': bandit.ucb_scores[0],
                'ucb_score_arm2': bandit.ucb_scores[1],
                'ucb_score_arm3': bandit.ucb_scores[2],
                'arm1_count': bandit.counts[0],
                'arm2_count': bandit.counts[1],
                'arm3_count': bandit.counts[2],
                'arm1_avg_reward': bandit.avg_rewards[0],
                'arm2_avg_reward': bandit.avg_rewards[1],
                'arm3_avg_reward': bandit.avg_rewards[2],
                'lambda_val': lambda_val
            })
            
            # 6. Drift Results Record
            drift_records.append({
                'seed': seed,
                'window_id': w_idx,
                'drift_type': cfg['drift_type'],
                'drift_severity': cfg['severity'],
                'drift_transition': cfg['transition'],
                'Wasserstein': drift_res['wasserstein_mean'],
                'KS': drift_res['ks_mean'],
                'PSI': drift_res['psi_mean'],
                'wasserstein_mean': drift_res['wasserstein_mean'],
                'ks_mean': drift_res['ks_mean'],
                'psi_mean': drift_res['psi_mean'],
                'rolling_f1': perf_drift['rolling_f1'],
                'rolling_error_rate': perf_drift['rolling_error_rate'],
                'performance_degradation': perf_drift['performance_degradation']
            })
            
            # --- Section 13 Specific Formatted Records ---
            # A. drift_config.csv: seed, window, drift_type, severity, transition, drift_parameters
            drift_config_records.append({
                'seed': seed,
                'window': w_idx,
                'drift_type': cfg['drift_type'],
                'severity': cfg['severity'],
                'transition': cfg['transition'],
                'drift_parameters': cfg.get('drift_parameters', 'baseline')
            })
            
            # B. predictions.csv: seed, window, model, y_true, y_pred, probability
            for m_key, p_arr, prob_arr in [
                ('Frozen RF', frozen_pred, frozen_prob),
                ('Retrained RF', retr_pred, retr_prob),
                ('Adaptive Ensemble', pred_ens, prob_ens)
            ]:
                for idx_sample in range(len(y_win)):
                    predictions_records.append({
                        'seed': seed,
                        'window': w_idx,
                        'model': m_key,
                        'y_true': int(y_win[idx_sample]),
                        'y_pred': int(p_arr[idx_sample]),
                        'probability': round(float(prob_arr[idx_sample]), 4)
                    })
                    
            # C. performance.csv: seed, window, model, f1, precision, recall, accuracy, auc
            for m_key, m_eval in [
                ('Frozen RF', frozen_m),
                ('Retrained RF', retr_m),
                ('Adaptive Ensemble', ens_m),
                ('UCB1', chosen_m),
                ('Oracle', strat_metrics[oracle_best_strat]),
                ('RandomForest_comp', comp_metrics['RandomForest']),
                ('ExtraTrees_comp', comp_metrics['ExtraTrees']),
                ('GradientBoosting_comp', comp_metrics['GradientBoosting'])
            ]:
                performance_records.append({
                    'seed': seed,
                    'window': w_idx,
                    'model': m_key,
                    'f1': m_eval['f1'],
                    'precision': m_eval.get('precision', 0.0),
                    'recall': m_eval.get('recall', 0.0),
                    'accuracy': m_eval['accuracy'],
                    'auc': m_eval.get('auc', 0.5)
                })
                
            # D. bandit.csv: seed, window, selected_arm, reward, UCB_value, drift_type, severity, selected_strategy
            bandit_clean_records.append({
                'seed': seed,
                'window': w_idx,
                'selected_arm': arm_chosen,
                'reward': reward,
                'UCB_value': bandit.ucb_scores[arm_chosen],
                'drift_type': cfg['drift_type'],
                'severity': cfg['severity'],
                'selected_strategy': chosen_strat_name
            })
            
            # E. ensemble_weights.csv: seed, window, RF_weight, ET_weight, GB_weight, weight_spread
            w_vals = list(current_weights.values())
            max_w = float(np.max(w_vals))
            min_w = float(np.min(w_vals))
            ensemble_weights_clean_records.append({
                'seed': seed,
                'window': w_idx,
                'RF_weight': current_weights['RandomForest'],
                'ET_weight': current_weights['ExtraTrees'],
                'GB_weight': current_weights['GradientBoosting'],
                'weight_spread': max_w - min_w,
                'max_weight': max_w,
                'min_weight': min_w
            })
            
            # F. resources.csv: seed, window, model, cpu_percent_avg, cpu_percent_peak, cpu_time, wall_time, ram
            for m_name, inf_r, tr_r in [
                ('Frozen RF', res_inf_frozen, res_train_frozen),
                ('Retrained RF', res_inf_retr, res_train_retr),
                ('Adaptive Ensemble', res_inf_ens, res_adapt_ens),
                ('RandomForest_comp', comp_res_inf['RandomForest'], comp_res_train['RandomForest']),
                ('ExtraTrees_comp', comp_res_inf['ExtraTrees'], comp_res_train['ExtraTrees']),
                ('GradientBoosting_comp', comp_res_inf['GradientBoosting'], comp_res_train['GradientBoosting']),
                ('UCB1', chosen_inf_res, chosen_train_res)
            ]:
                resources_clean_records.append({
                    'seed': seed,
                    'window': w_idx,
                    'model': m_name,
                    'cpu_percent_avg': max(inf_r['avg_cpu_percent'], tr_r['avg_cpu_percent']),
                    'cpu_percent_peak': max(inf_r['peak_cpu_percent'], tr_r['peak_cpu_percent']),
                    'cpu_time': inf_r['total_cpu_time'] + tr_r['total_cpu_time'],
                    'wall_time': inf_r['wall_clock_time'] + tr_r['wall_clock_time'],
                    'ram': max(inf_r['peak_ram_mb'], tr_r['peak_ram_mb'])
                })
                
            # G. retraining.csv: seed, window, model, retrained, training_cpu_time, training_wall_time, cpu_percent_avg, cpu_percent_peak
            retraining_records.append({
                'seed': seed,
                'window': w_idx,
                'model': 'Frozen RF',
                'retrained': False,
                'training_cpu_time': 0.0,
                'training_wall_time': 0.0,
                'cpu_percent_avg': 0.0,
                'cpu_percent_peak': 0.0
            })
            retraining_records.append({
                'seed': seed,
                'window': w_idx,
                'model': 'Retrained RF',
                'retrained': True,
                'training_cpu_time': res_train_retr['total_cpu_time'],
                'training_wall_time': res_train_retr['wall_clock_time'],
                'cpu_percent_avg': res_train_retr['avg_cpu_percent'],
                'cpu_percent_peak': res_train_retr['peak_cpu_percent']
            })
            retraining_records.append({
                'seed': seed,
                'window': w_idx,
                'model': 'Adaptive Ensemble',
                'retrained': adaptation_occurred,
                'training_cpu_time': res_adapt_ens['total_cpu_time'],
                'training_wall_time': res_adapt_ens['wall_clock_time'],
                'cpu_percent_avg': res_adapt_ens['avg_cpu_percent'],
                'cpu_percent_peak': res_adapt_ens['peak_cpu_percent']
            })
            for comp_name in ['RandomForest', 'ExtraTrees', 'GradientBoosting']:
                retraining_records.append({
                    'seed': seed,
                    'window': w_idx,
                    'model': comp_name,
                    'retrained': (comp_name in adapted_components),
                    'training_cpu_time': comp_res_train[comp_name]['total_cpu_time'],
                    'training_wall_time': comp_res_train[comp_name]['wall_clock_time'],
                    'cpu_percent_avg': comp_res_train[comp_name]['avg_cpu_percent'],
                    'cpu_percent_peak': comp_res_train[comp_name]['peak_cpu_percent']
                })
                
    print("\n=== Experiment Execution Completed Across All Seeds ===")
    
    # 5. Convert to DataFrames
    df_window = pd.DataFrame(window_records)
    df_models = pd.DataFrame(model_metric_records)
    df_resources = pd.DataFrame(resource_records)
    df_weights = pd.DataFrame(ensemble_weight_records)
    df_bandit = pd.DataFrame(bandit_records)
    df_drift = pd.DataFrame(drift_records)
    
    # Section 13 DataFrames
    df_drift_config = pd.DataFrame(drift_config_records)
    df_predictions = pd.DataFrame(predictions_records)
    df_performance = pd.DataFrame(performance_records)
    df_bandit_clean = pd.DataFrame(bandit_clean_records)
    df_weights_clean = pd.DataFrame(ensemble_weights_clean_records)
    df_resources_clean = pd.DataFrame(resources_clean_records)
    df_retraining = pd.DataFrame(retraining_records)
    
    # 6. Statistical Analysis
    print("Running paired statistical tests and hypothesis evaluations...")
    df_stats = run_paired_tests(df_window)
    
    # 7. Save All Section 13 CSV Results Files
    print(f"Saving Section 13 CSV files to {results_dir}...")
    df_drift_config.to_csv(os.path.join(results_dir, 'drift_config.csv'), index=False)
    df_predictions.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    df_performance.to_csv(os.path.join(results_dir, 'performance.csv'), index=False)
    df_bandit_clean.to_csv(os.path.join(results_dir, 'bandit.csv'), index=False)
    df_weights_clean.to_csv(os.path.join(results_dir, 'ensemble_weights.csv'), index=False)
    df_resources_clean.to_csv(os.path.join(results_dir, 'resources.csv'), index=False)
    df_retraining.to_csv(os.path.join(results_dir, 'retraining.csv'), index=False)
    
    # Also save complementary analysis files
    df_stats.to_csv(os.path.join(results_dir, 'statistical_results.csv'), index=False)
    df_window.to_csv(os.path.join(results_dir, 'window_results.csv'), index=False)
    df_drift.to_csv(os.path.join(results_dir, 'drift_results.csv'), index=False)
    df_models.to_csv(os.path.join(results_dir, 'model_metrics.csv'), index=False)
    df_resources.to_csv(os.path.join(results_dir, 'resource_metrics.csv'), index=False)
    df_bandit.to_csv(os.path.join(results_dir, 'bandit_results.csv'), index=False)
    print("All CSV result files successfully generated.")
    
    # 8. Generate All 22 Figures
    print(f"Generating all 22 publication-quality figures in {figures_dir}...")
    generate_all_figures(
        figures_dir, df_window, df_models, df_resources, df_weights, df_bandit, df_drift
    )
    print("All 22 figures successfully generated.")
    
    # 9. Lambda Sensitivity Analysis: lambda = [0, 0.01, 0.05, 0.10, 0.20]
    print("\n--- Running Lambda Sensitivity Analysis (lambda in [0.0, 0.01, 0.05, 0.10, 0.20]) ---")
    lambda_results = []
    for lam in [0.0, 0.01, 0.05, 0.10, 0.20]:
        sub_b = df_bandit.copy()
        rewards = []
        for _, r in sub_b.iterrows():
            sub_w = df_window[(df_window['seed'] == r['seed']) &
                              (df_window['window_id'] == r['window_id']) &
                              (df_window['strategy'] == r['arm_name'])].iloc[0]
            f1_val = sub_w['f1']
            cpu_val = sub_w['cpu_time']
            wall_val = sub_w['wall_time']
            retrained_val = sub_w['retrained']
            rew_val, _ = bandit.compute_reward(
                f1=f1_val,
                cpu_time=cpu_val,
                wall_time=wall_val,
                retrained=retrained_val,
                lambda_val=lam
            )
            rewards.append(rew_val)
            
        lambda_results.append({
            'lambda': lam,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards))
        })
    df_lambda = pd.DataFrame(lambda_results)
    df_lambda.to_csv(os.path.join(results_dir, 'lambda_sensitivity.csv'), index=False)
    print("Lambda sensitivity analysis complete:")
    print(df_lambda)
    
    # 10. Section 10: Randomized Drift + Model Selection Analysis Tables
    print("\n--- Aggregating Results by Drift Condition (Section 10) ---")
    # Table 1: Drift Type | Severity | Frozen RF F1 | Retrained RF F1 | Adaptive Ensemble F1 | Best Strategy
    condition_f1 = df_window[df_window['strategy'].isin(['Frozen RF', 'Retrained RF', 'Adaptive Ensemble'])].groupby(
        ['drift_type', 'drift_severity', 'strategy']
    )['f1'].mean().unstack()
    
    table1_rows = []
    for (dtype, sev), row in condition_f1.iterrows():
        f1_frozen = row.get('Frozen RF', 0.0)
        f1_retr = row.get('Retrained RF', 0.0)
        f1_ens = row.get('Adaptive Ensemble', 0.0)
        
        best_cand = {'Frozen RF': f1_frozen, 'Retrained RF': f1_retr, 'Adaptive Ensemble': f1_ens}
        best_strat = max(best_cand, key=best_cand.get)
        table1_rows.append({
            'drift_type': dtype,
            'severity': sev,
            'frozen_rf_f1': round(f1_frozen, 4),
            'retrained_rf_f1': round(f1_retr, 4),
            'adaptive_ensemble_f1': round(f1_ens, 4),
            'best_strategy': best_strat
        })
    df_table1 = pd.DataFrame(table1_rows)
    df_table1.to_csv(os.path.join(results_dir, 'drift_severity_f1_summary.csv'), index=False)
    print("\nTable 1: Performance by Drift Type & Severity:")
    print(df_table1.to_string(index=False))
    
    # Table 2: Drift Type | Best Strategy | Mean F1 | CPU Cost | Retraining Cost
    table2_rows = []
    for dtype in ['none', 'covariate', 'concept', 'prior', 'mixed']:
        sub_type = df_window[(df_window['drift_type'] == dtype) &
                             (df_window['strategy'].isin(['Frozen RF', 'Retrained RF', 'Adaptive Ensemble']))]
        if len(sub_type) == 0:
            continue
        strat_means = sub_type.groupby('strategy').agg({
            'f1': 'mean',
            'cpu_time': 'mean',
            'training_cpu_time': 'mean'
        })
        best_strat = strat_means['f1'].idxmax()
        best_stats = strat_means.loc[best_strat]
        table2_rows.append({
            'drift_type': dtype,
            'best_strategy': best_strat,
            'mean_f1': round(best_stats['f1'], 4),
            'cpu_cost': round(best_stats['cpu_time'], 4),
            'retraining_cost': round(best_stats['training_cpu_time'], 4)
        })
    df_table2 = pd.DataFrame(table2_rows)
    df_table2.to_csv(os.path.join(results_dir, 'drift_condition_summary.csv'), index=False)
    print("\nTable 2: Best Strategy by Drift Type with Computation Costs:")
    print(df_table2.to_string(index=False))
    
    return {
        'drift_config': df_drift_config,
        'predictions': df_predictions,
        'performance': df_performance,
        'bandit': df_bandit_clean,
        'ensemble_weights': df_weights_clean,
        'resources': df_resources_clean,
        'retraining': df_retraining,
        'window_results': df_window,
        'statistical_results': df_stats,
        'lambda_sensitivity': df_lambda,
        'drift_condition_summary': df_table2,
        'drift_severity_f1_summary': df_table1
    }


if __name__ == '__main__':
    run_experiment(seeds=[42, 43, 44, 45, 46])
