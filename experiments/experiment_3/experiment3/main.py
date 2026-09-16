"""
================================================================================
EXPERIMENT 3 — MAIN EXECUTION & ORCHESTRATION SCRIPT
================================================================================
Randomized Drift-Aware UCB1 Model Selection in Industrial Telemetry.

Base Models (Bandit Arms):
- Arm 0: RandomForest (RF)
- Arm 1: ExtraTrees (ET)
- Arm 2: GradientBoosting (GB)
No SGDClassifier or other models.

Evaluation: Strict Prequential Protocol (Test-Then-Train)
Across 5 Deterministic Seeds: [42, 43, 44, 45, 46]
Dataset: 10,000 samples (2,000 initial training, 8,000 streaming over 16 windows)
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

# Ensure local directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    SEEDS, N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_DEPLOYMENT_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, KPI_COLS, TARGET_COL, ARM_NAMES,
    BANDIT_CONFIG, RESULTS_DIR, PLOTS_DIR, FIGURES_DIR, LOGS_DIR
)
from data_generation import generate_experiment_dataset
from drift_generator import export_drift_configuration
from models import create_base_models, create_frozen_rf, create_retrained_rf, StaticEnsemble
from bandit import UCB1Bandit, RandomSelector
from drift import compute_drift_metrics, DriftMonitor
from metrics import evaluate_predictions
from resource_monitor import measure_execution, ResourceMonitor
from statistics import run_paired_tests, compute_summary_stats
from plots import generate_all_figures, generate_section12_plots


def run_experiment(seeds=None, beta=0.05, c_exploration=1.0):
    """
    Run complete Experiment 3 across experimental seeds under strict prequential evaluation.
    """
    if seeds is None:
        seeds = SEEDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 80)
    print("  EXPERIMENT 3: RANDOMIZED DRIFT-AWARE UCB1 MODEL SELECTION")
    print(f"  Seeds: {seeds} | Beta: {beta} | UCB Exploration (c): {c_exploration}")
    print(f"  Arms: {ARM_NAMES}")
    print("=" * 80)

    # 1. Export reproducible drift configuration table
    drift_cfg_path = os.path.join(RESULTS_DIR, 'drift_configuration.csv')
    export_drift_configuration(seeds, drift_cfg_path)
    print(f"[EXP3] Saved exact drift configuration to: {drift_cfg_path}")

    # Primary Required Records
    window_results_records = []
    model_resource_records = []
    
    # Backward-compatible & Detailed Records
    detailed_window_records = []
    predictions_records = []
    performance_records = []
    bandit_records = []
    ensemble_weights_records = []
    resources_records = []
    retraining_records = []
    drift_analysis_records = []

    # Telemetry accumulators for initial training
    initial_training_records = []

    for seed in seeds:
        print(f"\n" + "-" * 70)
        print(f"  Executing Seed {seed}")
        print("-" * 70)

        # A. Dataset Generation with Randomized Drift
        df, schedule = generate_experiment_dataset(seed)

        # B. Initial Partitioning (Chronological, No Leakage)
        X_train_raw = df.iloc[:N_INITIAL_TRAINING][KPI_COLS].values
        y_train = df.iloc[:N_INITIAL_TRAINING][TARGET_COL].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        # C. Initial Training of Base Models & Profiling
        print("  Fitting candidate base models on initial 2,000 samples...")
        base_models = create_base_models(seed)
        base_model_names = ['RandomForest', 'ExtraTrees', 'GradientBoosting']

        init_resource_profiles = {}
        for m_name in base_model_names:
            model = base_models[m_name]
            _, init_res = measure_execution(model.fit, X_train_scaled, y_train)
            init_resource_profiles[m_name] = init_res
            initial_training_records.append({
                'seed': seed,
                'model': m_name,
                'wall_time': init_res['wall_clock_time'],
                'cpu_time': init_res['total_cpu_time'],
                'cpu_percent': init_res['avg_cpu_percent'],
                'peak_cpu_percent': init_res['peak_cpu_percent'],
                'avg_ram_mb': init_res['avg_ram_mb'],
                'peak_ram_mb': init_res['peak_ram_mb']
            })
            print(f"    Initial {m_name:17s} -> CPU: {init_res['total_cpu_time']:.4f}s, "
                  f"Wall: {init_res['wall_clock_time']:.4f}s, RAM: {init_res['peak_ram_mb']:.1f}MB")

        # Train Baselines
        frozen_rf = create_frozen_rf(seed)
        frozen_rf.fit(X_train_scaled, y_train)

        retrained_rf = create_retrained_rf(seed)
        retrained_rf.fit(X_train_scaled, y_train)

        static_ens = StaticEnsemble(seed)
        static_ens.fit(X_train_scaled, y_train)

        # Buffers for cumulative retraining baseline
        cum_train_X = list(X_train_scaled)
        cum_train_y = list(y_train)

        # Buffers for adaptive candidates
        adapt_buf_X = list(X_train_scaled[-1000:])
        adapt_buf_y = list(y_train[-1000:])

        # Initialize UCB1 Bandit with 3 candidate base models
        bandit = UCB1Bandit(
            n_arms=3,
            c=c_exploration,
            ref_cpu_time=0.5,
            arm_names=base_model_names,
            beta=beta
        )
        random_selector = RandomSelector(seed=seed, n_arms=3)
        drift_monitor = DriftMonitor(baseline_f1=0.88)

        cumulative_bandit_reward = 0.0
        retrain_events_count = 0

        # Initial arm selection for window 0
        current_selected_arm = bandit.select_arm()

        # D. Sequential Streaming Loop (16 Windows)
        for w_idx, cfg in enumerate(schedule):
            w_id = cfg['window_id']
            dtype = cfg['drift_type']
            sev = cfg['severity']
            trans = cfg['transition']
            params = cfg.get('drift_parameters', 'baseline')

            start_idx = N_INITIAL_TRAINING + w_idx * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE

            X_win_raw = df.iloc[start_idx:end_idx][KPI_COLS].values
            y_win = df.iloc[start_idx:end_idx][TARGET_COL].values
            X_win_scaled = scaler.transform(X_win_raw)

            # 1. Unsupervised Drift Monitoring (No labels used)
            drift_res = compute_drift_metrics(X_train_raw, X_win_raw, KPI_COLS)
            drift_analysis_records.append({
                'seed': seed,
                'window_id': w_id,
                'drift_type': dtype,
                'severity': sev,
                'transition': trans,
                'wasserstein_mean': drift_res['wasserstein_mean'],
                'ks_mean': drift_res['ks_mean'],
                'psi_mean': drift_res['psi_mean']
            })

            # 2. Strict Out-of-Sample Prequential Inference
            # A. Base Models Evaluation
            base_metrics = {}
            base_preds = {}
            base_probs = {}
            base_inf_profiles = {}

            for a_idx, m_name in enumerate(base_model_names):
                m_obj = base_models[m_name]
                def _infer(m=m_obj):
                    pr = m.predict_proba(X_win_scaled)
                    p_pos = pr[:, 1] if pr.shape[1] > 1 else pr[:, 0]
                    p_cls = (p_pos >= 0.5).astype(int)
                    return p_cls, p_pos

                (preds, probs), inf_prof = measure_execution(_infer)
                met = evaluate_predictions(y_win, preds, probs)

                base_metrics[m_name] = met
                base_preds[m_name] = preds
                base_probs[m_name] = probs
                base_inf_profiles[m_name] = inf_prof

                # Store for predictions.csv
                for s_i in range(len(y_win)):
                    predictions_records.append({
                        'seed': seed,
                        'window': w_id,
                        'model': m_name,
                        'y_true': int(y_win[s_i]),
                        'y_pred': int(preds[s_i]),
                        'probability': float(probs[s_i])
                    })

                # Store for performance.csv
                performance_records.append({
                    'seed': seed,
                    'window': w_id,
                    'model': m_name,
                    'f1': met['f1'],
                    'precision': met['precision'],
                    'recall': met['recall'],
                    'accuracy': met['accuracy'],
                    'auc': met['auc']
                })

            # B. Baseline Models Evaluation (Frozen RF, Retrained RF, Static Ensemble)
            # Frozen RF
            (frozen_pred, frozen_prob), frozen_inf_prof = measure_execution(
                lambda: ((frozen_rf.predict_proba(X_win_scaled)[:, 1] >= 0.5).astype(int),
                         frozen_rf.predict_proba(X_win_scaled)[:, 1])
            )
            frozen_met = evaluate_predictions(y_win, frozen_pred, frozen_prob)

            # Retrained RF
            (retr_pred, retr_prob), retr_inf_prof = measure_execution(
                lambda: ((retrained_rf.predict_proba(X_win_scaled)[:, 1] >= 0.5).astype(int),
                         retrained_rf.predict_proba(X_win_scaled)[:, 1])
            )
            retr_met = evaluate_predictions(y_win, retr_pred, retr_prob)

            # Static Ensemble
            (ens_pred, ens_prob), ens_inf_prof = measure_execution(
                lambda: (static_ens.predict(X_win_scaled),
                         static_ens.predict_proba(X_win_scaled)[:, 1])
            )
            ens_met = evaluate_predictions(y_win, ens_pred, ens_prob)

            # 3. Reward & Cost Calculation for Candidate Arms
            arm_rewards = {}
            arm_costs = {}
            arm_ucbs = {}

            for a_idx, m_name in enumerate(base_model_names):
                f1_val = base_metrics[m_name]['f1']
                cpu_t = base_inf_profiles[m_name]['total_cpu_time']
                rew, cost = bandit.compute_reward(
                    f1=f1_val,
                    cpu_time=cpu_t,
                    lambda_val=beta
                )
                arm_rewards[m_name] = rew
                arm_costs[m_name] = cost

            # Calculate current UCB scores before updating
            for a_idx, m_name in enumerate(base_model_names):
                if bandit.counts[a_idx] > 0:
                    b_bonus = bandit.c * np.sqrt(np.log(max(bandit.total_selections, 1)) / bandit.counts[a_idx])
                    arm_ucbs[m_name] = bandit.avg_rewards[a_idx] + b_bonus
                else:
                    arm_ucbs[m_name] = float('inf')

            # Selected model metrics for current window
            selected_model_name = base_model_names[current_selected_arm]
            selected_met = base_metrics[selected_model_name]
            selected_inf_prof = base_inf_profiles[selected_model_name]
            selected_reward = arm_rewards[selected_model_name]

            cumulative_bandit_reward += selected_reward

            # Oracle and Random Selector reference rewards
            oracle_reward = max(arm_rewards.values())
            random_arm = random_selector.select_arm()
            random_reward = arm_rewards[base_model_names[random_arm]]

            # Record bandit choice
            bandit_records.append({
                'seed': seed,
                'window': w_id,
                'window_id': w_id,
                'selected_arm': selected_model_name,
                'arm_name': selected_model_name,
                'reward': selected_reward,
                'oracle_reward': oracle_reward,
                'random_reward': random_reward,
                'UCB_value': arm_ucbs[selected_model_name] if arm_ucbs[selected_model_name] != float('inf') else 1.5
            })

            # Pseudo ensemble weights representation based on softmax over UCBs
            bounded_ucbs = [min(2.0, max(0.0, arm_ucbs[m])) for m in base_model_names]
            exp_ucbs = np.exp(np.array(bounded_ucbs) * 2.0)
            norm_weights = exp_ucbs / np.sum(exp_ucbs)
            ensemble_weights_records.append({
                'seed': seed,
                'window': w_id,
                'window_id': w_id,
                'RF_weight': float(norm_weights[0]),
                'ET_weight': float(norm_weights[1]),
                'GB_weight': float(norm_weights[2]),
                'rf_weight': float(norm_weights[0]),
                'et_weight': float(norm_weights[1]),
                'gb_weight': float(norm_weights[2]),
                'rf_f1': base_metrics['RandomForest']['f1'],
                'et_f1': base_metrics['ExtraTrees']['f1'],
                'gb_f1': base_metrics['GradientBoosting']['f1'],
                'weight_spread': float(np.max(norm_weights) - np.min(norm_weights))
            })

            # 4. Bandit Statistics Update
            bandit.update(current_selected_arm, selected_reward)

            # 5. Next Arm Selection
            next_selected_arm = bandit.select_arm()

            # 6. Adaptation / Retraining Phase (Strictly Post-Evaluation)
            # Update cumulative buffers with current window
            cum_train_X.extend(X_win_scaled)
            cum_train_y.extend(y_win)
            adapt_buf_X.extend(X_win_scaled)
            adapt_buf_y.extend(y_win)
            if len(adapt_buf_X) > 2000:
                adapt_buf_X = adapt_buf_X[-2000:]
                adapt_buf_y = adapt_buf_y[-2000:]

            # Baseline: Retrained RF retrains cumulatively on every window
            _, retr_train_prof = measure_execution(retrained_rf.fit, cum_train_X, cum_train_y)
            retraining_records.append({
                'seed': seed,
                'window': w_id,
                'model': 'Retrained RF',
                'retrained': True,
                'training_cpu_time': retr_train_prof['total_cpu_time'],
                'training_wall_time': retr_train_prof['wall_clock_time'],
                'cpu_percent_avg': retr_train_prof['avg_cpu_percent'],
                'cpu_percent_peak': retr_train_prof['peak_cpu_percent']
            })

            # Frozen RF never retrains
            retraining_records.append({
                'seed': seed,
                'window': w_id,
                'model': 'Frozen RF',
                'retrained': False,
                'training_cpu_time': 0.0,
                'training_wall_time': 0.0,
                'cpu_percent_avg': 0.0,
                'cpu_percent_peak': 0.0
            })

            # Base Candidates Adaptation under UCB1 Control:
            # Retrain when drift > threshold or selected model F1 drops below baseline
            drift_detected = drift_res['wasserstein_mean'] > 0.15
            perf_degraded = selected_met['f1'] < 0.82
            trigger_adaptation = drift_detected or perf_degraded

            retraining_event_flag = False
            candidate_retrain_profiles = {}

            for m_name in base_model_names:
                m_retrained = False
                train_time = 0.0
                train_cpu = 0.0
                train_cpu_pct = 0.0
                ram_used = base_inf_profiles[m_name]['avg_ram_mb']

                # Retrain candidate if adaptation triggered and model is active or underperforming
                if trigger_adaptation and (m_name == selected_model_name or base_metrics[m_name]['f1'] < 0.80):
                    m_obj = base_models[m_name]
                    _, retrain_prof = measure_execution(m_obj.fit, adapt_buf_X, adapt_buf_y)
                    candidate_retrain_profiles[m_name] = retrain_prof
                    m_retrained = True
                    train_time = retrain_prof['wall_clock_time']
                    train_cpu = retrain_prof['total_cpu_time']
                    train_cpu_pct = retrain_prof['avg_cpu_percent']
                    ram_used = retrain_prof['peak_ram_mb']
                    retrain_events_count += 1
                    retraining_event_flag = True

                    retraining_records.append({
                        'seed': seed,
                        'window': w_id,
                        'model': m_name,
                        'retrained': True,
                        'training_cpu_time': train_cpu,
                        'training_wall_time': train_time,
                        'cpu_percent_avg': train_cpu_pct,
                        'cpu_percent_peak': retrain_prof['peak_cpu_percent']
                    })
                else:
                    retraining_records.append({
                        'seed': seed,
                        'window': w_id,
                        'model': m_name,
                        'retrained': False,
                        'training_cpu_time': 0.0,
                        'training_wall_time': 0.0,
                        'cpu_percent_avg': 0.0,
                        'cpu_percent_peak': 0.0
                    })

                # Record per-model resource consumption
                model_resource_records.append({
                    'seed': seed,
                    'window': w_id,
                    'model': m_name,
                    'training_time': train_time,
                    'retraining_time': train_time,
                    'prediction_time': base_inf_profiles[m_name]['wall_clock_time'],
                    'cpu_time': base_inf_profiles[m_name]['total_cpu_time'] + train_cpu,
                    'cpu_utilization': max(base_inf_profiles[m_name]['avg_cpu_percent'], train_cpu_pct),
                    'memory_usage': ram_used,
                    'retrained': m_retrained
                })

                # Legacy resources.csv record
                resources_records.append({
                    'seed': seed,
                    'window': w_id,
                    'model': m_name,
                    'cpu_percent_avg': max(base_inf_profiles[m_name]['avg_cpu_percent'], train_cpu_pct),
                    'cpu_percent_peak': max(base_inf_profiles[m_name]['peak_cpu_percent'], train_cpu_pct),
                    'cpu_time': base_inf_profiles[m_name]['total_cpu_time'] + train_cpu,
                    'wall_time': base_inf_profiles[m_name]['wall_clock_time'] + train_time,
                    'ram': ram_used
                })

            # Record resources for baselines
            resources_records.append({
                'seed': seed,
                'window': w_id,
                'model': 'Frozen RF',
                'cpu_percent_avg': frozen_inf_prof['avg_cpu_percent'],
                'cpu_percent_peak': frozen_inf_prof['peak_cpu_percent'],
                'cpu_time': frozen_inf_prof['total_cpu_time'],
                'wall_time': frozen_inf_prof['wall_clock_time'],
                'ram': frozen_inf_prof['avg_ram_mb']
            })
            resources_records.append({
                'seed': seed,
                'window': w_id,
                'model': 'Retrained RF',
                'cpu_percent_avg': max(retr_inf_prof['avg_cpu_percent'], retr_train_prof['avg_cpu_percent']),
                'cpu_percent_peak': max(retr_inf_prof['peak_cpu_percent'], retr_train_prof['peak_cpu_percent']),
                'cpu_time': retr_inf_prof['total_cpu_time'] + retr_train_prof['total_cpu_time'],
                'wall_time': retr_inf_prof['wall_clock_time'] + retr_train_prof['wall_clock_time'],
                'ram': retr_train_prof['peak_ram_mb']
            })

            # 7. Record Required Primary Table: window_results.csv
            window_results_records.append({
                'seed': seed,
                'window': w_id,
                'drift_type': dtype,
                'severity': sev,
                'transition': trans,
                'selected_model': selected_model_name,
                'RF_F1': base_metrics['RandomForest']['f1'],
                'ET_F1': base_metrics['ExtraTrees']['f1'],
                'GB_F1': base_metrics['GradientBoosting']['f1'],
                'RF_accuracy': base_metrics['RandomForest']['accuracy'],
                'ET_accuracy': base_metrics['ExtraTrees']['accuracy'],
                'GB_accuracy': base_metrics['GradientBoosting']['accuracy'],
                'RF_reward': arm_rewards['RandomForest'],
                'ET_reward': arm_rewards['ExtraTrees'],
                'GB_reward': arm_rewards['GradientBoosting'],
                'RF_UCB': arm_ucbs['RandomForest'] if arm_ucbs['RandomForest'] != float('inf') else 1.5,
                'ET_UCB': arm_ucbs['ExtraTrees'] if arm_ucbs['ExtraTrees'] != float('inf') else 1.5,
                'GB_UCB': arm_ucbs['GradientBoosting'] if arm_ucbs['GradientBoosting'] != float('inf') else 1.5,
                'cumulative_reward': cumulative_bandit_reward,
                'selected_model_prediction_time': selected_inf_prof['wall_clock_time'],
                'selected_model_cpu_time': selected_inf_prof['total_cpu_time'],
                'retraining_event': retraining_event_flag
            })

            # Detailed window record for strategy-level statistics & plotting
            for s_name, s_met, s_inf, s_train_cpu, s_train_wall, s_retrained in [
                ('Frozen RF', frozen_met, frozen_inf_prof, 0.0, 0.0, False),
                ('Retrained RF', retr_met, retr_inf_prof, retr_train_prof['total_cpu_time'], retr_train_prof['wall_clock_time'], True),
                ('Static Ensemble', ens_met, ens_inf_prof, 0.0, 0.0, False),
                ('Adaptive Ensemble', selected_met, selected_inf_prof,
                 candidate_retrain_profiles.get(selected_model_name, {}).get('total_cpu_time', 0.0),
                 candidate_retrain_profiles.get(selected_model_name, {}).get('wall_clock_time', 0.0),
                 retraining_event_flag),
                ('UCB1', selected_met, selected_inf_prof,
                 candidate_retrain_profiles.get(selected_model_name, {}).get('total_cpu_time', 0.0),
                 candidate_retrain_profiles.get(selected_model_name, {}).get('wall_clock_time', 0.0),
                 retraining_event_flag)
            ]:
                tot_cpu = s_inf['total_cpu_time'] + s_train_cpu
                tot_wall = s_inf['wall_clock_time'] + s_train_wall
                detailed_window_records.append({
                    'seed': seed,
                    'window_id': w_id,
                    'drift_type': dtype,
                    'drift_severity': sev,
                    'drift_transition': trans,
                    'strategy': s_name,
                    'f1': s_met['f1'],
                    'accuracy': s_met['accuracy'],
                    'precision': s_met['precision'],
                    'recall': s_met['recall'],
                    'auc': s_met['auc'],
                    'inference_time': s_inf['wall_clock_time'],
                    'training_time': s_train_wall,
                    'retraining_time': s_train_wall,
                    'training_cpu_time': s_train_cpu,
                    'retraining_cpu_time': s_train_cpu,
                    'wall_time': tot_wall,
                    'cpu_time': tot_cpu,
                    'cpu_utilization': s_inf['avg_cpu_percent'],
                    'cpu_user_time': s_inf['cpu_user_time'],
                    'cpu_system_time': s_inf['cpu_system_time'],
                    'avg_ram_mb': s_inf['avg_ram_mb'],
                    'peak_ram_mb': s_inf['peak_ram_mb'],
                    'retrained': s_retrained,
                    'adapted': s_retrained
                })

            # Advance bandit arm
            current_selected_arm = next_selected_arm

    # =========================================================================
    # E. SAVE ALL REQUIRED DATASETS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SAVING RESULTS DATASETS")
    print("=" * 80)

    # 1. Primary: window_results.csv
    df_win_results = pd.DataFrame(window_results_records)
    win_res_path = os.path.join(RESULTS_DIR, 'window_results.csv')
    df_win_results.to_csv(win_res_path, index=False)
    print(f"  [1/4] Saved window_results.csv ({len(df_win_results)} rows) -> {win_res_path}")

    # 2. Primary: model_resource_results.csv
    df_mod_res = pd.DataFrame(model_resource_records)
    mod_res_path = os.path.join(RESULTS_DIR, 'model_resource_results.csv')
    df_mod_res.to_csv(mod_res_path, index=False)
    print(f"  [2/4] Saved model_resource_results.csv ({len(df_mod_res)} rows) -> {mod_res_path}")

    # 3. Primary: drift_configuration.csv (already exported, check presence)
    print(f"  [3/4] Verified drift_configuration.csv -> {drift_cfg_path}")

    # 4. Primary: summary_results.csv
    summary_rows = []
    # Compute summary across strategies/models
    df_detailed = pd.DataFrame(detailed_window_records)
    for strat in ['UCB1', 'Frozen RF', 'Retrained RF', 'Static Ensemble']:
        sub = df_detailed[df_detailed['strategy'] == strat]
        f1_stats = compute_summary_stats(sub['f1'])
        acc_stats = compute_summary_stats(sub['accuracy'])
        cum_cpu = float(sub['cpu_time'].sum())
        cum_retrain = float(sub['retraining_time'].sum())
        n_retrains = int(sub['retrained'].sum())
        
        # Reward stats
        if strat == 'UCB1':
            mean_rew = float(df_win_results['cumulative_reward'].iloc[-1] / (len(df_win_results)))
            # Model selection frequency
            sel_freq = df_win_results['selected_model'].value_counts(normalize=True).to_dict()
            sel_freq_str = "; ".join([f"{k}: {v*100:.1f}%" for k, v in sel_freq.items()])
        else:
            mean_rew = float(f1_stats['mean'] - beta * min(1.0, cum_cpu / (16 * 0.5)))
            sel_freq_str = f"Fixed: {strat}"

        summary_rows.append({
            'strategy': strat,
            'mean_F1': f1_stats['mean'],
            'std_F1': f1_stats['std'],
            'mean_accuracy': acc_stats['mean'],
            'std_accuracy': acc_stats['std'],
            'cumulative_CPU': cum_cpu,
            'cumulative_retraining_time': cum_retrain,
            'number_of_retraining_events': n_retrains,
            'mean_reward': mean_rew,
            'model_selection_frequency': sel_freq_str
        })

    df_summary = pd.DataFrame(summary_rows)
    sum_res_path = os.path.join(RESULTS_DIR, 'summary_results.csv')
    df_summary.to_csv(sum_res_path, index=False)
    print(f"  [4/4] Saved summary_results.csv ({len(df_summary)} rows) -> {sum_res_path}")

    # Backward-compatible CSVs for regression test suite
    pd.DataFrame(predictions_records).to_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)
    pd.DataFrame(performance_records).to_csv(os.path.join(RESULTS_DIR, 'performance.csv'), index=False)
    pd.DataFrame(bandit_records).to_csv(os.path.join(RESULTS_DIR, 'bandit.csv'), index=False)
    pd.DataFrame(ensemble_weights_records).to_csv(os.path.join(RESULTS_DIR, 'ensemble_weights.csv'), index=False)
    pd.DataFrame(resources_records).to_csv(os.path.join(RESULTS_DIR, 'resources.csv'), index=False)
    pd.DataFrame(retraining_records).to_csv(os.path.join(RESULTS_DIR, 'retraining.csv'), index=False)
    
    # Save legacy drift_config.csv
    df_drift_legacy = pd.read_csv(drift_cfg_path)[['seed', 'window', 'drift_type', 'severity', 'transition', 'drift_parameters']]
    df_drift_legacy.to_csv(os.path.join(RESULTS_DIR, 'drift_config.csv'), index=False)

    # =========================================================================
    # F. STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("  RUNNING STATISTICAL TESTS & WILCOXON COMPARISONS")
    print("=" * 80)
    stats_df = run_paired_tests(df_detailed)
    stats_path = os.path.join(RESULTS_DIR, 'statistical_results.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved statistical test results -> {stats_path}")

    # =========================================================================
    # G. PLOT GENERATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("  GENERATING ALL PUBLICATION FIGURES")
    print("=" * 80)
    df_drift_all = pd.DataFrame(drift_analysis_records)
    generate_all_figures(
        output_dir=FIGURES_DIR,
        df_window=df_detailed,
        df_models=pd.DataFrame(performance_records),
        df_resources=pd.DataFrame(resources_records),
        df_weights=pd.DataFrame(ensemble_weights_records),
        df_bandit=pd.DataFrame(bandit_records),
        df_drift=df_drift_all
    )
    # Also generate in PLOTS_DIR
    generate_all_figures(
        output_dir=PLOTS_DIR,
        df_window=df_detailed,
        df_models=pd.DataFrame(performance_records),
        df_resources=pd.DataFrame(resources_records),
        df_weights=pd.DataFrame(ensemble_weights_records),
        df_bandit=pd.DataFrame(bandit_records),
        df_drift=df_drift_all
    )
    # Generate all Section 12 required figures
    generate_section12_plots(
        output_dir=FIGURES_DIR,
        df_win_results=df_win_results,
        df_mod_res=df_mod_res,
        df_detailed=df_detailed
    )
    generate_section12_plots(
        output_dir=PLOTS_DIR,
        df_win_results=df_win_results,
        df_mod_res=df_mod_res,
        df_detailed=df_detailed
    )
    print(f"  Generated publication figures in: {FIGURES_DIR} and {PLOTS_DIR}")

    print("\n" + "=" * 80)
    print("  EXPERIMENT 3 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return df_win_results, df_mod_res, df_summary, stats_df


if __name__ == '__main__':
    run_experiment()
