"""
================================================================================
EXPERIMENT 3 — UCB1-CONTROLLED HETEROGENEOUS ADAPTIVE ENSEMBLE
================================================================================
Compares three experimental approaches across 5 deterministic seeds (42-46):
  1. Frozen RF: Random Forest trained only on initial 2,000 samples. Never retrained.
  2. Continuously Retrained RF: Retrained at every window on all cumulative data.
  3. UCB1 Adaptive Ensemble: Multi-armed bandit selecting among 3 arms:
       Arm 0: RF
       Arm 1: ET
       Arm 2: Heterogeneous Ensemble (RF + ET + GB)
     with multi-temporal horizons (RF: 3500, ET: 2000, GB: 1200), diversity-aware
     soft voting, and drift-triggered adaptation.

Strict out-of-sample prequential evaluation:
  TEST -> MEASURE -> UPDATE -> ADAPT -> NEXT WINDOW
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    balanced_accuracy_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEEDS, N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE,
    KPI_COLS, TARGET_COL, ARM_NAMES, RESULTS_DIR, FIGURES_DIR,
    LAMBDA_COST, REF_CPU_TIME, C_EXPLORATION, DRIFT_THRESHOLD, DECISION_THRESHOLD
)
from data_generation import generate_experiment_dataset, export_drift_configuration
from models import create_frozen_rf, create_retrained_rf
from ensemble import AdaptiveEnsemble
from bandit import UCB1Bandit
from drift import compute_drift_metrics
from metrics import evaluate_predictions
from resource_monitor import measure_execution
from statistics import run_paired_tests
from plots import generate_all_figures


def compute_extended_metrics(y_true, y_pred, y_prob):
    """Compute base metrics plus PR-AUC and balanced accuracy."""
    base_m = evaluate_predictions(y_true, y_pred, y_prob)
    
    # PR-AUC
    try:
        if len(np.unique(y_true)) > 1 and y_prob is not None:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(rec_arr, prec_arr))
        else:
            pr_auc = float(base_m['f1'])
    except Exception:
        pr_auc = float(base_m['f1'])

    # Balanced accuracy
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = float(base_m['accuracy'])

    base_m['pr_auc'] = pr_auc
    base_m['balanced_accuracy'] = bal_acc
    return base_m


def run_single_seed(seed):
    """Execute Experiment 3 for a single seed across 16 streaming windows."""
    print(f"\n{'='*70}\n[Seed {seed}] Generating dataset with heterogeneous drift...\n{'='*70}")
    df, schedule = generate_experiment_dataset(seed)

    X_all = df[KPI_COLS].values
    y_all = df[TARGET_COL].values

    # Initial 2,000 baseline samples
    X_init = X_all[:N_INITIAL_TRAINING]
    y_init = y_all[:N_INITIAL_TRAINING]

    print(f"[Seed {seed}] Initial training on {N_INITIAL_TRAINING} samples (violation rate: {np.mean(y_init):.1%})...")

    # Initialize Approach 1: Frozen RF
    frozen_rf = create_frozen_rf(seed)
    _, frozen_init_res = measure_execution(frozen_rf.fit, X_init, y_init)

    # Initialize Approach 2: Retrained RF
    retrained_rf = create_retrained_rf(seed)
    _, retrained_init_res = measure_execution(retrained_rf.fit, X_init, y_init)

    # Initialize Approach 3: UCB1 Adaptive Ensemble
    ensemble = AdaptiveEnsemble(seed=seed)
    comp_init_res = ensemble.fit_initial(X_init, y_init)

    # Bandit selector for Approach 3 arms (RF, ET, Ensemble)
    bandit = UCB1Bandit(
        n_arms=3,
        c=C_EXPLORATION,
        lambda_cost=LAMBDA_COST,
        ref_cpu_time=REF_CPU_TIME,
        arm_names=['RF', 'ET', 'Ensemble']
    )

    cumulative_X = list(X_init)
    cumulative_y = list(y_init)

    window_rows = []
    bandit_rows = []
    weight_rows = []
    retrain_rows = []
    resource_rows = []
    diversity_rows = []

    # Initial resource entries
    resource_rows.append({
        'seed': seed, 'window': -1, 'approach': 'Frozen RF', 'component': 'RF',
        'training_cpu_time': frozen_init_res['total_cpu_time'], 'prediction_cpu_time': 0.0,
        'cpu_utilization': frozen_init_res['avg_cpu_percent'], 'ram_mb': frozen_init_res['avg_ram_mb'],
        'peak_ram_mb': frozen_init_res['peak_ram_mb']
    })
    resource_rows.append({
        'seed': seed, 'window': -1, 'approach': 'Retrained RF', 'component': 'RF',
        'training_cpu_time': retrained_init_res['total_cpu_time'], 'prediction_cpu_time': 0.0,
        'cpu_utilization': retrained_init_res['avg_cpu_percent'], 'ram_mb': retrained_init_res['avg_ram_mb'],
        'peak_ram_mb': retrained_init_res['peak_ram_mb']
    })
    for comp, res in comp_init_res.items():
        resource_rows.append({
            'seed': seed, 'window': -1, 'approach': 'UCB1 Adaptive Ensemble', 'component': comp,
            'training_cpu_time': res['total_cpu_time'], 'prediction_cpu_time': 0.0,
            'cpu_utilization': res['avg_cpu_percent'], 'ram_mb': res['avg_ram_mb'],
            'peak_ram_mb': res['peak_ram_mb']
        })

    sched_map = {cfg['window_id']: cfg for cfg in schedule}

    # ── 16 Streaming Windows ─────────────────────────────────────────────────
    for w in range(N_WINDOWS):
        cfg = sched_map.get(w, {'drift_type': 'none', 'severity': 'none', 'transition': 'stable'})
        dtype = cfg['drift_type']
        sev = cfg['severity']
        trans = cfg['transition']

        idx_s = N_INITIAL_TRAINING + w * WINDOW_SIZE
        idx_e = idx_s + WINDOW_SIZE
        X_win = X_all[idx_s:idx_e]
        y_win = y_all[idx_s:idx_e]

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: PREDICT (Strict Out-of-Sample Prequential Evaluation)
        # ──────────────────────────────────────────────────────────────────────

        # A. Frozen RF Prediction
        (frozen_pred, frozen_prob), frozen_inf_res = measure_execution(
            lambda: (frozen_rf.predict(X_win), frozen_rf.predict_proba(X_win)[:, 1])
        )

        # B. Retrained RF Prediction
        (retr_pred, retr_prob), retr_inf_res = measure_execution(
            lambda: (retrained_rf.predict(X_win), retrained_rf.predict_proba(X_win)[:, 1])
        )

        # C. Candidate Arms for Approach 3
        # Arm 0: RF
        (rf_pred, rf_prob), rf_inf_res = measure_execution(
            lambda: (
                (ensemble.predict_component_proba('RF', X_win) >= DECISION_THRESHOLD).astype(int),
                ensemble.predict_component_proba('RF', X_win)
            )
        )
        # Arm 1: ET
        (et_pred, et_prob), et_inf_res = measure_execution(
            lambda: (
                (ensemble.predict_component_proba('ET', X_win) >= DECISION_THRESHOLD).astype(int),
                ensemble.predict_component_proba('ET', X_win)
            )
        )
        # Arm 2: Ensemble (soft voting)
        (ens_pred, ens_prob, comp_preds, comp_probs), ens_inf_res = measure_execution(
            lambda: ensemble.predict(X_win, threshold=DECISION_THRESHOLD)
        )

        # Arm selection via UCB1 before evaluating current labels
        selected_arm = bandit.select_arm()
        arm_names = ['RF', 'ET', 'Ensemble']
        selected_model = arm_names[selected_arm]

        # Official Approach 3 prediction is from selected arm
        candidate_preds = [rf_pred, et_pred, ens_pred]
        candidate_probs = [rf_prob, et_prob, ens_prob]
        candidate_inf_res = [rf_inf_res, et_inf_res, ens_inf_res]

        app3_pred = candidate_preds[selected_arm]
        app3_prob = candidate_probs[selected_arm]
        app3_inf_res = candidate_inf_res[selected_arm]

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: MEASURE & EVALUATE (Using current window labels)
        # ──────────────────────────────────────────────────────────────────────
        frozen_m = compute_extended_metrics(y_win, frozen_pred, frozen_prob)
        retr_m = compute_extended_metrics(y_win, retr_pred, retr_prob)
        app3_m = compute_extended_metrics(y_win, app3_pred, app3_prob)

        rf_m = compute_extended_metrics(y_win, rf_pred, rf_prob)
        et_m = compute_extended_metrics(y_win, et_pred, et_prob)
        ens_m = compute_extended_metrics(y_win, ens_pred, ens_prob)

        # Rewards for each candidate arm
        rf_reward, rf_norm_cost = bandit.compute_reward(rf_m['f1'], rf_inf_res['total_cpu_time'])
        et_reward, et_norm_cost = bandit.compute_reward(et_m['f1'], et_inf_res['total_cpu_time'])
        ens_reward, ens_norm_cost = bandit.compute_reward(ens_m['f1'], ens_inf_res['total_cpu_time'])

        arm_rewards = [rf_reward, et_reward, ens_reward]

        # ──────────────────────────────────────────────────────────────────────
        # STEP 3: UPDATE UCB1 BANDIT
        # ──────────────────────────────────────────────────────────────────────
        rf_ucb = bandit.ucb_scores[0]
        et_ucb = bandit.ucb_scores[1]
        ens_ucb = bandit.ucb_scores[2]

        bandit.update(selected_arm, arm_rewards[selected_arm])

        bandit_rows.append({
            'seed': seed,
            'window_id': w,
            'selected_arm': selected_arm,
            'selected_model': selected_model,
            'RF_UCB': round(rf_ucb, 4) if not np.isinf(rf_ucb) else 999.0,
            'ET_UCB': round(et_ucb, 4) if not np.isinf(et_ucb) else 999.0,
            'Ensemble_UCB': round(ens_ucb, 4) if not np.isinf(ens_ucb) else 999.0,
            'RF_reward': round(rf_reward, 4),
            'ET_reward': round(et_reward, 4),
            'Ensemble_reward': round(ens_reward, 4),
            'selection_count_RF': int(bandit.counts[0]),
            'selection_count_ET': int(bandit.counts[1]),
            'selection_count_Ensemble': int(bandit.counts[2]),
            'drift_type': dtype,
            'severity': sev,
        })

        # ──────────────────────────────────────────────────────────────────────
        # STEP 4: UPDATE DIVERSITY-AWARE ENSEMBLE WEIGHTS & TRACK DIVERSITY
        # ──────────────────────────────────────────────────────────────────────
        _, current_weights, weight_spread, div_metrics = ensemble.update_weights(
            y_win, comp_preds, comp_probs
        )

        weight_rows.append({
            'seed': seed,
            'window_id': w,
            'RF_weight': round(current_weights['RF'], 4),
            'ET_weight': round(current_weights['ET'], 4),
            'GB_weight': round(current_weights['GB'], 4),
            'weight_spread': round(weight_spread, 4),
        })

        diversity_rows.append({
            'seed': seed,
            'window_id': w,
            'drift_type': dtype,
            'disagreement_rf_et': round(div_metrics['disagreement_rf_et'], 4),
            'disagreement_rf_gb': round(div_metrics['disagreement_rf_gb'], 4),
            'disagreement_et_gb': round(div_metrics['disagreement_et_gb'], 4),
            'corr_rf_et': round(div_metrics['corr_rf_et'], 4),
            'corr_rf_gb': round(div_metrics['corr_rf_gb'], 4),
            'corr_et_gb': round(div_metrics['corr_et_gb'], 4),
            'f1_rf': round(rf_m['f1'], 4),
            'f1_et': round(et_m['f1'], 4),
            'f1_gb': round(ensemble.history['GB']['f1'][-1], 4),
            'f1_ensemble': round(ens_m['f1'], 4),
            'ensemble_gain': round(ens_m['f1'] - max(rf_m['f1'], et_m['f1']), 4),
        })

        # ──────────────────────────────────────────────────────────────────────
        # STEP 5: ADAPT & RETRAIN (After Evaluation)
        # ──────────────────────────────────────────────────────────────────────

        # A. Frozen RF: Never retrain
        retrain_rows.append({
            'seed': seed,
            'window_id': w,
            'approach': 'Frozen RF',
            'retrained': False,
            'retraining_cpu_time': 0.0,
            'retraining_wall_time': 0.0,
        })

        # B. Continuously Retrained RF: Retrain on all cumulative data
        cumulative_X.extend(X_win)
        cumulative_y.extend(y_win)
        train_all_X = np.array(cumulative_X)
        train_all_y = np.array(cumulative_y)

        _, retr_rf_train_res = measure_execution(retrained_rf.fit, train_all_X, train_all_y)
        retrain_rows.append({
            'seed': seed,
            'window_id': w,
            'approach': 'Retrained RF',
            'retrained': True,
            'retraining_cpu_time': retr_rf_train_res['total_cpu_time'],
            'retraining_wall_time': retr_rf_train_res['wall_clock_time'],
        })

        # C. UCB1 Adaptive Ensemble: Drift-triggered adaptation
        # Lightweight unsupervised Wasserstein drift detector on feature space
        drift_metrics = compute_drift_metrics(X_init, X_win, KPI_COLS)
        w_dist = drift_metrics['wasserstein_mean']

        drift_detected = (w_dist > DRIFT_THRESHOLD)

        if drift_detected:
            retrained_flags, comp_retrain_res = ensemble.adapt(X_win, y_win)
            ens_retrain_cpu = sum(r['total_cpu_time'] for r in comp_retrain_res.values())
            ens_retrain_wall = sum(r['wall_clock_time'] for r in comp_retrain_res.values())
            ens_did_retrain = True
        else:
            ensemble.buffer_X.extend(X_win)
            ensemble.buffer_y.extend(y_win)
            ens_retrain_cpu = 0.0
            ens_retrain_wall = 0.0
            ens_did_retrain = False
            comp_retrain_res = {
                c: {'total_cpu_time': 0.0, 'wall_clock_time': 0.0, 'avg_cpu_percent': 0.0,
                    'avg_ram_mb': 0.0, 'peak_ram_mb': 0.0}
                for c in ['RF', 'ET', 'GB']
            }

        retrain_rows.append({
            'seed': seed,
            'window_id': w,
            'approach': 'UCB1 Adaptive Ensemble',
            'retrained': ens_did_retrain,
            'retraining_cpu_time': ens_retrain_cpu,
            'retraining_wall_time': ens_retrain_wall,
        })

        # ──────────────────────────────────────────────────────────────────────
        # STEP 6: RECORD WINDOW RESULTS & RESOURCES
        # ──────────────────────────────────────────────────────────────────────
        frozen_total_cpu = frozen_inf_res['total_cpu_time']
        frozen_total_wall = frozen_inf_res['wall_clock_time']

        retr_total_cpu = retr_inf_res['total_cpu_time'] + retr_rf_train_res['total_cpu_time']
        retr_total_wall = retr_inf_res['wall_clock_time'] + retr_rf_train_res['wall_clock_time']

        ens_total_cpu = app3_inf_res['total_cpu_time'] + ens_retrain_cpu
        ens_total_wall = app3_inf_res['wall_clock_time'] + ens_retrain_wall

        # Record Approach 1
        window_rows.append({
            'seed': seed, 'window_id': w, 'approach': 'Frozen RF',
            'f1': round(frozen_m['f1'], 4), 'accuracy': round(frozen_m['accuracy'], 4),
            'precision': round(frozen_m['precision'], 4), 'recall': round(frozen_m['recall'], 4),
            'auc': round(frozen_m['auc'], 4), 'pr_auc': round(frozen_m['pr_auc'], 4),
            'balanced_accuracy': round(frozen_m['balanced_accuracy'], 4),
            'cpu_time': round(frozen_total_cpu, 4), 'wall_time': round(frozen_total_wall, 4),
            'ram_mb': round(frozen_inf_res['avg_ram_mb'], 2),
            'drift_type': dtype, 'severity': sev, 'transition': trans,
        })

        # Record Approach 2
        window_rows.append({
            'seed': seed, 'window_id': w, 'approach': 'Retrained RF',
            'f1': round(retr_m['f1'], 4), 'accuracy': round(retr_m['accuracy'], 4),
            'precision': round(retr_m['precision'], 4), 'recall': round(retr_m['recall'], 4),
            'auc': round(retr_m['auc'], 4), 'pr_auc': round(retr_m['pr_auc'], 4),
            'balanced_accuracy': round(retr_m['balanced_accuracy'], 4),
            'cpu_time': round(retr_total_cpu, 4), 'wall_time': round(retr_total_wall, 4),
            'ram_mb': round(retr_rf_train_res['avg_ram_mb'], 2),
            'drift_type': dtype, 'severity': sev, 'transition': trans,
        })

        # Record Approach 3
        window_rows.append({
            'seed': seed, 'window_id': w, 'approach': 'UCB1 Adaptive Ensemble',
            'f1': round(app3_m['f1'], 4), 'accuracy': round(app3_m['accuracy'], 4),
            'precision': round(app3_m['precision'], 4), 'recall': round(app3_m['recall'], 4),
            'auc': round(app3_m['auc'], 4), 'pr_auc': round(app3_m['pr_auc'], 4),
            'balanced_accuracy': round(app3_m['balanced_accuracy'], 4),
            'cpu_time': round(ens_total_cpu, 4), 'wall_time': round(ens_total_wall, 4),
            'ram_mb': round(app3_inf_res['avg_ram_mb'], 2),
            'drift_type': dtype, 'severity': sev, 'transition': trans,
        })

        # Resource details per component
        resource_rows.append({
            'seed': seed, 'window': w, 'approach': 'Frozen RF', 'component': 'RF',
            'training_cpu_time': 0.0, 'prediction_cpu_time': frozen_inf_res['total_cpu_time'],
            'cpu_utilization': frozen_inf_res['avg_cpu_percent'],
            'ram_mb': frozen_inf_res['avg_ram_mb'], 'peak_ram_mb': frozen_inf_res['peak_ram_mb']
        })
        resource_rows.append({
            'seed': seed, 'window': w, 'approach': 'Retrained RF', 'component': 'RF',
            'training_cpu_time': retr_rf_train_res['total_cpu_time'],
            'prediction_cpu_time': retr_inf_res['total_cpu_time'],
            'cpu_utilization': retr_rf_train_res['avg_cpu_percent'],
            'ram_mb': retr_rf_train_res['avg_ram_mb'], 'peak_ram_mb': retr_rf_train_res['peak_ram_mb']
        })
        for c in ['RF', 'ET', 'GB']:
            train_c_cpu = comp_retrain_res[c]['total_cpu_time']
            resource_rows.append({
                'seed': seed, 'window': w, 'approach': 'UCB1 Adaptive Ensemble', 'component': c,
                'training_cpu_time': train_c_cpu,
                'prediction_cpu_time': app3_inf_res['total_cpu_time'] / 3.0,
                'cpu_utilization': comp_retrain_res[c]['avg_cpu_percent'],
                'ram_mb': app3_inf_res['avg_ram_mb'],
                'peak_ram_mb': app3_inf_res['peak_ram_mb']
            })

    mean_f1_frozen = np.mean([r['f1'] for r in window_rows if r['approach'] == 'Frozen RF'])
    mean_f1_retr = np.mean([r['f1'] for r in window_rows if r['approach'] == 'Retrained RF'])
    mean_f1_ens = np.mean([r['f1'] for r in window_rows if r['approach'] == 'UCB1 Adaptive Ensemble'])
    print(f"[Seed {seed}] Completed. F1 scores: Frozen={mean_f1_frozen:.4f}, Retrained={mean_f1_retr:.4f}, UCB1 Ensemble={mean_f1_ens:.4f}")

    return window_rows, bandit_rows, weight_rows, retrain_rows, resource_rows, diversity_rows


def run_experiment(seeds=SEEDS):
    """Main execution loop across all seeds, saving CSVs, tables, and figures."""
    start_total = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Export Drift Configuration
    drift_config_path = os.path.join(RESULTS_DIR, 'drift_config.csv')
    print(f"Exporting drift configuration to {drift_config_path}...")
    export_drift_configuration(seeds, drift_config_path)

    all_window_rows = []
    all_bandit_rows = []
    all_weight_rows = []
    all_retrain_rows = []
    all_resource_rows = []
    all_diversity_rows = []

    for seed in seeds:
        w_rows, b_rows, wt_rows, r_rows, res_rows, div_rows = run_single_seed(seed)
        all_window_rows.extend(w_rows)
        all_bandit_rows.extend(b_rows)
        all_weight_rows.extend(wt_rows)
        all_retrain_rows.extend(r_rows)
        all_resource_rows.extend(res_rows)
        all_diversity_rows.extend(div_rows)

    df_windows = pd.DataFrame(all_window_rows)
    df_bandit = pd.DataFrame(all_bandit_rows)
    df_weights = pd.DataFrame(all_weight_rows)
    df_retrain = pd.DataFrame(all_retrain_rows)
    df_resources = pd.DataFrame(all_resource_rows)
    df_diversity = pd.DataFrame(all_diversity_rows)

    # 2. Save CSVs
    print("\nSaving experimental result CSVs...")
    df_windows.to_csv(os.path.join(RESULTS_DIR, 'window_results.csv'), index=False)
    df_bandit.to_csv(os.path.join(RESULTS_DIR, 'bandit_results.csv'), index=False)
    df_weights.to_csv(os.path.join(RESULTS_DIR, 'ensemble_weights.csv'), index=False)
    df_retrain.to_csv(os.path.join(RESULTS_DIR, 'retraining.csv'), index=False)
    df_resources.to_csv(os.path.join(RESULTS_DIR, 'resource_details.csv'), index=False)
    df_diversity.to_csv(os.path.join(RESULTS_DIR, 'model_diversity.csv'), index=False)

    # 3. Model Diversity Diagnostics (Section 21)
    print("\n" + "="*80)
    print("SECTION 21 — MODEL DIVERSITY & COMPLEMENTARITY DIAGNOSTICS")
    print("="*80)
    print(f"Mean RF vs ET Disagreement: {df_diversity['disagreement_rf_et'].mean():.2%}")
    print(f"Mean RF vs GB Disagreement: {df_diversity['disagreement_rf_gb'].mean():.2%}")
    print(f"Mean ET vs GB Disagreement: {df_diversity['disagreement_et_gb'].mean():.2%}")
    print(f"Mean Individual RF F1:      {df_diversity['f1_rf'].mean():.4f}")
    print(f"Mean Individual ET F1:      {df_diversity['f1_et'].mean():.4f}")
    print(f"Mean Individual GB F1:      {df_diversity['f1_gb'].mean():.4f}")
    print(f"Mean Ensemble F1:           {df_diversity['f1_ensemble'].mean():.4f}")
    print(f"Mean Ensemble Gain:         {df_diversity['ensemble_gain'].mean():+.4f}")
    print("="*80)

    # 4. Statistical Hypothesis Testing
    print("\nRunning paired statistical tests (Wilcoxon + Holm-Bonferroni)...")
    df_stats = run_paired_tests(df_windows)
    df_stats.to_csv(os.path.join(RESULTS_DIR, 'statistical_results.csv'), index=False)

    # 5. Primary Result Table 1 (Approach Summary)
    summary_rows = []
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']
    for a in approaches:
        sub_w = df_windows[df_windows['approach'] == a]
        sub_r = df_retrain[df_retrain['approach'] == a]
        sub_res = df_resources[df_resources['approach'] == a]

        seed_f1 = sub_w.groupby('seed')['f1'].mean()
        seed_acc = sub_w.groupby('seed')['accuracy'].mean()
        seed_cpu = sub_w.groupby('seed')['cpu_time'].sum()
        seed_ram_peak = sub_res.groupby('seed')['peak_ram_mb'].max()
        seed_retr_events = sub_r.groupby('seed')['retrained'].sum()

        summary_rows.append({
            'Approach': a,
            'F1': f"{seed_f1.mean():.4f} ± {seed_f1.std():.4f}",
            'Accuracy': f"{seed_acc.mean():.4f} ± {seed_acc.std():.4f}",
            'CPU Time (s)': f"{seed_cpu.mean():.2f} ± {seed_cpu.std():.2f}",
            'Peak RAM (MB)': f"{seed_ram_peak.mean():.1f} ± {seed_ram_peak.std():.1f}",
            'Adapt/Retrain Events': f"{seed_retr_events.mean():.1f} ± {seed_retr_events.std():.1f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'summary_results.csv'), index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 3 — PRIMARY RESULT TABLE 1: OVERALL APPROACH COMPARISON")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)

    # 6. Primary Result Table 2 (Breakdown by Drift Type)
    drift_order = ['none', 'covariate', 'concept', 'mixed']
    drift_labels = {'none': 'No Drift', 'covariate': 'Covariate', 'concept': 'Concept', 'mixed': 'Mixed'}
    drift_rows = []
    for dt in drift_order:
        sub_dt = df_windows[df_windows['drift_type'] == dt]
        f1_f = sub_dt[sub_dt['approach'] == 'Frozen RF'].groupby('seed')['f1'].mean()
        f1_r = sub_dt[sub_dt['approach'] == 'Retrained RF'].groupby('seed')['f1'].mean()
        f1_e = sub_dt[sub_dt['approach'] == 'UCB1 Adaptive Ensemble'].groupby('seed')['f1'].mean()

        drift_rows.append({
            'Drift Type': drift_labels[dt],
            'Frozen RF F1': f"{f1_f.mean():.4f} ± {f1_f.std():.4f}",
            'Retrained RF F1': f"{f1_r.mean():.4f} ± {f1_r.std():.4f}",
            'Ensemble F1': f"{f1_e.mean():.4f} ± {f1_e.std():.4f}",
        })

    df_drift_summary = pd.DataFrame(drift_rows)
    df_drift_summary.to_csv(os.path.join(RESULTS_DIR, 'drift_type_summary.csv'), index=False)

    print("\n" + "="*80)
    print("EXPERIMENT 3 — PRIMARY RESULT TABLE 2: PERFORMANCE BY DRIFT TYPE")
    print("="*80)
    print(df_drift_summary.to_string(index=False))
    print("="*80)

    # 7. Generate Exactly the 9 Essential Figures
    print("\nGenerating the 9 essential figures in figures/...")
    generate_all_figures(FIGURES_DIR, df_windows, df_bandit, df_retrain)

    total_time = time.time() - start_total
    print(f"\nExperiment 3 successfully completed in {total_time:.1f} seconds.")
    return df_windows, df_bandit, df_summary, df_stats


if __name__ == '__main__':
    run_experiment()
