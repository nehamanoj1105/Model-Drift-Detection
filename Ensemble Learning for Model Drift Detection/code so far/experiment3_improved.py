"""
================================================================================
EXPERIMENT 3 (IMPROVED) — ENHANCED UCB1 ADAPTIVE ENSEMBLE
================================================================================
Compares three experimental approaches across 5 deterministic seeds (42-46):
  1. Frozen RF: Random Forest trained only on initial 2,000 samples. Never retrained.
  2. Continuously Retrained RF: Retrained at every window on all cumulative data.
  3. Enhanced UCB1 Adaptive Ensemble:
       - Discounted-UCB (D-UCB) with non-stationary discount factor gamma_d = 0.90, c = 0.25
       - Dual-objective balanced bandit reward: 0.5 * F1 + 0.5 * Accuracy - lambda * cost
       - Multi-temporal horizons: RF (3500), ET (2500), GB (1800)
       - Rank-weighted diverse soft voting: calibrated weights [0.18, 0.32, 0.50]
       - Dual drift adaptation: Wasserstein feature shift (W > 0.12) + prequential error shift (> 0.12)
       - Decision threshold tau = 0.455 calibrated for QoS telemetry imbalance

Saves all outputs separately:
  - CSV Results: experiment3/results_improved/
  - Figures:     experiment3/figures_improved/
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEEDS, N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE,
    KPI_COLS, TARGET_COL, DRIFT_THRESHOLD
)
from data_generation import generate_experiment_dataset
from models import create_frozen_rf, create_retrained_rf
from ensemble_improved import ImprovedAdaptiveEnsemble
from bandit_improved import DiscountedUCBBandit
from drift import compute_drift_metrics
from metrics import evaluate_predictions
from resource_monitor import measure_execution
from statistics import run_paired_tests
from plots_improved import generate_all_improved_figures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_IMPROVED_DIR = os.path.join(BASE_DIR, 'results_improved')
FIGURES_IMPROVED_DIR = os.path.join(BASE_DIR, 'figures_improved')
DECISION_THRESHOLD_IMPROVED = 0.455


def compute_extended_metrics(y_true, y_pred, y_prob):
    """Compute base metrics plus PR-AUC and balanced accuracy."""
    base = evaluate_predictions(y_true, y_pred, y_prob)
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = float(auc(rec_curve, prec_curve))
    except Exception:
        pr_auc_val = 0.0

    try:
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bacc = float(base['accuracy'])

    base['pr_auc'] = pr_auc_val
    base['balanced_accuracy'] = bacc
    return base


def run_improved_experiment():
    os.makedirs(RESULTS_IMPROVED_DIR, exist_ok=True)
    os.makedirs(FIGURES_IMPROVED_DIR, exist_ok=True)

    window_rows = []
    bandit_rows = []
    diversity_rows = []
    retrain_rows = []

    total_start = time.time()

    for seed in SEEDS:
        print(f"\n======================================================================")
        print(f"[Seed {seed}] Generating dataset with heterogeneous drift...")
        print(f"======================================================================")
        df, drift_schedule = generate_experiment_dataset(seed)
        X_all = df[KPI_COLS].values
        y_all = df[TARGET_COL].values

        X_init = X_all[:N_INITIAL_TRAINING]
        y_init = y_all[:N_INITIAL_TRAINING]

        init_violation = float(np.mean(y_init))
        print(f"[Seed {seed}] Initial training on {N_INITIAL_TRAINING} samples (violation rate: {init_violation:.1%})...")

        # Approach 1: Frozen RF
        frozen_rf = create_frozen_rf(seed)
        _, frozen_init_res = measure_execution(frozen_rf.fit, X_init, y_init)

        # Approach 2: Retrained RF
        retrained_rf = create_retrained_rf(seed)
        _, retr_init_res = measure_execution(retrained_rf.fit, X_init, y_init)

        # Approach 3: Enhanced Adaptive Ensemble & Discounted UCB Bandit
        ensemble = ImprovedAdaptiveEnsemble(seed, threshold=DECISION_THRESHOLD_IMPROVED)
        ens_init_res = ensemble.fit_initial(X_init, y_init)

        bandit = DiscountedUCBBandit(n_arms=3, gamma_d=0.90, c=0.25, lambda_cost=0.03)

        cumulative_X = list(X_init)
        cumulative_y = list(y_init)
        prev_app3_f1 = 0.85

        # Streaming loop over 16 windows
        for w in range(N_WINDOWS):
            idx_s = N_INITIAL_TRAINING + w * WINDOW_SIZE
            idx_e = idx_s + WINDOW_SIZE
            X_win = X_all[idx_s:idx_e]
            y_win = y_all[idx_s:idx_e]

            win_info = drift_schedule[w]
            dtype = win_info['drift_type']
            sev = win_info['severity']
            trans = win_info['transition']

            # STEP 1: PREDICT (Prequential Evaluation)
            # A. Frozen RF
            (frozen_pred, frozen_prob), frozen_inf_res = measure_execution(
                lambda: (frozen_rf.predict(X_win), frozen_rf.predict_proba(X_win)[:, 1])
            )

            # B. Retrained RF
            (retr_pred, retr_prob), retr_inf_res = measure_execution(
                lambda: (retrained_rf.predict(X_win), retrained_rf.predict_proba(X_win)[:, 1])
            )

            # C. Candidate Arms for Approach 3
            # Arm 0: RF
            (rf_pred, rf_prob), rf_inf_res = measure_execution(
                lambda: (
                    (ensemble.predict_component_proba('RF', X_win) >= DECISION_THRESHOLD_IMPROVED).astype(int),
                    ensemble.predict_component_proba('RF', X_win)
                )
            )
            # Arm 1: ET
            (et_pred, et_prob), et_inf_res = measure_execution(
                lambda: (
                    (ensemble.predict_component_proba('ET', X_win) >= DECISION_THRESHOLD_IMPROVED).astype(int),
                    ensemble.predict_component_proba('ET', X_win)
                )
            )
            # Arm 2: Ensemble (rank-weighted soft voting)
            (ens_pred, ens_prob, comp_preds, comp_probs), ens_inf_res = measure_execution(
                lambda: ensemble.predict(X_win, threshold=DECISION_THRESHOLD_IMPROVED)
            )

            selected_arm = bandit.select_arm()
            arm_names = ['RF', 'ET', 'Ensemble']
            selected_model = arm_names[selected_arm]

            candidate_preds = [rf_pred, et_pred, ens_pred]
            candidate_probs = [rf_prob, et_prob, ens_prob]
            candidate_inf_res = [rf_inf_res, et_inf_res, ens_inf_res]

            app3_pred = candidate_preds[selected_arm]
            app3_prob = candidate_probs[selected_arm]
            app3_inf_res = candidate_inf_res[selected_arm]

            # STEP 2: MEASURE & EVALUATE
            frozen_m = compute_extended_metrics(y_win, frozen_pred, frozen_prob)
            retr_m = compute_extended_metrics(y_win, retr_pred, retr_prob)
            app3_m = compute_extended_metrics(y_win, app3_pred, app3_prob)

            rf_m = compute_extended_metrics(y_win, rf_pred, rf_prob)
            et_m = compute_extended_metrics(y_win, et_pred, et_prob)
            ens_m = compute_extended_metrics(y_win, ens_pred, ens_prob)

            # STEP 3: UPDATE BANDIT (Dual-Objective: F1 + Accuracy)
            rf_reward, _ = bandit.compute_reward(rf_m['f1'], rf_m['accuracy'], rf_inf_res['total_cpu_time'])
            et_reward, _ = bandit.compute_reward(et_m['f1'], et_m['accuracy'], et_inf_res['total_cpu_time'])
            ens_reward, _ = bandit.compute_reward(ens_m['f1'], ens_m['accuracy'], ens_inf_res['total_cpu_time'])
            arm_rewards = [rf_reward, et_reward, ens_reward]

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
                'selection_count_RF': float(bandit.counts[0]),
                'selection_count_ET': float(bandit.counts[1]),
                'selection_count_Ensemble': float(bandit.counts[2]),
                'drift_type': dtype,
                'severity': sev,
            })

            # STEP 4: UPDATE WEIGHTS & TRACK DIVERSITY
            _, current_weights, weight_spread, div_metrics = ensemble.update_weights(
                y_win, comp_preds, comp_probs
            )

            diversity_rows.append({
                'seed': seed,
                'window_id': w,
                'drift_type': dtype,
                'disagreement_rf_et': round(div_metrics['disagreement_rf_et'], 4),
                'disagreement_rf_gb': round(div_metrics['disagreement_rf_gb'], 4),
                'disagreement_et_gb': round(div_metrics['disagreement_et_gb'], 4),
                'f1_rf': round(rf_m['f1'], 4),
                'f1_et': round(et_m['f1'], 4),
                'f1_gb': round(ensemble.history['GB']['f1'][-1], 4),
                'f1_ensemble': round(ens_m['f1'], 4),
            })

            # STEP 5: ADAPT & RETRAIN
            retrain_rows.append({
                'seed': seed,
                'window_id': w,
                'approach': 'Frozen RF',
                'retrained': False,
                'retraining_cpu_time': 0.0,
                'retraining_wall_time': 0.0,
            })

            # Continuously Retrained RF
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

            # Enhanced UCB1 Adaptive Ensemble: Dual Drift Detection
            drift_metrics = compute_drift_metrics(X_init, X_win, KPI_COLS)
            w_dist = drift_metrics['wasserstein_mean']
            cov_drift = (w_dist > DRIFT_THRESHOLD)
            perf_drop = (prev_app3_f1 - app3_m['f1'] > 0.12)

            if cov_drift or perf_drop:
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

            prev_app3_f1 = app3_m['f1']

            retrain_rows.append({
                'seed': seed,
                'window_id': w,
                'approach': 'UCB1 Ensemble',
                'retrained': ens_did_retrain,
                'retraining_cpu_time': ens_retrain_cpu,
                'retraining_wall_time': ens_retrain_wall,
            })

            # STEP 6: RECORD WINDOW RESULTS
            frozen_total_cpu = frozen_inf_res['total_cpu_time']
            frozen_total_wall = frozen_inf_res['wall_clock_time']

            retr_total_cpu = retr_inf_res['total_cpu_time'] + retr_rf_train_res['total_cpu_time']
            retr_total_wall = retr_inf_res['wall_clock_time'] + retr_rf_train_res['wall_clock_time']

            app3_total_cpu = app3_inf_res['total_cpu_time'] + ens_retrain_cpu
            app3_total_wall = app3_inf_res['wall_clock_time'] + ens_retrain_wall

            record_map = [
                ('Frozen RF', frozen_m, frozen_total_cpu, frozen_total_wall, frozen_inf_res['peak_ram_mb']),
                ('Retrained RF', retr_m, retr_total_cpu, retr_total_wall, retr_rf_train_res['peak_ram_mb']),
                ('UCB1 Ensemble', app3_m, app3_total_cpu, app3_total_wall, app3_inf_res['peak_ram_mb']),
            ]

            for app_name, m_dict, c_cpu, c_wall, ram in record_map:
                window_rows.append({
                    'seed': seed,
                    'window_id': w,
                    'approach': app_name,
                    'f1': round(m_dict['f1'], 4),
                    'accuracy': round(m_dict['accuracy'], 4),
                    'precision': round(m_dict['precision'], 4),
                    'recall': round(m_dict['recall'], 4),
                    'auc': round(m_dict['auc'], 4),
                    'pr_auc': round(m_dict['pr_auc'], 4),
                    'balanced_accuracy': round(m_dict['balanced_accuracy'], 4),
                    'cpu_time': round(c_cpu, 4),
                    'wall_time': round(c_wall, 4),
                    'ram_mb': round(ram, 2),
                    'drift_type': dtype,
                    'severity': sev,
                    'transition': trans,
                })

        # Seed Summary Printout
        seed_w_df = pd.DataFrame(window_rows)
        s_froz_f1 = seed_w_df[(seed_w_df['seed'] == seed) & (seed_w_df['approach'] == 'Frozen RF')]['f1'].mean()
        s_retr_f1 = seed_w_df[(seed_w_df['seed'] == seed) & (seed_w_df['approach'] == 'Retrained RF')]['f1'].mean()
        s_ens_f1 = seed_w_df[(seed_w_df['seed'] == seed) & (seed_w_df['approach'] == 'UCB1 Ensemble')]['f1'].mean()
        print(f"[Seed {seed}] Completed. F1 scores: Frozen={s_froz_f1:.4f}, Retrained={s_retr_f1:.4f}, UCB1 Ensemble={s_ens_f1:.4f}")

    # Build and export DataFrames
    window_df = pd.DataFrame(window_rows)
    bandit_df = pd.DataFrame(bandit_rows)
    diversity_df = pd.DataFrame(diversity_rows)
    retrain_df = pd.DataFrame(retrain_rows)

    window_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'window_results.csv'), index=False)
    bandit_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'bandit_results.csv'), index=False)
    diversity_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'model_diversity.csv'), index=False)
    retrain_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'retraining.csv'), index=False)

    # Statistical Significance Testing
    from scipy import stats
    comparisons = [
        ('UCB1 Ensemble', 'Frozen RF', 'f1'),
        ('UCB1 Ensemble', 'Frozen RF', 'accuracy'),
        ('UCB1 Ensemble', 'Retrained RF', 'f1'),
        ('UCB1 Ensemble', 'Retrained RF', 'accuracy'),
        ('UCB1 Ensemble', 'Retrained RF', 'cpu_time'),
        ('Retrained RF', 'Frozen RF', 'f1'),
        ('Retrained RF', 'Frozen RF', 'accuracy'),
        ('Retrained RF', 'Frozen RF', 'cpu_time'),
    ]
    results = []
    raw_p_values = []
    for strat_a, strat_b, metric in comparisons:
        sub_a = window_df[window_df['approach'] == strat_a].sort_values(['seed', 'window_id'])
        sub_b = window_df[window_df['approach'] == strat_b].sort_values(['seed', 'window_id'])
        merged = pd.merge(
            sub_a[['seed', 'window_id', metric]],
            sub_b[['seed', 'window_id', metric]],
            on=['seed', 'window_id'],
            suffixes=('_a', '_b'),
        )
        if len(merged) == 0:
            continue
        vals_a = merged[f'{metric}_a'].values
        vals_b = merged[f'{metric}_b'].values
        diffs = vals_a - vals_b
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        mean_diff = float(np.mean(diffs))

        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) > 5:
            try:
                w_stat, p_val = stats.wilcoxon(vals_a, vals_b, zero_method='wilcox')
                test_type = 'Wilcoxon signed-rank'
                test_stat = float(w_stat)
                n_nz = len(nonzero_diffs)
                total_rank = n_nz * (n_nz + 1) / 2
                effect_size = float(abs(1.0 - 2.0 * w_stat / total_rank)) if total_rank > 0 else 0.0
                effect_type = 'Rank-biserial r'
            except Exception:
                t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
                test_type = 'Paired t-test'
                test_stat = float(t_stat)
                std_diff = np.std(diffs, ddof=1)
                effect_size = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0
                effect_type = "Cohen's d"
        else:
            t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
            test_type = 'Paired t-test'
            test_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
            p_val = float(p_val) if not np.isnan(p_val) else 1.0
            std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 1.0
            effect_size = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0
            effect_type = "Cohen's d"

        raw_p_values.append(float(p_val))
        results.append({
            'comparison': f'{strat_a} vs {strat_b}',
            'metric': metric,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'test_type': test_type,
            'test_stat': test_stat,
            'p_value': float(p_val),
            'effect_size': float(effect_size),
            'effect_type': effect_type,
        })

    m = len(raw_p_values)
    if m > 0:
        sorted_idx = np.argsort(raw_p_values)
        adjusted = np.zeros(m)
        current_max = 0.0
        for rank, idx in enumerate(sorted_idx):
            adj_p = raw_p_values[idx] * (m - rank)
            adj_p = min(1.0, max(current_max, adj_p))
            current_max = adj_p
            adjusted[idx] = adj_p

        for idx, res in enumerate(results):
            res['p_value_adjusted'] = float(adjusted[idx])
            res['is_significant'] = bool(adjusted[idx] < 0.05)

    all_stats_df = pd.DataFrame(results)
    all_stats_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'statistical_results.csv'), index=False)

    # Summary Table 1
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']
    summary_rows = []
    for a in approaches:
        sub = window_df[window_df['approach'] == a]
        seed_f1s = sub.groupby('seed')['f1'].mean()
        seed_accs = sub.groupby('seed')['accuracy'].mean()
        seed_cpus = sub.groupby('seed')['cpu_time'].sum()
        seed_rams = sub.groupby('seed')['ram_mb'].max()
        r_sub = retrain_df[retrain_df['approach'] == a]
        seed_retrains = r_sub.groupby('seed')['retrained'].sum()

        summary_rows.append({
            'Approach': a,
            'F1': f"{seed_f1s.mean():.4f} ± {seed_f1s.std():.4f}",
            'Accuracy': f"{seed_accs.mean():.4f} ± {seed_accs.std():.4f}",
            'CPU Time (s)': f"{seed_cpus.mean():.2f} ± {seed_cpus.std():.2f}",
            'Peak RAM (MB)': f"{seed_rams.mean():.1f} ± {seed_rams.std():.1f}",
            'Adapt/Retrain Events': f"{seed_retrains.mean():.1f} ± {seed_retrains.std():.1f}"
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'summary_results.csv'), index=False)

    # Summary Table 2 (By Drift Type)
    drift_types = ['none', 'covariate', 'concept', 'mixed']
    drift_labels = {'none': 'No Drift', 'covariate': 'Covariate', 'concept': 'Concept', 'mixed': 'Mixed'}
    drift_rows = []
    for dt in drift_types:
        f_sub = window_df[(window_df['approach'] == 'Frozen RF') & (window_df['drift_type'] == dt)]
        r_sub = window_df[(window_df['approach'] == 'Retrained RF') & (window_df['drift_type'] == dt)]
        e_sub = window_df[(window_df['approach'] == 'UCB1 Ensemble') & (window_df['drift_type'] == dt)]

        f_f1 = f_sub.groupby('seed')['f1'].mean()
        r_f1 = r_sub.groupby('seed')['f1'].mean()
        e_f1 = e_sub.groupby('seed')['f1'].mean()

        drift_rows.append({
            'Drift Type': drift_labels[dt],
            'Frozen RF F1': f"{f_f1.mean():.4f} ± {f_f1.std():.4f}",
            'Retrained RF F1': f"{r_f1.mean():.4f} ± {r_f1.std():.4f}",
            'Ensemble F1': f"{e_f1.mean():.4f} ± {e_f1.std():.4f}"
        })
    drift_summary_df = pd.DataFrame(drift_rows)
    drift_summary_df.to_csv(os.path.join(RESULTS_IMPROVED_DIR, 'drift_type_summary.csv'), index=False)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 (IMPROVED) — PRIMARY RESULT TABLE 1: OVERALL APPROACH COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 (IMPROVED) — PRIMARY RESULT TABLE 2: PERFORMANCE BY DRIFT TYPE")
    print("=" * 80)
    print(drift_summary_df.to_string(index=False))
    print("=" * 80)

    # Generate Separate Figures
    print(f"\nGenerating 9 separate publication figures in {FIGURES_IMPROVED_DIR}...")
    generate_all_improved_figures(window_df, bandit_df, retrain_df, FIGURES_IMPROVED_DIR)

    elapsed = time.time() - total_start
    print(f"\nExperiment 3 (Improved) successfully completed in {elapsed:.1f} seconds.")


if __name__ == '__main__':
    run_improved_experiment()
