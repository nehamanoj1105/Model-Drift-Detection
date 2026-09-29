"""
================================================================================
EXPERIMENT 5 — ORACLE HISTORICAL MODEL RETRIEVAL VALIDATION BENCHMARK
================================================================================
Validation experiment to determine whether historical model-state retrieval
provides meaningful, held-out performance headroom over existing event-driven
and weighting approaches on ToN_IoT Weather streaming data.

Key Requirements:
  1. Reuse existing Experiment 4/5 Event-Driven drift detection architecture.
  2. Strict Prequential Control: Only checkpoints trained BEFORE target window W_t
     (C_{<t}) are eligible for selection. Zero future data leakage.
  3. Checkpoint registry tracking metadata: checkpoint_id, model_type,
     training_end_window, training_sample_range, timestamp.
  4. Single-model & Ensemble-state retrieval oracle evaluation.
  5. In-Sample vs. Held-Out Evaluation (250/250 window split) to audit selection overfitting.
  6. Precision Audit & Confusion Matrix breakdown (TP, TN, FP, FN).
  7. Headroom Quantification: H_weight vs H_retrieval_heldout.
  8. Window-paired statistical testing (Wilcoxon signed-rank test, 95% CI, effect size).
  9. Checkpoint Age vs Transfer Gain analysis.
 10. Automated Go/No-Go Decision Logic for RAPT repository.
================================================================================
"""

import os
import sys
import time
import json
import copy
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEEDS, N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_STREAM_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, RESULTS_DIR, FIGURES_DIR,
    DEFAULT_SIMILARITY_THRESHOLD, DECISION_THRESHOLD, SOFTMAX_TEMPERATURE
)
from data_loader import load_ton_iot_data, prepare_experiment_split
from preprocessing import Preprocessor
from ensemble_weighting import EnsembleWeightingRunner
from plots_exp5_retrieval import generate_all_retrieval_figures


class ModelCheckpointRegistry:
    """
    Strict prequential checkpoint registry storing historical model states.
    Ensures zero data leakage: for target window W_t, only checkpoints
    with training_end_window < t can be queried.
    """
    def __init__(self):
        self.checkpoints = []  # list of dicts

    def register_checkpoint(self, checkpoint_id, model_type, training_end_window,
                            training_sample_range, model_obj):
        """Deep-copies model object and registers metadata."""
        record = {
            'checkpoint_id': checkpoint_id,
            'model_type': model_type,
            'training_end_window': training_end_window,
            'training_sample_range': training_sample_range,
            'model_obj': copy.deepcopy(model_obj)
        }
        self.checkpoints.append(record)

    def get_eligible_checkpoints(self, target_window_id, model_type_filter=None):
        """Returns all checkpoints trained STRICTLY before target_window_id."""
        eligible = []
        for cp in self.checkpoints:
            if cp['training_end_window'] < target_window_id:
                if model_type_filter is None or cp['model_type'] == model_type_filter:
                    eligible.append(cp)
        return eligible


def predict_model_proba(model_obj, X):
    """Utility to get prediction probabilities P(Y=1|X) for a single model or ensemble tuple."""
    if isinstance(model_obj, dict):
        p_rf = predict_model_proba(model_obj['RF'], X)
        p_et = predict_model_proba(model_obj['ET'], X)
        p_gb = predict_model_proba(model_obj['GB'], X)
        prob = (p_rf + p_et + p_gb) / 3.0
    else:
        if hasattr(model_obj, 'predict_proba'):
            probs = model_obj.predict_proba(X)
            prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            prob = model_obj.predict(X).astype(float)
    return np.clip(prob, 1e-5, 1.0 - 1e-5)


def fast_compute_metrics(y_true, pred, prob=None):
    """High-performance vectorized calculation for binary classification metrics."""
    y_arr = np.asarray(y_true, dtype=int)
    pred_arr = np.asarray(pred, dtype=int)

    tp = int(np.sum((pred_arr == 1) & (y_arr == 1)))
    tn = int(np.sum((pred_arr == 0) & (y_arr == 0)))
    fp = int(np.sum((pred_arr == 1) & (y_arr == 0)))
    fn = int(np.sum((pred_arr == 0) & (y_arr == 1)))

    total = len(y_arr)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def find_fast_oracle_weights(comp_probs, y_true):
    """Vectorized grid search over weight simplex to find optimal weights."""
    grid = []
    for w1 in np.linspace(0.0, 1.0, 21):
        for w2 in np.linspace(0.0, 1.0 - w1, 21):
            w3 = 1.0 - w1 - w2
            grid.append((w1, w2, w3))
    W_mat = np.array(grid)  # (231, 3)
    P_mat = np.column_stack([comp_probs['RF'], comp_probs['ET'], comp_probs['GB']])  # (N, 3)

    ens_probs = W_mat @ P_mat.T  # (231, N)
    ens_preds = (ens_probs >= DECISION_THRESHOLD).astype(int)

    y_arr = np.asarray(y_true, dtype=int)
    tps = np.sum((ens_preds == 1) & (y_arr == 1), axis=1)
    fps = np.sum((ens_preds == 1) & (y_arr == 0), axis=1)
    fns = np.sum((ens_preds == 0) & (y_arr == 1), axis=1)

    precs = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
    recs = np.where(tps + fns > 0, tps / (tps + fns), 0.0)
    f1s = np.where(precs + recs > 0, 2.0 * precs * recs / (precs + recs), 0.0)

    best_idx = int(np.argmax(f1s))
    best_w1, best_w2, best_w3 = W_mat[best_idx]
    return {'RF': float(best_w1), 'ET': float(best_w2), 'GB': float(best_w3)}


def compute_paired_stats(vec_a, vec_b):
    """Computes window-paired statistical comparison."""
    diffs = np.array(vec_a) - np.array(vec_b)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    n = len(diffs)
    if n > 1 and std_diff > 1e-8:
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        try:
            stat, p_val = stats.wilcoxon(diffs)
        except Exception:
            p_val = 1.0
        effect_size = mean_diff / std_diff
    else:
        ci_lower = mean_diff
        ci_upper = mean_diff
        p_val = 1.0
        effect_size = 0.0

    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_val,
        'effect_size': effect_size,
        'n_obs': n
    }


def run_model_retrieval_validation(seeds=SEEDS, smoke_test=False, max_windows=None):
    """Executes the Oracle Historical Model Retrieval Validation Benchmark."""
    retrieval_results_dir = os.path.join(RESULTS_DIR, 'retrieval_validation')
    figures_retrieval_dir = os.path.join(RESULTS_DIR, 'figures_retrieval')
    os.makedirs(retrieval_results_dir, exist_ok=True)
    os.makedirs(figures_retrieval_dir, exist_ok=True)

    if smoke_test:
        print("!!! RUNNING IN SMOKE TEST MODE (1 seed, 5 windows) !!!")
        seeds = [42]
        max_windows = 5

    print("=" * 85)
    print("EXPERIMENT 5: ORACLE HISTORICAL MODEL RETRIEVAL VALIDATION")
    print(f"Dataset: ToN_IoT Weather | Total Sequential Samples: {N_TOTAL_SAMPLES:,}")
    print(f"Initial Training: {N_INITIAL_TRAINING:,} | Streaming Partition: {N_STREAM_SAMPLES:,}")
    print(f"Windows: {N_WINDOWS} (Window Size: {WINDOW_SIZE}) | Seeds: {seeds}")
    print("=" * 85)

    # 1. Load Data
    df_sorted = load_ton_iot_data()
    split_data = prepare_experiment_split(df_sorted)

    X_train_raw = split_data['X_train']
    y_train = split_data['y_train']
    X_stream_raw = split_data['X_stream']
    y_stream = split_data['y_stream']
    window_schedule = split_data['window_schedule']

    total_windows = min(len(window_schedule), max_windows) if max_windows else len(window_schedule)

    # 2. Preprocessing
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_stream = preprocessor.transform(X_stream_raw)

    all_window_records = []
    all_summary_records = []
    all_age_records = []

    cm_totals = {
        'Fixed Ensemble': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Global Adaptive Ensemble': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Regime-Aware Ensemble': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Current Event-Driven': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Oracle Weighting': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Oracle Model Retrieval (In-Sample)': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'Oracle Model Retrieval (Held-Out)': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    }

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN BENCHMARK LOOP
    # ──────────────────────────────────────────────────────────────────────────
    for seed in seeds:
        print(f"[Seed {seed}] Executing optimized retrieval benchmark...", flush=True)

        runner = EnsembleWeightingRunner(seed=seed, strategy_type='regime_aware', similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD)
        runner.fit_initial(X_train, y_train)

        registry = ModelCheckpointRegistry()
        for c_name in runner.component_names:
            registry.register_checkpoint(
                checkpoint_id=f"{c_name}_w-1",
                model_type=c_name,
                training_end_window=-1,
                training_sample_range="0-20000",
                model_obj=runner.models[c_name]
            )
        registry.register_checkpoint(
            checkpoint_id="Ensemble_w-1",
            model_type="Ensemble",
            training_end_window=-1,
            training_sample_range="0-20000",
            model_obj=runner.models
        )

        for w in range(total_windows):
            w_meta = window_schedule[w]
            idx_s = w * WINDOW_SIZE
            idx_e = idx_s + WINDOW_SIZE
            X_win = X_stream[idx_s:idx_e]
            y_win = y_stream[idx_s:idx_e]

            X_win_A, X_win_B = X_win[:250], X_win[250:]
            y_win_A, y_win_B = y_win[:250], y_win[250:]

            # Compute component probabilities once per window
            comp_probs = {}
            for name in runner.component_names:
                comp_probs[name] = runner.predict_component_proba(name, X_win)
            comp_preds = {name: (comp_probs[name] >= DECISION_THRESHOLD).astype(int) for name in runner.component_names}

            # 1. Fixed Ensemble
            prob_fixed = (comp_probs['RF'] + comp_probs['ET'] + comp_probs['GB']) / 3.0
            pred_fixed = (prob_fixed >= DECISION_THRESHOLD).astype(int)
            m_fixed = fast_compute_metrics(y_win, pred_fixed, prob_fixed)

            # 2. Global Adaptive Ensemble
            scores = np.array([runner.global_ema[c] for c in runner.component_names])
            exp_s = np.exp((scores - np.max(scores)) / SOFTMAX_TEMPERATURE)
            p_glob = exp_s / np.sum(exp_s)
            prob_glob = p_glob[0]*comp_probs['RF'] + p_glob[1]*comp_probs['ET'] + p_glob[2]*comp_probs['GB']
            pred_glob = (prob_glob >= DECISION_THRESHOLD).astype(int)
            m_glob = fast_compute_metrics(y_win, pred_glob, prob_glob)

            # 3. Regime-Aware Ensemble
            pred_regime, prob_regime, _, _, regime_w = runner.predict(X_win)
            m_regime = fast_compute_metrics(y_win, pred_regime, prob_regime)

            # 4. Current Event-Driven Baseline
            m_curr = m_fixed

            # 5. Oracle Weighting
            oracle_w = find_fast_oracle_weights(comp_probs, y_win)
            prob_oracle = oracle_w['RF']*comp_probs['RF'] + oracle_w['ET']*comp_probs['ET'] + oracle_w['GB']*comp_probs['GB']
            pred_oracle = (prob_oracle >= DECISION_THRESHOLD).astype(int)
            m_oracle = fast_compute_metrics(y_win, pred_oracle, prob_oracle)

            methods_metrics = {
                'Fixed Ensemble': m_fixed,
                'Global Adaptive Ensemble': m_glob,
                'Regime-Aware Ensemble': m_regime,
                'Current Event-Driven': m_curr,
                'Oracle Weighting': m_oracle
            }

            for m_name, m in methods_metrics.items():
                cm_totals[m_name]['tp'] += m['tp']
                cm_totals[m_name]['tn'] += m['tn']
                cm_totals[m_name]['fp'] += m['fp']
                cm_totals[m_name]['fn'] += m['fn']

                all_window_records.append({
                    'seed': seed,
                    'method': m_name,
                    'window_id': w,
                    'dominant_regime': w_meta['regime'],
                    'f1': m['f1'],
                    'accuracy': m['accuracy'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'tp': m['tp'], 'tn': m['tn'], 'fp': m['fp'], 'fn': m['fn'],
                    'is_heldout_eval': False
                })

            # Held-out current baseline (second 250 samples)
            prob_curr_B = (comp_probs['RF'][250:] + comp_probs['ET'][250:] + comp_probs['GB'][250:]) / 3.0
            pred_curr_B = (prob_curr_B >= DECISION_THRESHOLD).astype(int)
            m_curr_B = fast_compute_metrics(y_win_B, pred_curr_B, prob_curr_B)
            all_window_records.append({
                'seed': seed,
                'method': 'Current Event-Driven (Held-Out Baseline)',
                'window_id': w,
                'dominant_regime': w_meta['regime'],
                'f1': m_curr_B['f1'],
                'accuracy': m_curr_B['accuracy'],
                'precision': m_curr_B['precision'],
                'recall': m_curr_B['recall'],
                'tp': m_curr_B['tp'], 'tn': m_curr_B['tn'], 'fp': m_curr_B['fp'], 'fn': m_curr_B['fn'],
                'is_heldout_eval': True
            })

            # --- HISTORICAL MODEL RETRIEVAL EVALUATION ---
            single_cps = registry.get_eligible_checkpoints(w)

            best_insample_cp = None
            best_insample_f1 = -1.0
            best_insample_m = None

            best_sel_cp = None
            best_sel_f1_A = -1.0
            best_sel_m_B = None

            for cp in single_cps:
                prob_cp = predict_model_proba(cp['model_obj'], X_win)
                pred_cp = (prob_cp >= DECISION_THRESHOLD).astype(int)

                # In-Sample Full Window evaluation
                m_cp_full = fast_compute_metrics(y_win, pred_cp, prob_cp)
                if m_cp_full['f1'] > best_insample_f1:
                    best_insample_f1 = m_cp_full['f1']
                    best_insample_cp = cp
                    best_insample_m = m_cp_full

                # Held-Out Split evaluation (Selection on A, Eval on B)
                pred_A = pred_cp[:250]
                m_cp_A = fast_compute_metrics(y_win_A, pred_A)
                if m_cp_A['f1'] > best_sel_f1_A:
                    best_sel_f1_A = m_cp_A['f1']
                    best_sel_cp = cp
                    pred_B = pred_cp[250:]
                    best_sel_m_B = fast_compute_metrics(y_win_B, pred_B)

            # Record In-Sample Retrieval Results
            if best_insample_cp is not None:
                cm_totals['Oracle Model Retrieval (In-Sample)']['tp'] += best_insample_m['tp']
                cm_totals['Oracle Model Retrieval (In-Sample)']['tn'] += best_insample_m['tn']
                cm_totals['Oracle Model Retrieval (In-Sample)']['fp'] += best_insample_m['fp']
                cm_totals['Oracle Model Retrieval (In-Sample)']['fn'] += best_insample_m['fn']

                cp_age_in = w - best_insample_cp['training_end_window']
                gain_in = best_insample_f1 - m_curr['f1']

                all_window_records.append({
                    'seed': seed,
                    'method': 'Oracle Model Retrieval (In-Sample)',
                    'window_id': w,
                    'dominant_regime': w_meta['regime'],
                    'f1': best_insample_f1,
                    'accuracy': best_insample_m['accuracy'],
                    'precision': best_insample_m['precision'],
                    'recall': best_insample_m['recall'],
                    'tp': best_insample_m['tp'], 'tn': best_insample_m['tn'],
                    'fp': best_insample_m['fp'], 'fn': best_insample_m['fn'],
                    'best_checkpoint_id': best_insample_cp['checkpoint_id'],
                    'checkpoint_age': cp_age_in,
                    'f1_gain': gain_in,
                    'is_heldout_eval': False
                })

                all_age_records.append({
                    'seed': seed,
                    'window_id': w,
                    'eval_type': 'In-Sample',
                    'best_checkpoint_id': best_insample_cp['checkpoint_id'],
                    'model_type': best_insample_cp['model_type'],
                    'checkpoint_window': best_insample_cp['training_end_window'],
                    'target_window': w,
                    'checkpoint_age': cp_age_in,
                    'f1_insample': best_insample_f1,
                    'baseline_f1': m_curr['f1'],
                    'f1_gain': gain_in
                })

            # Record Held-Out Retrieval Results
            if best_sel_cp is not None:
                cm_totals['Oracle Model Retrieval (Held-Out)']['tp'] += best_sel_m_B['tp']
                cm_totals['Oracle Model Retrieval (Held-Out)']['tn'] += best_sel_m_B['tn']
                cm_totals['Oracle Model Retrieval (Held-Out)']['fp'] += best_sel_m_B['fp']
                cm_totals['Oracle Model Retrieval (Held-Out)']['fn'] += best_sel_m_B['fn']

                cp_age_ho = w - best_sel_cp['training_end_window']
                gain_ho = best_sel_m_B['f1'] - m_curr_B['f1']

                all_window_records.append({
                    'seed': seed,
                    'method': 'Oracle Model Retrieval (Held-Out)',
                    'window_id': w,
                    'dominant_regime': w_meta['regime'],
                    'f1': best_sel_m_B['f1'],
                    'accuracy': best_sel_m_B['accuracy'],
                    'precision': best_sel_m_B['precision'],
                    'recall': best_sel_m_B['recall'],
                    'tp': best_sel_m_B['tp'], 'tn': best_sel_m_B['tn'],
                    'fp': best_sel_m_B['fp'], 'fn': best_sel_m_B['fn'],
                    'best_checkpoint_id': best_sel_cp['checkpoint_id'],
                    'checkpoint_age': cp_age_ho,
                    'f1_gain': gain_ho,
                    'is_heldout_eval': True
                })

                all_age_records.append({
                    'seed': seed,
                    'window_id': w,
                    'eval_type': 'Held-Out',
                    'best_checkpoint_id': best_sel_cp['checkpoint_id'],
                    'model_type': best_sel_cp['model_type'],
                    'checkpoint_window': best_sel_cp['training_end_window'],
                    'target_window': w,
                    'checkpoint_age': cp_age_ho,
                    'f1_heldout': best_sel_m_B['f1'],
                    'baseline_f1': m_curr_B['f1'],
                    'f1_gain': gain_ho
                })

            # Post-prediction adaptation & drift check
            update_res = runner.update_and_adapt(X_win, y_win, comp_preds, m_curr['f1'], X_train)

            if update_res['drift_detected']:
                for c_name in runner.component_names:
                    registry.register_checkpoint(
                        checkpoint_id=f"{c_name}_w{w}",
                        model_type=c_name,
                        training_end_window=w,
                        training_sample_range=f"0-{20000 + (w + 1) * 500}",
                        model_obj=runner.models[c_name]
                    )
                registry.register_checkpoint(
                    checkpoint_id=f"Ensemble_w{w}",
                    model_type="Ensemble",
                    training_end_window=w,
                    training_sample_range=f"0-{20000 + (w + 1) * 500}",
                    model_obj=runner.models
                )

    # Convert to DataFrames
    df_window = pd.DataFrame(all_window_records)
    df_age = pd.DataFrame(all_age_records)

    methods_list = df_window['method'].unique()
    for seed in seeds:
        sub_s = df_window[df_window['seed'] == seed]
        for m in methods_list:
            sub_m = sub_s[sub_s['method'] == m]
            if len(sub_m) > 0:
                all_summary_records.append({
                    'seed': seed,
                    'method': m,
                    'f1': sub_m['f1'].mean(),
                    'accuracy': sub_m['accuracy'].mean(),
                    'precision': sub_m['precision'].mean(),
                    'recall': sub_m['recall'].mean(),
                    'tp': sub_m['tp'].sum(),
                    'tn': sub_m['tn'].sum(),
                    'fp': sub_m['fp'].sum(),
                    'fn': sub_m['fn'].sum()
                })

    df_summary = pd.DataFrame(all_summary_records)

    # Save CSVs
    df_window.to_csv(os.path.join(retrieval_results_dir, 'retrieval_window_metrics.csv'), index=False)
    df_summary.to_csv(os.path.join(retrieval_results_dir, 'retrieval_summary_metrics.csv'), index=False)
    df_age.to_csv(os.path.join(retrieval_results_dir, 'retrieval_checkpoint_age.csv'), index=False)

    # Headroom metrics
    mean_curr = df_summary[df_summary['method'] == 'Current Event-Driven']['f1'].mean()
    mean_curr_ho = df_window[df_window['method'] == 'Current Event-Driven (Held-Out Baseline)']['f1'].mean()

    mean_weight = df_summary[df_summary['method'] == 'Oracle Weighting']['f1'].mean()
    mean_insample = df_summary[df_summary['method'] == 'Oracle Model Retrieval (In-Sample)']['f1'].mean()
    mean_heldout = df_summary[df_summary['method'] == 'Oracle Model Retrieval (Held-Out)']['f1'].mean()

    h_weight = mean_weight - mean_curr
    h_insample = mean_insample - mean_curr
    h_heldout = mean_heldout - mean_curr_ho

    headroom_dict = {
        'h_weight': float(h_weight),
        'h_insample': float(h_insample),
        'h_heldout': float(h_heldout),
        'mean_curr': float(mean_curr),
        'mean_weight': float(mean_weight),
        'mean_insample': float(mean_insample),
        'mean_heldout': float(mean_heldout)
    }

    with open(os.path.join(retrieval_results_dir, 'headroom_summary.json'), 'w') as f:
        json.dump(headroom_dict, f, indent=4)

    # Statistical Tests
    curr_window_f1 = df_window[df_window['method'] == 'Current Event-Driven']['f1'].values
    curr_ho_window_f1 = df_window[df_window['method'] == 'Current Event-Driven (Held-Out Baseline)']['f1'].values

    weight_window_f1 = df_window[df_window['method'] == 'Oracle Weighting']['f1'].values
    insample_window_f1 = df_window[df_window['method'] == 'Oracle Model Retrieval (In-Sample)']['f1'].values
    heldout_window_f1 = df_window[df_window['method'] == 'Oracle Model Retrieval (Held-Out)']['f1'].values

    fixed_window_f1 = df_window[df_window['method'] == 'Fixed Ensemble']['f1'].values
    adaptive_window_f1 = df_window[df_window['method'] == 'Global Adaptive Ensemble']['f1'].values
    regime_window_f1 = df_window[df_window['method'] == 'Regime-Aware Ensemble']['f1'].values

    stat_tests = {
        'Fixed_vs_GlobalAdaptive': compute_paired_stats(adaptive_window_f1, fixed_window_f1),
        'Fixed_vs_RegimeAware': compute_paired_stats(regime_window_f1, fixed_window_f1),
        'Current_vs_OracleWeighting': compute_paired_stats(weight_window_f1, curr_window_f1),
        'Current_vs_RetrievalInSample': compute_paired_stats(insample_window_f1, curr_window_f1),
        'Current_vs_RetrievalHeldOut': compute_paired_stats(heldout_window_f1, curr_ho_window_f1),
    }

    with open(os.path.join(retrieval_results_dir, 'statistical_tests.json'), 'w') as f:
        json.dump(stat_tests, f, indent=4)

    # Automated Decision Logic
    p_heldout = stat_tests['Current_vs_RetrievalHeldOut']['p_value']

    if h_heldout >= 0.0030 and p_heldout < 0.05:
        decision_case = "Case A"
        recommendation = "Proceed to full RAPT model repository. Held-out historical model retrieval provides meaningful, statistically significant performance headroom."
    elif h_insample >= 0.0050 and h_heldout < 0.0020:
        decision_case = "Case C"
        recommendation = f"Do NOT build full RAPT on this dataset. In-sample oracle retrieval shows apparent gain (+{h_insample:.4f} F1) but collapses on held-out evaluation (+{h_heldout:.4f} F1), indicating target-window selection overfitting."
    else:
        decision_case = "Case B"
        recommendation = f"Do NOT build full RAPT model repository on ToN_IoT Weather dataset. Neither weight optimization (+{h_weight:.4f} F1) nor historical model retrieval (+{h_heldout:.4f} F1 held-out) provides meaningful transfer headroom over the event-driven baseline. Recommend evaluating RAPT on a dataset/task with stronger, abrupt regime recurrence."

    decision_summary = {
        'decision_case': decision_case,
        'recommendation': recommendation,
        'h_weight': h_weight,
        'h_insample': h_insample,
        'h_heldout': h_heldout,
        'p_value_heldout': p_heldout
    }

    with open(os.path.join(retrieval_results_dir, 'decision_logic.json'), 'w') as f:
        json.dump(decision_summary, f, indent=4)

    print("\n" + "=" * 85, flush=True)
    print("RETRIEVAL VALIDATION COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 85, flush=True)
    print(f"Current Baseline F1:       {mean_curr:.4f}", flush=True)
    print(f"Weight-Only Oracle F1:     {mean_weight:.4f}  (Headroom H_weight: +{h_weight:.4f})", flush=True)
    print(f"In-Sample Retrieval F1:   {mean_insample:.4f}  (Headroom H_retrieval_in: +{h_insample:.4f})", flush=True)
    print(f"Held-Out Retrieval F1:    {mean_heldout:.4f}  (Headroom H_retrieval_ho: +{h_heldout:.4f})", flush=True)
    print(f"Decision Case:             {decision_case}", flush=True)
    print(f"Recommendation:            {recommendation}", flush=True)
    print("=" * 85, flush=True)

    # Generate Figures
    print("\n[Figures] Generating publication-quality validation figures...", flush=True)
    cm_plot_dict = cm_totals['Oracle Model Retrieval (Held-Out)']
    generate_all_retrieval_figures(df_summary, df_window, df_age, cm_plot_dict, headroom_dict, figures_retrieval_dir)
    print(f"[Figures] Saved all 6 figures to: {figures_retrieval_dir}", flush=True)

    return {
        'df_summary': df_summary,
        'df_window': df_window,
        'df_age': df_age,
        'headroom_dict': headroom_dict,
        'stat_tests': stat_tests,
        'decision_summary': decision_summary
    }


if __name__ == '__main__':
    run_model_retrieval_validation()
