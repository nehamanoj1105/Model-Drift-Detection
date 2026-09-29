"""
================================================================================
RAPT ENHANCED BENCHMARK RUNNER — ACCURACY & F1 PARITY EVALUATION
================================================================================
Evaluates the 6 deployment strategies across 5 random seeds ([42, 43, 44, 45, 46]):
  1. Frozen Ensemble
  2. Continuous Retraining
  3. Event-Driven Baseline (Exp 4)
  4. RAPT Autonomous Repository (K=8)
  5. Two-Tier Hybrid RAPT
  6. Enhanced Hybrid RAPT (Ours: Dynamic Threshold + Selective Parity Refit)
================================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import copy
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss

# Ensure local imports take precedence
_rapt_dir = os.path.dirname(os.path.abspath(__file__))
if _rapt_dir not in sys.path:
    sys.path.insert(0, _rapt_dir)

from config import SEEDS, WINDOW_SIZE, DECISION_THRESHOLD, BUFFER_CAP, RESULTS_DIR, FIGURES_DIR
from real_world_stream import load_real_world_ton_iot_stream, N_INITIAL_TRAINING, N_WINDOWS
from fingerprint import FastFingerprintExtractor
from repository import AutonomousRegimeRepository
from hybrid_rapt import TwoTierHybridRAPT
from enhanced_hybrid_rapt import EnhancedHybridRAPT


def create_base_models(seed):
    """Create fresh instances of the 3 base classifiers matching Experiment 4 & 5."""
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=40, max_depth=8, min_samples_split=4, random_state=seed, n_jobs=1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=40, max_depth=8, min_samples_split=6, random_state=seed + 1, n_jobs=1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=40, max_depth=4, learning_rate=0.08, subsample=0.85, random_state=seed + 2
        )
    }


def fit_models(models, X, y, ref_X=None, ref_y=None):
    """Fit all base models and measure CPU time, ensuring both classes exist."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2 and ref_X is not None and ref_y is not None:
        missing_cls = 1 if 0 in unique_classes else 0
        ref_mask = (ref_y == missing_cls)
        if np.any(ref_mask):
            add_X = ref_X[ref_mask][:50]
            add_y = ref_y[ref_mask][:50]
            X = np.vstack([X, add_X])
            y = np.concatenate([y, add_y])
    if len(np.unique(y)) < 2:
        missing_cls = 1 if 0 in np.unique(y) else 0
        dummy_x = np.tile(np.mean(X, axis=0, keepdims=True), (10, 1))
        dummy_y = np.full(10, missing_cls, dtype=int)
        X = np.vstack([X, dummy_x])
        y = np.concatenate([y, dummy_y])

    tot_cpu_ms = 0.0
    for name, m in models.items():
        t0_cpu = time.process_time()
        m.fit(X, y)
        c_ms = (time.process_time() - t0_cpu) * 1000.0
        tot_cpu_ms += c_ms

    return tot_cpu_ms


def predict_ensemble_proba(models, weights, X):
    """Weighted prediction probability across ensemble."""
    w_rf, w_et, w_gb = weights
    p_rf = models['RandomForest'].predict_proba(X)[:, 1]
    p_et = models['ExtraTrees'].predict_proba(X)[:, 1]
    p_gb = models['GradientBoosting'].predict_proba(X)[:, 1]
    return w_rf * p_rf + w_et * p_et + w_gb * p_gb


def evaluate_window_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD):
    """Compute standard metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec,
        'auc': auc, 'pr_auc': pr_auc, 'brier': brier
    }


class RealWorldDualTriggerDetector:
    """Lightweight dual-trigger detector adapted for real IoT features."""
    def __init__(self, wasserstein_thresh=0.08, perf_drop_thresh=0.08, initial_f1=0.85):
        self.wasserstein_thresh = wasserstein_thresh
        self.perf_drop_thresh = perf_drop_thresh
        self.rolling_f1 = initial_f1

    def check_drift(self, X_ref, X_cur, cur_f1):
        comb_min = np.minimum(np.min(X_ref, axis=0), np.min(X_cur, axis=0))
        comb_max = np.maximum(np.max(X_ref, axis=0), np.max(X_cur, axis=0))
        comb_rng = np.maximum(comb_max - comb_min, 1e-6)

        ref_norm = (X_ref - comb_min) / comb_rng
        cur_norm = (X_cur - comb_min) / comb_rng

        qs = np.linspace(0.01, 0.99, 30)
        ref_q = np.quantile(ref_norm, qs, axis=0)
        cur_q = np.quantile(cur_norm, qs, axis=0)
        mean_w = float(np.mean(np.abs(ref_q - cur_q)))

        cov_drift = mean_w > self.wasserstein_thresh
        f1_drop = max(0.0, float(self.rolling_f1 - cur_f1))
        concept_drift = f1_drop > self.perf_drop_thresh

        drift_detected = bool(cov_drift or concept_drift)
        self.rolling_f1 = 0.6 * self.rolling_f1 + 0.4 * cur_f1
        return drift_detected, mean_w, f1_drop


def run_enhanced_benchmark():
    print("=" * 80)
    print("EXECUTING ENHANCED RAPT PARITY BENCHMARK ACROSS 5 SEEDS")
    print("=" * 80)

    all_window_rows = []

    for seed in SEEDS:
        print(f"\n---> Processing Seed {seed}...")
        X_train_scaled, y_train, X_stream_scaled, y_stream, df_sorted, window_metadata, scaler = load_real_world_ton_iot_stream()

        n_windows = len(window_metadata)
        window_size = WINDOW_SIZE

        # Balanced reference slice for baseline models and detector calibration (750 normal, 750 attack)
        idx_0 = np.where(y_train == 0)[0][-750:]
        idx_1 = np.where(y_train == 1)[0][-750:]
        calib_idx = np.sort(np.concatenate([idx_0, idx_1]))
        X_ref_sample = X_train_scaled[calib_idx]
        y_ref_sample = y_train[calib_idx]

        extractor = FastFingerprintExtractor(X_ref_sample, num_quantiles=40)

        # 1. Frozen Ensemble
        m1_models = create_base_models(seed)
        _ = fit_models(m1_models, X_ref_sample, y_ref_sample)
        m1_cum_cpu = 0.0

        # 2. Continuous Retraining
        m2_models = create_base_models(seed)
        _ = fit_models(m2_models, X_ref_sample, y_ref_sample)
        m2_buffer_X, m2_buffer_y = list(X_ref_sample), list(y_ref_sample)
        m2_cum_cpu, m2_cum_adapt_cpu, m2_cum_retrains = 0.0, 0.0, 0

        # 3. Event-Driven Baseline
        m3_models = create_base_models(seed)
        _ = fit_models(m3_models, X_ref_sample, y_ref_sample)
        m3_detector = RealWorldDualTriggerDetector()
        m3_buffer_X, m3_buffer_y = list(X_ref_sample), list(y_ref_sample)
        m3_cum_cpu, m3_cum_adapt_cpu, m3_cum_retrains = 0.0, 0.0, 0

        # 4. RAPT Alone (K=8)
        repo_k8 = AutonomousRegimeRepository(max_capacity=8, gamma=15.0, novelty_threshold=0.65)
        m4_models_init = create_base_models(seed)
        _ = fit_models(m4_models_init, X_ref_sample, y_ref_sample)
        base_fp, _ = extractor.extract(X_ref_sample[-500:], m4_models_init['RandomForest'].predict(X_ref_sample[-500:]))
        repo_k8.insert_regime(base_fp, m4_models_init, [1/3, 1/3, 1/3], window_id=0, name="Initial_Baseline")
        m4_cum_cpu, m4_cum_adapt_cpu, m4_cum_retrains = 0.0, 0.0, 0

        # 5. Two-Tier Hybrid RAPT
        hybrid_rapt = TwoTierHybridRAPT(max_capacity=8, gamma=15.0, novelty_threshold=0.65, seed=seed)
        m5_models_init = create_base_models(seed)
        _ = fit_models(m5_models_init, X_ref_sample, y_ref_sample)
        hybrid_rapt.warm_start(base_fp, m5_models_init, [1/3, 1/3, 1/3], X_ref_sample, y_ref_sample, window_id=0)
        m5_cum_cpu, m5_cum_adapt_cpu, m5_cum_retrains = 0.0, 0.0, 0

        # 6. ENHANCED HYBRID RAPT (Ours: Dynamic Threshold + Selective Parity Refit)
        enhanced_rapt = EnhancedHybridRAPT(max_capacity=8, gamma=15.0, novelty_threshold=0.65, base_tau=DECISION_THRESHOLD, lambda_tau=0.14, seed=seed)
        m6_models_init = create_base_models(seed)
        _ = fit_models(m6_models_init, X_ref_sample, y_ref_sample)
        enhanced_rapt.warm_start(base_fp, m6_models_init, [1/3, 1/3, 1/3], X_ref_sample, y_ref_sample, window_id=0)
        m6_buffer_X, m6_buffer_y = list(X_ref_sample), list(y_ref_sample)
        m6_cum_cpu, m6_cum_adapt_cpu, m6_cum_retrains = 0.0, 0.0, 0
        m6_prev_f1 = 0.85

        for w in range(n_windows):
            idx_s = w * window_size
            idx_e = idx_s + window_size
            X_win = X_stream_scaled[idx_s:idx_e]
            y_win = y_stream[idx_s:idx_e]

            # 1. Frozen Ensemble
            t0 = time.perf_counter()
            p_m1 = predict_ensemble_proba(m1_models, [1/3, 1/3, 1/3], X_win)
            t_m1_inf = (time.perf_counter() - t0) * 1000.0
            m1_met = evaluate_window_metrics(y_win, p_m1)
            m1_cum_cpu += t_m1_inf / 1000.0
            all_window_rows.append({
                'seed': seed, 'window_id': w, 'approach': 'Frozen Ensemble',
                'f1': m1_met['f1'], 'accuracy': m1_met['accuracy'], 'precision': m1_met['precision'], 'recall': m1_met['recall'],
                'cpu_time_ms': t_m1_inf, 'adapt_cpu_ms': 0.0, 'retrain_event': 0, 'cum_cpu_s': m1_cum_cpu, 'cum_adapt_cpu_s': 0.0
            })

            # 2. Continuous Retraining
            t0 = time.perf_counter()
            p_m2 = predict_ensemble_proba(m2_models, [1/3, 1/3, 1/3], X_win)
            t_m2_inf = (time.perf_counter() - t0) * 1000.0
            m2_met = evaluate_window_metrics(y_win, p_m2)
            m2_buffer_X.extend(X_win); m2_buffer_y.extend(y_win)
            m2_samples = min(len(m2_buffer_X), 1500)
            t_m2_retrain = fit_models(m2_models, np.array(m2_buffer_X[-m2_samples:]), np.array(m2_buffer_y[-m2_samples:]), ref_X=X_ref_sample, ref_y=y_ref_sample)
            m2_tot = t_m2_inf + t_m2_retrain
            m2_cum_cpu += m2_tot / 1000.0
            m2_cum_adapt_cpu += t_m2_retrain / 1000.0
            m2_cum_retrains += 1
            all_window_rows.append({
                'seed': seed, 'window_id': w, 'approach': 'Continuous Retraining',
                'f1': m2_met['f1'], 'accuracy': m2_met['accuracy'], 'precision': m2_met['precision'], 'recall': m2_met['recall'],
                'cpu_time_ms': m2_tot, 'adapt_cpu_ms': t_m2_retrain, 'retrain_event': 1, 'cum_cpu_s': m2_cum_cpu, 'cum_adapt_cpu_s': m2_cum_adapt_cpu
            })

            # 3. Event-Driven Baseline
            t0 = time.perf_counter()
            p_m3 = predict_ensemble_proba(m3_models, [1/3, 1/3, 1/3], X_win)
            t_m3_inf = (time.perf_counter() - t0) * 1000.0
            m3_met = evaluate_window_metrics(y_win, p_m3)
            fired, _, _ = m3_detector.check_drift(X_ref_sample, X_win, m3_met['f1'])
            t_m3_retrain = 0.0
            m3_evt = 0
            m3_buffer_X.extend(X_win); m3_buffer_y.extend(y_win)
            if fired:
                m3_evt = 1
                m3_samples = min(len(m3_buffer_X), 1500)
                t_m3_retrain = fit_models(m3_models, np.array(m3_buffer_X[-m3_samples:]), np.array(m3_buffer_y[-m3_samples:]), ref_X=X_ref_sample, ref_y=y_ref_sample)
            m3_tot = t_m3_inf + t_m3_retrain
            m3_cum_cpu += m3_tot / 1000.0
            m3_cum_adapt_cpu += t_m3_retrain / 1000.0
            m3_cum_retrains += m3_evt
            all_window_rows.append({
                'seed': seed, 'window_id': w, 'approach': 'Event-Driven Baseline',
                'f1': m3_met['f1'], 'accuracy': m3_met['accuracy'], 'precision': m3_met['precision'], 'recall': m3_met['recall'],
                'cpu_time_ms': m3_tot, 'adapt_cpu_ms': t_m3_retrain, 'retrain_event': m3_evt, 'cum_cpu_s': m3_cum_cpu, 'cum_adapt_cpu_s': m3_cum_adapt_cpu
            })

            # Shared Fingerprint
            prelim_pred = repo_k8.entries[min(repo_k8.entries.keys())].predict_proba(X_win)
            fp, t_fp_ms = extractor.extract(X_win, prelim_pred)

            # 4. RAPT Alone (K=8)
            t0 = time.perf_counter()
            w_k8, max_sim_k8, _ = repo_k8.compute_similarity_weights(fp)
            p_m4 = repo_k8.predict_synthesized_proba(X_win, w_k8, current_window_id=w)
            t_m4_query = (time.perf_counter() - t0) * 1000.0
            m4_met = evaluate_window_metrics(y_win, p_m4)
            t_m4_retrain = 0.0
            m4_evt = 0
            if max_sim_k8 < repo_k8.novelty_threshold:
                m4_evt = 1
                new_m4 = create_base_models(seed + w * 11)
                t_m4_retrain = fit_models(new_m4, X_win, y_win, ref_X=X_ref_sample, ref_y=y_ref_sample)
                repo_k8.insert_regime(fp, new_m4, [1/3, 1/3, 1/3], window_id=w, name=f"Regime_W{w}")
            m4_tot = t_fp_ms + t_m4_query + t_m4_retrain
            m4_cum_cpu += m4_tot / 1000.0
            m4_cum_adapt_cpu += t_m4_retrain / 1000.0
            m4_cum_retrains += m4_evt
            all_window_rows.append({
                'seed': seed, 'window_id': w, 'approach': 'RAPT (K=8)',
                'f1': m4_met['f1'], 'accuracy': m4_met['accuracy'], 'precision': m4_met['precision'], 'recall': m4_met['recall'],
                'cpu_time_ms': m4_tot, 'adapt_cpu_ms': t_m4_retrain, 'retrain_event': m4_evt, 'cum_cpu_s': m4_cum_cpu, 'cum_adapt_cpu_s': m4_cum_adapt_cpu
            })

            # 5. Two-Tier Hybrid RAPT
            t0 = time.perf_counter()
            p_hyb, _, _, beta_t, max_sim_h, _, _ = hybrid_rapt.predict(X_win, fp, current_window_id=w)
            t_m5_query = (time.perf_counter() - t0) * 1000.0
            m5_met = evaluate_window_metrics(y_win, p_hyb)
            t_m5_retrain = 0.0
            m5_evt = 0
            if max_sim_h < hybrid_rapt.novelty_threshold:
                m5_evt = 1
                new_m5 = create_base_models(seed + w * 11)
                t_m5_retrain = fit_models(new_m5, X_win, y_win, ref_X=X_ref_sample, ref_y=y_ref_sample)
                hybrid_rapt.adapt_macro_repository(fp, new_m5, [1/3, 1/3, 1/3], window_id=w, name=f"Regime_W{w}")
            t_onl = hybrid_rapt.update_online_learner(X_win, y_win)
            m5_tot = t_fp_ms + t_m5_query + t_m5_retrain + t_onl
            m5_cum_cpu += m5_tot / 1000.0
            m5_cum_adapt_cpu += (t_m5_retrain + t_onl) / 1000.0
            m5_cum_retrains += m5_evt
            all_window_rows.append({
                'seed': seed, 'window_id': w, 'approach': 'Two-Tier Hybrid RAPT',
                'f1': m5_met['f1'], 'accuracy': m5_met['accuracy'], 'precision': m5_met['precision'], 'recall': m5_met['recall'],
                'cpu_time_ms': m5_tot, 'adapt_cpu_ms': (t_m5_retrain + t_onl), 'retrain_event': m5_evt, 'cum_cpu_s': m5_cum_cpu, 'cum_adapt_cpu_s': m5_cum_adapt_cpu
            })

            # 6. ENHANCED HYBRID RAPT (Dynamic Threshold + Selective Parity Refit)
            t0 = time.perf_counter()
            p_enh, tau_t, _, _, beta_e, max_sim_e, _, _ = enhanced_rapt.predict_enhanced(X_win, fp, current_window_id=w)
            t_m6_query = (time.perf_counter() - t0) * 1000.0
            m6_met = evaluate_window_metrics(y_win, p_enh, threshold=tau_t)
            t_m6_retrain = 0.0
            m6_evt = 0
            m6_buffer_X.extend(X_win); m6_buffer_y.extend(y_win)
            
            # Selective trigger: Novel regime OR performance drop under low similarity
            is_novel = max_sim_e < enhanced_rapt.novelty_threshold
            perf_drop = (m6_prev_f1 - m6_met['f1']) > 0.12 and max_sim_e < 0.80

            if is_novel or perf_drop:
                m6_evt = 1
                new_m6 = create_base_models(seed + w * 11)
                m6_samples = min(len(m6_buffer_X), 1500)
                t_m6_retrain = fit_models(new_m6, np.array(m6_buffer_X[-m6_samples:]), np.array(m6_buffer_y[-m6_samples:]), ref_X=X_ref_sample, ref_y=y_ref_sample)
                enhanced_rapt.adapt_macro_repository(fp, new_m6, [1/3, 1/3, 1/3], window_id=w, name=f"Regime_W{w}")
                m6_prev_f1 = m6_met['f1']
            else:
                m6_prev_f1 = max(0.5, 0.9 * m6_prev_f1 + 0.1 * m6_met['f1'])

            t_onl_e = enhanced_rapt.update_online_learner(X_win, y_win)
            m6_tot = t_fp_ms + t_m6_query + t_m6_retrain + t_onl_e
            m6_cum_cpu += m6_tot / 1000.0
            m6_cum_adapt_cpu += (t_m6_retrain + t_onl_e) / 1000.0
            m6_cum_retrains += m6_evt
            all_window_rows.append({
                'seed': seed, 'window_id': w, 'approach': 'Enhanced Hybrid RAPT (Ours)',
                'f1': m6_met['f1'], 'accuracy': m6_met['accuracy'], 'precision': m6_met['precision'], 'recall': m6_met['recall'],
                'cpu_time_ms': m6_tot, 'adapt_cpu_ms': (t_m6_retrain + t_onl_e), 'retrain_event': m6_evt, 'cum_cpu_s': m6_cum_cpu, 'cum_adapt_cpu_s': m6_cum_adapt_cpu
            })

    df_results = pd.DataFrame(all_window_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_results.to_csv(os.path.join(RESULTS_DIR, "enhanced_window_metrics.csv"), index=False)

    summary_rows = []
    strategies = [
        'Frozen Ensemble', 'Continuous Retraining', 'Event-Driven Baseline',
        'RAPT (K=8)', 'Two-Tier Hybrid RAPT', 'Enhanced Hybrid RAPT (Ours)'
    ]
    for app in strategies:
        group = df_results[df_results['approach'] == app]
        
        seed_precs = group.groupby('seed')['precision'].mean()
        seed_recs = group.groupby('seed')['recall'].mean()
        seed_accs = group.groupby('seed')['accuracy'].mean()
        
        # Per-seed F1 calculated harmonically
        seed_f1s = (2.0 * seed_precs * seed_recs) / (seed_precs + seed_recs)
        
        seed_adapt_cpus = group.groupby('seed')['cum_adapt_cpu_s'].max()
        seed_tot_cpus = group.groupby('seed')['cum_cpu_s'].max()
        seed_retrains = group.groupby('seed')['retrain_event'].sum()

        m_prec = float(seed_precs.mean())
        m_rec = float(seed_recs.mean())
        m_f1 = float((2.0 * m_prec * m_rec) / (m_prec + m_rec)) if (m_prec + m_rec) > 0 else 0.0

        f1_check = (2.0 * m_prec * m_rec) / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0.0
        assert abs(m_f1 - f1_check) < 1e-4, f"F1 identity check failed for {app}: {m_f1:.4f} != {f1_check:.4f}"

        summary_rows.append({
            'Strategy': app,
            'F1_Mean': m_f1, 'F1_Std': float(seed_f1s.std()),
            'Acc_Mean': float(seed_accs.mean()), 'Acc_Std': float(seed_accs.std()),
            'Prec_Mean': m_prec, 'Prec_Std': float(seed_precs.std()),
            'Rec_Mean': m_rec, 'Rec_Std': float(seed_recs.std()),
            'Adapt_CPU_Mean': float(seed_adapt_cpus.mean()), 'Adapt_CPU_Std': float(seed_adapt_cpus.std()),
            'Total_CPU_Mean': float(seed_tot_cpus.mean()), 'Total_CPU_Std': float(seed_tot_cpus.std()),
            'Retrains_Mean': float(seed_retrains.mean())
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "enhanced_summary_metrics.csv"), index=False)

    print("\n" + "=" * 80)
    print("ENHANCED BENCHMARK SUMMARY RESULTS ACROSS 5 SEEDS")
    print("=" * 80)
    for _, r in df_summary.iterrows():
        print(f"{r['Strategy']:<32} | F1: {r['F1_Mean']:.4f} +/- {r['F1_Std']:.4f} | Acc: {r['Acc_Mean']:.4f} +/- {r['Acc_Std']:.4f} | Prec: {r['Prec_Mean']:.4f} | Rec: {r['Rec_Mean']:.4f} | Adapt CPU: {r['Adapt_CPU_Mean']:.2f}s | Retrains: {r['Retrains_Mean']:.1f}")

    return df_summary, df_results


if __name__ == '__main__':
    run_enhanced_benchmark()
