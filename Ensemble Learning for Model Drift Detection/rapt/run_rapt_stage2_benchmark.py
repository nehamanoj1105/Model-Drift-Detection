"""
================================================================================
RAPT STAGE 2 — MASTER BENCHMARK RUNNER
================================================================================
Benchmarks the Autonomous Regime Repository against the Experiment 4
Event-Driven Ensemble baseline across complex recurring drift episodes:
  1. Event-Driven Baseline (Dual-Trigger Detector + Buffer Retraining)
  2. RAPT Full Engine (K=8 Autonomous Repository + Continuous Transfer)
  3. RAPT Capacity Ablation (K=3 Constrained Repository + LRU Eviction Regret)

Evaluation Protocol:
  - 5 Random Seeds: [42, 43, 44, 45, 46]
  - 60 Deployment Windows (500 samples/win = 30,000 deployment samples)
  - 12 Recurring Episodes: Stationary <-> Covariate <-> Concept <-> Mixed <-> Partial
  - Sub-millisecond fingerprint latency directly instrumented (< 1.0 ms)
  - Endpoint sanity identity convergence verified
  - LRU eviction frequency and eviction regret quantified
  - Paired Wilcoxon signed-rank and paired t-tests with Benjamini-Hochberg FDR
  - Strictly single-threaded execution (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1)
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

# Ensure local imports take absolute precedence
_rapt_dir = os.path.dirname(os.path.abspath(__file__))
if _rapt_dir not in sys.path:
    sys.path.insert(0, _rapt_dir)

from config import (
    SEEDS, WINDOW_SIZE, DECISION_THRESHOLD, BUFFER_CAP,
    RESULTS_DIR, FIGURES_DIR, KPI_COLS, TARGET_COL
)
from fingerprint import FastFingerprintExtractor
from repository import AutonomousRegimeRepository
from recurring_stream import generate_recurring_dataset, RECURRING_EPISODE_SCHEDULE


# ==============================================================================
# 1. BASELINE & ENSEMBLE HELPERS
# ==============================================================================

def create_base_models(seed):
    """Create fresh instances of the 3 base classifiers matching Experiment 4."""
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
    elif len(unique_classes) < 2:
        missing_cls = 1 if 0 in unique_classes else 0
        dummy_x = np.mean(X, axis=0, keepdims=True)
        X = np.vstack([X, dummy_x])
        y = np.concatenate([y, [missing_cls]])

    t0 = time.perf_counter()
    for name, m in models.items():
        m.fit(X, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms


def predict_ensemble_proba(models, weights, X):
    """Compute soft-voting probability using normalized weights."""
    comp_probs = [m.predict_proba(X)[:, 1] for m in models.values()]
    w_norm = np.array(weights, dtype=np.float64) / sum(weights)
    p = np.zeros(len(X), dtype=np.float64)
    for w, prob in zip(w_norm, comp_probs):
        p += w * prob
    return np.clip(p, 1e-5, 1.0 - 1e-5)


class SimpleDualTriggerDetector:
    """
    Lightweight dual-trigger detector:
    Fires if 1D feature distribution Wasserstein distance exceeds threshold,
    OR if rolling prequential F1 drops below threshold.
    """
    def __init__(self, wasserstein_thresh=0.075, perf_drop_thresh=0.08, initial_f1=0.85):
        self.wasserstein_thresh = wasserstein_thresh
        self.perf_drop_thresh = perf_drop_thresh
        self.rolling_f1 = initial_f1

    def check_drift(self, X_ref, X_cur, cur_f1):
        # Mean 1D marginal Wasserstein distance across the 4 normalized KPIs
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


def evaluate_window_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD):
    """Compute standard classification metrics for a streaming window."""
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'auc': auc,
        'pr_auc': pr_auc,
        'brier': brier,
    }


# ==============================================================================
# 2. STATISTICAL TESTS & FDR CORRECTION
# ==============================================================================

def run_paired_hypothesis_test(vals_a, vals_b, metric_name, scope_label):
    """Perform paired Wilcoxon signed-rank and paired t-tests with Cohen's d."""
    vals_a = np.asarray(vals_a, dtype=np.float64)
    vals_b = np.asarray(vals_b, dtype=np.float64)
    diffs = vals_a - vals_b
    n = len(diffs)
    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    mean_diff = float(np.mean(diffs))

    # Wilcoxon signed-rank
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 5:
        try:
            w_stat, w_pval = stats.wilcoxon(vals_a, vals_b, zero_method='wilcox')
        except Exception:
            w_stat, w_pval = 0.0, 1.0
    else:
        w_stat, w_pval = 0.0, 1.0

    # Paired t-test
    try:
        t_stat, t_pval = stats.ttest_rel(vals_a, vals_b)
    except Exception:
        t_stat, t_pval = 0.0, 1.0

    # Cohen's d
    std_diff = np.std(diffs, ddof=1)
    cohens_d = float(abs(mean_diff / std_diff)) if std_diff > 1e-8 else 0.0

    return {
        'scope': scope_label,
        'metric': metric_name,
        'n_observations': n,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_delta': mean_diff,
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pval': float(w_pval),
        't_stat': float(t_stat),
        't_pval': float(t_pval),
        'cohens_d': float(cohens_d),
    }


def apply_benjamini_hochberg(p_values):
    """Benjamini-Hochberg False Discovery Rate (FDR) correction."""
    p_vals = np.asarray(p_values, dtype=np.float64)
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(m, dtype=np.float64)
    current_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = sorted_indices[rank]
        val = p_vals[idx] * m / (rank + 1)
        current_min = min(current_min, val)
        adj_p[idx] = min(1.0, current_min)
    return adj_p


# ==============================================================================
# 3. ENDPOINT SANITY IDENTITY CONVERGENCE CHECK
# ==============================================================================

def verify_endpoint_identity_convergence(seed=42):
    """
    Direct verification of requirement:
    When a regime is a near-exact repeat of a single stored regime (similarity ~ 1),
    verify the synthesized policy converges to that stored policy (identity check).
    """
    print("\n---> Running Endpoint Sanity Identity Convergence Check...")
    repo = AutonomousRegimeRepository(max_capacity=8, gamma=15.0, novelty_threshold=0.65)
    
    rng = np.random.RandomState(seed)
    X_ref = rng.normal(0, 1, (1000, 4))
    y_ref = (X_ref[:, 0] + X_ref[:, 1] > 0).astype(int)
    
    m1 = create_base_models(seed)
    fit_models(m1, X_ref, y_ref)
    fp1 = np.array([0.05, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.25], dtype=np.float32)
    id1 = repo.insert_regime(fp1, m1, [1/3, 1/3, 1/3], window_id=0, name="Regime_1")

    # Add a distinct second regime
    m2 = create_base_models(seed + 10)
    fit_models(m2, X_ref, 1 - y_ref)
    fp2 = np.array([0.80, 0.80, 0.80, 0.80, 0.40, 0.20, 0.20, 0.65], dtype=np.float32)
    id2 = repo.insert_regime(fp2, m2, [1/3, 1/3, 1/3], window_id=1, name="Regime_2")

    # Query with near-exact repeat of Regime 1 (similarity ~ 1)
    fp_query = fp1 + 1e-4 * rng.normal(0, 1, len(fp1))
    weights_dict, max_sim, closest_id = repo.compute_similarity_weights(fp_query)

    X_test = rng.normal(0, 1, (200, 4))
    p_synthesized = repo.predict_synthesized_proba(X_test, weights_dict)
    p_regime1 = repo.entries[id1].predict_proba(X_test)

    max_pred_discrepancy = float(np.max(np.abs(p_synthesized - p_regime1)))
    weight_regime1 = weights_dict[id1]

    print(f"     Max Similarity:          {max_sim:.6f} (target: ~1.000)")
    print(f"     Synthesized Weight (R1): {weight_regime1:.6f} (target: > 0.999)")
    print(f"     Max Policy Discrepancy:  {max_pred_discrepancy:.2e} (target: < 1e-4)")

    passed = (weight_regime1 > 0.999) and (max_pred_discrepancy < 1e-4)
    if passed:
        print("     [PASSED] Endpoint Sanity Identity Convergence holds strictly.")
    else:
        print("     [FAILED] Discrepancy exceeded tolerance.")
    return passed, max_sim, weight_regime1, max_pred_discrepancy


# ==============================================================================
# 4. MASTER BENCHMARK EXECUTION
# ==============================================================================

def run_stage2_master_benchmark(seeds=SEEDS, n_windows=60, window_size=500):
    print("=" * 85)
    print("STARTING RAPT STAGE 2 MASTER BENCHMARK: AUTONOMOUS REGIME REPOSITORY")
    print("=" * 85)
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Deployment Scale: {n_windows} windows x {window_size} samples = {n_windows * window_size:,} samples")
    print(f"Episodes: 12 recurring drift episodes (Covariate <-> Concept <-> Mixed <-> Partial)")
    print(f"Novelty Threshold: tau_novelty = 0.65 | Kernel Gamma: gamma = 15.0")
    print(f"Capacities Tested: K=8 (RAPT Autonomous) vs. K=3 (Capacity Constrained Ablation)")
    print("=" * 85)

    # 1. Run Endpoint Sanity Identity Check
    identity_passed, max_sim_test, weight_r1_test, max_err_test = verify_endpoint_identity_convergence(seeds[0])
    if not identity_passed:
        raise RuntimeError("Endpoint Sanity Identity Check Failed! Halting benchmark.")

    all_window_rows = []
    all_fingerprint_latencies = []

    # Map episode info per window
    win_to_episode = {}
    for ep in RECURRING_EPISODE_SCHEDULE:
        for w in range(ep['start_win'], ep['end_win'] + 1):
            if w < n_windows:
                win_to_episode[w] = ep

    for s_idx, seed in enumerate(seeds):
        t0_seed = time.time()
        print(f"\n---> [Seed {seed} ({s_idx + 1}/{len(seeds)})] Generating recurring drift stream...")
        X_init, y_init, X_stream, y_stream, df_stream = generate_recurring_dataset(
            seed=seed, n_initial_training=20_000, n_windows=n_windows, window_size=window_size
        )

        # Baseline reference slice for detector and extractor calibration
        X_ref_sample = X_init[-1000:]
        y_ref_sample = y_init[-1000:]

        # Initialize Fingerprint Extractor
        extractor = FastFingerprintExtractor(X_ref_sample, num_quantiles=40)

        # Pre-train initial baseline models on reference slice
        base_models_init = create_base_models(seed)
        _ = fit_models(base_models_init, X_ref_sample, y_ref_sample)
        base_fp_init, _ = extractor.extract(X_ref_sample[-500:], base_models_init['RandomForest'].predict(X_ref_sample[-500:]))

        # ----------------------------------------------------------------------
        # APPROACH 1: Event-Driven Ensemble Baseline (Experiment 4)
        # ----------------------------------------------------------------------
        ed_models = create_base_models(seed)
        _ = fit_models(ed_models, X_ref_sample, y_ref_sample)
        ed_detector = SimpleDualTriggerDetector(wasserstein_thresh=0.075, perf_drop_thresh=0.08, initial_f1=0.85)
        ed_buffer_X = list(X_ref_sample)
        ed_buffer_y = list(y_ref_sample)
        ed_cum_cpu = 0.0
        ed_cum_retrains = 0
        ed_cum_adapt_cpu = 0.0

        # ----------------------------------------------------------------------
        # APPROACH 2: RAPT (Autonomous Regime Repository, K=8)
        # ----------------------------------------------------------------------
        repo_k8 = AutonomousRegimeRepository(max_capacity=8, gamma=15.0, novelty_threshold=0.65)
        # Seed repository with baseline reference regime
        k8_models_init = create_base_models(seed)
        _ = fit_models(k8_models_init, X_ref_sample, y_ref_sample)
        repo_k8.insert_regime(base_fp_init, k8_models_init, [1/3, 1/3, 1/3], window_id=0, name="Stationary_Base")
        k8_cum_cpu = 0.0
        k8_cum_retrains = 0
        k8_cum_adapt_cpu = 0.0

        # ----------------------------------------------------------------------
        # APPROACH 3: RAPT Capacity Ablation (Constrained K=3)
        # ----------------------------------------------------------------------
        repo_k3 = AutonomousRegimeRepository(max_capacity=3, gamma=15.0, novelty_threshold=0.65)
        k3_models_init = create_base_models(seed)
        _ = fit_models(k3_models_init, X_ref_sample, y_ref_sample)
        repo_k3.insert_regime(base_fp_init, k3_models_init, [1/3, 1/3, 1/3], window_id=0, name="Stationary_Base")
        k3_cum_cpu = 0.0
        k3_cum_retrains = 0
        k3_cum_adapt_cpu = 0.0

        print(f"     Streaming across {n_windows} windows...")

        for w in range(n_windows):
            idx_s = w * window_size
            idx_e = idx_s + window_size
            X_win = X_stream[idx_s:idx_e]
            y_win = y_stream[idx_s:idx_e]

            ep_info = win_to_episode.get(w, {'name': 'Unknown', 'type': 'unknown', 'episode': -1})

            # ==================================================================
            # 1. EVALUATE EVENT-DRIVEN BASELINE
            # ==================================================================
            t_inf_s = time.perf_counter()
            p_ed = predict_ensemble_proba(ed_models, [1/3, 1/3, 1/3], X_win)
            t_ed_inf_ms = (time.perf_counter() - t_inf_s) * 1000.0
            m_ed = evaluate_window_metrics(y_win, p_ed)

            drift_fired, mean_w, f1_drop = ed_detector.check_drift(X_ref_sample, X_win, m_ed['f1'])

            ed_retrain_ms = 0.0
            ed_retrain_event = 0
            ed_samples_adapted = 0

            if drift_fired:
                ed_retrain_event = 1
                ed_buffer_X.extend(X_win)
                ed_buffer_y.extend(y_win)
                if len(ed_buffer_X) > BUFFER_CAP:
                    ed_buffer_X = ed_buffer_X[-BUFFER_CAP:]
                    ed_buffer_y = ed_buffer_y[-BUFFER_CAP:]
                ed_samples_adapted = min(len(ed_buffer_X), 1500)
                train_X = np.array(ed_buffer_X[-ed_samples_adapted:])
                train_y = np.array(ed_buffer_y[-ed_samples_adapted:])
                ed_retrain_ms = fit_models(ed_models, train_X, train_y)
            else:
                ed_buffer_X.extend(X_win)
                ed_buffer_y.extend(y_win)
                if len(ed_buffer_X) > BUFFER_CAP:
                    ed_buffer_X = ed_buffer_X[-BUFFER_CAP:]
                    ed_buffer_y = ed_buffer_y[-BUFFER_CAP:]

            ed_window_cpu_ms = t_ed_inf_ms + ed_retrain_ms
            ed_cum_cpu += ed_window_cpu_ms / 1000.0
            ed_cum_adapt_cpu += ed_retrain_ms / 1000.0
            ed_cum_retrains += ed_retrain_event

            all_window_rows.append({
                'seed': seed, 'window_id': w, 'episode_id': ep_info['episode'],
                'episode_name': ep_info['name'], 'episode_type': ep_info['type'],
                'approach': 'Event-Driven Baseline',
                'f1': m_ed['f1'], 'accuracy': m_ed['accuracy'], 'precision': m_ed['precision'],
                'recall': m_ed['recall'], 'auc': m_ed['auc'], 'pr_auc': m_ed['pr_auc'], 'brier': m_ed['brier'],
                'cpu_time_ms': ed_window_cpu_ms, 'adapt_cpu_ms': ed_retrain_ms,
                'retrain_event': ed_retrain_event, 'samples_adapted': ed_samples_adapted,
                'cum_cpu_s': ed_cum_cpu, 'cum_adapt_cpu_s': ed_cum_adapt_cpu, 'cum_retrains': ed_cum_retrains,
                'max_similarity': 0.0, 'active_regimes': 1, 'eviction_miss': 0, 'eviction_regret_ms': 0.0
            })

            # ==================================================================
            # FINGERPRINT EXTRACTION (Shared for RAPT evaluation)
            # ==================================================================
            # Preliminary prediction for violation rate feature
            prelim_prob = repo_k8.entries[min(repo_k8.entries.keys())].predict_proba(X_win)
            fp, t_fp_ms = extractor.extract(X_win, prelim_prob)
            all_fingerprint_latencies.append(t_fp_ms)

            # ==================================================================
            # 2. EVALUATE RAPT (K=8)
            # ==================================================================
            t_k8_s = time.perf_counter()
            weights_k8, max_sim_k8, closest_k8 = repo_k8.compute_similarity_weights(fp)
            p_k8 = repo_k8.predict_synthesized_proba(X_win, weights_k8, current_window_id=w)
            t_k8_query_ms = (time.perf_counter() - t_k8_s) * 1000.0
            m_k8 = evaluate_window_metrics(y_win, p_k8)

            k8_retrain_ms = 0.0
            k8_retrain_event = 0
            k8_samples_adapted = 0

            # Novelty check: adapt only when current regime is genuinely novel
            if max_sim_k8 < repo_k8.novelty_threshold:
                k8_retrain_event = 1
                new_k8_models = create_base_models(seed + w * 7)
                k8_samples_adapted = len(X_win)
                k8_retrain_ms = fit_models(new_k8_models, X_win, y_win, ref_X=X_ref_sample, ref_y=y_ref_sample)
                repo_k8.insert_regime(fp, new_k8_models, [1/3, 1/3, 1/3], window_id=w, name=f"Regime_W{w}")

            k8_window_cpu_ms = t_fp_ms + t_k8_query_ms + k8_retrain_ms
            k8_cum_cpu += k8_window_cpu_ms / 1000.0
            k8_cum_adapt_cpu += k8_retrain_ms / 1000.0
            k8_cum_retrains += k8_retrain_event

            # Auditing eviction regret on K=8 (should be zero under ample capacity)
            k8_miss, _ = repo_k8.check_eviction_regret(fp, window_id=w)

            all_window_rows.append({
                'seed': seed, 'window_id': w, 'episode_id': ep_info['episode'],
                'episode_name': ep_info['name'], 'episode_type': ep_info['type'],
                'approach': 'RAPT (K=8)',
                'f1': m_k8['f1'], 'accuracy': m_k8['accuracy'], 'precision': m_k8['precision'],
                'recall': m_k8['recall'], 'auc': m_k8['auc'], 'pr_auc': m_k8['pr_auc'], 'brier': m_k8['brier'],
                'cpu_time_ms': k8_window_cpu_ms, 'adapt_cpu_ms': k8_retrain_ms,
                'retrain_event': k8_retrain_event, 'samples_adapted': k8_samples_adapted,
                'cum_cpu_s': k8_cum_cpu, 'cum_adapt_cpu_s': k8_cum_adapt_cpu, 'cum_retrains': k8_cum_retrains,
                'max_similarity': max_sim_k8, 'active_regimes': repo_k8.size(),
                'eviction_miss': int(k8_miss), 'eviction_regret_ms': 0.0
            })

            # ==================================================================
            # 3. EVALUATE RAPT ABLATION (K=3 Constrained Capacity)
            # ==================================================================
            t_k3_s = time.perf_counter()
            weights_k3, max_sim_k3, closest_k3 = repo_k3.compute_similarity_weights(fp)
            p_k3 = repo_k3.predict_synthesized_proba(X_win, weights_k3, current_window_id=w)
            t_k3_query_ms = (time.perf_counter() - t_k3_s) * 1000.0
            m_k3 = evaluate_window_metrics(y_win, p_k3)

            k3_retrain_ms = 0.0
            k3_retrain_event = 0
            k3_samples_adapted = 0
            k3_eviction_regret_ms = 0.0

            # Check if this state matches a regime that was previously evicted
            k3_miss, miss_rec = repo_k3.check_eviction_regret(fp, window_id=w)

            if max_sim_k3 < repo_k3.novelty_threshold:
                k3_retrain_event = 1
                new_k3_models = create_base_models(seed + w * 7)
                k3_samples_adapted = len(X_win)
                k3_retrain_ms = fit_models(new_k3_models, X_win, y_win, ref_X=X_ref_sample, ref_y=y_ref_sample)
                repo_k3.insert_regime(fp, new_k3_models, [1/3, 1/3, 1/3], window_id=w, name=f"Regime_W{w}")
                if k3_miss:
                    k3_eviction_regret_ms = k3_retrain_ms

            k3_window_cpu_ms = t_fp_ms + t_k3_query_ms + k3_retrain_ms
            k3_cum_cpu += k3_window_cpu_ms / 1000.0
            k3_cum_adapt_cpu += k3_retrain_ms / 1000.0
            k3_cum_retrains += k3_retrain_event

            all_window_rows.append({
                'seed': seed, 'window_id': w, 'episode_id': ep_info['episode'],
                'episode_name': ep_info['name'], 'episode_type': ep_info['type'],
                'approach': 'RAPT Ablation (K=3)',
                'f1': m_k3['f1'], 'accuracy': m_k3['accuracy'], 'precision': m_k3['precision'],
                'recall': m_k3['recall'], 'auc': m_k3['auc'], 'pr_auc': m_k3['pr_auc'], 'brier': m_k3['brier'],
                'cpu_time_ms': k3_window_cpu_ms, 'adapt_cpu_ms': k3_retrain_ms,
                'retrain_event': k3_retrain_event, 'samples_adapted': k3_samples_adapted,
                'cum_cpu_s': k3_cum_cpu, 'cum_adapt_cpu_s': k3_cum_adapt_cpu, 'cum_retrains': k3_cum_retrains,
                'max_similarity': max_sim_k3, 'active_regimes': repo_k3.size(),
                'eviction_miss': int(k3_miss), 'eviction_regret_ms': k3_eviction_regret_ms
            })

        print(f"     [Seed {seed} Complete in {time.time() - t0_seed:.2f}s]")
        print(f"     - Event-Driven Baseline: F1={ed_cum_cpu:.2f}s CPU, Retrains={ed_cum_retrains}")
        print(f"     - RAPT (K=8):            F1={k8_cum_cpu:.2f}s CPU, Retrains={k8_cum_retrains} (Evictions: {len(repo_k8.eviction_history)})")
        print(f"     - RAPT Ablation (K=3):   F1={k3_cum_cpu:.2f}s CPU, Retrains={k3_cum_retrains} (Evictions: {len(repo_k3.eviction_history)}, Misses: {len(repo_k3.eviction_cache_misses)})")

    # ==============================================================================
    # 5. DATA AGGREGATION & EXPORT
    # ==============================================================================
    df_windows = pd.DataFrame(all_window_rows)
    csv_windows_path = os.path.join(RESULTS_DIR, 'stage2_window_metrics.csv')
    df_windows.to_csv(csv_windows_path, index=False)
    print(f"\nSaved window metrics ({len(df_windows)} rows) to: {csv_windows_path}")

    # Summary per approach per seed
    df_seed_summary = df_windows.groupby(['approach', 'seed']).agg({
        'f1': 'mean',
        'accuracy': 'mean',
        'cpu_time_ms': 'sum',
        'adapt_cpu_ms': 'sum',
        'retrain_event': 'sum',
        'eviction_miss': 'sum',
        'eviction_regret_ms': 'sum'
    }).reset_index()

    # Macro summary (mean +- std across 5 seeds)
    summary_rows = []
    baseline_seed_cpu = df_seed_summary[df_seed_summary['approach'] == 'Event-Driven Baseline'].set_index('seed')['cpu_time_ms']
    baseline_seed_adapt = df_seed_summary[df_seed_summary['approach'] == 'Event-Driven Baseline'].set_index('seed')['adapt_cpu_ms']
    baseline_seed_retrains = df_seed_summary[df_seed_summary['approach'] == 'Event-Driven Baseline'].set_index('seed')['retrain_event']

    for app in ['Event-Driven Baseline', 'RAPT (K=8)', 'RAPT Ablation (K=3)']:
        sub = df_seed_summary[df_seed_summary['approach'] == app].set_index('seed')
        
        # Adaptation savings relative to baseline on each seed
        adapt_savings = (baseline_seed_adapt - sub['adapt_cpu_ms']) / baseline_seed_adapt * 100.0
        retrain_savings = (baseline_seed_retrains - sub['retrain_event']) / baseline_seed_retrains * 100.0
        total_cpu_savings = (baseline_seed_cpu - sub['cpu_time_ms']) / baseline_seed_cpu * 100.0

        summary_rows.append({
            'approach': app,
            'mean_f1': float(sub['f1'].mean()),
            'std_f1': float(sub['f1'].std()),
            'mean_accuracy': float(sub['accuracy'].mean()),
            'std_accuracy': float(sub['accuracy'].std()),
            'mean_total_cpu_s': float(sub['cpu_time_ms'].mean() / 1000.0),
            'std_total_cpu_s': float(sub['cpu_time_ms'].std() / 1000.0),
            'mean_adapt_cpu_s': float(sub['adapt_cpu_ms'].mean() / 1000.0),
            'std_adapt_cpu_s': float(sub['adapt_cpu_ms'].std() / 1000.0),
            'mean_retrain_count': float(sub['retrain_event'].mean()),
            'std_retrain_count': float(sub['retrain_event'].std()),
            'mean_adapt_savings_pct': float(adapt_savings.mean()),
            'std_adapt_savings_pct': float(adapt_savings.std()),
            'mean_retrain_savings_pct': float(retrain_savings.mean()),
            'std_retrain_savings_pct': float(retrain_savings.std()),
            'mean_total_cpu_savings_pct': float(total_cpu_savings.mean()),
            'std_total_cpu_savings_pct': float(total_cpu_savings.std()),
            'mean_eviction_misses': float(sub['eviction_miss'].mean()),
            'mean_eviction_regret_ms': float(sub['eviction_regret_ms'].mean()),
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_summary_path = os.path.join(RESULTS_DIR, 'stage2_summary_metrics.csv')
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"Saved macro summary metrics to: {csv_summary_path}")

    # ==============================================================================
    # 6. PAIRED STATISTICAL HYPOTHESIS TESTING
    # ==============================================================================
    test_rows = []
    
    # Paired observations across 60 windows x 5 seeds = 300 observations
    ed_wins = df_windows[df_windows['approach'] == 'Event-Driven Baseline'].sort_values(['seed', 'window_id']).reset_index(drop=True)
    k8_wins = df_windows[df_windows['approach'] == 'RAPT (K=8)'].sort_values(['seed', 'window_id']).reset_index(drop=True)
    k3_wins = df_windows[df_windows['approach'] == 'RAPT Ablation (K=3)'].sort_values(['seed', 'window_id']).reset_index(drop=True)

    # Test 1: RAPT (K=8) vs Baseline: F1 Score
    test_rows.append(run_paired_hypothesis_test(
        k8_wins['f1'], ed_wins['f1'], 'f1_score', 'RAPT_K8_vs_Baseline_300obs'
    ))
    # Test 2: RAPT (K=8) vs Baseline: Adaptation CPU Time
    test_rows.append(run_paired_hypothesis_test(
        k8_wins['adapt_cpu_ms'], ed_wins['adapt_cpu_ms'], 'adapt_cpu_ms', 'RAPT_K8_vs_Baseline_300obs'
    ))
    # Test 3: RAPT (K=8) vs Baseline: Total Window CPU Time
    test_rows.append(run_paired_hypothesis_test(
        k8_wins['cpu_time_ms'], ed_wins['cpu_time_ms'], 'total_cpu_ms', 'RAPT_K8_vs_Baseline_300obs'
    ))
    # Test 4: RAPT (K=8) vs Baseline: Retrain Events
    test_rows.append(run_paired_hypothesis_test(
        k8_wins['retrain_event'], ed_wins['retrain_event'], 'retrain_events', 'RAPT_K8_vs_Baseline_300obs'
    ))
    # Test 5: RAPT (K=8) vs RAPT Ablation (K=3): Adaptation CPU Time
    test_rows.append(run_paired_hypothesis_test(
        k8_wins['adapt_cpu_ms'], k3_wins['adapt_cpu_ms'], 'adapt_cpu_ms', 'RAPT_K8_vs_RAPT_K3_300obs'
    ))
    # Test 6: RAPT (K=8) vs RAPT Ablation (K=3): Retrain Events
    test_rows.append(run_paired_hypothesis_test(
        k8_wins['retrain_event'], k3_wins['retrain_event'], 'retrain_events', 'RAPT_K8_vs_RAPT_K3_300obs'
    ))
    # Test 7: RAPT (K=8) vs Baseline on recurring episodes only (episodes 7-12, 150 obs)
    recurring_mask = (ed_wins['episode_id'] >= 7).to_numpy()
    test_rows.append(run_paired_hypothesis_test(
        k8_wins.loc[recurring_mask, 'adapt_cpu_ms'], ed_wins.loc[recurring_mask, 'adapt_cpu_ms'],
        'adapt_cpu_ms_recurring_episodes', 'RAPT_K8_vs_Baseline_Recurring_150obs'
    ))

    df_tests = pd.DataFrame(test_rows)
    df_tests['wilcoxon_pval_fdr'] = apply_benjamini_hochberg(df_tests['wilcoxon_pval'])
    df_tests['is_significant'] = df_tests['wilcoxon_pval_fdr'] < 0.05

    csv_tests_path = os.path.join(RESULTS_DIR, 'stage2_statistical_tests.csv')
    df_tests.to_csv(csv_tests_path, index=False)
    print(f"Saved paired statistical tests to: {csv_tests_path}")

    # ==============================================================================
    # 7. FINGERPRINT LATENCY PROFILING
    # ==============================================================================
    latencies = np.array(all_fingerprint_latencies, dtype=np.float64)
    latency_summary = {
        'count': len(latencies),
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'max_ms': float(np.max(latencies)),
        'pct_submillisecond': float(np.mean(latencies < 1.0) * 100.0),
    }
    df_latency = pd.DataFrame([latency_summary])
    csv_latency_path = os.path.join(RESULTS_DIR, 'stage2_fingerprint_latency.csv')
    df_latency.to_csv(csv_latency_path, index=False)
    print(f"Saved fingerprint latency profile to: {csv_latency_path}")

    # ==============================================================================
    # 8. CONSOLE BENCHMARK REPORT
    # ==============================================================================
    print("\n" + "=" * 85)
    print("STAGE 2 MASTER BENCHMARK RESULTS SUMMARY (Mean +- Std across 5 Seeds)")
    print("=" * 85)
    print(df_summary[[
        'approach', 'mean_f1', 'mean_accuracy', 'mean_adapt_cpu_s', 'mean_retrain_count',
        'mean_adapt_savings_pct', 'mean_total_cpu_savings_pct', 'mean_eviction_misses'
    ]].to_string(index=False))

    print("\n" + "=" * 85)
    print("FINGERPRINT COMPUTATION LATENCY PROFILE (Strict Sub-Millisecond Instrument)")
    print("=" * 85)
    print(f"Total Evaluations:       {latency_summary['count']}")
    print(f"Mean Latency:            {latency_summary['mean_ms']:.4f} ms")
    print(f"Median (p50) Latency:    {latency_summary['median_ms']:.4f} ms")
    print(f"95th Percentile:         {latency_summary['p95_ms']:.4f} ms")
    print(f"99th Percentile:         {latency_summary['p99_ms']:.4f} ms")
    print(f"Maximum Latency:         {latency_summary['max_ms']:.4f} ms")
    print(f"Sub-Millisecond Success: {latency_summary['pct_submillisecond']:.2f}% (< 1.0 ms)")

    print("\n" + "=" * 85)
    print("PAIRED STATISTICAL HYPOTHESIS TESTS (Wilcoxon + FDR Correction)")
    print("=" * 85)
    print(df_tests[[
        'scope', 'metric', 'mean_a', 'mean_b', 'mean_delta',
        'wilcoxon_pval', 'wilcoxon_pval_fdr', 'cohens_d', 'is_significant'
    ]].to_string(index=False))
    print("=" * 85)

    return df_windows, df_summary, df_tests, df_latency


if __name__ == '__main__':
    run_stage2_master_benchmark()
