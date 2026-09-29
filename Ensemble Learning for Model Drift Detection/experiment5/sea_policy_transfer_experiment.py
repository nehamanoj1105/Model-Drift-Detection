"""
================================================================================
SEA RECURRING-CONCEPT POLICY TRANSFER BENCHMARK (CORRECTED)
================================================================================
Main benchmark runner for SEA Concept Recurrence Transfer Experiment.
Evaluates 6 methods across 5 seeds ([42, 43, 44, 45, 46]):
    1. Fixed Equal Ensemble
    2. Global Adaptive Ensemble
    3. Cold Adaptation Baseline (Event-Driven)
    4. Similarity-Based Policy Transfer (Deployable)
    5. Similarity-Based Model-State Transfer (Deployable)
    6. Oracle Recurrence Transfer (Upper Bound)

Key Refinement:
    Uses Joint Distribution Fingerprints F(X, Y) to detect P(Y|X) concept drift
    where marginal P(X) is static across SEA thresholds.
================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr, pearsonr
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

from sea_stream_generator import generate_sea_recurring_stream


# ------------------------------------------------------------------------------
# JOINT DISTRIBUTION FINGERPRINT GENERATION P(X, Y)
# ------------------------------------------------------------------------------
def compute_window_fingerprint(X, y):
    """
    Computes a 25-dimensional Joint P(X, Y) fingerprint:
      - Class balance P(Y=1): 1 dim
      - Feature quantiles for Y=1 (10, 25, 50, 75, 90) x 3 = 15 dims
      - Feature means for Y=1 (3) and Y=0 (3) = 6 dims
      - Sum feature f1+f2 mean for Y=1 and Y=0 = 2 dims
    This reliably distinguishes Concept A (th=7), Concept B (th=10), Concept C (th=13)
    where marginal P(X) is uniform U(0,10) and identical across concepts.
    """
    pos_rate = float(np.mean(y))
    
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    
    if len(X_pos) > 5:
        q10_pos = np.percentile(X_pos, 10, axis=0)
        q25_pos = np.percentile(X_pos, 25, axis=0)
        q50_pos = np.percentile(X_pos, 50, axis=0)
        q75_pos = np.percentile(X_pos, 75, axis=0)
        q90_pos = np.percentile(X_pos, 90, axis=0)
        mean_pos = np.mean(X_pos, axis=0)
        sum_pos = float(np.mean(X_pos[:, 0] + X_pos[:, 1]))
    else:
        q10_pos = q25_pos = q50_pos = q75_pos = q90_pos = mean_pos = np.zeros(3)
        sum_pos = 0.0
        
    if len(X_neg) > 5:
        mean_neg = np.mean(X_neg, axis=0)
        sum_neg = float(np.mean(X_neg[:, 0] + X_neg[:, 1]))
    else:
        mean_neg = np.zeros(3)
        sum_neg = 0.0
        
    fp = np.concatenate([
        [pos_rate],
        q10_pos, q25_pos, q50_pos, q75_pos, q90_pos,
        mean_pos, mean_neg,
        [sum_pos, sum_neg]
    ])
    return fp


def compute_fingerprint_similarity(fp1, fp2):
    """
    Cosine similarity between two joint fingerprints.
    """
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(fp1, fp2) / (norm1 * norm2))


# ------------------------------------------------------------------------------
# FAST METRICS
# ------------------------------------------------------------------------------
def compute_fast_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + fp) > 0 else 0.0
    
    macro_f1 = (f1_1 + f1_0) / 2.0
    prec = (prec_1 + prec_0) / 2.0
    rec = (rec_1 + rec_0) / 2.0
    
    return {
        'macro_f1': float(macro_f1),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# ------------------------------------------------------------------------------
# BENCHMARK RUNNER CLASS
# ------------------------------------------------------------------------------
class SEABenchmarkRunner:
    def __init__(self, seed=42, sim_threshold=0.97):
        self.seed = seed
        self.sim_threshold = sim_threshold
        self.windows, self.df_meta = generate_sea_recurring_stream(seed=seed)
        
    def _create_base_models(self, sub_seed):
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=sub_seed, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=sub_seed, n_jobs=-1)
        gb = HistGradientBoostingClassifier(max_iter=50, random_state=sub_seed)
        return [rf, et, gb]

    def run(self):
        methods = [
            'Fixed Ensemble',
            'Global Adaptive Ensemble',
            'Cold Adaptation Baseline',
            'Similarity Policy Transfer',
            'Similarity Model-State Transfer',
            'Oracle Recurrence Transfer'
        ]
        
        results = {m: [] for m in methods}
        retrain_counts = {m: 0 for m in methods}
        cpu_times = {m: 0.0 for m in methods}
        
        # Base models per method
        models = {m: self._create_base_models(self.seed + idx * 100) for idx, m in enumerate(methods)}
        
        # Policies: array of [w_rf, w_et, w_gb]
        policies = {m: np.array([1/3, 1/3, 1/3]) for m in methods}
        
        # Sliding memory buffers of exposed data for continuous online fitting
        # {method: (X_buf, y_buf)}
        data_buffers = {m: (None, None) for m in methods}
        
        # Concept memory stores
        memory_similarity_policy = [] # list of dicts: {'fp': fp, 'policy': policy, 'concept_id': concept_id}
        memory_similarity_model = []  # list of dicts: {'fp': fp, 'policy': policy, 'models': [rf, et, gb], 'concept_id': concept_id}
        memory_oracle = {}            # dict keyed by concept_id: {'policy': policy, 'models': [rf, et, gb]}
        
        # Initial fit on Window 0
        w0 = self.windows[0]
        X0, y0 = w0['X'], w0['y']
        for m in methods:
            t0 = time.time()
            for model in models[m]:
                model.fit(X0, y0)
            data_buffers[m] = (X0.copy(), y0.copy())
            cpu_times[m] += time.time() - t0
            
        transfer_logs = []
        
        for w in self.windows:
            w_id = w['window_id']
            concept_id = w['concept_id']
            is_block_start = w['is_block_start']
            X, y = w['X'], w['y']
            
            for m in methods:
                t0 = time.time()
                active_models = models[m]
                active_policy = policies[m]
                
                # --------------------------------------------------------------
                # PRE-PREDICTION ADAPTATION / RETRIEVAL (At Block Boundaries)
                # --------------------------------------------------------------
                if is_block_start and w_id > 0:
                    # Previous window data was exposed at end of w_id - 1
                    prev_w = self.windows[w_id - 1]
                    X_prev, y_prev = prev_w['X'], prev_w['y']
                    fp_prev = compute_window_fingerprint(X_prev, y_prev)
                    
                    if m == 'Cold Adaptation Baseline':
                        # Cold start: reset models and policy, train ONLY on prev window
                        active_policy = np.array([1/3, 1/3, 1/3])
                        policies[m] = active_policy
                        for model in active_models:
                            model.fit(X_prev, y_prev)
                        data_buffers[m] = (X_prev.copy(), y_prev.copy())
                        retrain_counts[m] += 1
                        
                    elif m == 'Oracle Recurrence Transfer':
                        if concept_id in memory_oracle:
                            saved = memory_oracle[concept_id]
                            policies[m] = saved['policy'].copy()
                            active_policy = policies[m]
                            # Restore base models
                            models[m] = [m_saved for m_saved in saved['models']]
                            active_models = models[m]
                            data_buffers[m] = (X_prev.copy(), y_prev.copy())
                        else:
                            # Cold adaptation for unseen concept
                            active_policy = np.array([1/3, 1/3, 1/3])
                            policies[m] = active_policy
                            for model in active_models:
                                model.fit(X_prev, y_prev)
                            data_buffers[m] = (X_prev.copy(), y_prev.copy())
                            retrain_counts[m] += 1
                            
                    elif m == 'Similarity Policy Transfer':
                        best_sim = -1.0
                        best_mem = None
                        for mem in memory_similarity_policy:
                            sim = compute_fingerprint_similarity(fp_prev, mem['fp'])
                            if sim > best_sim:
                                best_sim = sim
                                best_mem = mem
                                
                        if best_sim >= self.sim_threshold and best_mem is not None:
                            policies[m] = best_mem['policy'].copy()
                            active_policy = policies[m]
                            # Retrain base models on prev window
                            for model in active_models:
                                model.fit(X_prev, y_prev)
                            data_buffers[m] = (X_prev.copy(), y_prev.copy())
                            retrain_counts[m] += 1
                        else:
                            active_policy = np.array([1/3, 1/3, 1/3])
                            policies[m] = active_policy
                            for model in active_models:
                                model.fit(X_prev, y_prev)
                            data_buffers[m] = (X_prev.copy(), y_prev.copy())
                            retrain_counts[m] += 1
                            
                    elif m == 'Similarity Model-State Transfer':
                        best_sim = -1.0
                        best_mem = None
                        for mem in memory_similarity_model:
                            sim = compute_fingerprint_similarity(fp_prev, mem['fp'])
                            if sim > best_sim:
                                best_sim = sim
                                best_mem = mem
                                
                        if best_sim >= self.sim_threshold and best_mem is not None:
                            policies[m] = best_mem['policy'].copy()
                            active_policy = policies[m]
                            models[m] = [m_saved for m_saved in best_mem['models']]
                            active_models = models[m]
                            data_buffers[m] = (X_prev.copy(), y_prev.copy())
                        else:
                            active_policy = np.array([1/3, 1/3, 1/3])
                            policies[m] = active_policy
                            for model in active_models:
                                model.fit(X_prev, y_prev)
                            data_buffers[m] = (X_prev.copy(), y_prev.copy())
                            retrain_counts[m] += 1

                # --------------------------------------------------------------
                # PREDICT & EVALUATE ON CURRENT WINDOW W_t
                # --------------------------------------------------------------
                probas = np.column_stack([model.predict_proba(X)[:, 1] for model in active_models])
                ens_proba = probas @ active_policy
                y_pred = (ens_proba >= 0.5).astype(int)
                
                metrics = compute_fast_metrics(y, y_pred)
                
                # Individual model accuracies for policy updates
                m_accs = np.array([np.mean((p >= 0.5).astype(int) == y) for p in probas.T])
                
                # --------------------------------------------------------------
                # POST-EVALUATION CONTINUOUS ADAPTATION & MEMORY SAVING
                # --------------------------------------------------------------
                # Update data buffer (keep max 1000 samples = 2 windows)
                X_buf, y_buf = data_buffers[m]
                if X_buf is None:
                    X_buf, y_buf = X.copy(), y.copy()
                else:
                    X_buf = np.vstack([X_buf, X])[-1000:]
                    y_buf = np.concatenate([y_buf, y])[-1000:]
                data_buffers[m] = (X_buf, y_buf)
                
                # Continuously refit base models during intra-block streaming
                if not is_block_start and (w_id % 2 == 0):
                    for model in active_models:
                        model.fit(X_buf, y_buf)
                        
                # Update policy weights
                if m == 'Global Adaptive Ensemble':
                    policy_lr = 0.1
                    exp_accs = np.exp(m_accs * 5.0)
                    new_w = exp_accs / np.sum(exp_accs)
                    policies[m] = (1.0 - policy_lr) * policies[m] + policy_lr * new_w
                    policies[m] /= np.sum(policies[m])
                elif m != 'Fixed Ensemble':
                    exp_accs = np.exp(m_accs * 3.0)
                    new_w = exp_accs / np.sum(exp_accs)
                    policies[m] = 0.8 * policies[m] + 0.2 * new_w
                    policies[m] /= np.sum(policies[m])
                    
                # Save concept memory at the end of concept block (e.g. w_id = 9, 19, 29...)
                if (w_id + 1) % 10 == 0:
                    fp_block = compute_window_fingerprint(X_buf, y_buf)
                    if m == 'Oracle Recurrence Transfer':
                        memory_oracle[concept_id] = {
                            'policy': policies[m].copy(),
                            'models': [m_saved for m_saved in active_models]
                        }
                    elif m == 'Similarity Policy Transfer':
                        memory_similarity_policy.append({
                            'fp': fp_block.copy(),
                            'policy': policies[m].copy(),
                            'concept_id': concept_id
                        })
                    elif m == 'Similarity Model-State Transfer':
                        memory_similarity_model.append({
                            'fp': fp_block.copy(),
                            'policy': policies[m].copy(),
                            'models': [m_saved for m_saved in active_models],
                            'concept_id': concept_id
                        })
                        
                cpu_times[m] += time.time() - t0
                
                results[m].append({
                    'seed': self.seed,
                    'method': m,
                    'window_id': w_id,
                    'block_idx': w['block_idx'],
                    'concept_id': concept_id,
                    'is_recurrence': w['is_recurrence'],
                    'window_in_block': w['window_in_block'],
                    'macro_f1': metrics['macro_f1'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'w_rf': float(policies[m][0]),
                    'w_et': float(policies[m][1]),
                    'w_gb': float(policies[m][2]),
                    'tp': metrics['tp'], 'tn': metrics['tn'],
                    'fp': metrics['fp'], 'fn': metrics['fn']
                })
                
        df_results_list = []
        for m in methods:
            df_m = pd.DataFrame(results[m])
            df_m['cpu_time'] = cpu_times[m]
            df_m['retrain_count'] = retrain_counts[m]
            df_results_list.append(df_m)
            
        df_all = pd.concat(df_results_list, ignore_index=True)
        return df_all, pd.DataFrame(transfer_logs)


# ------------------------------------------------------------------------------
# BENCHMARK RUNNER & COMPILATION
# ------------------------------------------------------------------------------
def run_sea_experiment(seeds=[42, 43, 44, 45, 46], sim_threshold=0.97):
    print("=" * 80)
    print("RUNNING SEA RECURRING-CONCEPT POLICY TRANSFER BENCHMARK (CORRECTED)")
    print(f"Seeds: {seeds} | Window Size: 500 | Concept Sequence: A->B->C (x3)")
    print("=" * 80)
    
    all_results = []
    all_transfers = []
    
    for seed in seeds:
        print(f"[Seed {seed}] Running 6 methods over 90 windows...")
        runner = SEABenchmarkRunner(seed=seed, sim_threshold=sim_threshold)
        df_res, df_trans = runner.run()
        all_results.append(df_res)
        all_transfers.append(df_trans)
        
    df_all_results = pd.concat(all_results, ignore_index=True)
    df_all_transfers = pd.concat(all_transfers, ignore_index=True) if len(all_transfers) > 0 else pd.DataFrame()
    
    # Create results directories
    res_dir = os.path.join(os.path.dirname(__file__), 'results', 'sea_transfer')
    fig_dir = os.path.join(res_dir, 'figures')
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Save raw window results
    df_all_results.to_csv(os.path.join(res_dir, 'sea_window_metrics.csv'), index=False)
    
    # 1. Summary Metrics per Method
    summary_list = []
    for method, grp in df_all_results.groupby('method'):
        summary_list.append({
            'method': method,
            'macro_f1_mean': grp['macro_f1'].mean(),
            'macro_f1_std': grp['macro_f1'].std(),
            'accuracy_mean': grp['accuracy'].mean(),
            'accuracy_std': grp['accuracy'].std(),
            'precision_mean': grp['precision'].mean(),
            'precision_std': grp['precision'].std(),
            'recall_mean': grp['recall'].mean(),
            'recall_std': grp['recall'].std(),
            'cpu_time_mean': grp.groupby('seed')['cpu_time'].first().mean(),
            'retrain_count_mean': grp.groupby('seed')['retrain_count'].first().mean(),
            'tp_total': grp['tp'].sum(),
            'tn_total': grp['tn'].sum(),
            'fp_total': grp['fp'].sum(),
            'fn_total': grp['fn'].sum()
        })
    df_summary = pd.DataFrame(summary_list).sort_values(by='macro_f1_mean', ascending=False)
    df_summary.to_csv(os.path.join(res_dir, 'sea_summary_metrics.csv'), index=False)
    
    # 2. Recurrence Recovery Curves (Blocks 3, 4, 5, 6, 7, 8)
    df_recurrence = df_all_results[df_all_results['is_recurrence'] == True].copy()
    recovery_list = []
    for (method, win_in_block), grp in df_recurrence.groupby(['method', 'window_in_block']):
        recovery_list.append({
            'method': method,
            'window_after_recurrence': win_in_block + 1,
            'rec_f1_mean': grp['macro_f1'].mean(),
            'rec_f1_std': grp['macro_f1'].std(),
            'rec_acc_mean': grp['accuracy'].mean(),
            'rec_acc_std': grp['accuracy'].std()
        })
    df_recovery = pd.DataFrame(recovery_list).sort_values(by=['method', 'window_after_recurrence'])
    df_recovery.to_csv(os.path.join(res_dir, 'sea_recovery_curves.csv'), index=False)
    
    # 3. Statistical Tests
    stat_tests = compute_paired_statistical_tests(df_all_results)
    with open(os.path.join(res_dir, 'sea_statistical_tests.json'), 'w') as f:
        json.dump(stat_tests, f, indent=4)
        
    # 4. Decision Logic
    decision = evaluate_pre_registered_hypotheses(df_summary, df_recovery, stat_tests)
    with open(os.path.join(res_dir, 'sea_decision_logic.json'), 'w') as f:
        json.dump(decision, f, indent=4)
        
    print("\n" + "=" * 80)
    print("SEA BENCHMARK SUMMARY RESULTS (CORRECTED)")
    print("=" * 80)
    print(df_summary[['method', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean', 'cpu_time_mean', 'retrain_count_mean']].to_string(index=False))
    print("\n" + "=" * 80)
    print(f"DECISION: {decision['decision_case']}")
    print(f"RECOMMENDATION: {decision['recommendation']}")
    print("=" * 80)
    
    return df_all_results, df_summary, df_recovery, stat_tests, decision


def compute_paired_statistical_tests(df_all):
    methods = [
        'Cold Adaptation Baseline',
        'Similarity Policy Transfer',
        'Similarity Model-State Transfer',
        'Oracle Recurrence Transfer'
    ]
    
    tests = {}
    baseline_df = df_all[df_all['method'] == 'Cold Adaptation Baseline'].sort_values(by=['seed', 'window_id'])
    y_base = baseline_df['macro_f1'].values
    
    for m in methods[1:]:
        m_df = df_all[df_all['method'] == m].sort_values(by=['seed', 'window_id'])
        y_m = m_df['macro_f1'].values
        
        diff = y_m - y_base
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))
        
        rng = np.random.RandomState(42)
        boot_means = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(1000)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        
        try:
            stat, p_val = wilcoxon(y_m, y_base)
            p_val = float(p_val)
        except Exception:
            p_val = 1.0
            
        effect_size = mean_diff / std_diff if std_diff > 0 else 0.0
        
        tests[f"Cold_vs_{m.replace(' ', '')}"] = {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_val,
            'effect_size': effect_size,
            'n_obs': len(diff)
        }
        
    p_df = df_all[df_all['method'] == 'Similarity Policy Transfer'].sort_values(by=['seed', 'window_id'])
    ms_df = df_all[df_all['method'] == 'Similarity Model-State Transfer'].sort_values(by=['seed', 'window_id'])
    y_p = p_df['macro_f1'].values
    y_ms = ms_df['macro_f1'].values
    diff_ms_p = y_ms - y_p
    try:
        _, p_val_ms_p = wilcoxon(y_ms, y_p)
        p_val_ms_p = float(p_val_ms_p)
    except Exception:
        p_val_ms_p = 1.0
        
    tests["PolicyTransfer_vs_ModelStateTransfer"] = {
        'mean_diff': float(np.mean(diff_ms_p)),
        'std_diff': float(np.std(diff_ms_p)),
        'p_value': p_val_ms_p,
        'effect_size': float(np.mean(diff_ms_p) / np.std(diff_ms_p)) if np.std(diff_ms_p) > 0 else 0.0,
        'n_obs': len(diff_ms_p)
    }
    
    return tests


def evaluate_pre_registered_hypotheses(df_summary, df_recovery, stat_tests):
    f1_dict = dict(zip(df_summary['method'], df_summary['macro_f1_mean']))
    
    f1_cold = f1_dict.get('Cold Adaptation Baseline', 0.0)
    f1_policy = f1_dict.get('Similarity Policy Transfer', 0.0)
    f1_model = f1_dict.get('Similarity Model-State Transfer', 0.0)
    f1_oracle = f1_dict.get('Oracle Recurrence Transfer', 0.0)
    
    gap_policy = f1_policy - f1_cold
    gap_model = f1_model - f1_cold
    gap_oracle = f1_oracle - f1_cold
    
    p_model = stat_tests.get('Cold_vs_SimilarityModel-StateTransfer', {}).get('p_value', 1.0)
    p_policy = stat_tests.get('Cold_vs_SimilarityPolicyTransfer', {}).get('p_value', 1.0)
    
    rec1_cold = df_recovery[(df_recovery['method'] == 'Cold Adaptation Baseline') & (df_recovery['window_after_recurrence'] == 1)]['rec_f1_mean'].values[0]
    rec1_model = df_recovery[(df_recovery['method'] == 'Similarity Model-State Transfer') & (df_recovery['window_after_recurrence'] == 1)]['rec_f1_mean'].values[0]
    rec1_policy = df_recovery[(df_recovery['method'] == 'Similarity Policy Transfer') & (df_recovery['window_after_recurrence'] == 1)]['rec_f1_mean'].values[0]
    rec1_oracle = df_recovery[(df_recovery['method'] == 'Oracle Recurrence Transfer') & (df_recovery['window_after_recurrence'] == 1)]['rec_f1_mean'].values[0]
    
    rec1_gain_model = rec1_model - rec1_cold
    rec1_gain_policy = rec1_policy - rec1_cold
    rec1_gain_oracle = rec1_oracle - rec1_cold
    
    h1_supported = (gap_model > 0.01 and p_model < 0.05 and rec1_gain_model > 0.02)
    h2_supported = (gap_policy > 0.005 and gap_model > gap_policy)
    h3_supported = (abs(f1_model - f1_oracle) < 0.01)
    
    if h1_supported and h3_supported:
        decision_case = "Case A (RAPT Fully Justified)"
        recommendation = "Proceed to full RAPT implementation. Model-state transfer under recurring concepts produces statistically significant recovery acceleration (+{:.4f} F1 at Window +1, p={:.4e}).".format(rec1_gain_model, p_model)
    elif h2_supported and not h1_supported:
        decision_case = "Case B (Policy Transfer Only)"
        recommendation = "Policy weight transfer provides benefit (+{:.4f} F1), but model-state transfer provides no additional gain. Focus RAPT design on policy weight transfer.".format(gap_policy)
    else:
        decision_case = "Case C (Similarity / Transfer Limitation)"
        recommendation = "Model-state transfer provides modest or insignificant gain (+{:.4f} F1, p={:.4f}). Inspect fingerprint similarity threshold or feature space representation.".format(gap_model, p_model)
        
    return {
        'decision_case': decision_case,
        'recommendation': recommendation,
        'h1_supported': bool(h1_supported),
        'h2_supported': bool(h2_supported),
        'h3_supported': bool(h3_supported),
        'f1_cold': f1_cold,
        'f1_policy': f1_policy,
        'f1_model': f1_model,
        'f1_oracle': f1_oracle,
        'gap_policy': gap_policy,
        'gap_model': gap_model,
        'gap_oracle': gap_oracle,
        'rec1_cold': float(rec1_cold),
        'rec1_model': float(rec1_model),
        'rec1_policy': float(rec1_policy),
        'rec1_oracle': float(rec1_oracle),
        'rec1_gain_model': float(rec1_gain_model),
        'rec1_gain_policy': float(rec1_gain_policy),
        'rec1_gain_oracle': float(rec1_gain_oracle),
        'p_model': p_model,
        'p_policy': p_policy
    }


if __name__ == '__main__':
    run_sea_experiment()
