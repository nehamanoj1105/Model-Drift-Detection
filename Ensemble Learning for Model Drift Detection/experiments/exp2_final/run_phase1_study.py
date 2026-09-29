"""
Phase 1 Master Study Runner for exp2_final:
Runs S1 Main, N1 Noise, N2 Fresh Concepts, and S2 9B (Exp1 Protocol) across 5 seeds (42, 43, 44, 45, 46).
Evaluates all methods under Policy B (X_init + X_regime_history):
- Frozen Baseline
- Event-Driven Baseline
- Local Retraining Baseline
- Probability-Guided Transfer
- Dynamic Oracle Transfer
- Softmax Similarity-Weighted Blending
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loaders import load_s1_stream
from similarity import predict_softmax_weighted
from oracle import DynamicOracleSelector
from gates import verify_g1_leakage, verify_g2_reproduction

class HeterogeneousBaseEnsemblePolicyB:
    """Exp1 Heterogeneous BaseEnsemble under Policy B (X_init + X_regime)."""
    def __init__(self, seed=42):
        self.rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=seed, n_jobs=1)
        self.et = ExtraTreesClassifier(n_estimators=30, max_depth=10, random_state=seed, n_jobs=1)
        self.hgb = HistGradientBoostingClassifier(max_iter=30, max_depth=5, random_state=seed)

    def fit(self, X_init, y_init, X_regime=None, y_regime=None):
        if X_regime is not None and len(X_regime) > 0:
            X_concat = np.concatenate([X_init, X_regime], axis=0)
            y_concat = np.concatenate([y_init, y_regime], axis=0)
        else:
            X_concat, y_concat = X_init, y_init
            
        self.rf.fit(X_concat, y_concat)
        self.et.fit(X_concat, y_concat)
        self.hgb.fit(X_concat, y_concat)
        return self

    def predict(self, X):
        p1 = self.rf.predict_proba(X)
        p2 = self.et.predict_proba(X)
        p3 = self.hgb.predict_proba(X)
        p_avg = (p1 + p2 + p3) / 3.0
        return np.argmax(p_avg, axis=1)

    def predict_proba(self, X):
        p1 = self.rf.predict_proba(X)
        p2 = self.et.predict_proba(X)
        p3 = self.hgb.predict_proba(X)
        return (p1 + p2 + p3) / 3.0

class OnlineTransferabilityEstimator:
    """Strict online rolling-origin LogisticRegression estimator for transfer prediction."""
    def __init__(self, tau=0.60, seed=42):
        self.tau = tau
        self.seed = seed
        self.is_fitted = False
        self._model = None
        self._scaler = StandardScaler()
        self._scaler_fitted = False
        self._X_history = []
        self._y_history = []
        self._dp_history = []

    def record_outcome(self, dp_idx, features, delta_f1):
        label = 1 if delta_f1 > 0.010 else 0
        self._dp_history.append(dp_idx)
        self._X_history.append(np.array(features, dtype=np.float64))
        self._y_history.append(label)

    def _refit(self, current_dp_idx):
        valid_mask = [dp < current_dp_idx for dp in self._dp_history]
        if not any(valid_mask):
            self.is_fitted = False
            return

        X_train = np.array([x for x, m in zip(self._X_history, valid_mask) if m])
        y_train = np.array([y for y, m in zip(self._y_history, valid_mask) if m])
        n_pos = int(y_train.sum())

        if n_pos < 3 or len(np.unique(y_train)) < 2 or len(y_train) < 6:
            self.is_fitted = False
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        self._scaler_fitted = True

        try:
            self._model = LogisticRegression(
                random_state=self.seed,
                max_iter=500,
                class_weight='balanced',
                C=0.1,
                solver='lbfgs'
            )
            self._model.fit(X_scaled, y_train)
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def predict_proba_positive(self, features, current_dp_idx):
        self._refit(current_dp_idx)
        if not self.is_fitted or self._model is None:
            return 0.50
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        if self._scaler_fitted:
            X = self._scaler.transform(X)
        try:
            proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
        except Exception:
            return 0.50

    def should_transfer(self, p_pos):
        return p_pos >= self.tau

def run_single_seed_evaluation(df, feature_cols, target_col, regime_col, K, H, warmup, preproc_initial=False, seed=42, eps=0.010):
    N = len(df)
    if preproc_initial:
        scaler = StandardScaler()
        X_raw = df[feature_cols].values.astype(np.float64)
        scaler.fit(X_raw[:warmup])
        X_all = scaler.transform(X_raw)
    else:
        X_all = df[feature_cols].values.astype(np.float64)
        
    y_all = df[target_col].values.astype(int)
    regimes = df[regime_col].values
    
    X_init = X_all[:warmup]
    y_init = y_all[:warmup]
    
    # Fit Initial / Frozen Model
    init_m = HeterogeneousBaseEnsemblePolicyB(seed=seed).fit(X_init, y_init)
    init_m.regime_id = regimes[0]
    y_pred_frozen = init_m.predict(X_all)
    frozen_f1 = float(f1_score(y_all[warmup:], y_pred_frozen[warmup:], average='macro', zero_division=0))
    
    # Identify regime boundaries
    seg_starts = [warmup]
    for idx in range(warmup + 1, N):
        if regimes[idx] != regimes[idx - 1]:
            seg_starts.append(idx)
    seg_starts.append(N)
    
    # Precompute regime mean features
    unique_regimes = set(regimes)
    cand_means = {}
    for r_id in unique_regimes:
        idx_r_all = np.where(regimes == r_id)[0]
        if len(idx_r_all) > 0:
            cand_means[r_id] = np.mean(X_all[idx_r_all], axis=0)

    # Event-Driven Stream Predictions (Policy B)
    ed_predictions = list(y_pred_frozen[:warmup])
    ed_curr_model = init_m
    for i in range(len(seg_starts) - 1):
        s_idx, e_idx = seg_starts[i], seg_starts[i + 1]
        curr_r = regimes[s_idx]
        if s_idx > warmup:
            idx_r = np.where(regimes[:s_idx] == curr_r)[0]
            if len(idx_r) >= 20:
                ed_curr_model = HeterogeneousBaseEnsemblePolicyB(seed=seed).fit(X_init, y_init, X_all[idx_r], y_all[idx_r])
                ed_curr_model.regime_id = curr_r
        preds_seg = ed_curr_model.predict(X_all[s_idx:e_idx])
        ed_predictions.extend(preds_seg)
    ed_f1 = float(f1_score(y_all[warmup:], ed_predictions[warmup:], average='macro', zero_division=0))

    # Local Retraining Stream Predictions (Policy B)
    local_retrain_preds = list(y_pred_frozen[:warmup])
    local_cache = {}
    pool_checkpoints = {regimes[0]: init_m}
    fit_counts = {}

    for i in range(len(seg_starts) - 1):
        s_idx, e_idx = seg_starts[i], seg_starts[i + 1]
        curr_r = regimes[s_idx]
        idx_r = np.where(regimes[:s_idx] == curr_r)[0]
        if len(idx_r) >= 20:
            cache_key = (curr_r, len(idx_r))
            if cache_key not in local_cache:
                m_fit = HeterogeneousBaseEnsemblePolicyB(seed=seed).fit(X_init, y_init, X_all[idx_r], y_all[idx_r])
                m_fit.regime_id = curr_r
                local_cache[cache_key] = m_fit
            local_curr_m = local_cache[cache_key]
        else:
            local_curr_m = pool_checkpoints.get(curr_r, init_m)
        preds_seg = local_curr_m.predict(X_all[s_idx:e_idx])
        local_retrain_preds.extend(preds_seg)
    local_retrain_f1 = float(f1_score(y_all[warmup:], local_retrain_preds[warmup:], average='macro', zero_division=0))

    # Decision Points Simulation for Online Methods
    decision_points = list(range(warmup, N - H, K))
    g1_log = []
    
    pred_cache = {}
    def get_cand_pred(m_obj, dp_idx):
        k_cand = (id(m_obj), dp_idx)
        if k_cand not in pred_cache:
            pred_cache[k_cand] = m_obj.predict(X_all[dp_idx:dp_idx+H])
        return pred_cache[k_cand]

    prob_estimator = OnlineTransferabilityEstimator(tau=0.60, seed=seed)
    oracle_selector = DynamicOracleSelector()

    oracle_f1_list = []
    prob_guided_f1_list = []
    similarity_weighted_f1_list = []

    all_prob_preds = []
    all_prob_labels = []

    for dp in decision_points:
        curr_r = regimes[dp]
        
        idx_r = np.where(regimes[:dp] == curr_r)[0]
        if len(idx_r) >= 20:
            if curr_r not in fit_counts or (len(idx_r) - fit_counts[curr_r]) >= max(500, K * 10):
                m_loc = HeterogeneousBaseEnsemblePolicyB(seed=seed).fit(X_init, y_init, X_all[idx_r], y_all[idx_r])
                m_loc.regime_id = curr_r
                pool_checkpoints[curr_r] = m_loc
                fit_counts[curr_r] = len(idx_r)
            local_m = pool_checkpoints[curr_r]
        else:
            local_m = pool_checkpoints.get(curr_r, init_m)
            
        g1_log.append({
            'window_idx': dp,
            'eval_window_end': dp + H,
            'fit_max_window_idx': max(idx_r) if len(idx_r) > 0 else None
        })
        
        candidates = [m for r_id, m in pool_checkpoints.items() if r_id != curr_r]
        X_eval = X_all[dp:dp+H]
        y_eval = y_all[dp:dp+H]

        y_pred_loc = local_m.predict(X_eval)
        f1_loc = f1_score(y_eval, y_pred_loc, average='macro', zero_division=0)
        curr_mean_feat = np.mean(X_all[max(0, dp-20):dp], axis=0)

        if not candidates:
            oracle_f1_list.append(f1_loc)
            prob_guided_f1_list.append(f1_loc)
            similarity_weighted_f1_list.append(f1_loc)
            continue

        # 1. Dynamic Oracle
        best_cand_oracle, best_oracle_f1 = oracle_selector.evaluate_and_select(candidates, X_eval, y_eval, local_model=local_m)
        oracle_f1_list.append(best_oracle_f1)

        # 2. Probability-Guided Online Transfer Selection
        cand_probs = []
        cand_features = []
        for cand in candidates:
            cand_r_id = getattr(cand, 'regime_id', None)
            dist = np.linalg.norm(curr_mean_feat - cand_means[cand_r_id]) if cand_r_id in cand_means else 1.0
            feat_vec = [dist, f1_loc]
            cand_features.append((cand, feat_vec, dist))
            p_pos = prob_estimator.predict_proba_positive(feat_vec, current_dp_idx=dp)
            cand_probs.append(p_pos)

        best_cand_idx = np.argmax(cand_probs)
        best_p_val = cand_probs[best_cand_idx]

        if prob_estimator.should_transfer(best_p_val):
            selected_cand = candidates[best_cand_idx]
            y_pred_prob = get_cand_pred(selected_cand, dp)
        else:
            y_pred_prob = y_pred_loc
            
        prob_f1 = f1_score(y_eval, y_pred_prob, average='macro', zero_division=0)
        prob_guided_f1_list.append(prob_f1)

        # 3. Softmax Similarity-Weighted Blending
        sim_scores = {}
        for idx_c, (cand, feat_vec, dist) in enumerate(cand_features):
            sim_scores[idx_c] = float(1.0 / (1.0 + dist))
        preds_sim = predict_softmax_weighted(candidates, sim_scores, X_eval)
        sim_f1 = f1_score(y_eval, preds_sim, average='macro', zero_division=0)
        similarity_weighted_f1_list.append(sim_f1)

        # Post-Evaluation: Record Outcomes into Estimator History (Strictly Causal)
        for idx_c, (cand, feat_vec, dist) in enumerate(cand_features):
            y_pred_cand = get_cand_pred(cand, dp)
            f1_cand = f1_score(y_eval, y_pred_cand, average='macro', zero_division=0)
            diff = f1_cand - f1_loc
            prob_estimator.record_outcome(dp, feat_vec, diff)
            all_prob_preds.append(cand_probs[idx_c])
            all_prob_labels.append(1 if diff > eps else 0)

    # Calculate overall AUROC and 95% Bootstrap CI
    if len(all_prob_labels) > 20 and len(set(all_prob_labels)) > 1:
        try:
            prob_auroc = float(roc_auc_score(all_prob_labels, all_prob_preds))
        except Exception:
            prob_auroc = 0.50
            
        rng_bs = np.random.default_rng(seed)
        bs_aurocs = []
        y_arr = np.array(all_prob_labels)
        p_arr = np.array(all_prob_preds)
        for _ in range(100):
            bs_idx = rng_bs.choice(len(y_arr), size=len(y_arr), replace=True)
            if len(set(y_arr[bs_idx])) > 1:
                bs_aurocs.append(roc_auc_score(y_arr[bs_idx], p_arr[bs_idx]))
        auroc_ci_low = float(np.percentile(bs_aurocs, 2.5)) if bs_aurocs else 0.50
        auroc_ci_high = float(np.percentile(bs_aurocs, 97.5)) if bs_aurocs else 0.50
    else:
        prob_auroc = 0.50
        auroc_ci_low = 0.50
        auroc_ci_high = 0.50

    oracle_macro_f1 = float(np.mean(oracle_f1_list)) if oracle_f1_list else frozen_f1
    prob_guided_macro_f1 = float(np.mean(prob_guided_f1_list)) if prob_guided_f1_list else frozen_f1
    similarity_weighted_macro_f1 = float(np.mean(similarity_weighted_f1_list)) if similarity_weighted_f1_list else frozen_f1

    # Gate G1 verification
    g1_res = verify_g1_leakage(g1_log)

    return {
        'seed': seed,
        'frozen_f1': frozen_f1,
        'event_driven_f1': ed_f1,
        'local_retrain_f1': local_retrain_f1,
        'probability_guided_f1': prob_guided_macro_f1,
        'oracle_f1': oracle_macro_f1,
        'similarity_weighted_f1': similarity_weighted_macro_f1,
        'auroc': prob_auroc,
        'auroc_ci_low': auroc_ci_low,
        'auroc_ci_high': auroc_ci_high,
        'g1_passed': g1_res['passed'],
    }

def run_phase1_study():
    print("=== STARTING EXPERIMENT 2 FINAL - PHASE 1 STUDY ===", flush=True)
    seeds = [42, 43, 44, 45, 46]
    
    results = {}
    
    # 1. S1 Main
    print("\n--- Running S1 Main (Synthetic Concept Recurrence) ---", flush=True)
    s1_seed_results = []
    for s in seeds:
        df_s1, hash_s1 = load_s1_stream(seed=s, variant="S1_Main")
        feat_s1 = [c for c in df_s1.columns if c.startswith('f_')]
        res = run_single_seed_evaluation(df_s1, feat_s1, 'target', 'regime', K=10, H=25, warmup=100, seed=s)
        res['hash_full'] = hash_s1
        s1_seed_results.append(res)
        print(f"Seed {s}: Frozen={res['frozen_f1']:.4f}, ED={res['event_driven_f1']:.4f}, Local={res['local_retrain_f1']:.4f}, ProbGuided={res['probability_guided_f1']:.4f}, Oracle={res['oracle_f1']:.4f}", flush=True)

    # 2. N1 Noise Null
    print("\n--- Running N1 Noise (Null Stream) ---", flush=True)
    n1_seed_results = []
    for s in seeds:
        df_n1, hash_n1 = load_s1_stream(seed=s, variant="N1_Noise")
        res = run_single_seed_evaluation(df_n1, feat_s1, 'target', 'regime', K=10, H=25, warmup=100, seed=s)
        res['hash_full'] = hash_n1
        n1_seed_results.append(res)
        print(f"Seed {s}: Frozen={res['frozen_f1']:.4f}, Local={res['local_retrain_f1']:.4f}, ProbGuided={res['probability_guided_f1']:.4f}, AUROC={res['auroc']:.4f} [{res['auroc_ci_low']:.4f}, {res['auroc_ci_high']:.4f}]", flush=True)

    # 3. N2 Fresh Concepts Null
    print("\n--- Running N2 Fresh Concepts (Null Stream) ---", flush=True)
    n2_seed_results = []
    for s in seeds:
        df_n2, hash_n2 = load_s1_stream(seed=s, variant="N2_FreshConcepts")
        res = run_single_seed_evaluation(df_n2, feat_s1, 'target', 'regime', K=10, H=25, warmup=100, seed=s)
        res['hash_full'] = hash_n2
        n2_seed_results.append(res)
        print(f"Seed {s}: Frozen={res['frozen_f1']:.4f}, Local={res['local_retrain_f1']:.4f}, ProbGuided={res['probability_guided_f1']:.4f}, AUROC={res['auroc']:.4f} [{res['auroc_ci_low']:.4f}, {res['auroc_ci_high']:.4f}]", flush=True)

    # 4. S2 9B (Exp1 Protocol: n_init=99, StreamingPreprocessor)
    print("\n--- Running S2 9B Original (Exp1 Protocol: n_init=99) ---", flush=True)
    path_9b = os.path.join(_HERE, "..", "exp9b", "data", "processed_exp9b_stream.csv")
    s2_seed_results = []
    for s in seeds:
        df_s2 = pd.read_csv(path_9b)
        feat_s2 = ['mean_latency', 'std_latency', 'p90_latency', 'packet_loss_rate', 'effective_throughput']
        target_s2 = 'qos_target'
        regime_s2 = 'regime_id' if 'regime_id' in df_s2.columns else 'regime_letter'
        res = run_single_seed_evaluation(df_s2, feat_s2, target_s2, regime_s2, K=2, H=20, warmup=99, preproc_initial=True, seed=s)
        res['hash_full'] = "Local_CSV_9B_Exp1_Protocol"
        s2_seed_results.append(res)
        print(f"Seed {s}: Frozen={res['frozen_f1']:.4f}, ED={res['event_driven_f1']:.4f}, Local={res['local_retrain_f1']:.4f}, ProbGuided={res['probability_guided_f1']:.4f}, Oracle={res['oracle_f1']:.4f}", flush=True)

    # Aggregating Macro F1 (Mean ± Std across 5 seeds)
    def aggregate_metrics(seed_list):
        keys = ['frozen_f1', 'event_driven_f1', 'local_retrain_f1', 'probability_guided_f1', 'oracle_f1', 'similarity_weighted_f1', 'auroc']
        agg = {}
        for k in keys:
            vals = [x[k] for x in seed_list if k in x]
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        agg['auroc_ci_low'] = float(np.mean([x['auroc_ci_low'] for x in seed_list]))
        agg['auroc_ci_high'] = float(np.mean([x['auroc_ci_high'] for x in seed_list]))
        agg['g1_passed_all'] = all(x['g1_passed'] for x in seed_list)
        return agg

    summary = {
        'S1_Main': {'seed_results': s1_seed_results, 'aggregated': aggregate_metrics(s1_seed_results)},
        'N1_Noise': {'seed_results': n1_seed_results, 'aggregated': aggregate_metrics(n1_seed_results)},
        'N2_FreshConcepts': {'seed_results': n2_seed_results, 'aggregated': aggregate_metrics(n2_seed_results)},
        'S2_9B': {'seed_results': s2_seed_results, 'aggregated': aggregate_metrics(s2_seed_results)},
    }

    # Verify G2 Reproduction on S2 9B (Exp1 Ref: ED=0.9350, Frozen=0.8961)
    g2_ed_res = verify_g2_reproduction(summary['S2_9B']['aggregated']['event_driven_f1']['mean'], 0.9350, tolerance=0.005)
    summary['S2_9B']['g2_reproduction'] = g2_ed_res

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    out_file = os.path.join(_HERE, "results", "phase1_summary.json")

    def default_converter(o):
        if hasattr(o, 'item'):
            return o.item()
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=default_converter)
        f.flush()
        os.fsync(f.fileno())

    print("\n==========================================================================================", flush=True)
    print("=== PHASE 1 STUDY SUMMARY TABLE (Policy B, 5-Seed Averages) ===", flush=True)
    print("==========================================================================================", flush=True)
    print(f"{'Stream ID':<18} | {'Frozen':<12} | {'Event-Driven':<12} | {'Local Retrain':<12} | {'Prob-Guided':<12} | {'Oracle':<12} | {'Softmax Sim':<12}")
    print("-" * 105)
    for name, s_data in summary.items():
        agg = s_data['aggregated']
        print(f"{name:<18} | {agg['frozen_f1']['mean']:.4f}±{agg['frozen_f1']['std']:.4f} | {agg['event_driven_f1']['mean']:.4f}±{agg['event_driven_f1']['std']:.4f} | {agg['local_retrain_f1']['mean']:.4f}±{agg['local_retrain_f1']['std']:.4f} | {agg['probability_guided_f1']['mean']:.4f}±{agg['probability_guided_f1']['std']:.4f} | {agg['oracle_f1']['mean']:.4f}±{agg['oracle_f1']['std']:.4f} | {agg['similarity_weighted_f1']['mean']:.4f}±{agg['similarity_weighted_f1']['std']:.4f}")
    print("=" * 105, flush=True)

    print(f"\nPhase 1 JSON summary saved to {out_file}.")
    print("\n*** STOPPING AFTER S2 FOR USER REVIEW AS INSTRUCTED ***\n")

if __name__ == '__main__':
    run_phase1_study()
