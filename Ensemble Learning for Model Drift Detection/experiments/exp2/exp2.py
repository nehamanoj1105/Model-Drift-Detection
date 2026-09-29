"""
Master Experiment 2 System Controller
Executes online prequential streaming benchmarks across all 11 methods, maintaining strict zero-look-ahead anti-leakage boundaries.
Logs episode transfer outcomes, prediction results, CPU adaptation costs, and probability calibration statistics.
"""

import os
import sys
sys.path.insert(0, os.getcwd())

import time
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score

from experiments.exp2.regime_representation import RegimeRepresentationExtractor
from experiments.exp2.similarity import SimilarityRanker
from experiments.exp2.transferability import ProbabilisticTransferEstimator

class RegimeCheckpointExp2:
    """
    Stores historical regime checkpoint metadata, models, representation vector, and reuse history.
    """
    def __init__(self, checkpoint_id, regime_key, representation, rf_model, et_model, weights, created_window):
        self.checkpoint_id = checkpoint_id
        self.regime_key = regime_key
        self.representation = representation
        self.rf_model = copy.deepcopy(rf_model)
        self.et_model = copy.deepcopy(et_model)
        self.weights = list(weights)
        self.created_window = created_window
        
        # Reuse & Transfer Statistics
        self.observation_count = 1
        self.successful_transfers = 0
        self.failed_transfers = 0
        self.historical_f1 = 1.0
        self.historical_acc = 1.0

    def update_history(self, delta_f1, delta_thresh=0.005):
        self.observation_count += 1
        if delta_f1 > delta_thresh:
            self.successful_transfers += 1
        elif delta_f1 < -delta_thresh:
            self.failed_transfers += 1

    def to_metadata_dict(self, current_window):
        return {
            'checkpoint_id': self.checkpoint_id,
            'regime_key': self.regime_key,
            'weights': self.weights,
            'created_window': self.created_window,
            'observation_count': self.observation_count,
            'successful_transfers': self.successful_transfers,
            'failed_transfers': self.failed_transfers,
            'historical_f1': self.historical_f1,
            'historical_acc': self.historical_acc,
            'checkpoint_age': current_window - self.created_window
        }


class RAPTExp2Runner:
    """
    Streaming prequential evaluation engine for Experiment 2.
    Supports 11 benchmark methods with strict anti-leakage guarantees.
    """
    def __init__(self, feature_cols, target_col, dataset_type='9a', seed=42, tau=0.60, delta_thresh=0.005, horizon=50):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.dataset_type = dataset_type
        self.seed = seed
        self.tau = tau
        self.delta_thresh = delta_thresh
        self.horizon = horizon

        self.rep_extractor = RegimeRepresentationExtractor(feature_cols)
        self.similarity_ranker = SimilarityRanker(feature_cols)
        self.transfer_estimator = ProbabilisticTransferEstimator(feature_cols, tau=tau)

    def _train_ensemble(self, X_train, y_train):
        rf = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=self.seed, n_jobs=1)
        et = ExtraTreesClassifier(n_estimators=20, max_depth=8, random_state=self.seed, n_jobs=1)
        
        # Ensure multi-class support even if single class in small slice
        classes_present = np.unique(y_train)
        rf.fit(X_train, y_train)
        et.fit(X_train, y_train)
        return rf, et

    def _predict_ensemble(self, rf, et, weights, X_eval):
        p_rf = rf.predict_proba(X_eval)
        p_et = et.predict_proba(X_eval)
        
        # Handle shape mismatch if classes differ in training
        n_samples = len(X_eval)
        n_classes = max(p_rf.shape[1], p_et.shape[1], 3)
        
        full_p_rf = np.zeros((n_samples, n_classes))
        full_p_et = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(rf.classes_):
            if c < n_classes:
                full_p_rf[:, c] = p_rf[:, idx]
        for idx, c in enumerate(et.classes_):
            if c < n_classes:
                full_p_et[:, c] = p_et[:, idx]

        p_blend = weights[0] * full_p_rf + weights[1] * full_p_et
        preds = np.argmax(p_blend, axis=1)
        
        # Disagreement & Entropy
        disagreement = float(np.mean(np.argmax(full_p_rf, axis=1) != np.argmax(full_p_et, axis=1)))
        p_safe = np.clip(p_blend, 1e-9, 1.0)
        entropy = float(-np.mean(np.sum(p_safe * np.log(p_safe), axis=1)))
        
        return preds, disagreement, entropy

    def run_stream(self, df_stream, method_name='Probability-Guided Top-1', telemetry_cache=None):
        """
        Executes prequential streaming benchmark for a given method on df_stream.
        """
        np.random.seed(self.seed)
        
        # Results containers
        per_window_results = []
        transfer_pairs = []
        
        # Identify window column name
        win_col = 'stream_window_id' if 'stream_window_id' in df_stream.columns else 'window_id'
        window_ids = sorted(df_stream[win_col].unique())

        # Pre-group window DataFrames for instant O(1) lookup
        window_dfs = {w_id: group for w_id, group in df_stream.groupby(win_col, sort=False)}
        
        # Initial model training on Window 0
        w0_data = window_dfs[window_ids[0]]
        X_w0 = w0_data[self.feature_cols].values
        y_w0 = w0_data[self.target_col].values.astype(int)
        
        t0_start = time.perf_counter()
        active_rf, active_et = self._train_ensemble(X_w0, y_w0)
        active_weights = [0.5, 0.5]
        t0_train_time = time.perf_counter() - t0_start
        
        # Check if w0 representation is cached
        w0_key = (self.dataset_type, window_ids[0])
        if telemetry_cache is not None and w0_key in telemetry_cache:
            w0_rep = telemetry_cache[w0_key]
        else:
            w0_rep = self.rep_extractor.extract_telemetry_features(w0_data)
            if telemetry_cache is not None:
                telemetry_cache[w0_key] = w0_rep
        
        # Checkpoint pool
        checkpoint_pool = []
        ckpt_0 = RegimeCheckpointExp2(
            checkpoint_id=0,
            regime_key=w0_data['regime_label'].iloc[0] if 'regime_label' in w0_data.columns else 'A',
            representation=w0_rep,
            rf_model=active_rf,
            et_model=active_et,
            weights=active_weights,
            created_window=0
        )
        checkpoint_pool.append(ckpt_0)

        # Sliding window buffer for event-driven adaptation / retraining
        recent_buffer_df = [w0_data]
        
        total_adapt_cpu = t0_train_time
        retrain_count = 0
        total_transfers = 0
        positive_transfers = 0
        negative_transfers = 0
        
        # Reset transferability estimator for online learning
        if 'Probability' in method_name:
            self.transfer_estimator = ProbabilisticTransferEstimator(self.feature_cols, tau=self.tau)

        # Telemetry feature cache per window id
        if telemetry_cache is not None:
            self.telemetry_cache = telemetry_cache
        elif not hasattr(self, 'telemetry_cache'):
            self.telemetry_cache = {}

        for idx, w_id in enumerate(window_ids):
            w_df = window_dfs[w_id]
            X_win = w_df[self.feature_cols].values
            y_win = w_df[self.target_col].values.astype(int)
            
            reg_key = w_df['regime_label'].iloc[0] if 'regime_label' in w_df.columns else w_df.get('regime_id', pd.Series(['A'])).iloc[0]
            
            # Detect regime change / transition point
            is_transition = False
            if idx > 0:
                prev_w_df = window_dfs[window_ids[idx-1]]
                prev_reg = prev_w_df['regime_label'].iloc[0] if 'regime_label' in prev_w_df.columns else prev_w_df.get('regime_id', pd.Series(['A'])).iloc[0]
                if reg_key != prev_reg:
                    is_transition = True

            # Step 1: Pre-transfer Telemetry Representation (Cached)
            t_rep_start = time.perf_counter()
            cache_key = (self.dataset_type, w_id)
            if cache_key in self.telemetry_cache:
                target_rep = self.telemetry_cache[cache_key]
            else:
                target_rep = self.rep_extractor.extract_telemetry_features(w_df)
                self.telemetry_cache[cache_key] = target_rep
            
            # Extract candidate checkpoints representation list ONLY on transition windows
            candidate_checkpoints = []
            if is_transition and len(checkpoint_pool) > 0:
                for ckpt in checkpoint_pool:
                    c_meta = ckpt.to_metadata_dict(w_id)
                    preds_cand, dis_cand, ent_cand = self._predict_ensemble(ckpt.rf_model, ckpt.et_model, ckpt.weights, X_win)
                    c_meta['disagreement'] = dis_cand
                    c_meta['entropy'] = ent_cand
                    c_rep = self.rep_extractor.combine_with_model_metadata(ckpt.representation, c_meta, w_id)
                    candidate_checkpoints.append({
                        'checkpoint_obj': ckpt,
                        'representation': c_rep,
                        'weights': ckpt.weights
                    })
            
            rep_cpu = time.perf_counter() - t_rep_start
            
            # Step 2: Decision logic based on method_name
            t_decision_start = time.perf_counter()
            decision = 'KEEP_ACTIVE'
            selected_source_id = None
            pred_prob = 0.0
            chosen_rf, chosen_et = active_rf, active_et
            chosen_weights = active_weights

            if is_transition and method_name != 'Frozen':
                if method_name == 'Full Retraining':
                    decision = 'RETRAIN'
                    X_buf = pd.concat(recent_buffer_df[-3:])[self.feature_cols].values
                    y_buf = pd.concat(recent_buffer_df[-3:])[self.target_col].values.astype(int)
                    chosen_rf, chosen_et = self._train_ensemble(X_buf, y_buf)
                    chosen_weights = [0.5, 0.5]
                    retrain_count += 1
                    
                elif method_name == 'Event-Driven':
                    decision = 'RETRAIN'
                    X_buf = pd.concat(recent_buffer_df[-2:])[self.feature_cols].values
                    y_buf = pd.concat(recent_buffer_df[-2:])[self.target_col].values.astype(int)
                    chosen_rf, chosen_et = self._train_ensemble(X_buf, y_buf)
                    chosen_weights = [0.5, 0.5]
                    retrain_count += 1

                elif method_name == 'RAPT-E':
                    best_ckpt_item, best_sim, _ = self.similarity_ranker.select_top1(target_rep, candidate_checkpoints)
                    if best_sim >= 0.85:
                        decision = 'TRANSFER'
                        ckpt_obj = best_ckpt_item['checkpoint_obj']
                        chosen_rf, chosen_et = ckpt_obj.rf_model, ckpt_obj.et_model
                        chosen_weights = ckpt_obj.weights
                        selected_source_id = ckpt_obj.checkpoint_id
                        total_transfers += 1
                    else:
                        decision = 'RETRAIN'
                        X_buf = pd.concat(recent_buffer_df[-2:])[self.feature_cols].values
                        y_buf = pd.concat(recent_buffer_df[-2:])[self.target_col].values.astype(int)
                        chosen_rf, chosen_et = self._train_ensemble(X_buf, y_buf)
                        chosen_weights = [0.5, 0.5]
                        retrain_count += 1

                elif method_name == 'Similarity-Only':
                    best_ckpt_item, best_sim, _ = self.similarity_ranker.select_top1(target_rep, candidate_checkpoints)
                    if best_sim >= 0.70:
                        decision = 'TRANSFER'
                        ckpt_obj = best_ckpt_item['checkpoint_obj']
                        chosen_rf, chosen_et = ckpt_obj.rf_model, ckpt_obj.et_model
                        chosen_weights = ckpt_obj.weights
                        selected_source_id = ckpt_obj.checkpoint_id
                        total_transfers += 1
                    else:
                        decision = 'ABSTAIN'
                        retrain_count += 1

                elif method_name == 'Similarity-Weighted':
                    blend_info = self.similarity_ranker.compute_weighted_blend(target_rep, candidate_checkpoints)
                    if blend_info and 'weights' in blend_info:
                        decision = 'TRANSFER_WEIGHTED'
                        chosen_weights = blend_info['weights']
                        best_ckpt_item = blend_info['ranked_candidates'][0]['checkpoint']
                        ckpt_obj = best_ckpt_item['checkpoint_obj']
                        chosen_rf, chosen_et = ckpt_obj.rf_model, ckpt_obj.et_model
                        selected_source_id = ckpt_obj.checkpoint_id
                        total_transfers += 1

                elif method_name == 'Historical Reliability':
                    # Choose policy with highest transfer success rate
                    cand_objs = [c['checkpoint_obj'] for c in candidate_checkpoints]
                    cand_objs.sort(key=lambda c: c.successful_transfers / (c.observation_count + 1e-6), reverse=True)
                    best_rel = cand_objs[0]
                    decision = 'TRANSFER'
                    chosen_rf, chosen_et = best_rel.rf_model, best_rel.et_model
                    chosen_weights = best_rel.weights
                    selected_source_id = best_rel.checkpoint_id
                    total_transfers += 1

                elif method_name == 'Random Historical':
                    rand_c = np.random.choice([c['checkpoint_obj'] for c in candidate_checkpoints])
                    decision = 'TRANSFER'
                    chosen_rf, chosen_et = rand_c.rf_model, rand_c.et_model
                    chosen_weights = rand_c.weights
                    selected_source_id = rand_c.checkpoint_id
                    total_transfers += 1

                elif method_name == 'Probability-Guided Top-1':
                    sel_ckpt, dec_res, pred_prob, meta = self.transfer_estimator.select_source(target_rep, candidate_checkpoints)
                    if dec_res == 'TRANSFER' and sel_ckpt:
                        decision = 'TRANSFER'
                        ckpt_obj = sel_ckpt['checkpoint_obj']
                        chosen_rf, chosen_et = ckpt_obj.rf_model, ckpt_obj.et_model
                        chosen_weights = ckpt_obj.weights
                        selected_source_id = ckpt_obj.checkpoint_id
                        total_transfers += 1
                    else:
                        decision = 'ABSTAIN'
                        # Fallback to local retraining
                        X_buf = pd.concat(recent_buffer_df[-2:])[self.feature_cols].values
                        y_buf = pd.concat(recent_buffer_df[-2:])[self.target_col].values.astype(int)
                        chosen_rf, chosen_et = self._train_ensemble(X_buf, y_buf)
                        chosen_weights = [0.5, 0.5]
                        retrain_count += 1

                elif method_name == 'Probability-Guided Weighted':
                    blend_res, dec_res, pred_prob, meta = self.transfer_estimator.compute_probability_weighted_blend(target_rep, candidate_checkpoints)
                    if dec_res == 'TRANSFER_BLENDED' and blend_res:
                        decision = 'TRANSFER_WEIGHTED'
                        chosen_weights = blend_res['weights']
                        best_item = meta['evaluated_candidates'][0]['checkpoint']
                        ckpt_obj = best_item['checkpoint_obj']
                        chosen_rf, chosen_et = ckpt_obj.rf_model, ckpt_obj.et_model
                        selected_source_id = ckpt_obj.checkpoint_id
                        total_transfers += 1
                    else:
                        decision = 'ABSTAIN'
                        X_buf = pd.concat(recent_buffer_df[-2:])[self.feature_cols].values
                        y_buf = pd.concat(recent_buffer_df[-2:])[self.target_col].values.astype(int)
                        chosen_rf, chosen_et = self._train_ensemble(X_buf, y_buf)
                        chosen_weights = [0.5, 0.5]
                        retrain_count += 1

                elif method_name == 'Oracle Transfer':
                    # Post-hoc evaluation of all candidate checkpoints to select true highest future F1 candidate
                    best_f1_oracle = -1.0
                    best_oracle_ckpt = None
                    for c_item in candidate_checkpoints:
                        c_obj = c_item['checkpoint_obj']
                        c_preds, _, _ = self._predict_ensemble(c_obj.rf_model, c_obj.et_model, c_obj.weights, X_win)
                        c_f1 = f1_score(y_win, c_preds, average='macro', zero_division=0)
                        if c_f1 > best_f1_oracle:
                            best_f1_oracle = c_f1
                            best_oracle_ckpt = c_obj
                    
                    if best_oracle_ckpt:
                        decision = 'TRANSFER_ORACLE'
                        chosen_rf, chosen_et = best_oracle_ckpt.rf_model, best_oracle_ckpt.et_model
                        chosen_weights = best_oracle_ckpt.weights
                        selected_source_id = best_oracle_ckpt.checkpoint_id
                        total_transfers += 1

            decision_cpu = time.perf_counter() - t_decision_start
            
            # Step 3: Prequential Prediction
            preds, disagreement, entropy = self._predict_ensemble(chosen_rf, chosen_et, chosen_weights, X_win)
            f1_win = f1_score(y_win, preds, average='macro', zero_division=0)
            acc_win = accuracy_score(y_win, preds)
            
            total_adapt_cpu += (rep_cpu + decision_cpu)
            
            # Update active models
            active_rf, active_et = chosen_rf, chosen_et
            active_weights = chosen_weights
            recent_buffer_df.append(w_df)

            # Step 4: Post-Hoc Transfer Outcome Calculation & Ground Truth Recording
            if is_transition and selected_source_id is not None:
                # Compute baseline local performance without transfer
                X_loc = pd.concat(recent_buffer_df[-2:])[self.feature_cols].values
                y_loc = pd.concat(recent_buffer_df[-2:])[self.target_col].values.astype(int)
                rf_loc, et_loc = self._train_ensemble(X_loc, y_loc)
                preds_loc, _, _ = self._predict_ensemble(rf_loc, et_loc, [0.5, 0.5], X_win)
                f1_baseline = f1_score(y_win, preds_loc, average='macro', zero_division=0)
                
                delta_f1 = f1_win - f1_baseline
                
                if delta_f1 > self.delta_thresh:
                    label = 1
                    positive_transfers += 1
                elif delta_f1 < -self.delta_thresh:
                    label = -1
                    negative_transfers += 1
                else:
                    label = 0
                
                # Retrieve source checkpoint object
                src_ckpt = next((c for c in checkpoint_pool if c.checkpoint_id == selected_source_id), None)
                if src_ckpt:
                    src_ckpt.update_history(delta_f1, delta_thresh=self.delta_thresh)
                    
                    # Pair features for transfer_pairs.csv logging
                    pair_feats = self.transfer_estimator.feature_builder.build_pair_features(
                        src_ckpt.representation, target_rep
                    )
                    
                    transfer_pairs.append({
                        'episode_id': len(transfer_pairs) + 1,
                        'window_id': w_id,
                        'source_regime': src_ckpt.regime_key,
                        'target_regime': reg_key,
                        'source_checkpoint_age': w_id - src_ckpt.created_window,
                        'source_historical_F1': src_ckpt.historical_f1,
                        'source_usage_count': src_ckpt.observation_count,
                        'distribution_distance': pair_feats['dist_euclidean'],
                        'wasserstein_distance': pair_feats['dist_wasserstein'],
                        'JS_distance': pair_feats['dist_js'],
                        'correlation_distance': pair_feats['dist_correlation'],
                        'temporal_distance': pair_feats['dist_temporal'],
                        'combined_distance': pair_feats['dist_combined'],
                        'combined_similarity': pair_feats['sim_combined'],
                        'model_disagreement': disagreement,
                        'source_confidence': 1.0 - src_ckpt.representation.get('prediction_entropy', 0.0),
                        'target_confidence': 1.0 - entropy,
                        'predicted_probability': pred_prob,
                        'delta_F1': delta_f1,
                        'transfer_label': label,
                        'transfer_CPU': decision_cpu,
                        'recovery_time': 0.0
                    })
                    
                    # Record in transferability estimator for online model learning
                    if 'Probability' in method_name:
                        self.transfer_estimator.record_outcome(pair_feats, delta_f1, label)

            # Checkpoint Pool Updating: Save new checkpoint if transition or pool empty
            if is_transition or len(checkpoint_pool) == 0:
                new_ckpt_id = len(checkpoint_pool)
                new_ckpt = RegimeCheckpointExp2(
                    checkpoint_id=new_ckpt_id,
                    regime_key=reg_key,
                    representation=target_rep,
                    rf_model=active_rf,
                    et_model=active_et,
                    weights=active_weights,
                    created_window=w_id
                )
                checkpoint_pool.append(new_ckpt)

            per_window_results.append({
                'window_id': w_id,
                'method': method_name,
                'regime': reg_key,
                'is_transition': is_transition,
                'f1_score': f1_win,
                'accuracy': acc_win,
                'decision': decision,
                'selected_source': selected_source_id,
                'predicted_prob': pred_prob,
                'disagreement': disagreement,
                'entropy': entropy,
                'cpu_cost': rep_cpu + decision_cpu
            })

        # Summarize Streaming Run
        df_win_res = pd.DataFrame(per_window_results)
        macro_f1 = float(df_win_res['f1_score'].mean())
        mean_acc = float(df_win_res['accuracy'].mean())
        
        ntr = float(negative_transfers / (total_transfers + 1e-6)) if total_transfers > 0 else 0.0
        str_rate = float(positive_transfers / (total_transfers + 1e-6)) if total_transfers > 0 else 0.0

        summary = {
            'method': method_name,
            'dataset_type': self.dataset_type,
            'seed': self.seed,
            'macro_f1': macro_f1,
            'mean_accuracy': mean_acc,
            'adapt_cpu': total_adapt_cpu,
            'retrain_count': retrain_count,
            'total_transfers': total_transfers,
            'positive_transfers': positive_transfers,
            'negative_transfers': negative_transfers,
            'negative_transfer_rate': ntr,
            'successful_transfer_rate': str_rate,
            'tau': self.tau
        }

        return summary, df_win_res, pd.DataFrame(transfer_pairs)
