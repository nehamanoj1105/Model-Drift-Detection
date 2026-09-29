"""
Probabilistic Transferability Estimator for Experiment 2
Constructs directional pairwise features X_{s,t}, maintains online transfer database D_meta(t),
trains calibrated probabilistic transferability estimators, and implements 2-stage candidate selection
with abstention thresholding.
"""

import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError

from experiments.exp2.similarity import RegimeSimilarityEngine

class TransferFeatureBuilder:
    """
    Constructs directional pairwise features X_{s,t} for source regime s and target regime t.
    Preserves source/target directionality and enforces zero-look-ahead anti-leakage rules.
    """
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.similarity_engine = RegimeSimilarityEngine(feature_cols)

    def build_pair_features(self, source_rep, target_rep, source_df=None, target_df=None, pool_reps=None):
        """
        Builds feature dictionary X_{s,t} from source and target representations.
        """
        X = {}

        # 1. Similarity & Telemetry Distances
        dists = self.similarity_engine.compute_distances(source_rep, target_rep, source_df=source_df, target_df=target_df)
        X['dist_euclidean'] = dists['euclidean_distance']
        X['dist_wasserstein'] = dists['wasserstein_distance']
        X['dist_js'] = dists['js_distance']
        X['dist_correlation'] = dists['correlation_distance']
        X['dist_temporal'] = dists['temporal_distance']
        X['dist_combined'] = dists['combined_distance']
        X['sim_combined'] = dists['combined_similarity']

        # 2. Directional Mean & Std Feature Differences (Target - Source)
        mean_diffs = []
        std_diffs = []
        for col in self.feature_cols:
            s_m = source_rep.get(f"{col}_mean", 0.0)
            t_m = target_rep.get(f"{col}_mean", 0.0)
            s_s = source_rep.get(f"{col}_std", 0.0)
            t_s = target_rep.get(f"{col}_std", 0.0)
            mean_diffs.append(t_m - s_m)
            std_diffs.append(t_s - s_s)
            
        X['mean_diff_avg'] = float(np.mean(mean_diffs))
        X['std_diff_avg'] = float(np.mean(std_diffs))
        X['abs_mean_diff_avg'] = float(np.mean(np.abs(mean_diffs)))
        X['abs_std_diff_avg'] = float(np.mean(np.abs(std_diffs)))

        # 3. Source Historical Quality & Metadata
        X['source_historical_f1'] = source_rep.get('historical_f1', 1.0)
        X['source_historical_acc'] = source_rep.get('historical_acc', 1.0)
        X['source_checkpoint_age'] = source_rep.get('checkpoint_age', 0)
        X['source_usage_count'] = source_rep.get('observation_count', 1)
        X['source_successful_transfers'] = source_rep.get('successful_transfers', 0)
        X['source_failed_transfers'] = source_rep.get('failed_transfers', 0)
        X['source_transfer_success_rate'] = source_rep.get('transfer_success_rate', 0.5)
        X['source_rf_weight'] = source_rep.get('model_weights_rf', 0.5)

        # 4. Target Novelty (Minimum distance to any historical regime in pool)
        if pool_reps and len(pool_reps) > 1:
            pool_dists = []
            for other_rep in pool_reps:
                d = self.similarity_engine.compute_distances(other_rep, target_rep)['combined_distance']
                pool_dists.append(d)
            X['target_novelty_min_dist'] = float(np.min(pool_dists))
            X['target_novelty_mean_dist'] = float(np.mean(pool_dists))
        else:
            X['target_novelty_min_dist'] = X['dist_combined']
            X['target_novelty_mean_dist'] = X['dist_combined']

        # 5. Model Uncertainty & Disagreement
        X['source_disagreement'] = source_rep.get('model_disagreement', 0.0)
        X['source_entropy'] = source_rep.get('prediction_entropy', 0.0)
        X['target_disagreement'] = target_rep.get('model_disagreement', 0.0)
        X['target_entropy'] = target_rep.get('prediction_entropy', 0.0)
        X['disagreement_diff'] = X['target_disagreement'] - X['source_disagreement']

        return X


class ProbabilisticTransferEstimator:
    """
    Online meta-learning model predicting P(positive transfer | X_{s,t}).
    Trains online on accumulated transfer episodes D_meta(t) and enforces abstention gate.
    """
    def __init__(self, feature_cols, min_samples_to_fit=8, tau=0.60, k_candidates=5):
        self.feature_cols = feature_cols
        self.feature_builder = TransferFeatureBuilder(feature_cols)
        self.min_samples_to_fit = min_samples_to_fit
        self.tau = tau
        self.k_candidates = k_candidates
        
        # Internal online transfer dataset D_meta
        self.historical_episodes = []
        
        # Scikit-learn model setup
        self.base_estimator = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def record_outcome(self, pair_features, delta_f1, label, cpu_cost=0.0, recovery_time=0.0):
        """
        Adds observed transfer outcome episode to online dataset D_meta(t).
        label: +1 for positive transfer (delta_f1 > 0.005), 0 for neutral, -1 for negative.
        """
        episode = {
            'features': pair_features,
            'delta_f1': delta_f1,
            'label': label,
            'binary_positive_label': 1 if label > 0 else 0,
            'cpu_cost': cpu_cost,
            'recovery_time': recovery_time
        }
        self.historical_episodes.append(episode)
        
        # Trigger online model refit
        self._refit_model()

    def _refit_model(self):
        """
        Refits calibrated logistic regression using all historical transfer episodes before t.
        """
        if len(self.historical_episodes) < self.min_samples_to_fit:
            self.is_fitted = False
            return

        # Prepare X, y
        X_list = [ep['features'] for ep in self.historical_episodes]
        y_list = [ep['binary_positive_label'] for ep in self.historical_episodes]
        
        # Require both positive and non-positive examples to fit classifier
        unique_y = set(y_list)
        if len(unique_y) < 2:
            self.is_fitted = False
            return

        df_X = pd.DataFrame(X_list).fillna(0.0)
        self.feature_names = list(df_X.columns)
        X_mat = df_X.values
        y_vec = np.array(y_list, dtype=int)

        try:
            # Cross-validation calibration if sufficient samples, else sigmoid calibration
            n_cv = min(3, max(2, min(np.bincount(y_vec))))
            if n_cv >= 2 and len(y_vec) >= 12:
                calibrated = CalibratedClassifierCV(estimator=self.base_estimator, method='sigmoid', cv=n_cv)
                calibrated.fit(X_mat, y_vec)
                self.model = calibrated
            else:
                lr = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
                lr.fit(X_mat, y_vec)
                self.model = lr
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def predict_probability(self, source_rep, target_rep, source_df=None, target_df=None, pool_reps=None):
        """
        Predicts P(positive transfer | X_{s,t}) for a given candidate source and current target.
        Returns (prob_positive, pair_features).
        """
        pair_features = self.feature_builder.build_pair_features(
            source_rep, target_rep, source_df=source_df, target_df=target_df, pool_reps=pool_reps
        )

        if not self.is_fitted or self.model is None:
            # Cold-start fallback: Use calibrated transformation of combined similarity
            sim = pair_features['sim_combined']
            hist_f1 = pair_features['source_historical_f1']
            prob_pos = float(0.7 * sim + 0.3 * hist_f1)
            prob_pos = min(0.95, max(0.05, prob_pos))
            return prob_pos, pair_features

        df_feat = pd.DataFrame([pair_features])[self.feature_names].fillna(0.0)
        try:
            probs = self.model.predict_proba(df_feat.values)[0]
            # Probabilities of class 1 (positive transfer)
            prob_pos = float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception:
            sim = pair_features['sim_combined']
            prob_pos = float(sim)

        return prob_pos, pair_features

    def evaluate_candidates(self, target_rep, candidate_checkpoints, target_df=None):
        """
        2-Stage Selection Process:
        Stage 1: Cheap similarity filtering to top-K candidates.
        Stage 2: Evaluate P(positive transfer | X_{s,t}) on top-K candidates.
        
        Returns sorted list of evaluated candidate dicts.
        """
        if not candidate_checkpoints:
            return []

        # Extract pool representations for novelty features
        pool_reps = [ckpt['representation'] for ckpt in candidate_checkpoints]

        # Stage 1: Fast similarity ranking
        similarity_engine = self.feature_builder.similarity_engine
        stage1_ranked = []
        for ckpt in candidate_checkpoints:
            s_rep = ckpt['representation']
            s_df = ckpt.get('df_slice', None)
            dists = similarity_engine.compute_distances(s_rep, target_rep, source_df=s_df, target_df=target_df)
            stage1_ranked.append({
                'checkpoint': ckpt,
                'similarity': dists['combined_similarity'],
                'metrics': dists
            })
        stage1_ranked.sort(key=lambda x: x['similarity'], reverse=True)

        # Select top-K for Stage 2
        top_k = stage1_ranked[:min(self.k_candidates, len(stage1_ranked))]

        # Stage 2: Transferability probability estimation
        stage2_results = []
        for item in top_k:
            ckpt = item['checkpoint']
            s_rep = ckpt['representation']
            s_df = ckpt.get('df_slice', None)
            
            prob_pos, pair_features = self.predict_probability(
                s_rep, target_rep, source_df=s_df, target_df=target_df, pool_reps=pool_reps
            )
            
            stage2_results.append({
                'checkpoint': ckpt,
                'probability': prob_pos,
                'similarity': item['similarity'],
                'metrics': item['metrics'],
                'pair_features': pair_features
            })

        # Sort descending by predicted transfer probability
        stage2_results.sort(key=lambda x: x['probability'], reverse=True)
        return stage2_results

    def select_source(self, target_rep, candidate_checkpoints, target_df=None):
        """
        Main Decision Rule:
        Selects s* = argmax_s P_{s,t}.
        Enforces abstention threshold: If P_{s*,t} < tau, rejects transfer (returns decision = 'ABSTAIN').
        """
        if not candidate_checkpoints:
            return None, 'NO_CANDIDATES', 0.0, {}

        evaluated = self.evaluate_candidates(target_rep, candidate_checkpoints, target_df=target_df)
        if not evaluated:
            return None, 'NO_CANDIDATES', 0.0, {}

        best = evaluated[0]
        best_prob = best['probability']

        if best_prob >= self.tau:
            decision = 'TRANSFER'
            selected_ckpt = best['checkpoint']
        else:
            decision = 'ABSTAIN'
            selected_ckpt = None

        metadata = {
            'evaluated_candidates': evaluated,
            'best_candidate': best,
            'best_probability': best_prob,
            'tau': self.tau,
            'cold_start': not self.is_fitted
        }

        return selected_ckpt, decision, best_prob, metadata

    def compute_probability_weighted_blend(self, target_rep, candidate_checkpoints, target_df=None):
        """
        Probability-Guided Weighted Transfer: Blends candidate policy weights based on normalized P_{s,t}.
        """
        if not candidate_checkpoints:
            return None, 'NO_CANDIDATES', 0.0, {}

        evaluated = self.evaluate_candidates(target_rep, candidate_checkpoints, target_df=target_df)
        probs = np.array([item['probability'] for item in evaluated], dtype=float)
        
        max_prob = float(np.max(probs))
        if max_prob < self.tau:
            return None, 'ABSTAIN', max_prob, {'evaluated_candidates': evaluated}

        # Normalize probabilities for blending
        total_p = float(np.sum(probs))
        blend_weights = probs / (total_p + 1e-9)

        blended_rf = sum(w * item['checkpoint']['weights'][0] for w, item in zip(blend_weights, evaluated))
        blended_et = sum(w * item['checkpoint']['weights'][1] for w, item in zip(blend_weights, evaluated))
        
        tot_w = blended_rf + blended_et
        if tot_w > 0:
            blended_rf /= tot_w
            blended_et /= tot_w
        else:
            blended_rf, blended_et = 0.5, 0.5

        result_policy = {
            'weights': [blended_rf, blended_et],
            'evaluated_candidates': evaluated,
            'blend_weights': blend_weights.tolist()
        }

        return result_policy, 'TRANSFER_BLENDED', max_prob, result_policy
