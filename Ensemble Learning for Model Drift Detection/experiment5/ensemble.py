"""
================================================================================
EXPERIMENT 5 — HETEROGENEOUS ADAPTIVE ENSEMBLE
================================================================================
Architecture:
  - Component Models: Random Forest (RF), Extra Trees (ET), Gradient Boosting (GB)
  - Multi-Temporal Historical Horizons:
      * RF: 3,500 samples (long-term structural stability)
      * ET: 2,500 samples (medium-term covariate invariance)
      * GB: 1,800 samples (short-term rapid nonlinear concept adaptation)
  - Rank-Weighted Diverse Soft Voting:
      * Ranking by rolling joint utility: 0.5 * F1 + 0.5 * Accuracy
      * Calibrated rank weights: [0.18, 0.32, 0.50]
      * Calibrated decision threshold: tau = 0.50
================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from config import (
    TEMPORAL_HORIZONS,
    DECISION_THRESHOLD,
    RANK_WEIGHTS,
    WEIGHT_ALPHA
)
from models import create_candidate_models
from resource_monitor import measure_execution


class HeterogeneousAdaptiveEnsemble:
    """
    Heterogeneous Ensemble combining RF, ET, and GB with multi-temporal horizons
    and rank-weighted soft voting.
    """

    def __init__(self, seed, is_frozen=False, threshold=DECISION_THRESHOLD,
                 horizons=None, rank_weights=None, alpha=WEIGHT_ALPHA):
        self.seed = seed
        self.is_frozen = is_frozen
        self.threshold = threshold
        self.horizons = horizons or dict(TEMPORAL_HORIZONS)
        self.rank_weights = rank_weights or list(RANK_WEIGHTS)
        self.alpha = alpha

        raw_models = create_candidate_models(seed)
        self.models = {
            'RF': raw_models['RandomForest'],
            'ET': raw_models['ExtraTrees'],
            'GB': raw_models['GradientBoosting'],
        }
        self.component_names = ['RF', 'ET', 'GB']

        # Initial weights: equal 1/3
        self.weights = {c: 1.0 / 3.0 for c in self.component_names}
        self.ema_scores = {c: 0.85 for c in self.component_names}
        self.history = {c: {'f1': [], 'accuracy': []} for c in self.component_names}

        self.buffer_X = []
        self.buffer_y = []
        self.retrain_counts = {c: 0 for c in self.component_names}
        self.total_retrain_events = 0
        self.total_adaptation_samples = 0

    def fit_initial(self, X_train, y_train):
        """Fit all 3 base models on initial 20,000 training samples."""
        self.buffer_X = list(X_train)
        self.buffer_y = list(y_train)

        train_res = {}
        for name in self.component_names:
            _, res = measure_execution(self.models[name].fit, X_train, y_train)
            train_res[name] = res

        return train_res

    def predict_component_proba(self, name, X):
        """Get predicted probability P(Y=1|X) for a single component."""
        model = self.models[name]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            p = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            p = model.predict(X).astype(float)
        return np.clip(p, 1e-5, 1.0 - 1e-5)

    def predict_proba(self, X):
        """
        Compute ensemble probability via rank-weighted soft voting:
          P_ensemble = w_RF * P_RF + w_ET * P_ET + w_GB * P_GB
        """
        comp_probs = {}
        for name in self.component_names:
            comp_probs[name] = self.predict_component_proba(name, X)

        prob_ensemble = (
            self.weights['RF'] * comp_probs['RF'] +
            self.weights['ET'] * comp_probs['ET'] +
            self.weights['GB'] * comp_probs['GB']
        )
        return prob_ensemble, comp_probs

    def predict(self, X, threshold=None):
        """
        Predict binary labels and probabilities for all components and the ensemble.
        """
        th = threshold if threshold is not None else self.threshold
        prob_ensemble, comp_probs = self.predict_proba(X)
        pred_ensemble = (prob_ensemble >= th).astype(int)
        comp_preds = {name: (comp_probs[name] >= th).astype(int) for name in self.component_names}
        return pred_ensemble, prob_ensemble, comp_preds, comp_probs

    def compute_diversity_metrics(self, comp_preds, comp_probs):
        """Compute pairwise prediction disagreements and probability correlations."""
        p_rf, p_et, p_gb = comp_preds['RF'], comp_preds['ET'], comp_preds['GB']
        pr_rf, pr_et, pr_gb = comp_probs['RF'], comp_probs['ET'], comp_probs['GB']

        d_rf_et = float(np.mean(p_rf != p_et))
        d_rf_gb = float(np.mean(p_rf != p_gb))
        d_et_gb = float(np.mean(p_et != p_gb))

        def _safe_corr(a, b):
            if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                return float(np.corrcoef(a, b)[0, 1])
            return 1.0

        c_rf_et = _safe_corr(pr_rf, pr_et)
        c_rf_gb = _safe_corr(pr_rf, pr_gb)
        c_et_gb = _safe_corr(pr_et, pr_gb)

        return {
            'disagreement_rf_et': d_rf_et,
            'disagreement_rf_gb': d_rf_gb,
            'disagreement_et_gb': d_et_gb,
            'corr_rf_et': c_rf_et,
            'corr_rf_gb': c_rf_gb,
            'corr_et_gb': c_et_gb,
        }

    def update_weights(self, y_true, comp_preds, comp_probs=None):
        """
        Update rolling component performance and rank weights based on joint utility:
          Score_i = 0.5 * F1_i + 0.5 * Accuracy_i
        """
        if self.is_frozen:
            return {}, self.weights, 0.0, {}

        div_metrics = self.compute_diversity_metrics(comp_preds, comp_probs)

        comp_metrics = {}
        for name in self.component_names:
            p = comp_preds[name]
            f1 = float(f1_score(y_true, p, zero_division=0))
            acc = float(accuracy_score(y_true, p))
            self.history[name]['f1'].append(f1)
            self.history[name]['accuracy'].append(acc)

            joint_util = 0.5 * f1 + 0.5 * acc
            self.ema_scores[name] = self.alpha * joint_util + (1.0 - self.alpha) * self.ema_scores[name]
            comp_metrics[name] = {'f1': f1, 'accuracy': acc, 'joint_util': joint_util}

        # Rank components: 0 is lowest, 2 is highest
        scores = [self.ema_scores[c] for c in self.component_names]
        ranks = np.argsort(np.argsort(scores))

        for idx, name in enumerate(self.component_names):
            rank_pos = ranks[idx]
            self.weights[name] = float(self.rank_weights[rank_pos])

        w_vals = list(self.weights.values())
        weight_spread = float(np.max(w_vals) - np.min(w_vals))

        return comp_metrics, self.weights, weight_spread, div_metrics

    def adapt(self, X_win, y_win):
        """
        Retrain component models using multi-temporal horizons on accumulated buffer.
        """
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)

        if self.is_frozen:
            empty_res = {
                'wall_clock_time': 0.0, 'total_cpu_time': 0.0, 'cpu_user_time': 0.0,
                'cpu_system_time': 0.0, 'avg_cpu_percent': 0.0, 'peak_cpu_percent': 0.0,
                'avg_ram_mb': 0.0, 'peak_ram_mb': 0.0,
            }
            return {c: False for c in self.component_names}, {c: dict(empty_res) for c in self.component_names}, 0

        all_X = np.array(self.buffer_X)
        all_y = np.array(self.buffer_y)

        retrained = {}
        resources = {}
        window_samples_trained = 0

        for name in self.component_names:
            horizon = self.horizons.get(name, 2000)
            curr_h = horizon
            t_X = all_X[-curr_h:]
            t_y = all_y[-curr_h:]

            # If single class or severely unbalanced, extend horizon into buffer
            while (len(np.unique(t_y)) < 2 or np.min(np.bincount(t_y)) < 20) and curr_h < len(all_y):
                curr_h = min(len(all_y), curr_h + 1000)
                t_X = all_X[-curr_h:]
                t_y = all_y[-curr_h:]

            # Fallback anchor if still single class
            if len(np.unique(t_y)) < 2:
                other_class = 1 - int(t_y[0])
                other_idx = np.where(all_y == other_class)[0]
                if len(other_idx) > 0:
                    anchor_idx = other_idx[:min(100, len(other_idx))]
                    t_X = np.concatenate([t_X, all_X[anchor_idx]])
                    t_y = np.concatenate([t_y, all_y[anchor_idx]])

            # Refit component model
            _, res = measure_execution(self.models[name].fit, t_X, t_y)
            resources[name] = res
            retrained[name] = True
            self.retrain_counts[name] += 1
            self.total_retrain_events += 1
            window_samples_trained += len(t_X)

        self.total_adaptation_samples += window_samples_trained
        return retrained, resources, window_samples_trained
