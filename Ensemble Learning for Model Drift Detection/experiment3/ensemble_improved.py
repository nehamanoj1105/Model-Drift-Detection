"""
================================================================================
EXPERIMENT 3 (IMPROVED) — RANK-WEIGHTED DIVERSE ADAPTIVE ENSEMBLE
================================================================================
Features:
  - Base Models: Random Forest (RF), Extra Trees (ET), Gradient Boosting (GB)
    (Strictly identical candidate models as Experiment 3)
  - Multi-Temporal Horizons:
      * RF: Long horizon (3,500 samples)
      * ET: Medium horizon (2,500 samples)
      * GB: Short horizon (1,800 samples)
  - Rank-Weighted Soft Voting:
      * Base models are ranked by rolling joint utility: 0.5 * F1 + 0.5 * Accuracy
      * Dynamic rank weights: Rank 1 (best) = 0.50, Rank 2 = 0.32, Rank 3 = 0.18
      * Combined with threshold tau = 0.455 calibrated for QoS telemetry imbalance
================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from config import TEMPORAL_HORIZONS
from models import create_candidate_models
from resource_monitor import measure_execution


class ImprovedAdaptiveEnsemble:
    """
    Enhanced Adaptive Ensemble with multi-temporal perspectives,
    rank-weighted soft voting, and joint F1-Accuracy EMA tracking.
    """

    def __init__(self, seed, horizons=None, threshold=0.455,
                 rank_weights=None, alpha=0.7):
        self.seed = seed
        self.horizons = horizons or dict(TEMPORAL_HORIZONS)
        self.threshold = threshold
        self.rank_weights = rank_weights or [0.18, 0.32, 0.50]  # [3rd, 2nd, 1st]
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

    def fit_initial(self, X_train, y_train):
        """Fit all 3 component models on the initial 2,000 training samples."""
        self.buffer_X = list(X_train)
        self.buffer_y = list(y_train)

        train_res = {}
        for name in self.component_names:
            _, res = measure_execution(self.models[name].fit, X_train, y_train)
            train_res[name] = res

        return train_res

    def predict_component_proba(self, name, X):
        """Get predicted probability P(Y=1|X) for a single model."""
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
        Returns (prob_ensemble, comp_probs_dict)
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

        div_rf = (d_rf_et + d_rf_gb) / 2.0
        div_et = (d_rf_et + d_et_gb) / 2.0
        div_gb = (d_rf_gb + d_et_gb) / 2.0

        return {
            'disagreement_rf_et': d_rf_et,
            'disagreement_rf_gb': d_rf_gb,
            'disagreement_et_gb': d_et_gb,
            'corr_rf_et': c_rf_et,
            'corr_rf_gb': c_rf_gb,
            'corr_et_gb': c_et_gb,
            'diversity': {'RF': div_rf, 'ET': div_et, 'GB': div_gb},
        }

    def update_weights(self, y_true, comp_preds, comp_probs=None):
        """
        Rank-weighted updating based on rolling joint utility:
          Score_i = 0.5 * F1_i + 0.5 * Accuracy_i
          EMA_Score_i = alpha * Score_i + (1 - alpha) * EMA_Score_i
        Ranks base models and allocates rank weights [0.18, 0.32, 0.50].
        """
        div_metrics = self.compute_diversity_metrics(comp_preds, comp_probs)

        comp_metrics = {}
        for name in self.component_names:
            p = comp_preds[name]
            f1 = float(f1_score(y_true, p, zero_division=0))
            acc = float(accuracy_score(y_true, p))
            self.history[name]['f1'].append(f1)
            self.history[name]['accuracy'].append(acc)

            # Balanced joint utility
            joint_util = 0.5 * f1 + 0.5 * acc
            self.ema_scores[name] = self.alpha * joint_util + (1.0 - self.alpha) * self.ema_scores[name]
            comp_metrics[name] = {'f1': f1, 'accuracy': acc, 'joint_util': joint_util}

        # Rank components based on EMA score
        scores = [self.ema_scores[c] for c in self.component_names]
        ranks = np.argsort(np.argsort(scores))  # 0 is lowest score, 2 is highest score

        for idx, name in enumerate(self.component_names):
            rank_pos = ranks[idx]  # 0, 1, or 2
            self.weights[name] = float(self.rank_weights[rank_pos])

        w_vals = list(self.weights.values())
        weight_spread = float(np.max(w_vals) - np.min(w_vals))

        return comp_metrics, self.weights, weight_spread, div_metrics

    def adapt(self, X_win, y_win, components_to_retrain=None):
        """
        Adapt models using multi-temporal horizons:
          - RF trains on last 3,500 samples
          - ET trains on last 2,500 samples
          - GB trains on last 1,800 samples
        """
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)

        all_X = np.array(self.buffer_X)
        all_y = np.array(self.buffer_y)

        if components_to_retrain is None:
            components_to_retrain = self.component_names

        retrained = {}
        resources = {}
        empty_res = {
            'wall_clock_time': 0.0,
            'total_cpu_time': 0.0,
            'cpu_user_time': 0.0,
            'cpu_system_time': 0.0,
            'avg_cpu_percent': 0.0,
            'peak_cpu_percent': 0.0,
            'avg_ram_mb': 0.0,
            'peak_ram_mb': 0.0,
        }

        for name in self.component_names:
            if name in components_to_retrain:
                horizon = self.horizons.get(name, 2000)
                t_X = all_X[-horizon:]
                t_y = all_y[-horizon:]

                _, res = measure_execution(self.models[name].fit, t_X, t_y)
                resources[name] = res
                retrained[name] = True
                self.retrain_counts[name] += 1
                self.total_retrain_events += 1
            else:
                resources[name] = dict(empty_res)
                retrained[name] = False

        return retrained, resources
