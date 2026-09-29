"""
================================================================================
EXPERIMENT 8 — SELECTIVE ADAPTIVE ENSEMBLE ENGINE
================================================================================
Implements the SelectiveAdaptiveEnsemble class capable of executing discrete
per-model adaptation actions ('KEEP', 'PARTIAL', 'FULL') under prequential constraints.
================================================================================
"""

import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from models import create_ensemble_models, fit_model_with_action


class SelectiveAdaptiveEnsemble:
    """
    Ensemble of Random Forest (RF), Extra Trees (ET), and Gradient Boosting (GB)
    with per-model selective adaptation and diversity-aware softmax weighting.
    """

    def __init__(self, seed, alpha=0.7, beta=8.0, gamma=0.15, threshold=0.46):
        self.seed = seed
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

        self.models = create_ensemble_models(seed)
        self.component_names = ['RF', 'ET', 'GB']

        self.weights = {c: 1.0 / 3.0 for c in self.component_names}
        self.ema_scores = {c: 0.85 for c in self.component_names}
        self.history = {c: {'f1': [], 'accuracy': []} for c in self.component_names}

        # Temporal historical buffers
        self.buffer_X = []
        self.buffer_y = []

        self.retrain_counts = {c: {'FULL': 0, 'PARTIAL': 0, 'KEEP': 0} for c in self.component_names}
        self.total_retrain_cpu_time = 0.0

    def fit_initial(self, X_train, y_train):
        """Fit all models on initial stationary baseline data."""
        self.buffer_X = list(X_train)
        self.buffer_y = list(y_train)

        fit_times = {}
        for name in self.component_names:
            t0 = time.perf_counter()
            self.models[name].fit(X_train, y_train)
            t1 = time.perf_counter()
            fit_times[name] = t1 - t0

        return fit_times

    def predict_component_proba(self, name, X):
        model = self.models[name]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            p = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            p = model.predict(X).astype(float)
        return np.clip(p, 1e-5, 1.0 - 1e-5)

    def predict_proba(self, X):
        prob_ensemble = np.zeros(len(X), dtype=float)
        comp_probs = {}
        for name in self.component_names:
            p = self.predict_component_proba(name, X)
            comp_probs[name] = p
            prob_ensemble += self.weights[name] * p
        return prob_ensemble, comp_probs

    def predict(self, X, threshold=None):
        th = threshold if threshold is not None else self.threshold
        prob_ensemble, comp_probs = self.predict_proba(X)
        pred_ensemble = (prob_ensemble >= th).astype(int)
        comp_preds = {name: (comp_probs[name] >= th).astype(int) for name in self.component_names}
        return pred_ensemble, prob_ensemble, comp_preds, comp_probs

    def compute_diversity_metrics(self, comp_preds):
        p_rf, p_et, p_gb = comp_preds['RF'], comp_preds['ET'], comp_preds['GB']
        d_rf_et = float(np.mean(p_rf != p_et))
        d_rf_gb = float(np.mean(p_rf != p_gb))
        d_et_gb = float(np.mean(p_et != p_gb))

        return {
            'RF': (d_rf_et + d_rf_gb) / 2.0,
            'ET': (d_rf_et + d_et_gb) / 2.0,
            'GB': (d_rf_gb + d_et_gb) / 2.0,
        }

    def update_weights(self, y_true, comp_preds):
        diversities = self.compute_diversity_metrics(comp_preds)

        for name in self.component_names:
            f1 = f1_score(y_true, comp_preds[name], average='macro', zero_division=0)
            self.history[name]['f1'].append(f1)
            self.ema_scores[name] = self.alpha * f1 + (1.0 - self.alpha) * self.ema_scores[name]

        scores = np.array([
            self.ema_scores[c] + self.gamma * diversities[c]
            for c in self.component_names
        ])

        exp_scores = np.exp(self.beta * (scores - np.max(scores)))
        new_weights = exp_scores / np.sum(exp_scores)

        for idx, name in enumerate(self.component_names):
            self.weights[name] = float(new_weights[idx])

        return self.weights

    def adapt_selective(self, X_win, y_win, action_dict, partial_sample_ratio=0.35):
        """
        Executes selective adaptation according to action_dict:
          {'RF': 'KEEP'|'PARTIAL'|'FULL', 'ET': ..., 'GB': ...}
        
        Appends X_win, y_win to buffer AFTER predicting.
        """
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)

        execution_times = {}
        samples_used = {}
        estimators_used = {}

        for name in self.component_names:
            action = action_dict.get(name, 'KEEP')
            t0 = time.perf_counter()
            new_model, n_samp, n_est = fit_model_with_action(
                name, self.models[name], action,
                self.buffer_X, self.buffer_y, self.seed,
                partial_sample_ratio=partial_sample_ratio
            )
            t1 = time.perf_counter()
            cpu_t = t1 - t0

            if action != 'KEEP':
                self.models[name] = new_model

            execution_times[name] = cpu_t
            samples_used[name] = n_samp
            estimators_used[name] = n_est
            self.retrain_counts[name][action] += 1
            self.total_retrain_cpu_time += cpu_t

        return {
            'execution_times': execution_times,
            'samples_used': samples_used,
            'estimators_used': estimators_used,
            'total_adaptation_cpu': sum(execution_times.values()),
        }
