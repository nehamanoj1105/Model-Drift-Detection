"""
================================================================================
EXPERIMENT 5 — ENSEMBLE WEIGHTING STRATEGIES
================================================================================
Implements 4 Ensemble Weighting Strategies:
  1. Fixed Ensemble (Baseline: Equal static weights 1/3, 1/3, 1/3)
  2. Global Adaptive Ensemble (Global EMA F1 Softmax weights)
  3. Regime-Aware Ensemble (Distributional quantile fingerprinting + regime memory)
  4. Oracle Regime Weights (Offline optimal weight vector per window)

All strategies preserve the Event-Driven Ensemble drift detection and retraining
mechanism from Experiment 4.
================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from config import (
    TEMPORAL_HORIZONS, DECISION_THRESHOLD, WEIGHT_ALPHA,
    SOFTMAX_TEMPERATURE, DEFAULT_SIMILARITY_THRESHOLD
)
from models import create_candidate_models
from drift import DualTriggerDriftDetector
from fingerprint import extract_quantile_fingerprint, RegimeMemory
from resource_monitor import measure_execution


class EnsembleWeightingRunner:
    """
    Unified Ensemble Runner executing one of the four weighting strategies:
      - 'fixed'
      - 'global_adaptive'
      - 'regime_aware'
      - 'oracle'
    """

    def __init__(self, seed, strategy_type='fixed', similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                 horizons=None, alpha=WEIGHT_ALPHA, temperature=SOFTMAX_TEMPERATURE):
        self.seed = seed
        self.strategy_type = strategy_type
        self.similarity_threshold = similarity_threshold
        self.horizons = horizons or dict(TEMPORAL_HORIZONS)
        self.alpha = alpha
        self.temperature = temperature

        raw_models = create_candidate_models(seed)
        self.models = {
            'RF': raw_models['RandomForest'],
            'ET': raw_models['ExtraTrees'],
            'GB': raw_models['GradientBoosting'],
        }
        self.component_names = ['RF', 'ET', 'GB']

        # Weighting mechanisms
        self.fixed_weights = {c: 1.0 / 3.0 for c in self.component_names}
        self.global_weights = {c: 1.0 / 3.0 for c in self.component_names}
        self.global_ema = {c: 0.85 for c in self.component_names}

        # Regime memory for regime-aware strategy
        self.regime_memory = RegimeMemory(
            similarity_threshold=similarity_threshold,
            alpha=alpha,
            temperature=temperature,
            component_names=self.component_names
        )

        # Drift detector (Event-Driven Baseline)
        self.drift_detector = DualTriggerDriftDetector(initial_baseline_f1=0.85)

        # Buffer tracking
        self.buffer_X = []
        self.buffer_y = []
        self.retrain_counts = {c: 0 for c in self.component_names}
        self.total_retrain_events = 0
        self.total_adaptation_samples = 0

        # State per window
        self.current_weights = dict(self.fixed_weights)
        self.current_regime_info = {
            'regime_id': 0,
            'similarity': 1.0,
            'is_retrieved': False
        }

    def fit_initial(self, X_train, y_train):
        """Fits base models on initial 20,000 training partition."""
        self.buffer_X = list(X_train)
        self.buffer_y = list(y_train)

        train_res = {}
        for name in self.component_names:
            _, res = measure_execution(self.models[name].fit, X_train, y_train)
            train_res[name] = res

        return train_res

    def predict_component_proba(self, name, X):
        """Gets component prediction probability P(Y=1|X)."""
        model = self.models[name]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            p = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        else:
            p = model.predict(X).astype(float)
        return np.clip(p, 1e-5, 1.0 - 1e-5)

    def _find_oracle_weights(self, comp_probs, y_true):
        """
        Evaluates grid search over weight simplex to find optimal weights for window y_true.
        """
        best_f1 = -1.0
        best_w = {c: 1.0 / 3.0 for c in self.component_names}

        grid = []
        for w1 in np.linspace(0.0, 1.0, 21):
            for w2 in np.linspace(0.0, 1.0 - w1, 21):
                w3 = 1.0 - w1 - w2
                grid.append((w1, w2, w3))

        for w1, w2, w3 in grid:
            prob_ens = w1 * comp_probs['RF'] + w2 * comp_probs['ET'] + w3 * comp_probs['GB']
            pred_ens = (prob_ens >= DECISION_THRESHOLD).astype(int)
            score = float(f1_score(y_true, pred_ens, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_w = {'RF': float(w1), 'ET': float(w2), 'GB': float(w3)}

        return best_w

    def predict(self, X_win, y_win_oracle=None):
        """
        Predicts prequentially on streaming window X_win.
        Strictly zero data leakage: weights are selected BEFORE observing y_win.
        """
        comp_probs = {}
        for name in self.component_names:
            comp_probs[name] = self.predict_component_proba(name, X_win)

        # Select weights according to strategy
        if self.strategy_type == 'fixed':
            weights = dict(self.fixed_weights)
            self.current_regime_info = {'regime_id': 0, 'similarity': 1.0, 'is_retrieved': False}

        elif self.strategy_type == 'global_adaptive':
            weights = dict(self.global_weights)
            self.current_regime_info = {'regime_id': 0, 'similarity': 1.0, 'is_retrieved': False}

        elif self.strategy_type == 'regime_aware':
            fp = extract_quantile_fingerprint(X_win)
            r_id, sim, r_weights, is_retrieved = self.regime_memory.query_regime(fp)
            weights = dict(r_weights)
            self.current_regime_info = {
                'regime_id': r_id,
                'similarity': sim,
                'is_retrieved': is_retrieved,
                'fingerprint': fp
            }

        elif self.strategy_type == 'oracle':
            if y_win_oracle is not None:
                weights = self._find_oracle_weights(comp_probs, y_win_oracle)
            else:
                weights = dict(self.fixed_weights)
            self.current_regime_info = {'regime_id': 0, 'similarity': 1.0, 'is_retrieved': False}
        else:
            raise ValueError(f"Unknown strategy_type: {self.strategy_type}")

        self.current_weights = dict(weights)

        # Combine component probabilities
        prob_ensemble = (
            weights['RF'] * comp_probs['RF'] +
            weights['ET'] * comp_probs['ET'] +
            weights['GB'] * comp_probs['GB']
        )
        pred_ensemble = (prob_ensemble >= DECISION_THRESHOLD).astype(int)
        comp_preds = {name: (comp_probs[name] >= DECISION_THRESHOLD).astype(int) for name in self.component_names}

        return pred_ensemble, prob_ensemble, comp_preds, comp_probs, weights

    def update_and_adapt(self, X_win, y_win, comp_preds, ens_f1, X_train_ref):
        """
        Post-prediction phase:
          1. Calculates component F1 scores on observed y_win.
          2. Updates strategy weight vectors (Global EMA / Regime Memory).
          3. Checks dual-trigger drift detector.
          4. If drift detected, retrains base models on historical buffers.
        """
        # Calculate component F1 scores
        comp_f1s = {}
        for name in self.component_names:
            comp_f1s[name] = float(f1_score(y_win, comp_preds[name], zero_division=0))

        # 1. Update Strategy Weights
        if self.strategy_type == 'global_adaptive':
            for c in self.component_names:
                self.global_ema[c] = self.alpha * self.global_ema[c] + (1.0 - self.alpha) * comp_f1s[c]
            scores = np.array([self.global_ema[c] for c in self.component_names])
            exp_s = np.exp((scores - np.max(scores)) / self.temperature)
            probs = exp_s / np.sum(exp_s)
            for idx, c in enumerate(self.component_names):
                self.global_weights[c] = float(probs[idx])

        elif self.strategy_type == 'regime_aware':
            r_id = self.current_regime_info['regime_id']
            fp = self.current_regime_info.get('fingerprint', extract_quantile_fingerprint(X_win))
            self.regime_memory.update_regime(r_id, fp, comp_f1s)

        # 2. Add window to historical buffer
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)

        # 3. Dual-trigger drift check (Event-Driven mechanism)
        drift_decision = self.drift_detector.check_drift(X_train_ref, X_win, ens_f1)
        drift_detected = drift_decision['drift_detected']

        retrained = {c: False for c in self.component_names}
        resources = {}
        adaptation_time = 0.0

        if drift_detected:
            all_X = np.array(self.buffer_X)
            all_y = np.array(self.buffer_y)

            for name in self.component_names:
                horizon = self.horizons.get(name, 2000)
                curr_h = horizon
                t_X = all_X[-curr_h:]
                t_y = all_y[-curr_h:]

                # Balance check
                while (len(np.unique(t_y)) < 2 or np.min(np.bincount(t_y)) < 20) and curr_h < len(all_y):
                    curr_h = min(len(all_y), curr_h + 1000)
                    t_X = all_X[-curr_h:]
                    t_y = all_y[-curr_h:]

                if len(np.unique(t_y)) < 2:
                    other_class = 1 - int(t_y[0])
                    other_idx = np.where(all_y == other_class)[0]
                    if len(other_idx) > 0:
                        anchor_idx = other_idx[:min(100, len(other_idx))]
                        t_X = np.concatenate([t_X, all_X[anchor_idx]])
                        t_y = np.concatenate([t_y, all_y[anchor_idx]])

                _, res = measure_execution(self.models[name].fit, t_X, t_y)
                resources[name] = res
                retrained[name] = True
                self.retrain_counts[name] += 1
                self.total_retrain_events += 1
                adaptation_time += res['total_cpu_time']

        return {
            'drift_detected': drift_detected,
            'drift_decision': drift_decision,
            'retrained': retrained,
            'adaptation_time': adaptation_time,
            'comp_f1s': comp_f1s
        }
