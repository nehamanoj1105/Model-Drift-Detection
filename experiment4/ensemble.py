"""
================================================================================
EXPERIMENT 4 — HETEROGENEOUS ADAPTIVE ENSEMBLE & NOVEL ADAPTATION STRATEGIES
================================================================================
Implements:
  1. HeterogeneousAdaptiveEnsemble (Base):
     RF (3500), ET (2500), GB (1800) with dynamic rank-weighted soft voting.
     Bounded buffer cap (BUFFER_CAP=5000) prevents unbounded memory growth.
  2. WarmStartEnsemble:
     Appends 20 new trees/stages per retrain event and prunes oldest 20 trees,
     drastically reducing refitting latency while maintaining model diversity.
  3. ComponentSelectiveEnsemble:
     Tracks component-level rolling F1 and retrains ONLY degraded components.
  4. TwoTierHybridEnsemble:
     Pairs batch ensemble with an always-on online streaming learner (HAT) that
     updates every sample, dynamically elevating online voting weight immediately
     post-drift to eliminate onset adaptation regret.
================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from config import (
    TEMPORAL_HORIZONS,
    DECISION_THRESHOLD,
    RANK_WEIGHTS,
    WEIGHT_ALPHA,
    BUFFER_CAP,
    COMPONENT_DROP_THRESHOLD,
    WARM_START_TREES_PER_EVENT,
)
from models import create_candidate_models, create_warm_start_models
from resource_monitor import measure_execution
from streaming_models import OnlineLearnerWrapper


class HeterogeneousAdaptiveEnsemble:
    """
    Heterogeneous Ensemble combining RF, ET, and GB with multi-temporal horizons
    and rank-weighted soft voting. Bounded buffer prevents memory leaks.
    """

    def __init__(self, seed, is_frozen=False, threshold=DECISION_THRESHOLD,
                 horizons=None, rank_weights=None, alpha=WEIGHT_ALPHA,
                 buffer_cap=BUFFER_CAP):
        self.seed = seed
        self.is_frozen = is_frozen
        self.threshold = threshold
        self.horizons = horizons or dict(TEMPORAL_HORIZONS)
        self.rank_weights = rank_weights or list(RANK_WEIGHTS)
        self.alpha = alpha
        self.buffer_cap = buffer_cap

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
        if self.buffer_cap is not None:
            self.buffer_X = list(X_train[-self.buffer_cap:])
            self.buffer_y = list(y_train[-self.buffer_cap:])
        else:
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

    def update_weights(self, y_true, comp_preds, comp_probs=None):
        """
        Update rolling component performance and rank weights based on joint utility:
          Score_i = 0.5 * F1_i + 0.5 * Accuracy_i
        """
        if self.is_frozen:
            return {}, self.weights, 0.0, {}

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

        return comp_metrics, self.weights, weight_spread, {}

    def adapt(self, X_win, y_win):
        """
        Retrain component models using multi-temporal horizons on bounded buffer.
        """
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)

        # Enforce bounded buffer capacity
        if self.buffer_cap is not None and len(self.buffer_X) > self.buffer_cap:
            self.buffer_X = self.buffer_X[-self.buffer_cap:]
            self.buffer_y = self.buffer_y[-self.buffer_cap:]

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
            t_X = all_X[-horizon:]
            t_y = all_y[-horizon:]

            _, res = measure_execution(self.models[name].fit, t_X, t_y)
            resources[name] = res
            retrained[name] = True
            self.retrain_counts[name] += 1
            self.total_retrain_events += 1
            window_samples_trained += len(t_X)

        self.total_adaptation_samples += window_samples_trained
        return retrained, resources, window_samples_trained


class WarmStartEnsemble(HeterogeneousAdaptiveEnsemble):
    """
    Warm-Start Ensemble:
    Instead of fitting trees from scratch, fits only 20 new trees/stages per trigger
    on recent window data, then prunes the oldest 20 trees to maintain constant size.
    """

    def __init__(self, seed, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP,
                 trees_per_event=WARM_START_TREES_PER_EVENT):
        super().__init__(seed, is_frozen=False, threshold=threshold, buffer_cap=buffer_cap)
        self.trees_per_event = trees_per_event
        # Replace base models with warm_start=True versions
        raw_models = create_warm_start_models(seed)
        self.models = {
            'RF': raw_models['RandomForest'],
            'ET': raw_models['ExtraTrees'],
            'GB': raw_models['GradientBoosting'],
        }

    def adapt(self, X_win, y_win):
        """
        Incrementally train 20 new estimators on new window data and prune oldest 20.
        """
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        if self.buffer_cap is not None and len(self.buffer_X) > self.buffer_cap:
            self.buffer_X = self.buffer_X[-self.buffer_cap:]
            self.buffer_y = self.buffer_y[-self.buffer_cap:]

        retrained = {}
        resources = {}
        window_samples_trained = 0

        for name in self.component_names:
            model = self.models[name]
            target_n = model.n_estimators + self.trees_per_event
            model.n_estimators = target_n

            # Fit new estimators on the new window data
            _, res = measure_execution(model.fit, X_win, y_win)
            resources[name] = res

            # Prune oldest estimators to preserve original capacity (50 trees/stages)
            if name in ['RF', 'ET']:
                if len(model.estimators_) > 50:
                    model.estimators_ = model.estimators_[-50:]
                    model.n_estimators = len(model.estimators_)
            elif name == 'GB':
                if len(model.estimators_) > 50:
                    model.estimators_ = model.estimators_[-50:]
                    model.n_estimators = len(model.estimators_)

            retrained[name] = True
            self.retrain_counts[name] += 1
            self.total_retrain_events += 1
            window_samples_trained += len(X_win)

        self.total_adaptation_samples += window_samples_trained
        return retrained, resources, window_samples_trained


class ComponentSelectiveEnsemble(HeterogeneousAdaptiveEnsemble):
    """
    Component-Selective Retraining Ensemble:
    Tracks individual component rolling performance. On a drift trigger, retrains
    ONLY the component(s) whose performance actually dropped past threshold.
    """

    def __init__(self, seed, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP,
                 comp_drop_threshold=COMPONENT_DROP_THRESHOLD):
        super().__init__(seed, is_frozen=False, threshold=threshold, buffer_cap=buffer_cap)
        self.comp_drop_threshold = comp_drop_threshold
        self.component_retrain_log = []

    def adapt_selective(self, X_win, y_win, latest_comp_metrics=None):
        """
        Retrain only components exhibiting performance degradation.
        """
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        if self.buffer_cap is not None and len(self.buffer_X) > self.buffer_cap:
            self.buffer_X = self.buffer_X[-self.buffer_cap:]
            self.buffer_y = self.buffer_y[-self.buffer_cap:]

        # Identify components needing retraining
        selected_components = []
        if latest_comp_metrics:
            for name in self.component_names:
                c_f1 = latest_comp_metrics.get(name, {}).get('f1', 0.8)
                drop = max(0.0, self.ema_scores[name] - c_f1)
                if drop >= self.comp_drop_threshold:
                    selected_components.append(name)

        # Fallback: if no component exceeded drop threshold, retrain the lowest scoring component
        if not selected_components:
            scores = {name: self.ema_scores[name] for name in self.component_names}
            lowest_comp = min(scores, key=scores.get)
            selected_components.append(lowest_comp)

        self.component_retrain_log.append(list(selected_components))

        all_X = np.array(self.buffer_X)
        all_y = np.array(self.buffer_y)

        retrained = {c: False for c in self.component_names}
        resources = {}
        window_samples_trained = 0

        for name in selected_components:
            horizon = self.horizons.get(name, 2000)
            t_X = all_X[-horizon:]
            t_y = all_y[-horizon:]

            _, res = measure_execution(self.models[name].fit, t_X, t_y)
            resources[name] = res
            retrained[name] = True
            self.retrain_counts[name] += 1
            self.total_retrain_events += 1
            window_samples_trained += len(t_X)

        self.total_adaptation_samples += window_samples_trained
        return retrained, resources, window_samples_trained, selected_components


class TwoTierHybridEnsemble:
    """
    Two-Tier Cheap-Plus-Expensive Ensemble:
    Combines batch RF+ET+GB ensemble with an always-on online streaming learner (HAT).
    The online learner updates on every single sample at near-zero cost (~0.05ms).
    During the window immediately post-drift, the online learner's voting weight
    is temporarily boosted to 0.40 to mitigate adaptation onset regret.
    """

    def __init__(self, seed, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP):
        self.seed = seed
        self.threshold = threshold
        self.batch_ensemble = HeterogeneousAdaptiveEnsemble(seed, threshold=threshold, buffer_cap=buffer_cap)
        self.online_learner = OnlineLearnerWrapper(learner_type='HAT', seed=seed)
        self.online_weight = 0.10
        self.baseline_online_weight = 0.10
        self.post_drift_boost_weight = 0.40
        self.decay_factor = 0.70
        self.history_online_weights = []

    def fit_initial(self, X_train, y_train):
        """Fit batch ensemble and warm-start online learner."""
        batch_res = self.batch_ensemble.fit_initial(X_train, y_train)
        self.online_learner.fit_initial(X_train, y_train)
        return batch_res

    def predict(self, X):
        """
        Soft vote combining batch ensemble and online streaming learner:
          P_hybrid = (1 - w_online) * P_batch + w_online * P_online
        """
        _, p_batch, comp_preds, comp_probs = self.batch_ensemble.predict(X)
        p_online = self.online_learner.predict_proba(X)

        p_hybrid = (1.0 - self.online_weight) * p_batch + self.online_weight * p_online
        pred_hybrid = (p_hybrid >= self.threshold).astype(int)

        self.history_online_weights.append(float(self.online_weight))
        return pred_hybrid, p_hybrid, comp_preds, comp_probs

    def update_weights(self, y_true, comp_preds, comp_probs=None):
        return self.batch_ensemble.update_weights(y_true, comp_preds, comp_probs)

    def adapt(self, X_win, y_win, drift_fired=False):
        """
        Always update online learner.
        Adapt batch ensemble ONLY if drift_fired=True.
        Dynamically adjust online weight.
        """
        # 1. Always update online streaming learner per-sample
        _, online_res = measure_execution(self.online_learner.learn, X_win, y_win)

        # 2. Batch ensemble adaptation
        if drift_fired:
            retrained, batch_res, samples = self.batch_ensemble.adapt(X_win, y_win)
            self.online_weight = self.post_drift_boost_weight
        else:
            self.batch_ensemble.buffer_X.extend(X_win)
            self.batch_ensemble.buffer_y.extend(y_win)
            if self.batch_ensemble.buffer_cap is not None and len(self.batch_ensemble.buffer_X) > self.batch_ensemble.buffer_cap:
                self.batch_ensemble.buffer_X = self.batch_ensemble.buffer_X[-self.batch_ensemble.buffer_cap:]
                self.batch_ensemble.buffer_y = self.batch_ensemble.buffer_y[-self.batch_ensemble.buffer_cap:]
            retrained = {c: False for c in self.batch_ensemble.component_names}
            batch_res = {}
            samples = 0
            self.online_weight = max(self.baseline_online_weight, self.online_weight * self.decay_factor)

        # Combine timing
        total_retrain_cpu = sum(r['total_cpu_time'] for r in batch_res.values()) + online_res['total_cpu_time']
        total_retrain_wall = sum(r['wall_clock_time'] for r in batch_res.values()) + online_res['wall_clock_time']

        return retrained, batch_res, samples, total_retrain_cpu, total_retrain_wall
