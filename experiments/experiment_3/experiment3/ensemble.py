"""
================================================================================
EXPERIMENT 3 — ADAPTIVE HETEROGENEOUS ENSEMBLE
================================================================================
Strategy 3:
- Heterogeneous ensemble of Random Forest, ExtraTrees, Gradient Boosting.
- Adaptive performance-based weights:
  Weight_RF + Weight_ET + Weight_GB = 1.0
  Weight update dynamically shifts probability toward higher-performing models.
- Adaptive retraining decision:
  Only adapts/retrains when drift or performance degradation is detected.
  Selectively retrains underperforming components to minimize CPU/time cost.
================================================================================
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from metrics import evaluate_predictions
from models import create_candidate_models
from resource_monitor import ResourceMonitor, measure_execution


class AdaptiveEnsemble:
    """
    Heterogeneous ensemble with dynamic performance-based weighting
    and selective drift-triggered adaptation.
    """

    def __init__(self, seed, temperature=15.0, drift_threshold=0.15, deg_threshold=0.08, ema_alpha=0.7):
        self.seed = seed
        self.temperature = temperature
        self.drift_threshold = drift_threshold
        self.deg_threshold = deg_threshold
        self.ema_alpha = ema_alpha
        
        self.models = create_candidate_models(seed)
        self.model_names = ['RandomForest', 'ExtraTrees', 'GradientBoosting']
        
        # Initial weights: equal
        self.weights = {m: 1.0 / len(self.model_names) for m in self.model_names}
        self.ema_scores = {m: 0.85 for m in self.model_names}
        
        # Track historical component performance
        self.history = {m: {'f1': [], 'acc': [], 'reward': []} for m in self.model_names}
        self.retrain_counts = {m: 0 for m in self.model_names}
        self.total_retrain_events = 0
        
        # Baseline reference performance
        self.baseline_f1 = 0.90
        
        # Cumulative training buffer (starts with initial training data)
        self.buffer_X = []
        self.buffer_y = []

    def fit_initial(self, X_train, y_train):
        """Fit all component models on initial 2,000 samples."""
        self.buffer_X = list(X_train)
        self.buffer_y = list(y_train)
        
        monitor = ResourceMonitor()
        monitor.start()
        for name in self.model_names:
            self.models[name].fit(X_train, y_train)
        res = monitor.stop()
        return res

    def predict_proba(self, X):
        """
        Compute ensemble predicted probability:
        P(Y=1|X) = sum_m (w_m * P_m(Y=1|X))
        """
        prob_ensemble = np.zeros(len(X))
        comp_probs = {}
        for name in self.model_names:
            p = self.models[name].predict_proba(X)[:, 1]
            comp_probs[name] = p
            prob_ensemble += self.weights[name] * p
        return prob_ensemble, comp_probs

    def predict(self, X):
        """
        Inference with individual component profiling and ensemble combination.
        Returns predictions, probabilities, component predictions/probabilities,
        component resource profiles, and aggregate ensemble resource profile.
        """
        prob_ensemble = np.zeros(len(X))
        comp_probs = {}
        comp_preds = {}
        comp_res_inf = {}
        
        for name in self.model_names:
            model = self.models[name]
            p, res = measure_execution(lambda m=model: m.predict_proba(X)[:, 1])
            comp_probs[name] = p
            comp_preds[name] = (p >= 0.5).astype(int)
            comp_res_inf[name] = res
            prob_ensemble += self.weights[name] * p
            
        pred_ensemble = (prob_ensemble >= 0.5).astype(int)
        
        # Aggregate inference resources for the ensemble
        total_wall = sum(comp_res_inf[m]['wall_clock_time'] for m in self.model_names)
        total_cpu = sum(comp_res_inf[m]['total_cpu_time'] for m in self.model_names)
        cpu_user = sum(comp_res_inf[m]['cpu_user_time'] for m in self.model_names)
        cpu_sys = sum(comp_res_inf[m]['cpu_system_time'] for m in self.model_names)
        avg_cpu = float(np.mean([comp_res_inf[m]['avg_cpu_percent'] for m in self.model_names]))
        peak_cpu = float(max([comp_res_inf[m]['peak_cpu_percent'] for m in self.model_names]))
        avg_ram = float(np.mean([comp_res_inf[m]['avg_ram_mb'] for m in self.model_names]))
        peak_ram = float(max([comp_res_inf[m]['peak_ram_mb'] for m in self.model_names]))
        
        aggregate_res_inf = {
            'wall_clock_time': total_wall,
            'total_cpu_time': total_cpu,
            'cpu_user_time': cpu_user,
            'cpu_system_time': cpu_sys,
            'avg_cpu_percent': avg_cpu,
            'peak_cpu_percent': peak_cpu,
            'avg_ram_mb': avg_ram,
            'peak_ram_mb': peak_ram,
        }
        
        return pred_ensemble, prob_ensemble, comp_preds, comp_probs, comp_res_inf, aggregate_res_inf

    def update_weights(self, y_true, comp_preds, comp_probs=None, lambda_cost=0.05):
        """
        Evaluate recent component performance and update normalized ensemble weights
        using a softmax over recent performance.
        """
        comp_metrics = {}
        recent_scores = []
        
        for name in self.model_names:
            prob = comp_probs.get(name) if comp_probs is not None else None
            m_eval = evaluate_predictions(y_true, comp_preds[name], prob)
            f1 = m_eval['f1']
            acc = m_eval['accuracy']
            prec = m_eval['precision']
            rec = m_eval['recall']
            auc = m_eval['auc']
            
            # Recent reward
            reward = f1 - lambda_cost * (0.2 if self.retrain_counts[name] > 0 else 0.0)
            
            self.history[name]['f1'].append(f1)
            self.history[name]['acc'].append(acc)
            self.history[name]['reward'].append(reward)
            
            # Update exponential moving average of performance for responsive adaptation
            if not hasattr(self, 'ema_scores') or name not in self.ema_scores:
                if not hasattr(self, 'ema_scores'):
                    self.ema_scores = {}
                self.ema_scores[name] = f1
            else:
                alpha = getattr(self, 'ema_alpha', 0.7)
                self.ema_scores[name] = alpha * f1 + (1.0 - alpha) * self.ema_scores[name]
                
            recent_scores.append(self.ema_scores[name])
            
            comp_metrics[name] = {
                'f1': f1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'auc': auc,
                'reward': reward,
                'rolling_f1': float(np.mean(self.history[name]['f1'][-3:])),
                'ema_f1': self.ema_scores[name]
            }
            
        # Softmax weighting
        scores_arr = np.array(recent_scores)
        exp_scores = np.exp(self.temperature * (scores_arr - np.max(scores_arr)))
        new_weights = exp_scores / np.sum(exp_scores)
        
        for idx, name in enumerate(self.model_names):
            self.weights[name] = float(new_weights[idx])
            
        w_values = list(self.weights.values())
        weight_diff = float(np.max(w_values) - np.min(w_values))
        
        return comp_metrics, self.weights, weight_diff

    def adapt_if_needed(self, X_win, y_win, drift_metrics, ensemble_f1):
        """
        Section 9: Adaptive retraining decision.
        Evaluates drift and performance degradation to selectively retrain components.
        Measures execution for each candidate component individually and computes aggregate.
        Returns:
        - adaptation_occurred (bool)
        - adapted_components (str)
        - comp_res_train (dict of component resource metrics)
        - aggregate_res_train (dict of ensemble resource metrics)
        """
        # Append window data to buffer
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        
        w_mean = drift_metrics.get('wasserstein_mean', 0.0)
        deg = max(0.0, self.baseline_f1 - ensemble_f1)
        
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
        comp_res_train = {name: dict(empty_res) for name in self.model_names}
        
        # Check if adaptation should trigger
        should_adapt = (w_mean > self.drift_threshold) or (deg > self.deg_threshold)
        
        if not should_adapt:
            return False, 'None', comp_res_train, dict(empty_res)
            
        # Identify underperforming components to selectively adapt
        component_scores = {m: np.mean(self.history[m]['f1'][-2:]) if len(self.history[m]['f1']) > 0 else 1.0
                            for m in self.model_names}
        min_score = min(component_scores.values())
        
        # Select components within 0.03 of the minimum score to retrain
        to_retrain = [m for m, sc in component_scores.items() if sc <= min_score + 0.03]
        if not to_retrain:
            to_retrain = [min(component_scores, key=component_scores.get)]
            
        # Use recent sliding buffer (up to 2,000 samples) to train efficiently and adapt fast
        max_buffer = 2000
        train_X = np.array(self.buffer_X[-max_buffer:])
        train_y = np.array(self.buffer_y[-max_buffer:])
        
        for name in self.model_names:
            if name in to_retrain:
                _, res_m = measure_execution(self.models[name].fit, train_X, train_y)
                comp_res_train[name] = res_m
                self.retrain_counts[name] += 1
                self.total_retrain_events += 1
            else:
                comp_res_train[name] = dict(empty_res)
                
        # Aggregate adaptation resources across retrained components
        retrained_names = [name for name in self.model_names if name in to_retrain]
        total_wall = sum(comp_res_train[m]['wall_clock_time'] for m in retrained_names)
        total_cpu = sum(comp_res_train[m]['total_cpu_time'] for m in retrained_names)
        cpu_user = sum(comp_res_train[m]['cpu_user_time'] for m in retrained_names)
        cpu_sys = sum(comp_res_train[m]['cpu_system_time'] for m in retrained_names)
        avg_cpu = float(np.mean([comp_res_train[m]['avg_cpu_percent'] for m in retrained_names])) if retrained_names else 0.0
        peak_cpu = float(max([comp_res_train[m]['peak_cpu_percent'] for m in retrained_names])) if retrained_names else 0.0
        avg_ram = float(np.mean([comp_res_train[m]['avg_ram_mb'] for m in retrained_names])) if retrained_names else 0.0
        peak_ram = float(max([comp_res_train[m]['peak_ram_mb'] for m in retrained_names])) if retrained_names else 0.0
        
        aggregate_res_train = {
            'wall_clock_time': total_wall,
            'total_cpu_time': total_cpu,
            'cpu_user_time': cpu_user,
            'cpu_system_time': cpu_sys,
            'avg_cpu_percent': avg_cpu,
            'peak_cpu_percent': peak_cpu,
            'avg_ram_mb': avg_ram,
            'peak_ram_mb': peak_ram,
        }
        
        adapted_str = ','.join(to_retrain)
        return True, adapted_str, comp_res_train, aggregate_res_train
