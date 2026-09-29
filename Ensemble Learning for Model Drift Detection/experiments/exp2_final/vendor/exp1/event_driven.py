"""
Event-Driven Adaptive Ensemble Baseline Module for Exp 9B
Detects performance drops / error spikes streaming over time and retrains ensemble on historical buffer.
Identical mechanism to Exp 9A for rigorous cross-experiment comparability.
"""

import time
import numpy as np
from heterogeneous_ensemble import create_base_ensemble

class EventDrivenEnsemble:
    """
    Event-driven adaptive ensemble with rolling error drift detector.
    """
    def __init__(self, seed=42, buffer_capacity=1000, error_window_size=20, error_threshold_k=2.0):
        self.seed = seed
        self.buffer_capacity = buffer_capacity
        self.error_window_size = error_window_size
        self.error_threshold_k = error_threshold_k
        
        self.ensemble = create_base_ensemble(seed=seed)
        self.is_fitted = False
        
        # Historical buffer
        self.buffer_X = []
        self.buffer_y = []
        
        # Detector tracking
        self.recent_errors = []
        self.retrain_events = 0
        self.total_trees_trained = 0
        self.cumulative_cpu_time = 0.0
        self.cumulative_wall_time = 0.0
        
    def fit_initial(self, X_init, y_init):
        """Perform initial training on historical prefix."""
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()
        
        self.ensemble.fit(X_init, y_init)
        self.is_fitted = True
        
        t_end_cpu = time.process_time()
        t_end_wall = time.perf_counter()
        
        cpu_spent = t_end_cpu - t_start_cpu
        wall_spent = t_end_wall - t_start_wall
        
        self.cumulative_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        self.total_trees_trained += self.ensemble.get_num_trees()
        
        # Initialize buffer
        self.buffer_X = list(X_init[-self.buffer_capacity:])
        self.buffer_y = list(y_init[-self.buffer_capacity:])
        return cpu_spent, wall_spent

    def predict(self, X_win):
        """Predict class labels for window."""
        return self.ensemble.predict(X_win)

    def predict_proba(self, X_win):
        """Predict class probabilities for window."""
        return self.ensemble.predict_proba(X_win)

    def update_and_adapt(self, X_win, y_win, win_error):
        """
        Prequential post-prediction update step:
          1. Record window error in rolling window.
          2. Append window data to historical buffer.
          3. Check if rolling error exceeds trigger condition: error > mu + k * sigma
          4. If triggered: retrain ensemble from scratch on historical buffer.
        """
        # 1. Update rolling error window
        self.recent_errors.append(win_error)
        if len(self.recent_errors) > self.error_window_size:
            self.recent_errors.pop(0)
            
        # 2. Update buffer
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        if len(self.buffer_X) > self.buffer_capacity:
            self.buffer_X = self.buffer_X[-self.buffer_capacity:]
            self.buffer_y = self.buffer_y[-self.buffer_capacity:]
            
        # 3. Check trigger
        triggered = False
        cpu_spent = 0.0
        wall_spent = 0.0
        
        if len(self.recent_errors) >= 5:
            mu_err = np.mean(self.recent_errors[:-1])
            sigma_err = np.std(self.recent_errors[:-1])
            curr_err = self.recent_errors[-1]
            
            # Trigger condition: error > mu + k * sigma (with min std floor)
            threshold = mu_err + self.error_threshold_k * max(sigma_err, 0.05)
            if curr_err > threshold:
                triggered = True
                
        # 4. Retrain if triggered
        if triggered:
            self.retrain_events += 1
            train_X = np.array(self.buffer_X)
            train_y = np.array(self.buffer_y)
            
            t_start_wall = time.perf_counter()
            t_start_cpu = time.process_time()
            
            # Retrain clean ensemble instance with updated seed offset
            self.ensemble = create_base_ensemble(seed=self.seed + self.retrain_events * 13)
            self.ensemble.fit(train_X, train_y)
            
            t_end_cpu = time.process_time()
            t_end_wall = time.perf_counter()
            
            cpu_spent = t_end_cpu - t_start_cpu
            wall_spent = t_end_wall - t_start_wall
            
            self.cumulative_cpu_time += cpu_spent
            self.cumulative_wall_time += wall_spent
            self.total_trees_trained += self.ensemble.get_num_trees()
            
            # Reset rolling errors after retraining
            self.recent_errors = [win_error]
            
        return triggered, cpu_spent, wall_spent
