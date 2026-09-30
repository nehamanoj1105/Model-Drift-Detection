"""
Event-Driven Adaptive Ensemble baseline for the REVISED Experiment 9A.

Identical drift detector to the existing 9A/9B harness: a rolling error window
triggers a full retrain when the current window error exceeds mu + k*sigma of
the recent error history. Retraining uses the class-anchored buffer
(see models_9a) so all adaptive models share the same retraining substrate.
"""

import time
import numpy as np
from models_9a import create_base_ensemble, make_train_buffer


class EventDrivenEnsemble:
    def __init__(self, seed=42, buffer_capacity=1000, error_window_size=20,
                 error_threshold_k=2.0, anchor_X=None, anchor_y=None):
        self.seed = seed
        self.buffer_capacity = buffer_capacity
        self.error_window_size = error_window_size
        self.error_threshold_k = error_threshold_k
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y

        self.ensemble = create_base_ensemble(seed=seed)
        self.is_fitted = False
        self.buffer_X = []
        self.buffer_y = []
        self.recent_errors = []
        self.retrain_events = 0
        self.total_trees_trained = 0
        self.cumulative_cpu_time = 0.0
        self.cumulative_wall_time = 0.0

    def fit_initial(self, X_init, y_init):
        t0w, t0c = time.perf_counter(), time.process_time()
        self.ensemble.fit(X_init, y_init)
        self.is_fitted = True
        cpu = time.process_time() - t0c
        wall = time.perf_counter() - t0w
        self.cumulative_cpu_time += cpu
        self.cumulative_wall_time += wall
        self.total_trees_trained += self.ensemble.get_num_trees()
        self.buffer_X = list(X_init[-self.buffer_capacity:])
        self.buffer_y = list(y_init[-self.buffer_capacity:])
        return cpu, wall

    def predict(self, X_win):
        return self.ensemble.predict(X_win)

    def predict_proba(self, X_win):
        return self.ensemble.predict_proba(X_win)

    def update_and_adapt(self, X_win, y_win, win_error):
        self.recent_errors.append(win_error)
        if len(self.recent_errors) > self.error_window_size:
            self.recent_errors.pop(0)

        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        if len(self.buffer_X) > self.buffer_capacity:
            self.buffer_X = self.buffer_X[-self.buffer_capacity:]
            self.buffer_y = self.buffer_y[-self.buffer_capacity:]

        triggered = False
        cpu_spent, wall_spent = 0.0, 0.0

        if len(self.recent_errors) >= 5:
            mu_err = np.mean(self.recent_errors[:-1])
            sigma_err = np.std(self.recent_errors[:-1])
            curr_err = self.recent_errors[-1]
            threshold = mu_err + self.error_threshold_k * max(sigma_err, 0.05)
            if curr_err > threshold:
                triggered = True

        if triggered:
            self.retrain_events += 1
            train_X, train_y = make_train_buffer(
                np.array(self.buffer_X), np.array(self.buffer_y),
                self.anchor_X, self.anchor_y, self.buffer_capacity,
            )
            t0w, t0c = time.perf_counter(), time.process_time()
            self.ensemble = create_base_ensemble(seed=self.seed + self.retrain_events * 13)
            self.ensemble.fit(train_X, train_y)
            cpu_spent = time.process_time() - t0c
            wall_spent = time.perf_counter() - t0w
            self.cumulative_cpu_time += cpu_spent
            self.cumulative_wall_time += wall_spent
            self.total_trees_trained += self.ensemble.get_num_trees()
            self.recent_errors = [win_error]

        return triggered, cpu_spent, wall_spent
