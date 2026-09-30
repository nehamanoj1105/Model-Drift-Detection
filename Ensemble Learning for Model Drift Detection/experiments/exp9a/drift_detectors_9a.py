"""
Historical drift-detection baselines for Experiment 9A (three telecom datasets).

Each detector answers only WHEN to adapt; the predictive model, the retraining
buffer (class-anchored), the buffer capacity and the minimum retrain interval
are identical across detectors and to the primary models. Detector parameters
are fixed a priori (documented below); no parameter is tuned on the test stream.

Detectors
---------
ADWIN         river.drift.ADWIN(delta=0.002) on the window error signal.
Page-Hinkley  river.drift.PageHinkley(min_instances=30, delta=0.005,
              threshold=50, alpha=0.9999) on the window error signal.
EDD           Early Drift Detection Method (custom, Baena-Garcia et al. 2006):
              tracks the mean/std of the distance between consecutive errors and
              signals drift when (p_i + 2*s_i) > (p_min + 3*s_min).
EDMA          Error-Distance Moving Average (custom): EWMA of the window error;
              drift when the current error exceeds EWMA + k*sigma_EWMA.

The window error signal is 1 - window accuracy, i.e. the same signal used by the
Event-Driven baseline.
"""

import time
import numpy as np

from river.drift import ADWIN, PageHinkley

from models_9a import create_base_ensemble, make_train_buffer


class _CustomEDD:
    def __init__(self, min_instances=30, warning_level=2.0, drift_level=3.0):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset(self):
        self.n = 0
        self.last_error = None
        self.mean_dist = 0.0
        self.var_dist = 0.0
        self.p_min = None
        self.s_min = None
        self.drift_detected = False
        self.warning_detected = False

    def update(self, error):
        self.drift_detected = False
        self.warning_detected = False
        self.n += 1
        if self.last_error is not None:
            dist = float(error != self.last_error)
            self.mean_dist += (dist - self.mean_dist) / self.n
            self.var_dist += (dist - self.mean_dist) * (dist - self.mean_dist)
            s = np.sqrt(self.var_dist / self.n) if self.n > 1 else 0.0
            p = self.mean_dist
            if self.p_min is None or (p + self.warning_level * s) <= (self.p_min + self.warning_level * self.s_min):
                self.p_min = p
                self.s_min = s
            if (p + self.drift_level * s) > (self.p_min + self.drift_level * self.s_min) and self.n >= self.min_instances:
                self.drift_detected = True
            elif (p + self.warning_level * s) > (self.p_min + self.warning_level * self.s_min):
                self.warning_detected = True
        self.last_error = error
        return self.drift_detected


class _CustomEDMA:
    def __init__(self, min_instances=30, alpha=0.2, k=2.0):
        self.min_instances = min_instances
        self.alpha = alpha
        self.k = k
        self.reset()

    def reset(self):
        self.n = 0
        self.ewma = None
        self.ewvar = 0.0
        self.drift_detected = False

    def update(self, error):
        self.drift_detected = False
        self.n += 1
        if self.ewma is None:
            self.ewma = float(error)
            return False
        dev = error - self.ewma
        self.ewma += self.alpha * dev
        self.ewvar = (1 - self.alpha) * (self.ewvar + self.alpha * dev * dev)
        sd = np.sqrt(self.ewvar) if self.ewvar > 0 else 0.0
        if self.n >= self.min_instances and error > self.ewma + self.k * max(sd, 0.02):
            self.drift_detected = True
        return self.drift_detected


def make_detector(name):
    if name == "ADWIN":
        return ADWIN(delta=0.002)
    if name == "Page-Hinkley":
        return PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
    if name == "EDD":
        return _CustomEDD()
    if name == "EDMA":
        return _CustomEDMA()
    raise ValueError(name)


class DetectorAdaptiveModel:
    """Detector-triggered retraining on the shared class-anchored buffer."""

    def __init__(self, name, seed=42, buffer_capacity=1000, min_retrain_interval=3,
                 anchor_X=None, anchor_y=None):
        self.name = name
        self.seed = seed
        self.buffer_capacity = buffer_capacity
        self.min_retrain_interval = min_retrain_interval
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.detector = make_detector(name)

        self.ensemble = create_base_ensemble(seed=seed)
        self.is_fitted = False
        self.buffer_X = []
        self.buffer_y = []
        self.retrain_events = 0
        self.detected_events = 0
        self.total_trees_trained = 0
        self.cumulative_cpu_time = 0.0
        self.cumulative_wall_time = 0.0
        self._last_retrain_window = -10 ** 9

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

    def update_and_adapt(self, window_id, X_win, y_win, win_error):
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        if len(self.buffer_X) > self.buffer_capacity:
            self.buffer_X = self.buffer_X[-self.buffer_capacity:]
            self.buffer_y = self.buffer_y[-self.buffer_capacity:]

        triggered = bool(self.detector.update(win_error))
        if triggered:
            self.detected_events += 1

        did_retrain = False
        cpu_spent, wall_spent = 0.0, 0.0
        if triggered and (window_id - self._last_retrain_window) >= self.min_retrain_interval:
            did_retrain = True
            self.retrain_events += 1
            self._last_retrain_window = window_id
            train_X, train_y = make_train_buffer(
                np.array(self.buffer_X), np.array(self.buffer_y),
                self.anchor_X, self.anchor_y, self.buffer_capacity)
            t0w, t0c = time.perf_counter(), time.process_time()
            self.ensemble = create_base_ensemble(seed=self.seed + self.retrain_events * 17)
            self.ensemble.fit(train_X, train_y)
            cpu_spent = time.process_time() - t0c
            wall_spent = time.perf_counter() - t0w
            self.cumulative_cpu_time += cpu_spent
            self.cumulative_wall_time += wall_spent
            self.total_trees_trained += self.ensemble.get_num_trees()

        return triggered, did_retrain, cpu_spent, wall_spent
