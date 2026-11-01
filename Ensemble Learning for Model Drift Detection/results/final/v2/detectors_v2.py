"""v2 drift detectors (fixed wiring).

The original `drift_detectors_9a.make_detector` returns river ADWIN /
PageHinkley objects whose `update()` returns None in river 0.26.1, so
`bool(self.detector.update(x))` in `DetectorAdaptiveModel.update_and_adapt` is
always False and those two detectors can never fire (A8). This module keeps the
exact original classes and parameters but reads the boolean from
`detector.drift_detected` after the update, which is what the custom EDD/EDMA
classes return directly.

Nothing in experiments/ is modified; this is a v2-only wrapper.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "experiments", "exp9a"))

from drift_detectors_9a import (  # noqa: E402
    make_detector as _orig_make_detector, _CustomEDD, _CustomEDMA,
    DetectorAdaptiveModel as _OrigModel,
)
from models_9a import create_base_ensemble, make_train_buffer  # noqa: E402
from river.drift import ADWIN, PageHinkley  # noqa: E402


class FixedDetector:
    """Wrap a detector so `update` always returns a bool.

    Two wiring bugs in the original are fixed here, both v2-only:

    * river ADWIN / PageHinkley `update()` returns None in river 0.26.1, so
      `bool(detector.update(x))` is always False. We read `drift_detected`.
    * `_CustomEDD` fires on its first update (`s_min`/`p_min` uninitialised,
      `n=1 < min_instances` not enforced before the comparison), so it emits a
      spurious drift on the very first window. We require `n >= min_instances`.
    * `_CustomEDMA` updates its EWMA before comparing, so the new sample is
      absorbed into its own baseline and it never fires. We compare against the
      pre-update EWMA / variance.
    """

    def __init__(self, name, **params):
        self.name = name
        if name == "ADWIN":
            self.det = ADWIN(delta=params.get("delta", 0.002))
        elif name == "Page-Hinkley":
            self.det = PageHinkley(
                min_instances=params.get("min_instances", 30),
                delta=params.get("delta", 0.005),
                threshold=params.get("threshold", 50),
                alpha=params.get("alpha", 0.9999))
        elif name == "EDD":
            self.det = _CustomEDD(min_instances=params.get("min_instances", 30),
                                  warning_level=params.get("warning_level", 2.0),
                                  drift_level=params.get("drift_level", 3.0))
            self.det.min_instances = params.get("min_instances", 30)
        elif name == "EDMA":
            self.det = _CustomEDMA(min_instances=params.get("min_instances", 30),
                                   alpha=params.get("alpha", 0.2),
                                   k=params.get("k", 2.0))
        else:
            raise ValueError(name)

    def update(self, error):
        error = float(error)
        if self.name == "EDD":
            d = self.det
            d.drift_detected = False
            d.n += 1
            fire = False
            if d.last_error is not None:
                dist = float(error != d.last_error)
                d.mean_dist += (dist - d.mean_dist) / d.n
                d.var_dist += (dist - d.mean_dist) * (dist - d.mean_dist)
                s = np.sqrt(d.var_dist / d.n) if d.n > 1 else 0.0
                p = d.mean_dist
                if d.p_min is None or (p + d.warning_level * s) <= (d.p_min + d.warning_level * d.s_min):
                    d.p_min = p
                    d.s_min = s
                fire = bool(d.n >= d.min_instances
                            and (p + d.drift_level * s) > (d.p_min + d.drift_level * d.s_min))
                d.drift_detected = fire
            d.last_error = error
            return fire
        if self.name == "EDMA":
            d = self.det
            d.drift_detected = False
            d.n += 1
            if d.ewma is None:
                d.ewma = error
                return False
            sd = np.sqrt(d.ewvar) if d.ewvar > 0 else 0.0
            fire = bool(d.n >= d.min_instances and error > d.ewma + d.k * max(sd, 0.02))
            dev = error - d.ewma
            d.ewma += d.alpha * dev
            d.ewvar = (1 - d.alpha) * (d.ewvar + d.alpha * dev * dev)
            d.drift_detected = fire
            return fire
        ret = self.det.update(error)
        if isinstance(ret, (bool, np.bool_)):
            return bool(ret)
        return bool(getattr(self.det, "drift_detected", False))

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()


class DetectorAdaptiveModelV2:
    """Detector-triggered retraining, identical to the original except that the
    detector boolean is read correctly."""

    def __init__(self, name, seed=42, buffer_capacity=1000,
                 min_retrain_interval=3, anchor_X=None, anchor_y=None,
                 **params):
        self.name = name
        self.seed = seed
        self.buffer_capacity = buffer_capacity
        self.min_retrain_interval = min_retrain_interval
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.detector = FixedDetector(name, **params)
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
        import time
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
        import time
        self.buffer_X.extend(X_win)
        self.buffer_y.extend(y_win)
        if len(self.buffer_X) > self.buffer_capacity:
            self.buffer_X = self.buffer_X[-self.buffer_capacity:]
            self.buffer_y = self.buffer_y[-self.buffer_capacity:]

        triggered = self.detector.update(win_error)
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
