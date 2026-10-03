"""
Drift detectors driving the shared retraining harness.

Exact provenance (for correcting the paper's detector names):
  ADWIN          -> river.drift.ADWIN(delta=0.002)
  Page-Hinkley   -> river.drift.PageHinkley(min_instances=30, delta=0.005,
                    threshold=50, alpha=0.9999)
  EDDM           -> river.drift.binary.EDDM(warm_start=30, alpha=0.95, beta=0.9).
                    The paper's label "EDD" denotes the Early Drift Detection
                    Method, i.e. River's EDDM (Baena-Garcia et al. 2006). River's
                    EDDM exposes warm_start/alpha/beta, NOT warning/drift levels;
                    those belong to DDM.
  ECDD-EWMA      -> custom Exponentially-Weighted-Moving-Average control chart
                    (Ross et al. 2012). River 0.26 ships no ECDD/EWMA drift
                    detector, so it is implemented here. The paper's label
                    "EDMA" is this estimator and should be renamed ECDD-EWMA.

ADWIN, Page-Hinkley and ECDD-EWMA are fed the window error (1 - window
accuracy), one value per window, the same signal the Event-Driven baseline uses.
EDDM is a binary detector and is fed the per-instance error bit stream, its
native input, because the per-window aggregate discards the within-window error
structure it relies on. Model, class-anchored buffer, buffer capacity and minimum
retrain interval are identical across detectors.

Note: River 0.26's ADWIN / PageHinkley / EDDM `update` return None and expose the
outcome on `drift_detected`; `_detector_fired` normalizes both conventions.
"""

from __future__ import annotations

import time
import numpy as np
from river.drift import ADWIN, PageHinkley
from river.drift.binary import EDDM

from models import create_ensemble, make_train_buffer


class PersistEDDM:
    """river.drift.binary.EDDM adapted to the per-window error signal.

    River's EDDM only advances its internal distance statistics when the error
    bit is 1 (`total += pred`), so an all-correct stream never updates it. The
    harness feeds one aggregated error per window; we pass the bit and expose a
    drift flag each window.
    """

    def __init__(self, warm_start=30, alpha=0.95, beta=0.9):
        self._d = EDDM(warm_start=warm_start, alpha=alpha, beta=beta)

    def update(self, error):
        # River's EDDM.update returns None and sets .drift_detected; surface it.
        self._d.update(bool(error))
        return bool(self._d.drift_detected)


class ECDDEWMA:
    """Exponentially-Weighted-Moving-Average drift detector (Ross et al. 2012)."""

    def __init__(self, alpha=0.2, k=2.0, min_instances=30):
        self.alpha, self.k, self.min_instances = alpha, k, min_instances
        self.n = 0
        self.ewma = None
        self.ewvar = 0.0

    def update(self, error):
        self.n += 1
        if self.ewma is None:
            self.ewma = float(error)
            return False
        dev = error - self.ewma
        self.ewma += self.alpha * dev
        self.ewvar = (1 - self.alpha) * (self.ewvar + self.alpha * dev * dev)
        sd = np.sqrt(self.ewvar) if self.ewvar > 0 else 0.0
        return bool(self.n >= self.min_instances and error > self.ewma + self.k * max(sd, 0.02))


RIVER_CLASS_NAMES = {
    "ADWIN": "river.drift.ADWIN",
    "Page-Hinkley": "river.drift.PageHinkley",
    "EDDM": "river.drift.binary.EDDM",
    "ECDD-EWMA": "custom (no River class in 0.26; Ross et al. 2012)",
}


def make_detector(name, cfg):
    d = cfg["detectors"]
    if name == "ADWIN":
        return ADWIN(delta=d["adwin"]["delta"])
    if name == "Page-Hinkley":
        p = d["page_hinkley"]
        return PageHinkley(min_instances=p["min_instances"], delta=p["delta"],
                           threshold=p["threshold"], alpha=p["alpha"])
    if name == "EDDM":
        e = d["eddm"]
        return PersistEDDM(warm_start=e["warm_start"], alpha=e["alpha"], beta=e["beta"])
    if name == "ECDD-EWMA":
        e = d["ecdd_ewma"]
        return ECDDEWMA(alpha=e["alpha"], k=e["k"], min_instances=e["min_instances"])
    raise ValueError(name)


def _detector_fired(detector, x):
    """Normalize a drift-detector update to a boolean.

    River 0.26's ADWIN / PageHinkley / EDDM `update` return None and instead
    expose the outcome on the `drift_detected` attribute; the custom ECDD-EWMA
    returns the boolean itself. Handle both so no detector is silently inert.
    """
    ret = detector.update(x)
    if ret is not None:
        return bool(ret)
    return bool(getattr(detector, "drift_detected", False))


class DetectorModel:
    def __init__(self, name, seed, cfg, anchor):
        self.name, self.seed, self.cfg = name, seed, cfg
        self.anchor_X, self.anchor_y = anchor
        self.detector = make_detector(name, cfg)
        self.cap = cfg["buffer_capacity"]
        self.min_interval = cfg["detector_min_retrain_interval"]
        self.retrains = self.detected = 0
        self.trees_trained = 0
        self._last_retrain = -10 ** 9

    def fit_initial(self, X, y):
        t = time.process_time()
        self.model = create_ensemble(self.seed, self.cfg).fit(X, y)
        self.trees_trained += self.model.get_num_trees()
        self.buffer_X = list(X[-self.cap:])
        self.buffer_y = list(y[-self.cap:])
        return {"adapt_cpu": time.process_time() - t, "event_type": "retrain"}

    def predict(self, X):
        return self.model.predict(X)

    def update(self, X, y, ctx):
        self.buffer_X.extend(X)
        self.buffer_y.extend(y)
        if len(self.buffer_X) > self.cap:
            self.buffer_X = self.buffer_X[-self.cap:]
            self.buffer_y = self.buffer_y[-self.cap:]

        if self.name == "EDDM":
            # EDDM is a binary detector: feed it the per-instance error bit
            # stream (River's design), not the per-window aggregate, so it can
            # actually fire on a 500-observation window.
            bits = (np.asarray(ctx["y_pred"]) != np.asarray(y)).astype(int)
            fired = False
            for b in bits:
                if _detector_fired(self.detector, b):
                    fired = True
        else:
            err = 1.0 - float(np.mean(ctx["y_pred"] == y))
            fired = _detector_fired(self.detector, err)
        if fired:
            self.detected += 1
        wid = ctx["window_id"]
        if not fired or (wid - self._last_retrain) < self.min_interval:
            return {"event_type": "none", "adapt_cpu": 0.0}

        self.retrains += 1
        self._last_retrain = wid
        tX, tY = make_train_buffer(np.array(self.buffer_X), np.array(self.buffer_y),
                                   self.anchor_X, self.anchor_y, self.cap)
        t = time.process_time()
        self.model = create_ensemble(self.seed + self.retrains * 17, self.cfg).fit(tX, tY)
        cpu = time.process_time() - t
        self.trees_trained += self.model.get_num_trees()
        return {"event_type": "retrain", "adapt_cpu": cpu}
