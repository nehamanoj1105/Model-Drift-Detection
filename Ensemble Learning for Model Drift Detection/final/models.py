"""
Base ensemble, primary models, RAPT variants and detector-driven models.

Every model shares one base architecture (RandomForest + ExtraTrees, soft voting,
same tree count/depth), one class-anchored buffer, one buffer capacity and one
initial prefix. Differences are confined to the adaptation mechanism.

All adaptation is timed with time.process_time() (adapt_cpu) and every model
exposes the same interface:

    fit_initial(X, y)
    predict(X) -> labels
    update(X_win, y_win, ctx) -> dict(event_type=..., adapt_cpu=..., n_trees=...)

`event_type` is one of {retrain, reuse, refresh, none}.
"""

from __future__ import annotations

import copy
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score

EVENT_TYPES = ("retrain", "reuse", "refresh", "none")


# ---------------------------------------------------------------------------
class Ensemble:
    def __init__(self, seed=42, n_estimators=50, max_depth=7, weights=None):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.weights = list(weights) if weights is not None else [0.5, 0.5]
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         random_state=seed, n_jobs=1)
        self.et = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=seed, n_jobs=1)
        self.is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        p_rf, p_et = self.rf.predict_proba(X), self.et.predict_proba(X)
        w1, w2 = self.weights
        s = w1 + w2
        if s <= 0:
            w1, w2, s = 0.5, 0.5, 1.0
        return (w1 / s) * p_rf + (w2 / s) * p_et

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_num_trees(self):
        return len(self.rf.estimators_) + len(self.et.estimators_)


def create_ensemble(seed, cfg):
    e = cfg["ensemble"]
    return Ensemble(seed=seed, n_estimators=e["n_estimators"],
                    max_depth=e["max_depth"], weights=e["weights"])


def make_class_anchor(X_init, y_init, per_class=15, seed=0):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y_init):
        ci = np.where(y_init == c)[0]
        idx.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False).tolist())
    idx = np.array(sorted(idx))
    return X_init[idx], y_init[idx]


def make_train_buffer(recent_X, recent_y, anchor_X, anchor_y, cap):
    recent_X = np.asarray(recent_X)
    recent_y = np.asarray(recent_y)
    if len(recent_X) > cap:
        recent_X, recent_y = recent_X[-cap:], recent_y[-cap:]
    if anchor_X is not None and len(anchor_X):
        if recent_X.size == 0:
            return anchor_X.copy(), anchor_y.copy()
        return np.vstack([recent_X, anchor_X]), np.concatenate([recent_y, anchor_y])
    return recent_X, recent_y


class BufferMixin:
    def _init_buffer(self, X, y, cap):
        self.buffer_X = list(X[-cap:])
        self.buffer_y = list(y[-cap:])
        self.cap = cap

    def _push_buffer(self, X, y):
        self.buffer_X.extend(X)
        self.buffer_y.extend(y)
        if len(self.buffer_X) > self.cap:
            self.buffer_X = self.buffer_X[-self.cap:]
            self.buffer_y = self.buffer_y[-self.cap:]


# ---------------------------------------------------------------------------
# Primary models
# ---------------------------------------------------------------------------
class Frozen(BufferMixin):
    name = "Frozen"

    def __init__(self, seed, cfg, anchor):
        self.seed, self.cfg, self.anchor = seed, cfg, anchor
        self._init_buffer(*anchor, cfg["buffer_capacity"]) if anchor[0] is not None \
            else (setattr(self, "buffer_X", []), setattr(self, "buffer_y", []),
                  setattr(self, "cap", cfg["buffer_capacity"]))

    def fit_initial(self, X, y):
        t = time.process_time()
        self.model = create_ensemble(self.seed, self.cfg).fit(X, y)
        self.trees_trained = self.model.get_num_trees()
        return {"adapt_cpu": time.process_time() - t, "event_type": "retrain"}

    def predict(self, X):
        return self.model.predict(X)

    def update(self, X, y, ctx):
        return {"event_type": "none", "adapt_cpu": 0.0}


class EventDriven(BufferMixin):
    name = "Event-Driven"

    def __init__(self, seed, cfg, anchor):
        self.seed, self.cfg, self.anchor = seed, cfg, anchor
        self.anchor_X, self.anchor_y = anchor
        self._init_buffer(self.anchor_X, self.anchor_y, cfg["buffer_capacity"])
        p = cfg["event_driven"]
        self.win, self.k = p["error_window_size"], p["error_threshold_k"]
        self.recent = []
        self.retrains = 0

    def fit_initial(self, X, y):
        t = time.process_time()
        self.model = create_ensemble(self.seed, self.cfg).fit(X, y)
        self.trees_trained = self.model.get_num_trees()
        return {"adapt_cpu": time.process_time() - t, "event_type": "retrain"}

    def predict(self, X):
        return self.model.predict(X)

    def update(self, X, y, ctx):
        err = 1.0 - float(np.mean(ctx["y_pred"] == y))
        self.recent.append(err)
        if len(self.recent) > self.win:
            self.recent.pop(0)
        self._push_buffer(X, y)
        trig = False
        if len(self.recent) >= 5:
            mu, sd = float(np.mean(self.recent[:-1])), float(np.std(self.recent[:-1]))
            trig = self.recent[-1] > mu + self.k * max(sd, 0.05)
        if not trig:
            return {"event_type": "none", "adapt_cpu": 0.0}
        self.retrains += 1
        tX, tY = make_train_buffer(np.array(self.buffer_X), np.array(self.buffer_y),
                                   self.anchor_X, self.anchor_y, self.cap)
        t = time.process_time()
        self.model = create_ensemble(self.seed + self.retrains * 13, self.cfg).fit(tX, tY)
        cpu = time.process_time() - t
        self.trees_trained += self.model.get_num_trees()
        self.recent = [err]
        return {"event_type": "retrain", "adapt_cpu": cpu}


class FullRetraining(BufferMixin):
    name = "Full Retraining"

    def __init__(self, seed, cfg, anchor):
        self.seed, self.cfg = seed, cfg
        self.anchor_X, self.anchor_y = anchor
        self._init_buffer(self.anchor_X, self.anchor_y, cfg["buffer_capacity"])
        self.retrains = 0
        self.prev_regime = None

    def fit_initial(self, X, y):
        t = time.process_time()
        self.model = create_ensemble(self.seed, self.cfg).fit(X, y)
        self.trees_trained = self.model.get_num_trees()
        self.prev_regime = None
        return {"adapt_cpu": time.process_time() - t, "event_type": "retrain"}

    def predict(self, X):
        return self.model.predict(X)

    def update(self, X, y, ctx):
        reg = ctx["regime"]
        transition = self.prev_regime is not None and reg != self.prev_regime
        self.prev_regime = reg
        self._push_buffer(X, y)
        if not transition:
            return {"event_type": "none", "adapt_cpu": 0.0}
        self.retrains += 1
        tX, tY = make_train_buffer(np.array(self.buffer_X), np.array(self.buffer_y),
                                   self.anchor_X, self.anchor_y, self.cap)
        t = time.process_time()
        self.model = create_ensemble(self.seed + self.retrains * 11, self.cfg).fit(tX, tY)
        cpu = time.process_time() - t
        self.trees_trained += self.model.get_num_trees()
        return {"event_type": "retrain", "adapt_cpu": cpu}


# ---------------------------------------------------------------------------
# RAPT core
# ---------------------------------------------------------------------------
class Checkpoint:
    def __init__(self, regime, model, weights, window_id, fingerprint,
                 origin_X=None, origin_y=None, buffer_counts=None):
        self.regime, self.model = regime, model
        self.weights = list(weights)
        self.created_window = window_id
        self.last_access = window_id
        self.accesses = 1
        self.fingerprint = np.asarray(fingerprint, dtype=np.float64)
        self.origin_X = origin_X
        self.origin_y = origin_y
        self.buffer_counts = buffer_counts or {}


class RAPT(BufferMixin):
    """Regime-keyed policy repository.

    Reuse on a regime-id match (optionally gated by fingerprint similarity);
    train + store on a novel regime. Post-reuse weight calibration on the most
    recent buffered observations. Optional in-regime refresh policies are enabled
    by subclasses through `_refresh_policy`.
    """

    name = "RAPT"

    def __init__(self, seed, cfg, anchor):
        self.seed, self.cfg = seed, cfg
        self.anchor_X, self.anchor_y = anchor
        self._init_buffer(self.anchor_X, self.anchor_y, cfg["buffer_capacity"])
        r = cfg["rapt"]
        self.cal_n, self.cal_min = r["calibration_last_n"], r["calibration_min_n"]
        self.repo = {}
        self.cur_regime = None
        self.retrains = 0
        self.reuses = 0
        self.refreshes = 0
        self.trees_trained = 0
        self.trees_reused = 0
        self._recent_acc = []
        self._relative_ref = None
        self._since_refresh = 0
        self._last_seen_regime = None
        self.policy_events = []
        # Per-component adaptation CPU (spec: breakdown in cpu_breakdown.csv).
        self.cpu_parts = {"fingerprint": 0.0, "gate": 0.0, "recalibrate": 0.0,
                          "refit": 0.0, "checkpoint": 0.0, "refresh": 0.0, "other": 0.0}

    # -- subclass hooks ----------------------------------------------------
    def _should_reuse(self, ckpt, fingerprint):
        return True

    def _refresh_policy(self, ctx):
        return None  # returns (event_type, cpu) or None

    def _make_model(self, window_id):
        return create_ensemble(self.seed + window_id * 19, self.cfg)

    def fit_initial(self, X, y):
        t = time.process_time()
        self.model = create_ensemble(self.seed, self.cfg).fit(X, y)
        self.trees_trained += self.model.get_num_trees()
        cpu = time.process_time() - t
        fp = self._fingerprint(X)
        self.cur_regime = None
        return {"adapt_cpu": cpu, "event_type": "retrain"}

    def _fingerprint(self, X):
        return np.mean(X, axis=0)

    def predict(self, X):
        return self.model.predict(X)

    def update(self, X, y, ctx):
        window_id, regime = ctx["window_id"], ctx["regime"]
        _t0 = time.process_time()
        fp = self._fingerprint(X)
        self.cpu_parts["fingerprint"] += time.process_time() - _t0
        event = {"event_type": "none", "adapt_cpu": 0.0}

        if self.cur_regime is None:
            self.cur_regime = regime
            self.repo[regime] = Checkpoint(regime, self.model, [0.5, 0.5], window_id, fp)
        elif regime != self.cur_regime:
            event = self._transition(regime, window_id, fp, ctx)

        self._push_buffer(X, y)

        ref = self._refresh_policy(ctx)
        if ref is not None:
            event = {"event_type": ref[0], "adapt_cpu": event.get("adapt_cpu", 0.0) + ref[1]}

        self._recent_acc.append(float(np.mean(ctx["y_pred"] == y)))
        if len(self._recent_acc) > 5:
            self._recent_acc.pop(0)
        return event

    def _novel_n(self):
        """Observations used to refit a novel regime (base RAPT default)."""
        return self.cfg["rapt"]["novel_refit_n"]

    def _transition(self, regime, window_id, fp, ctx=None):
        t = time.process_time()
        ck = self.repo.get(regime)
        # Diagnostics: staleness of a stored policy on its own stored regime, and
        # the class counts of the buffer it was trained on.
        if ck is not None and ck.origin_X is not None and len(ck.origin_X):
            ck_stored_f1 = float(f1_score(ck.origin_y, ck.model.predict(ck.origin_X),
                                          average="macro", zero_division=0))
        else:
            ck_stored_f1 = np.nan
        _tg = time.process_time()
        reuse = regime in self.repo and self._should_reuse(self.repo[regime], fp)
        self.cpu_parts["gate"] += time.process_time() - _tg
        if reuse:
            ck = self.repo[regime]
            ck.last_access, ck.accesses = window_id, ck.accesses + 1
            self.model = copy.copy(ck.model)
            self.model.weights = list(ck.weights)
            self.reuses += 1
            self.trees_reused += ck.model.get_num_trees()
            if len(self.buffer_X) >= self.cal_min:
                _tc = time.process_time()
                Xb = np.array(self.buffer_X[-self.cal_n:])
                yb = np.array(self.buffer_y[-self.cal_n:])
                a_rf = float(np.mean(self.model.rf.predict(Xb) == yb))
                a_et = float(np.mean(self.model.et.predict(Xb) == yb))
                if a_rf + a_et > 0:
                    self.model.weights = [a_rf / (a_rf + a_et), a_et / (a_rf + a_et)]
                self.cpu_parts["recalibrate"] += time.process_time() - _tc
            event = "reuse"
            self.policy_events.append({
                "regime": regime, "window_idx": window_id, "event": "reuse",
                "stored_policy_origin_f1": ck_stored_f1,
                "buffer_counts": ck.buffer_counts,
                "origin_window": ck.created_window})
        else:
            self.retrains += 1
            # Novel-regime refit buffer (base RAPT default = 500 observations).
            n = self._novel_n()
            Xb = np.array(self.buffer_X[-n:])
            yb = np.array(self.buffer_y[-n:])
            tX, tY = make_train_buffer(Xb, yb, self.anchor_X, self.anchor_y, self.cap)
            self.model = self._make_model(window_id)
            _tf = time.process_time()
            if len(tX) > 0:
                self.model.fit(tX, tY)
                self.trees_trained += self.model.get_num_trees()
            self.cpu_parts["refit"] += time.process_time() - _tf
            _tk = time.process_time()
            counts = {int(c): int((tY == c).sum()) for c in np.unique(tY)}
            oX = np.asarray(ctx["X_win"]) if ctx is not None else np.array(self.buffer_X[-n:])
            oY = np.asarray(ctx["y_win"]) if ctx is not None else np.array(self.buffer_y[-n:])
            self.repo[regime] = Checkpoint(regime, self.model, [0.5, 0.5], window_id, fp,
                                           origin_X=oX, origin_y=oY,
                                           buffer_counts=counts)
            self.cpu_parts["checkpoint"] += time.process_time() - _tk
            event = "retrain"
            self.policy_events.append({
                "regime": regime, "window_idx": window_id, "event": "retrain",
                "stored_policy_origin_f1": np.nan,
                "buffer_counts": counts, "origin_window": window_id})
        self.cur_regime = regime
        self._since_refresh = 0
        return {"event_type": event, "adapt_cpu": time.process_time() - t}


# ---- RAPT-Full: fingerprint similarity gate -------------------------------
class RAPTFul(RAPT):
    name = "RAPT-Full"

    def _should_reuse(self, ckpt, fingerprint):
        d = fingerprint.shape[0]
        dist = float(np.linalg.norm(fingerprint - ckpt.fingerprint))
        sim = 1.0 - dist / np.sqrt(d)
        return sim >= self.cfg["rapt"]["similarity_gate_threshold"]


# ---- RAPT-Evidence: relative-degradation refresh --------------------------
class RAPTEvidence(RAPT):
    name = "RAPT-Evidence"

    def _refresh_policy(self, ctx):
        r = self.cfg["rapt"]
        if len(self._recent_acc) < r["relative_window"]:
            return None
        acc = float(np.mean(self._recent_acc[-r["relative_window"]:]))
        if self._relative_ref is None:
            self._relative_ref = acc
            return None
        if acc > self._relative_ref:
            self._relative_ref = acc
        if acc < self._relative_ref - r["relative_margin"]:
            cpu = self._do_refresh(ctx)
            self._relative_ref = acc
            return ("refresh", cpu)
        self._relative_ref = r["relative_decay"] * self._relative_ref + \
            (1 - r["relative_decay"]) * acc
        return None

    def _do_refresh(self, ctx):
        tX, tY = make_train_buffer(np.array(self.buffer_X), np.array(self.buffer_y),
                                   self.anchor_X, self.anchor_y, self.cap)
        t = time.process_time()
        self.model = create_ensemble(self.seed + ctx["window_id"] * 23, self.cfg)
        self.model.fit(tX, tY)
        self.refreshes += 1
        self.trees_trained += self.model.get_num_trees()
        if self.cur_regime is not None:
            self.repo[self.cur_regime] = Checkpoint(self.cur_regime, self.model,
                                                    [0.5, 0.5], ctx["window_id"],
                                                    self._fingerprint(ctx.get("X_win")))
        return time.process_time() - t


# ---- RAPT-Floor: absolute-floor refresh -----------------------------------
class RAPTFloor(RAPT):
    name = "RAPT-Floor"

    def _refresh_policy(self, ctx):
        r = self.cfg["rapt"]
        if len(self._recent_acc) < 3:
            return None
        if float(np.mean(self._recent_acc[-3:])) < r["floor_threshold"]:
            tX, tY = make_train_buffer(np.array(self.buffer_X), np.array(self.buffer_y),
                                       self.anchor_X, self.anchor_y, self.cap)
            t = time.process_time()
            self.model = create_ensemble(self.seed + ctx["window_id"] * 29, self.cfg)
            self.model.fit(tX, tY)
            cpu = time.process_time() - t
            self.refreshes += 1
            self.trees_trained += self.model.get_num_trees()
            self._recent_acc = []
            if self.cur_regime is not None:
                self.repo[self.cur_regime] = Checkpoint(
                    self.cur_regime, self.model, [0.5, 0.5], ctx["window_id"],
                    self._fingerprint(ctx["X_win"]))
            return ("refresh", cpu)
        return None


# ---- Periodic refresh variants --------------------------------------------
class _PeriodicRefresh(RAPT):
    def __init__(self, seed, cfg, anchor):
        super().__init__(seed, cfg, anchor)
        r = cfg["rapt"]
        self.period = r["periodic_period_windows"]
        self.refit_trees = r["full_refit_trees"]
        self.refit_buffer = r["full_refit_buffer"]

    def _refresh_policy(self, ctx):
        self._since_refresh += 1
        if self._since_refresh < self.period:
            return None
        rX = np.array(self.buffer_X[-self.refit_buffer:])
        rY = np.array(self.buffer_y[-self.refit_buffer:])
        tX, tY = make_train_buffer(rX, rY, self.anchor_X, self.anchor_y, self.cap)
        t = time.process_time()
        self.model = create_ensemble(self.seed + ctx["window_id"] * 31, self.cfg)
        self.model.n_estimators = self.refit_trees
        self.model.rf = RandomForestClassifier(n_estimators=self.refit_trees,
                                               max_depth=self.cfg["ensemble"]["max_depth"],
                                               random_state=self.seed + ctx["window_id"] * 31,
                                               n_jobs=1)
        self.model.et = ExtraTreesClassifier(n_estimators=self.refit_trees,
                                             max_depth=self.cfg["ensemble"]["max_depth"],
                                             random_state=self.seed + ctx["window_id"] * 31,
                                             n_jobs=1)
        if len(tX) > 0:
            self.model.fit(tX, tY)
        cpu = time.process_time() - t
        self.refreshes += 1
        self.trees_trained += self.model.get_num_trees()
        self._since_refresh = 0
        if self.cur_regime is not None:
            self.repo[self.cur_regime] = Checkpoint(self.cur_regime, self.model,
                                                    [0.5, 0.5], ctx["window_id"],
                                                    self._fingerprint(ctx["X_win"]))
        return ("refresh", cpu)


class RAPTPeriodicFullRefit(_PeriodicRefresh):
    name = "RAPT-Periodic-FullRefit"

    def __init__(self, seed, cfg, anchor):
        super().__init__(seed, cfg, anchor)
        r = cfg["rapt"]
        self.refit_trees = r["full_refit_trees"]
        self.refit_buffer = r["full_refit_buffer"]


class RAPTCheap(_PeriodicRefresh):
    name = "RAPT-Cheap"

    def __init__(self, seed, cfg, anchor):
        super().__init__(seed, cfg, anchor)
        r = cfg["rapt"]
        self.refit_trees = r["cheap_refit_trees"]
        self.refit_buffer = r["cheap_refit_buffer"]


# ---- RAPT-Incremental: River Hoeffding adaptive tree ----------------------
class RAPTIncremental(BufferMixin):
    name = "RAPT-Incremental"

    def __init__(self, seed, cfg, anchor):
        from river.tree import HoeffdingAdaptiveTreeClassifier
        self.seed, self.cfg = seed, cfg
        self.anchor_X, self.anchor_y = anchor
        self._init_buffer(self.anchor_X, self.anchor_y, cfg["buffer_capacity"])
        p = cfg["rapt_incremental"]
        self._mk = lambda: HoeffdingAdaptiveTreeClassifier(
            grace_period=p["grace_period"], delta=p["delta"], seed=seed)
        self.repo = {}
        self.cur_regime = None
        self.retrains = self.reuses = self.refreshes = 0
        self.trees_trained = self.trees_reused = 0
        self.classes_ = None

    def fit_initial(self, X, y):
        t = time.process_time()
        self.model = self._mk()
        self.classes_ = np.unique(y)
        for xi, yi in zip(X, y):
            self.model.learn_one(dict(enumerate(xi)), int(yi))
        return {"adapt_cpu": time.process_time() - t, "event_type": "retrain"}

    def predict(self, X):
        out = []
        for xi in X:
            p = self.model.predict_one(dict(enumerate(xi)))
            out.append(0 if p is None else int(p))
        return np.array(out)

    def update(self, X, y, ctx):
        regime = ctx["regime"]
        event = "none"
        t = time.process_time()
        if self.cur_regime is None:
            self.cur_regime = regime
            self.repo[regime] = self.model
        elif regime != self.cur_regime:
            if regime in self.repo:
                self.model = self.repo[regime]
                self.reuses += 1
                event = "reuse"
            else:
                self.model = self._mk()
                self.repo[regime] = self.model
                self.retrains += 1
                event = "retrain"
            self.cur_regime = regime
        self._push_buffer(X, y)
        for xi, yi in zip(X, y):
            self.model.learn_one(dict(enumerate(xi)), int(yi))
        cpu = time.process_time() - t
        if event == "none":
            return {"event_type": "refresh", "adapt_cpu": cpu}
        return {"event_type": event, "adapt_cpu": cpu}


# ---- RAPT-Enhanced (primary companion) ------------------------------------
class RAPTEnhanced(RAPT):
    """RAPT + buffer-blended novelty refit + selective parity refit."""

    name = "RAPT-Enhanced"

    def __init__(self, seed, cfg, anchor):
        super().__init__(seed, cfg, anchor)
        p = cfg["rapt_enhanced"]
        self.novelty_n = p["novelty_refit_n"]
        self.parity_threshold = p["parity_threshold"]
        self.parity_window = p["parity_window"]
        self.parity_refits = 0
        self._was_reused = False

    def _novel_n(self):
        return self.novelty_n

    def _transition(self, regime, window_id, fp, ctx=None):
        ev = super()._transition(regime, window_id, fp, ctx)
        self._was_reused = ev["event_type"] == "reuse"
        return ev

    def _refresh_policy(self, ctx):
        if not self._was_reused:
            return None
        if len(self._recent_acc) < self.parity_window:
            return None
        if float(np.mean(self._recent_acc[-self.parity_window:])) >= self.parity_threshold:
            return None
        rX = np.array(self.buffer_X[-self.novelty_n:])
        rY = np.array(self.buffer_y[-self.novelty_n:])
        tX, tY = make_train_buffer(rX, rY, self.anchor_X, self.anchor_y, self.cap)
        t = time.process_time()
        self.model = create_ensemble(self.seed + ctx["window_id"] * 37, self.cfg)
        if len(tX) > 0:
            self.model.fit(tX, tY)
        cpu = time.process_time() - t
        self.parity_refits += 1
        self.refreshes += 1
        self.trees_trained += self.model.get_num_trees()
        self._recent_acc = []
        self._was_reused = False
        if self.cur_regime is not None:
            self.repo[self.cur_regime] = Checkpoint(
                self.cur_regime, self.model, [0.5, 0.5], ctx["window_id"],
                self._fingerprint(ctx["X_win"]))
        return ("refresh", cpu)


PRIMARY = {
    "Frozen": Frozen, "Event-Driven": EventDriven, "Full Retraining": FullRetraining,
    "RAPT": RAPT, "RAPT-Enhanced": RAPTEnhanced,
}
ABLATION = {
    "RAPT-Full": RAPTFul, "RAPT-Evidence": RAPTEvidence, "RAPT-Floor": RAPTFloor,
    "RAPT-Periodic-FullRefit": RAPTPeriodicFullRefit, "RAPT-Cheap": RAPTCheap,
    "RAPT-Incremental": RAPTIncremental,
}
