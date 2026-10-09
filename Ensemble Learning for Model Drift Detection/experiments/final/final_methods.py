"""
Methods for the unified final harness.

Every method shares:
  * the same base architecture (RandomForest + ExtraTrees soft vote),
  * the same class-anchored retraining buffer (models_9a.make_train_buffer),
  * test-then-train ordering (predict, then adapt with the same window).

Adaptation CPU and prediction CPU are tracked separately, and the per-window
record also stores total_cpu = adapt + predict so the cost comparison is fair
(a method that infers twice is charged twice).
"""

import time
import numpy as np

from models_9a import create_base_ensemble, make_train_buffer

try:
    from river.drift import ADWIN, PageHinkley
except Exception:  # pragma: no cover
    ADWIN = PageHinkley = None

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def make_ensemble(seed, n_estimators=50, max_depth=7, class_weight=None):
    """Same architecture as models_9a.HeterogeneousEnsemble but with an
    optional class_weight, used only for the NordicDat class-weight check."""
    from models_9a import HeterogeneousEnsemble
    if class_weight is None:
        return create_base_ensemble(seed=seed, n_estimators=n_estimators)
    e = HeterogeneousEnsemble(seed=seed, n_estimators=n_estimators, max_depth=max_depth)
    e.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  random_state=seed, n_jobs=1, class_weight=class_weight)
    e.et = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=seed, n_jobs=1, class_weight=class_weight)
    return e


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------
class _Base:
    def __init__(self, seed, anchor_X, anchor_y, cfg):
        self.seed = seed
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.cfg = cfg
        self.cap = cfg.get("buffer_capacity", 1000)
        self.buf_X, self.buf_y = [], []
        self.ens = None
        self.last_flags = {}

    def _train(self, seed, X, y, n_trees=None, buf=None):
        ens = create_base_ensemble(seed=seed, n_estimators=(n_trees or 50))
        trX, trY = make_train_buffer(np.asarray(X), np.asarray(y),
                                     self.anchor_X, self.anchor_y, buf or self.cap)
        if len(trX) > 0 and len(np.unique(trY)) >= 1:
            ens.fit(trX, trY)
        return ens

    def fit_initial(self, X_init, y_init, init_regime=None):
        t0 = time.process_time()
        self.ens = create_base_ensemble(seed=self.seed)
        self.ens.fit(X_init, y_init)
        self.init_cpu = time.process_time() - t0
        self.buf_X, self.buf_y = list(X_init), list(y_init)
        self.last_flags = {}
        return self.init_cpu

    def predict(self, X):
        return self.ens.predict(X)

    def predict_window(self, X_win, buf_X, buf_y):
        """Return (predictions, inference_cpu). Overridden where inference is
        heavier than a single ensemble call (e.g. RAPT-Anchor)."""
        t0 = time.process_time()
        yp = self.ens.predict(X_win)
        return yp, time.process_time() - t0

    def _push(self, X_win, y_win):
        self.buf_X.extend(X_win); self.buf_y.extend(y_win)
        if len(self.buf_X) > self.cap:
            self.buf_X = self.buf_X[-self.cap:]
            self.buf_y = self.buf_y[-self.cap:]

    def _floor_refit(self, w, X_win, y_win, buf_X, buf_y, flag="retrain"):
        """In-regime absolute-floor trigger (STEP 1d). Refits when the rolling
        window accuracy over `floor_window` windows drops below
        `floor_acc_threshold`. Off unless cfg['enable_floor'] is True."""
        if not self.cfg.get("enable_floor", False):
            self._floor_hist = []
            return 0.0
        acc = float(np.mean(self.ens.predict(X_win) == np.asarray(y_win)))
        hist = getattr(self, "_floor_hist", None)
        if hist is None:
            hist = self._floor_hist = []
        hist.append(acc)
        fw = self.cfg.get("floor_window", 3)
        if len(hist) > fw:
            hist.pop(0)
        if (len(hist) >= fw and np.mean(hist) < self.cfg.get("floor_acc_threshold", 0.6)
                and (w - getattr(self, "_last_floor", -10**9)) >= self.cfg.get("floor_min_interval", 2)):
            self._last_floor = w
            self._floor_hist = []
            t0 = time.process_time()
            self.ens = self._train(self.seed + w * 31, buf_X, buf_y)
            self.last_flags[flag] = 1
            return time.process_time() - t0
        return 0.0

    def post_predict(self, w, idx, X_win, y_win, *a, **k):
        return 0.0

    def on_regime_change(self, *a, **k):
        return 0.0


class Frozen(_Base):
    def post_predict(self, w, idx, X_win, y_win, *a, **k):
        self._push(X_win, y_win)
        self.last_flags = {"retrain": 0, "reuse": 0}
        return self._floor_refit(w, X_win, y_win, self.buf_X, self.buf_y)


class FrozenPeriodic(_Base):
    def post_predict(self, w, idx, X_win, y_win, *a, **k):
        self._push(X_win, y_win)
        self.last_flags = {"retrain": 0, "reuse": 0}
        if w % self.cfg["periodic_interval"] == 0:
            t0 = time.process_time()
            self.ens = self._train(self.seed + w * 7, self.buf_X, self.buf_y,
                                   n_trees=self.cfg["periodic_n_trees"],
                                   buf=self.cfg["periodic_buffer"])
            self.last_flags["retrain"] = 1
            return time.process_time() - t0
        return 0.0


class EventDriven(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.errs = []

    def post_predict(self, w, idx, X_win, y_win, *a, err=None, **k):
        self._push(X_win, y_win)
        triggered = self._trigger(err)
        self.last_flags = {"retrain": 0, "reuse": 0}
        ac = 0.0
        if triggered:
            self.last_flags["retrain"] = 1
            t0 = time.process_time()
            self.ens = self._train(self.seed + w * 13, self.buf_X, self.buf_y)
            self.errs = [err]
            ac = time.process_time() - t0
        return ac + self._floor_refit(w, X_win, y_win, self.buf_X, self.buf_y)

    def _trigger(self, err):
        self.errs.append(err if err is not None else 1.0)
        if len(self.errs) > self.cfg.get("error_window_size", 20):
            self.errs.pop(0)
        if len(self.errs) < 5:
            return False
        mu = np.mean(self.errs[:-1]); sg = np.std(self.errs[:-1])
        return self.errs[-1] > mu + self.cfg.get("error_threshold_k", 2.0) * max(sg, 0.05)


class EventDrivenReuse(EventDriven):
    """Event-Driven trigger, but the trigger reuses a stored policy when the
    current regime has one; otherwise it trains and stores."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.repo = {}

    def fit_initial(self, X_init, y_init, init_regime=None):
        c = super().fit_initial(X_init, y_init, init_regime)
        if init_regime is not None:
            self.repo[init_regime] = self.ens
        return c

    def post_predict(self, w, idx, X_win, y_win, *a, err=None, regime=None, **k):
        self._push(X_win, y_win)
        triggered = self._trigger(err)
        self.last_flags = {"retrain": 0, "reuse": 0}
        if not triggered:
            return self._floor_refit(w, X_win, y_win, self.buf_X, self.buf_y)
        t0 = time.process_time()
        if regime is not None and regime in self.repo:
            self.ens = self.repo[regime]
            self.last_flags["reuse"] = 1
        else:
            self.ens = self._train(self.seed + w * 13, self.buf_X, self.buf_y)
            self.last_flags["retrain"] = 1
            if regime is not None:
                self.repo[regime] = self.ens
        self.errs = [err]
        return time.process_time() - t0 + self._floor_refit(
            w, X_win, y_win, self.buf_X, self.buf_y)


class FullRetraining(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prev = None

    def post_predict(self, w, idx, X_win, y_win, *a, regime=None, **k):
        self._push(X_win, y_win)
        self.last_flags = {"retrain": 0, "reuse": 0}
        ac = 0.0
        if regime is not None and regime != self.prev:
            self.prev = regime
            self.last_flags["retrain"] = 1
            t0 = time.process_time()
            self.ens = self._train(self.seed + w * 11, self.buf_X, self.buf_y)
            ac = time.process_time() - t0
        return ac + self._floor_refit(w, X_win, y_win, self.buf_X, self.buf_y)


# ---------------------------------------------------------------------------
# RAPT family (regime-boundary keyed)
# ---------------------------------------------------------------------------
class _RAPTBase(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.repo = {}
        self.cur = None
        self.prefix_ens = None
        self._reused = False

    def fit_initial(self, X_init, y_init, init_regime=None):
        c = super().fit_initial(X_init, y_init, init_regime)
        self.prefix_ens = self.ens
        if init_regime is not None:
            self.repo[init_regime] = self.ens
            self.cur = init_regime
        self.last_flags = {}
        return c

    def on_regime_change(self, reg, w, X_win, y_win, buf_X, buf_y, prev_regime=None):
        self.last_flags = {"retrain": 0, "reuse": 0}
        t0 = time.process_time()
        if reg in self.repo:
            self.ens = self.repo[reg]
            self._reused = True
            self.last_flags["reuse"] = 1
        else:
            self.ens = self._train(self.seed + w * 19, buf_X, buf_y)
            self.repo[reg] = self.ens
            self._reused = False
            self.last_flags["retrain"] = 1
        self.cur = reg
        return time.process_time() - t0


class RAPT(_RAPTBase):
    def post_predict(self, w, idx, X_win, y_win, buf_X, buf_y, **k):
        self._push(X_win, y_win)
        return self._floor_refit(w, X_win, y_win, buf_X, buf_y)


class RAPTEnhanced(_RAPTBase):
    """Base RAPT + parity refit: a reused policy whose rolling accuracy over
    `parity_window` windows falls below `parity_threshold` is refit and its
    checkpoint replaced."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._recent = []

    def on_regime_change(self, *a, **k):
        self._recent = []
        return super().on_regime_change(*a, **k)

    def post_predict(self, w, idx, X_win, y_win, buf_X, buf_y, **k):
        self._push(X_win, y_win)
        acc = float(np.mean(self.ens.predict(X_win) == np.asarray(y_win)))
        self._recent.append(acc)
        pwin = self.cfg.get("parity_window", 3)
        if len(self._recent) > pwin:
            self._recent.pop(0)
        if (self._reused and len(self._recent) >= pwin
                and np.mean(self._recent) < self.cfg.get("parity_threshold", 0.5)):
            t0 = time.process_time()
            self.ens = self._train(self.seed + w * 23, buf_X, buf_y)
            self.repo[self.cur] = self.ens
            self.last_flags["retrain"] = 1
            self._recent = []
            return time.process_time() - t0
        return 0.0


class RAPTGuard(_RAPTBase):
    """On a returning regime, score the stored ensemble and the prefix-trained
    ensemble on the last `guard_lookback` labelled rows; reuse the stored model
    only if it beats the prefix model by `guard_margin`; otherwise fall back to
    the prefix model and refit the regime checkpoint."""

    def on_regime_change(self, reg, w, X_win, y_win, buf_X, buf_y, prev_regime=None):
        self.last_flags = {"retrain": 0, "reuse": 0, "gate_allow": 0, "gate_block": 0}
        if not hasattr(self, "gate_log"):
            self.gate_log = []
        t0 = time.process_time()
        if reg in self.repo:
            lb = self.cfg["guard_lookback"]
            xb = np.asarray(buf_X[-lb:]); yb = np.asarray(buf_y[-lb:])
            self.last_flags["gate"] = 1
            if len(yb) >= self.cfg["guard_min_rows"]:
                stored = float(np.mean(self.repo[reg].predict(xb) == yb))
                prefix = float(np.mean(self.prefix_ens.predict(xb) == yb))
                decision = "allow" if stored >= prefix + self.cfg["guard_margin"] else "block"
                self.gate_log.append(dict(window_id=int(w), regime=str(reg),
                                          stored_acc=stored, prefix_acc=prefix,
                                          decision=decision))
                if decision == "allow":
                    self.ens = self.repo[reg]; self._reused = True
                    self.last_flags["reuse"] = 1; self.last_flags["gate_allow"] = 1
                else:
                    self.ens = self._train(self.seed + w * 29, buf_X, buf_y)
                    self.repo[reg] = self.ens; self._reused = False
                    self.last_flags["retrain"] = 1; self.last_flags["gate_block"] = 1
            else:
                self.ens = self.repo[reg]; self._reused = True
                self.last_flags["reuse"] = 1; self.last_flags["gate_allow"] = 1
        else:
            self.ens = self._train(self.seed + w * 19, buf_X, buf_y)
            self.repo[reg] = self.ens; self._reused = False
            self.last_flags["retrain"] = 1
        self.cur = reg
        return time.process_time() - t0

    def post_predict(self, w, idx, X_win, y_win, buf_X, buf_y, **k):
        self._push(X_win, y_win)
        return 0.0

    def predict_with_cost(self, X_win, buf_X, buf_y):
        """Pick, from a labelled lookback, whichever of prefix/active is better,
        then predict with it. Charges both scores + the prediction."""
        lb = self.cfg["anchor_lookback"]
        xb = np.asarray(buf_X[-lb:]); yb = np.asarray(buf_y[-lb:])
        t0 = time.process_time()
        if len(yb) >= 10:
            a = float(np.mean(self.ens.predict(xb) == yb))
            p = float(np.mean(self.prefix_ens.predict(xb) == yb))
            chosen = self.ens if a >= p else self.prefix_ens
        else:
            chosen = self.ens
        yp = chosen.predict(X_win)
        return yp, time.process_time() - t0


class RAPTAnchor(_RAPTBase):
    """Anchor to the prefix model. Each window, both the active RAPT policy and
    the frozen prefix model are scored on the labelled lookback; the better one
    predicts. Inference is therefore doubled, and is charged as such."""

    def predict_window(self, X_win, buf_X, buf_y):
        lb = self.cfg["anchor_lookback"]
        xb = np.asarray(buf_X[-lb:]); yb = np.asarray(buf_y[-lb:])
        t0 = time.process_time()
        if len(yb) >= 10:
            a = float(np.mean(self.ens.predict(xb) == yb))
            p = float(np.mean(self.prefix_ens.predict(xb) == yb))
            chosen = self.ens if a >= p else self.prefix_ens
        else:
            chosen = self.ens
        yp = chosen.predict(X_win)
        return yp, time.process_time() - t0

    def post_predict(self, w, idx, X_win, y_win, buf_X, buf_y, **k):
        self._push(X_win, y_win)
        return 0.0


# ---------------------------------------------------------------------------
# Historical detectors (trigger-only; shared buffer/model)
# ---------------------------------------------------------------------------
class _EDD:
    def __init__(self, warning=2.0, drift=3.0, min_instances=30):
        self.warning, self.drift, self.min_instances = warning, drift, min_instances
        self.reset()

    def reset(self):
        self.n = 0; self.last = None; self.mean = 0.0; self.var = 0.0
        self.pmin = None; self.smin = None

    def update(self, e):
        self.n += 1
        if self.last is not None:
            d = float(e != self.last)
            self.mean += (d - self.mean) / self.n
            self.var += (d - self.mean) ** 2
            s = np.sqrt(self.var / self.n) if self.n > 1 else 0.0
            p = self.mean
            if self.pmin is None or (p + self.warning * s) <= (self.pmin + self.warning * self.smin):
                self.pmin, self.smin = p, s
            if (p + self.drift * s) > (self.pmin + self.drift * self.smin) and self.n >= self.min_instances:
                self.last = e; return True
        self.last = e
        return False


class _EDMA:
    def __init__(self, alpha=0.2, k=2.0, min_instances=30):
        self.alpha, self.k, self.min_instances = alpha, k, min_instances
        self.n = 0; self.ewma = None; self.ewvar = 0.0

    def update(self, e):
        self.n += 1
        if self.ewma is None:
            self.ewma = float(e); return False
        dev = e - self.ewma
        self.ewma += self.alpha * dev
        self.ewvar = (1 - self.alpha) * (self.ewvar + self.alpha * dev * dev)
        sd = np.sqrt(self.ewvar) if self.ewvar > 0 else 0.0
        return self.n >= self.min_instances and e > self.ewma + self.k * max(sd, 0.02)


class DetectorModel(_Base):
    def __init__(self, name, *a, **k):
        super().__init__(*a, **k)
        self.name = name
        cfg = self.cfg
        if name == "ADWIN":
            assert ADWIN is not None
            self.det = ADWIN(delta=cfg["adwin_delta"])
        elif name == "Page-Hinkley":
            assert PageHinkley is not None
            self.det = PageHinkley(min_instances=cfg["ph_min_instances"],
                                   delta=cfg["ph_delta"],
                                   threshold=cfg["ph_threshold"],
                                   alpha=cfg["ph_alpha"])
        elif name == "EDD":
            self.det = _EDD(cfg["edd_warning"], cfg["edd_drift"])
        elif name == "EDMA":
            self.det = _EDMA(cfg["edma_alpha"], cfg["edma_k"])
        else:
            raise ValueError(name)
        self.last_retrain = -10 ** 9

    def post_predict(self, w, idx, X_win, y_win, *a, err=None, **k):
        self._push(X_win, y_win)
        if err is None:
            err = 0.0
        fired = False
        # ADWIN / Page-Hinkley operate on the per-sample 0/1 error stream, which
        # is the granularity they are designed for.
        for e in np.atleast_1d(err):
            self.det.update(float(e))
            if getattr(self.det, "drift_detected", False):
                fired = True
        self.last_flags = {"retrain": 0, "detected": int(fired)}
        if fired and (w - self.last_retrain) >= self.cfg["detector_min_retrain_interval"]:
            self.last_retrain = w
            self.last_flags["retrain"] = 1
            t0 = time.process_time()
            self.ens = self._train(self.seed + w * 17, self.buf_X, self.buf_y)
            return time.process_time() - t0
        return 0.0


# ---------------------------------------------------------------------------
def build_method(name, seed=42, anchor_X=None, anchor_y=None, cfg=None):
    cfg = dict(cfg or {})
    cfg.setdefault("buffer_capacity", 1000)
    table = {
        "Frozen": Frozen,
        "Frozen+Periodic": FrozenPeriodic,
        "Event-Driven": EventDriven,
        "Event-Driven+Reuse": EventDrivenReuse,
        "Full Retraining": FullRetraining,
        "RAPT": RAPT,
        "RAPT-Enhanced": RAPTEnhanced,
        "RAPT-Guard": RAPTGuard,
        "RAPT-Anchor": RAPTAnchor,
    }
    if name in table:
        return table[name](seed, anchor_X, anchor_y, cfg)
    return DetectorModel(name, seed, anchor_X, anchor_y, cfg)
