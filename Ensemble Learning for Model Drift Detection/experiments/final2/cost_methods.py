"""Cost-reduction job methods (no new model families).

Adds only *cheap variants* of existing methods and a cheaper RAPT-Guard gate.
Everything else (streams, protocol, base architecture) is shared with the
existing harness; the base model class is never modified.

Cheap refit mode is read from cfg['cheap_n_trees'] / cfg['cheap_buffer'] and is
chosen on dev (see cost_dev_refit.py).

The cheaper gate (RAPTGuardCheap) differs from RAPT-Guard only in *bookkeeping*:
  * the gate is consulted only at regime returns (never on error triggers),
  * the prefix model's predictions on the lookback rows are cached (the prefix
    model is frozen, so its predictions are valid forever),
  * the gate is skipped entirely when the rolling window error is already below
    the trigger (the model is fine, so reuse the stored policy unchecked).
The decision rule and margin are unchanged from RAPT-Guard.
"""

import time
import numpy as np

from final_methods import (EventDriven, EventDrivenReuse, FullRetraining,
                           RAPTGuard, _Base)
from models_9a import create_base_ensemble


def cheap_params(cfg):
    return int(cfg.get("cheap_n_trees", 20)), int(cfg.get("cheap_buffer", 300))


class _CheapRefit:
    """Mixin: refit with the frozen cheap mode instead of the default 50/full."""

    def _cheap_train(self, seed, X, y):
        n_trees, buf = cheap_params(self.cfg)
        t0 = time.process_time()
        self.ens = self._train(seed, X, y, n_trees=n_trees, buf=buf)
        return time.process_time() - t0


class FullRetrainingCheap(_CheapRefit, FullRetraining):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._first = True

    def fit_initial(self, X_init, y_init, init_regime=None):
        self._first = True
        return super().fit_initial(X_init, y_init, init_regime)

    def post_predict(self, w, idx, X_win, y_win, *a, regime=None, **k):
        self._push(X_win, y_win)
        self.last_flags = {"retrain": 0, "reuse": 0}
        ac = 0.0
        if regime is not None and regime != self.prev:
            self.last_flags["retrain"] = 1
            ac = self._cheap_train(self.seed + w * 11, self.buf_X, self.buf_y)
            self.prev = regime
        return ac


class EventDrivenCheap(_CheapRefit, EventDriven):
    def post_predict(self, w, idx, X_win, y_win, *a, err=None, **k):
        self._push(X_win, y_win)
        triggered = self._trigger(err)
        self.last_flags = {"retrain": 0, "reuse": 0}
        ac = 0.0
        if triggered:
            self.last_flags["retrain"] = 1
            ac = self._cheap_train(self.seed + w * 13, self.buf_X, self.buf_y)
            self.errs = [err]
        return ac


class EventDrivenReuseCheap(_CheapRefit, EventDrivenReuse):
    """Event-Driven trigger + reuse, cheap refit. On a trigger: reuse the stored
    policy when the current regime has one, else cheap-refit and store."""

    def post_predict(self, w, idx, X_win, y_win, *a, err=None, regime=None, **k):
        self._push(X_win, y_win)
        triggered = self._trigger(err)
        self.last_flags = {"retrain": 0, "reuse": 0}
        ac = 0.0
        if triggered:
            if regime is not None and regime in self.repo:
                self.ens = self.repo[regime]
                self.last_flags["reuse"] = 1
            else:
                self.last_flags["retrain"] = 1
                ac = self._cheap_train(self.seed + w * 13, self.buf_X, self.buf_y)
                if regime is not None:
                    self.repo[regime] = self.ens
            self.errs = [err]
        return ac


class FrozenCheap(_CheapRefit, _Base):
    """Frozen model built with the cheap base config (20 trees / buffer 300).
    Purely a fairness control for Frozen: no adaptation, but the base learner
    matches the cheap refits so the CPU/F1 comparison is apples-to-apples."""

    def fit_initial(self, X_init, y_init, init_regime=None):
        n_trees, _ = cheap_params(self.cfg)
        t0 = time.process_time()
        self.ens = create_base_ensemble(seed=self.seed, n_estimators=n_trees)
        self.ens.fit(X_init, y_init)
        self.init_cpu = time.process_time() - t0
        self.buf_X, self.buf_y = list(X_init), list(y_init)
        self.last_flags = {}
        return self.init_cpu

    def post_predict(self, w, idx, X_win, y_win, *a, **k):
        self._push(X_win, y_win)
        self.last_flags = {"retrain": 0, "reuse": 0}
        return 0.0


class RAPTGuardCheap(_CheapRefit, RAPTGuard):
    """RAPT-Guard with a cheaper gate. Same decision rule and margin; the gate
    is only consulted at regime returns, the prefix model's per-window
    predictions are cached (the prefix model is frozen), and the gate is skipped
    when the rolling window error is already below the trigger."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._win_prefix = {}        # window_id -> prefix predictions (frozen)
        self._win_sizes = []         # [(window_id, n_rows)] push order
        self._win_acc = []           # rolling active-policy accuracy

    def record_window(self, w, X_win, y_win):
        """Cache the prefix predictions for this window and the active policy's
        accuracy, so the gate never recomputes prefix predictions for a window."""
        self._win_prefix[int(w)] = self.prefix_ens.predict(X_win)
        self._win_sizes.append((int(w), int(len(X_win))))
        acc = float(np.mean(self.ens.predict(X_win) == np.asarray(y_win)))
        self._win_acc.append(acc)
        if len(self._win_acc) > 25:
            self._win_acc.pop(0)

    def _cached_prefix_for_lookback(self, lb, fallback_X):
        """Concatenate cached prefix predictions for the most recent `lb` buffer
        rows. Rows whose window was never cached (the initial prefix) fall back
        to a single fresh prediction on those rows only."""
        preds, acc = [], 0
        for wid, n in reversed(self._win_sizes):
            if acc >= lb:
                break
            p = self._win_prefix.get(wid)
            if p is None:
                break
            preds.append(p); acc += n
        if not preds:
            return self.prefix_ens.predict(fallback_X)
        joined = np.concatenate(list(reversed(preds)))
        if len(joined) >= lb:
            return joined[-lb:]
        # top up the shortfall with a fresh prediction on the missing rows
        miss = lb - len(joined)
        head = self.prefix_ens.predict(np.asarray(fallback_X)[:miss])
        return np.concatenate([head, joined])

    def on_regime_change(self, reg, w, X_win, y_win, buf_X, buf_y, prev_regime=None):
        self.last_flags = {"retrain": 0, "reuse": 0, "gate_allow": 0,
                           "gate_block": 0, "gate_skipped": 0}
        if not hasattr(self, "gate_log"):
            self.gate_log = []
        t0 = time.process_time()
        self.cur = reg
        if reg not in self.repo:
            n_trees, buf = cheap_params(self.cfg)
            self.ens = self._train(self.seed + w * 19, buf_X, buf_y,
                                   n_trees=n_trees, buf=buf)
            self.repo[reg] = self.ens
            self._reused = False
            self.last_flags["retrain"] = 1
            return time.process_time() - t0

        stored_model = self.repo[reg]
        skip_win = int(self.cfg.get("gate_skip_window", 3))
        skip_acc = float(self.cfg.get("gate_skip_acc", 1.0))
        # The raw-accuracy skip is disabled by config (gate_skip_acc >= 1.0)
        # because raw accuracy is majority-dominated on imbalanced streams and
        # would let stale policies through unchecked.
        if (skip_acc < 1.0 and len(self._win_acc) >= skip_win and
                float(np.mean(self._win_acc[-skip_win:])) >= skip_acc):
            self.ens = stored_model
            self._reused = True
            self.last_flags["reuse"] = 1
            self.last_flags["gate_skipped"] = 1
            return time.process_time() - t0

        lb = self.cfg["guard_lookback"]
        min_rows = self.cfg["guard_min_rows"]
        margin = self.cfg["guard_margin"]
        xb = np.asarray(buf_X[-lb:]); yb = np.asarray(buf_y[-lb:])
        self.last_flags["gate"] = 1
        if len(yb) >= min_rows:
            stored = float(np.mean(stored_model.predict(xb) == yb))
            prefix = float(np.mean(self._cached_prefix_for_lookback(lb, xb) == yb))
            decision = "allow" if stored >= prefix + margin else "block"
            self.gate_log.append(dict(window_id=int(w), regime=str(reg),
                                      stored_acc=stored, prefix_acc=prefix,
                                      decision=decision))
            if decision == "allow":
                self.ens = stored_model
                self._reused = True
                self.last_flags["reuse"] = 1; self.last_flags["gate_allow"] = 1
            else:
                n_trees, buf = cheap_params(self.cfg)
                self.ens = self._train(self.seed + w * 29, buf_X, buf_y,
                                       n_trees=n_trees, buf=buf)
                self.repo[reg] = self.ens; self._reused = False
                self.last_flags["retrain"] = 1; self.last_flags["gate_block"] = 1
        else:
            self.ens = stored_model
            self._reused = True
            self.last_flags["reuse"] = 1; self.last_flags["gate_allow"] = 1
        return time.process_time() - t0


METHODS = ["Frozen", "Frozen-cheap", "Full Retraining", "Full Retraining-cheap",
           "Event-Driven", "Event-Driven-cheap", "Event-Driven+Reuse",
           "Page-Hinkley", "RAPT", "RAPT-Guard"]


def build_cost_method(name, seed=42, anchor_X=None, anchor_y=None, cfg=None):
    cfg = dict(cfg or {})
    cfg.setdefault("buffer_capacity", 1000)
    from final_methods import build_method
    table = {
        "Frozen-cheap": FrozenCheap,
        "Full Retraining-cheap": FullRetrainingCheap,
        "Event-Driven-cheap": EventDrivenCheap,
        "Event-Driven+Reuse": EventDrivenReuseCheap,
        "RAPT-Guard": RAPTGuardCheap,
        "RAPT-Guard-std": RAPTGuard,
    }
    if name in table:
        return table[name](seed, anchor_X, anchor_y, cfg)
    return build_method(name, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
