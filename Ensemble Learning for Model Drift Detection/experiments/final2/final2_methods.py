"""Task A method: RAPT-ED-Guard.

Combines the Event-Driven error trigger with regime-id memory and the Guard
safety gate:

  * Trigger condition: the current window's error exceeds the mean + 2 sd of
    the last `error_window_size` window errors (the existing Event-Driven
    trigger), OR the regime id changes (a regime return).
  * On any such event, run the Guard check: score the stored policy for the
    current regime and the prefix-trained policy on the most recent
    `guard_lookback` labelled rows. Reuse the stored policy only if
    stored_acc >= prefix_acc + guard_margin; otherwise refit from the shared
    buffer and overwrite the regime checkpoint.

No base model is changed; the base architecture, buffer and ordering are shared
with every other method.
"""

import time
import numpy as np

from final_methods import _Base, build_method  # noqa: F401


class RAPTEDGuard(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.repo = {}
        self.prefix_ens = None
        self.cur = None
        self._reused = False
        self.errs = []
        self.gate_log = []

    def fit_initial(self, X_init, y_init, init_regime=None):
        c = super().fit_initial(X_init, y_init, init_regime)
        self.prefix_ens = self.ens
        if init_regime is not None:
            self.repo[init_regime] = self.ens
            self.cur = init_regime
        self.errs = []
        return c

    def predict_window(self, X_win, buf_X, buf_y):
        t0 = time.process_time()
        yp = self.ens.predict(X_win)
        return yp, time.process_time() - t0

    def _trigger(self, err):
        self.errs.append(err if err is not None else 1.0)
        if len(self.errs) > self.cfg.get("error_window_size", 20):
            self.errs.pop(0)
        if len(self.errs) < 5:
            return False
        mu = np.mean(self.errs[:-1]); sg = np.std(self.errs[:-1])
        return self.errs[-1] > mu + self.cfg.get("error_threshold_k", 2.0) * max(sg, 0.05)

    def _guard(self, reg, w, X_win, y_win, buf_X, buf_y):
        """Return (adapt_cpu, reused). Uses the 50-row lookback gate.

        The lookback decision uses only past labelled rows. The incoming
        window's labels are used solely to *evaluate* the decision afterwards
        (bad allow / bad block), never to make it."""
        self.last_flags = {"retrain": 0, "reuse": 0, "gate_allow": 0, "gate_block": 0}
        t0 = time.process_time()
        lb = self.cfg.get("edguard_lookback", 50)
        min_rows = self.cfg.get("edguard_min_rows", 30)
        margin = self.cfg.get("edguard_margin", 0.0)
        tol = self.cfg.get("gate_outcome_tol", 0.02)
        entry = dict(window_id=int(w), regime=str(reg))
        if reg in self.repo:
            stored_model = self.repo[reg]
            xb = np.asarray(buf_X[-lb:]); yb = np.asarray(buf_y[-lb:])
            if len(yb) >= min_rows:
                stored = float(np.mean(stored_model.predict(xb) == yb))
                prefix = float(np.mean(self.prefix_ens.predict(xb) == yb))
                allow = stored >= prefix + margin
                entry.update(stored_acc=stored, prefix_acc=prefix,
                             decision="allow" if allow else "block")
                if allow:
                    self.ens = stored_model
                    self._reused = True
                    self.last_flags["reuse"] = 1; self.last_flags["gate_allow"] = 1
                else:
                    self.ens = self._train(self.seed + w * 29, buf_X, buf_y)
                    self.repo[reg] = self.ens
                    self._reused = False
                    self.last_flags["retrain"] = 1; self.last_flags["gate_block"] = 1
            else:
                self.ens = stored_model
                self._reused = True
                self.last_flags["reuse"] = 1; self.last_flags["gate_allow"] = 1
                entry.update(stored_acc=np.nan, prefix_acc=np.nan, decision="allow")
        else:
            self.ens = self._train(self.seed + w * 19, buf_X, buf_y)
            self.repo[reg] = self.ens
            self._reused = False
            self.last_flags["retrain"] = 1
            entry = None
        if entry is not None:
            sw = float(np.mean(stored_model.predict(X_win) == y_win))
            pw = float(np.mean(self.prefix_ens.predict(X_win) == y_win))
            reuse_hurts = sw < pw - tol
            reuse_helps = sw > pw + tol
            dec = entry["decision"]
            entry.update(win_stored=sw, win_prefix=pw,
                         bad_allow=int(dec == "allow" and reuse_hurts),
                         bad_block=int(dec == "block" and reuse_helps))
            self.gate_log.append(entry)
        self.cur = reg
        return time.process_time() - t0

    def post_predict(self, w, idx, X_win, y_win, buf_X, buf_y,
                     err=None, regime=None, prev_regime=None, **k):
        self._push(X_win, y_win)
        self.last_flags = {}
        triggered = self._trigger(err)
        regime_return = (regime is not None and regime != self.cur)
        if not (triggered or regime_return):
            return 0.0
        # a trigger resets the error window, mirroring Event-Driven
        self.errs = [err]
        return self._guard(regime, w, X_win, y_win, buf_X, buf_y)
