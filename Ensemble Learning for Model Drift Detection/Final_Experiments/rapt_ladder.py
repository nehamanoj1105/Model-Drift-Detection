"""
RAPT mechanism ladder for Final_Experiments.

The base controller (Tier-2 policy repository) is the SAME algorithm used in
the existing 9A/9B experiments; the ladder only ADDS the three mechanisms
identified as missing from the 9A port of the canonical two-tier RAPT
(`rapt/hybrid_rapt.py`). Each rung adds exactly one mechanism:

  RAPT_T2       Tier-2 policy repository only  (= current 9A RAPT)
  RAPT_T1       + Tier-1 online micro-learner, beta-blended by regime similarity
  RAPT_T1_REFIT + always-on parity refit safety net
  RAPT_FULL     + fingerprint similarity gate on reuse

All mechanisms are deterministic and identical across seeds. Nothing is tuned
on the evaluation stream.
"""

import copy
import time
import numpy as np
from river.tree import HoeffdingAdaptiveTreeClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "exp9a"))
from models_9a import create_base_ensemble, make_train_buffer


# ---------------------------------------------------------------------------
# Tier 1 -- always-on online micro-learner (ported from rapt/hybrid_rapt.py)
# ---------------------------------------------------------------------------
class OnlineMicroLearner:
    """River HoeffdingAdaptiveTreeClassifier wrapper (multi-class aware)."""

    def __init__(self, seed=42):
        self.model = HoeffdingAdaptiveTreeClassifier(seed=seed)

    @staticmethod
    def _row(x):
        return {f"f{j}": float(v) for j, v in enumerate(x)}

    def warm_start(self, X, y, limit=2000):
        n = min(len(X), limit)
        for i in range(n):
            self.model.learn_one(self._row(X[i]), int(y[i]))

    def learn_window(self, X, y):
        for i in range(len(X)):
            self.model.learn_one(self._row(X[i]), int(y[i]))

    def predict_proba(self, X, classes):
        """Return (n_samples, n_classes) aligned to `classes` (unseen -> 0)."""
        out = np.zeros((len(X), len(classes)), dtype=np.float64)
        pos = {int(c): j for j, c in enumerate(classes)}
        for i in range(len(X)):
            p = self.model.predict_proba_one(self._row(X[i]))
            for c, v in (p or {}).items():
                if int(c) in pos:
                    out[i, pos[int(c)]] = float(v)
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        return out / row_sum


# ---------------------------------------------------------------------------
# Regime fingerprint + similarity (ported/adapted from rapt/fingerprint.py)
# ---------------------------------------------------------------------------
class RegimeFingerprint:
    """Low-dimensional, normalized fingerprint for the similarity gate.

    Averages the window to one vector and z-scores it against the training
    prefix, so the RBF scale (gamma) is comparable across features. This is the
    documented adaptation of the reference 8-d physical fingerprint for a
    19-feature tabular stream; the raw un-normalised fingerprint saturates.
    """

    def __init__(self, X_ref):
        self.mean = np.mean(X_ref, axis=0)
        self.std = np.maximum(np.std(X_ref, axis=0), 1e-6)

    def extract(self, X_win):
        v = (np.mean(X_win, axis=0) - self.mean) / self.std
        return v.astype(np.float64)


def rbf_similarity(f_a, f_b, gamma=1.0):
    d = float(np.sum((f_a - f_b) ** 2))
    return float(np.exp(-gamma * d))


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------
class LadderRAPT:
    """Configurable RAPT. Flags select the mechanisms (see module docstring)."""

    def __init__(self, seed=42, use_tier1=False, use_refit=False, use_gate=False,
                 refit_mode="absolute", rel_drop=0.10,
                 anchor_X=None, anchor_y=None, buffer_capacity=1000,
                 gamma=1.0, novelty_threshold=0.65, refit_n=1500,
                 parity_threshold=0.5, parity_window=3):
        self.seed = seed
        self.use_tier1 = use_tier1
        self.use_refit = use_refit
        self.use_gate = use_gate
        self.refit_mode = refit_mode
        self.rel_drop = rel_drop
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.buffer_capacity = buffer_capacity
        self.gamma = gamma
        self.novelty_threshold = novelty_threshold
        self.refit_n = refit_n
        self.parity_threshold = parity_threshold
        self.parity_window = parity_window

        self.repository = {}
        self.fingerprints = {}
        self.current_regime_id = None
        self.active_ensemble = None
        self.online = OnlineMicroLearner(seed=seed) if use_tier1 else None
        self.fp = None
        self.beta = 0.0

        self.created_policy_count = 0
        self.reused_policy_count = 0
        self.trees_trained_count = 0
        self.trees_reused_count = 0
        self.adaptation_cpu_time = 0.0
        self.cumulative_cpu_time = 0.0
        self.parity_refits = 0
        self._recent_acc = []
        self._base_acc = None
        self._current_reused = False

    # -- initial ------------------------------------------------------------
    def fit_initial(self, regime_id, X_init, y_init):
        t0 = time.process_time()
        self.active_ensemble = create_base_ensemble(seed=self.seed)
        self.active_ensemble.fit(X_init, y_init)
        cpu = time.process_time() - t0
        self.cumulative_cpu_time += cpu
        self.trees_trained_count += self.active_ensemble.get_num_trees()
        self.repository[regime_id] = (self.active_ensemble, list(self.active_ensemble.weights))
        self.current_regime_id = regime_id
        self.created_policy_count += 1
        if self.use_tier1:
            self.online.warm_start(X_init, y_init)
        if self.use_gate:
            self.fp = RegimeFingerprint(X_init)
            self.fingerprints[regime_id] = self.fp.extract(X_init)
        return cpu, 0.0

    # -- transition ---------------------------------------------------------
    def _similarity_beta(self, X_buffer):
        if not self.use_gate or self.fp is None or not self.fingerprints:
            return 1.0, True   # no gate -> always allow reuse
        f_t = self.fp.extract(np.asarray(X_buffer[-self.buffer_capacity:]))
        s = {k: rbf_similarity(f_t, v, self.gamma)
             for k, v in self.fingerprints.items()}
        best_k = max(s, key=s.get)
        s_max = s[best_k]
        return s_max, s_max >= self.novelty_threshold

    def handle_regime_transition(self, new_regime_id, X_buffer=None, y_buffer=None):
        t0 = time.process_time()
        allow_reuse = True
        if self.use_gate and X_buffer is not None and len(X_buffer) > 0:
            _, allow_reuse = self._similarity_beta(X_buffer)

        if new_regime_id in self.repository and allow_reuse:
            ens, weights = self.repository[new_regime_id]
            self.active_ensemble = copy.copy(ens)
            self.active_ensemble.weights = list(weights)
            self.reused_policy_count += 1
            self.trees_reused_count += self.active_ensemble.get_num_trees()
            self._current_reused = True
            # optional calibration of RF/ET soft-vote weights on recent buffer
            if X_buffer is not None and len(X_buffer) >= 20:
                Xb = np.asarray(X_buffer[-50:]); yb = np.asarray(y_buffer[-50:])
                a_rf = np.mean(np.argmax(self.active_ensemble.rf.predict_proba(Xb), 1) == yb)
                a_et = np.mean(np.argmax(self.active_ensemble.et.predict_proba(Xb), 1) == yb)
                if a_rf + a_et > 0:
                    self.active_ensemble.weights = [a_rf / (a_rf + a_et), a_et / (a_rf + a_et)]
        else:
            self.active_ensemble = self._train_policy(X_buffer, y_buffer)
            self.trees_trained_count += self.active_ensemble.get_num_trees()
            self.created_policy_count += 1
            self.repository[new_regime_id] = (self.active_ensemble,
                                              list(self.active_ensemble.weights))
            if self.use_gate and self.fp is not None and X_buffer is not None and len(X_buffer):
                self.fingerprints[new_regime_id] = self.fp.extract(
                    np.asarray(X_buffer[-self.buffer_capacity:]))
            self._current_reused = False
        self.current_regime_id = new_regime_id
        self._recent_acc = []
        cpu = time.process_time() - t0
        self.cumulative_cpu_time += cpu
        self.adaptation_cpu_time += cpu
        return self._current_reused, cpu, 0.0

    def _train_policy(self, X_buffer, y_buffer):
        ens = create_base_ensemble(seed=self.seed + self.created_policy_count * 19)
        if X_buffer is not None and len(X_buffer) > 50:
            n = self.refit_n if self.use_refit else 500
            Xr = np.asarray(X_buffer)[-n:]; yr = np.asarray(y_buffer)[-n:]
            trX, trY = make_train_buffer(Xr, yr, self.anchor_X, self.anchor_y,
                                         self.buffer_capacity)
            ens.fit(trX, trY)
        return ens

    # -- streaming ----------------------------------------------------------
    def predict_proba(self, X_win):
        p_rapt = self.active_ensemble.predict_proba(X_win)
        if not self.use_tier1 or self.beta < 1e-3:
            return p_rapt
        p_on = self.online.predict_proba(X_win, self.active_ensemble.classes_)
        return (1.0 - self.beta) * p_rapt + self.beta * p_on

    def predict(self, X_win):
        return self.active_ensemble.classes_[np.argmax(self.predict_proba(X_win), axis=1)]

    def update(self, X_win, y_win, X_buffer=None, y_buffer=None):
        t0 = time.process_time()
        cpu = 0.0
        if self.use_tier1:
            self.online.learn_window(X_win, y_win)
            # beta driven by similarity to the closest stored regime
            if self.use_gate and self.fp is not None and X_buffer is not None and len(X_buffer):
                s_max, _ = self._similarity_beta(X_buffer)
                if s_max >= self.novelty_threshold:
                    self.beta = float(np.clip(
                        0.15 * (1.0 - s_max) / (1.0 - self.novelty_threshold + 1e-6), 0.0, 0.15))
                else:
                    self.beta = float(np.clip(
                        0.15 + 0.85 * (self.novelty_threshold - s_max) / (self.novelty_threshold + 1e-6),
                        0.15, 0.85))
            else:
                self.beta = 0.15
        if self.use_refit:
            acc = float(np.mean(self.predict(X_win) == np.asarray(y_win)))
            # Track a pre-drift baseline of accuracy for the active policy so the
            # relative trigger can fire on a 3-class stream (absolute 0.5 cannot).
            if len(self._recent_acc) < self.parity_window:
                self._base_acc = 0.5 * self._base_acc + 0.5 * acc if self._base_acc else acc
            self._recent_acc.append(acc)
            if len(self._recent_acc) > self.parity_window:
                self._recent_acc.pop(0)
            if self.refit_mode == "relative":
                degraded = (self._base_acc is not None
                            and np.mean(self._recent_acc) < self._base_acc - self.rel_drop)
            else:
                degraded = np.mean(self._recent_acc) < self.parity_threshold
            if (X_buffer is not None and len(self._recent_acc) >= self.parity_window
                    and degraded):
                self.active_ensemble = self._train_policy(X_buffer, y_buffer)
                self.repository[self.current_regime_id] = (
                    self.active_ensemble, list(self.active_ensemble.weights))
                self.trees_trained_count += self.active_ensemble.get_num_trees()
                self.parity_refits += 1
                self._recent_acc = []
                self._base_acc = None
                cpu += time.process_time() - t0
                self.adaptation_cpu_time += cpu
                self.cumulative_cpu_time += cpu
        return cpu


def build_rapt(variant, **kw):
    rel = variant == "RAPT_REL_REFIT"
    return LadderRAPT(
        use_tier1=variant in ("RAPT_T1", "RAPT_T1_REFIT", "RAPT_FULL", "RAPT_REL_REFIT"),
        use_refit=variant in ("RAPT_T1_REFIT", "RAPT_FULL", "RAPT_REL_REFIT"),
        use_gate=variant in ("RAPT_FULL", "RAPT_REL_REFIT"),
        refit_mode="relative" if rel else "absolute",
        **kw,
    )
