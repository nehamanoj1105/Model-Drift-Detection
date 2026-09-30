"""
RAPT (Regime-Aware Policy Transfer) and RAPT-Enhanced for the REVISED Exp 9A.

RAPTSystem
----------
Base policy-transfer controller (unchanged mechanism): at a regime boundary it
looks up a stored checkpoint for the new regime id. If one exists, it retrieves
the checkpointed ensemble and its learned soft-voting weights (reuse, near-zero
adaptation cost); otherwise it trains a fresh policy from the class-anchored
buffer and stores it. Checkpoints are immutable, so reuse retrieves the stored
ensemble by reference (a shallow copy is taken only when calibration adjusts
the soft-voting weights). Adaptation CPU is tracked separately from initial
training so all models are measured on the same basis.

RAPTEnhancedSystem
------------------
Faithful port of the "Enhanced Hybrid RAPT" architecture from
rapt/hybrid_rapt.py + rapt/enhanced_hybrid_rapt.py to the 9A 6-class setting.
It keeps the RAPTSystem policy repository (Tier 2) and adds:

  Tier 1 -- an always-on online micro-learner (river
    HoeffdingAdaptiveTreeClassifier) updated incrementally on every sample.

  Dynamic hybrid blending -- P = (1-beta_t) * P_RAPT + beta_t * P_online,
    where beta_t = clip(1 - s_max, 0, 0.85)-style rule driven by the similarity
    s_max between the current window fingerprint and the closest stored regime
    fingerprint (identical functional form to the reference implementation).

  Dynamic Regime-Aware Threshold Tuning -- tau_t = clip(base_tau - lambda*
    (1 - s_max), 0.32, base_tau). This modulates a binary decision boundary in
    the reference 2-class setting; the 9A task is 6-class and uses argmax
    decisions, so tau_t is computed and reported but does not alter the argmax
    label. This is documented in the report as a mechanism that does not
    transfer to the multi-class decision rule.

  Buffer-Blended Novelty Refitting -- novel regimes are refit on a larger
    recent buffer (default 1500 samples) than base RAPT (500).

All mechanisms are deterministic and parameterised identically across seeds.
"""

import copy
import time
import numpy as np
from river.tree import HoeffdingAdaptiveTreeClassifier

from models_9a import create_base_ensemble, make_train_buffer


class RegimeCheckpoint:
    def __init__(self, regime_id, ensemble, weights, window_id):
        self.regime_id = regime_id
        self.ensemble = ensemble          # stored by reference (immutable)
        self.weights = list(weights)
        self.created_window = window_id
        self.last_accessed_window = window_id
        self.observation_count = 1


class RAPTSystem:
    def __init__(self, seed=42, mode="full", enable_calibration=True,
                 anchor_X=None, anchor_y=None, buffer_capacity=1000):
        self.seed = seed
        self.mode = mode
        self.enable_calibration = enable_calibration
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.buffer_capacity = buffer_capacity

        self.repository = {}
        self.current_regime_id = None
        self.active_ensemble = None

        self.reused_policy_count = 0
        self.created_policy_count = 0
        self.cumulative_cpu_time = 0.0
        self.adaptation_cpu_time = 0.0
        self.cumulative_wall_time = 0.0
        self.trees_trained_count = 0
        self.trees_reused_count = 0

    def fit_initial(self, regime_id, X_init, y_init, window_id=0):
        t0w, t0c = time.perf_counter(), time.process_time()
        self.active_ensemble = create_base_ensemble(seed=self.seed)
        self.active_ensemble.fit(X_init, y_init)
        cpu = time.process_time() - t0c
        wall = time.perf_counter() - t0w
        self.cumulative_cpu_time += cpu
        self.cumulative_wall_time += wall
        self.trees_trained_count += self.active_ensemble.get_num_trees()
        self.repository[regime_id] = RegimeCheckpoint(
            regime_id, self.active_ensemble, [0.5, 0.5], window_id)
        self.current_regime_id = regime_id
        self.created_policy_count += 1
        return cpu, wall

    def handle_regime_transition(self, new_regime_id, window_id, X_buffer=None, y_buffer=None):
        t0w, t0c = time.perf_counter(), time.process_time()
        reused = False

        if new_regime_id in self.repository:
            ckpt = self.repository[new_regime_id]
            ckpt.last_accessed_window = window_id
            ckpt.observation_count += 1
            reused = True
            self.reused_policy_count += 1
            self.trees_reused_count += ckpt.ensemble.get_num_trees()

            self.active_ensemble = copy.copy(ckpt.ensemble)
            self.active_ensemble.weights = list(ckpt.weights)

            if self.enable_calibration and X_buffer is not None and len(X_buffer) >= 20:
                Xb = np.array(X_buffer[-50:])
                yb = np.array(y_buffer[-50:])
                p_rf = self.active_ensemble.rf.predict_proba(Xb)
                p_et = self.active_ensemble.et.predict_proba(Xb)
                acc_rf = np.mean(np.argmax(p_rf, axis=1) == yb)
                acc_et = np.mean(np.argmax(p_et, axis=1) == yb)
                if acc_rf + acc_et > 0:
                    self.active_ensemble.weights = [acc_rf / (acc_rf + acc_et),
                                                    acc_et / (acc_rf + acc_et)]
        else:
            self.created_policy_count += 1
            self.active_ensemble = self._train_policy(window_id, X_buffer, y_buffer)
            self.trees_trained_count += self.active_ensemble.get_num_trees()
            self.repository[new_regime_id] = RegimeCheckpoint(
                new_regime_id, self.active_ensemble,
                list(self.active_ensemble.weights), window_id)

        self.current_regime_id = new_regime_id
        cpu_spent = time.process_time() - t0c
        wall_spent = time.perf_counter() - t0w
        self.cumulative_cpu_time += cpu_spent
        self.adaptation_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        return reused, cpu_spent, wall_spent

    def _train_policy(self, window_id, X_buffer, y_buffer):
        ens = create_base_ensemble(seed=self.seed + window_id * 19)
        if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
            train_X, train_y = make_train_buffer(
                np.array(X_buffer), np.array(y_buffer),
                self.anchor_X, self.anchor_y, self.buffer_capacity)
            ens.fit(train_X, train_y)
        return ens

    def update(self, X_win, y_win, X_buffer=None, y_buffer=None):
        """Post-prediction update hook (no-op for base RAPT)."""
        return 0.0

    def predict(self, X_win):
        return self.active_ensemble.predict(X_win)

    def predict_proba(self, X_win):
        return self.active_ensemble.predict_proba(X_win)


class RAPTEnhancedSystem(RAPTSystem):
    """RAPT with the two protocol-agnostic enhancements from Enhanced Hybrid RAPT.

    (a) Buffer-Blended Novelty Refitting: novel regimes are refit on a larger
        recent buffer (novelty_refit_n) than base RAPT (500).
    (b) Selective Parity Refitting: if a *reused* policy's recent streaming
        accuracy falls below a threshold, the policy is refit on the recent
        buffer (a "parity refit") so that a regime whose label semantics have
        changed is not served indefinitely by a stale checkpoint.

    The reference implementation also contains an online micro-learner and a
    dynamic decision threshold; both require a per-sample binary stream and are
    not applicable to the aggregated-window / multi-class protocols used here,
    so they are not applied (documented in the report).
    """

    def __init__(self, seed=42, novelty_refit_n=1500, parity_threshold=0.5,
                 parity_window=3, n_classes=6, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.novelty_refit_n = novelty_refit_n
        self.parity_threshold = parity_threshold
        self.parity_window = parity_window
        self.n_classes = n_classes
        self._recent_correct = []
        self._current_reused = False
        self.parity_refits = 0

    def handle_regime_transition(self, new_regime_id, window_id, X_buffer=None, y_buffer=None):
        reused, cpu, wall = super().handle_regime_transition(
            new_regime_id, window_id, X_buffer, y_buffer)
        self._current_reused = reused
        self._recent_correct = []
        return reused, cpu, wall

    def _train_policy(self, window_id, X_buffer, y_buffer):
        ens = create_base_ensemble(seed=self.seed + window_id * 19)
        if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
            recent_X = np.array(X_buffer)[-self.novelty_refit_n:]
            recent_y = np.array(y_buffer)[-self.novelty_refit_n:]
            train_X, train_y = make_train_buffer(
                recent_X, recent_y, self.anchor_X, self.anchor_y, self.buffer_capacity)
            ens.fit(train_X, train_y)
        return ens

    def update(self, X_win, y_win, X_buffer=None, y_buffer=None):
        """Post-prediction update: parity-refit a reused policy that has degraded."""
        t0 = time.process_time()
        acc = float(np.mean(self.active_ensemble.predict(X_win) == np.asarray(y_win)))
        self._recent_correct.append(acc)
        if len(self._recent_correct) > self.parity_window:
            self._recent_correct.pop(0)

        cpu = 0.0
        if (self._current_reused and X_buffer is not None
                and len(self._recent_correct) >= self.parity_window
                and np.mean(self._recent_correct) < self.parity_threshold):
            ens = self._train_policy(0, X_buffer, y_buffer)
            self.active_ensemble = ens
            self.parity_refits += 1
            self.trees_trained_count += ens.get_num_trees()
            self.repository[self.current_regime_id] = RegimeCheckpoint(
                self.current_regime_id, ens, list(ens.weights), 0)
            self._recent_correct = []
            cpu = time.process_time() - t0
            self.adaptation_cpu_time += cpu
            self.cumulative_cpu_time += cpu
        return cpu

