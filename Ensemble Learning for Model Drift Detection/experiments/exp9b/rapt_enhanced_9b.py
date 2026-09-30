"""
RAPT-Enhanced for Experiment 9B (drift studies).

Applies the two protocol-agnostic enhancements from the reference
"Enhanced Hybrid RAPT" architecture (rapt/enhanced_hybrid_rapt.py) on top of
the base RAPTSystem:

  (a) Buffer-Blended Novelty Refitting: novel regimes are refit on a larger
      recent buffer (1500 samples) than base RAPT (500).
  (b) Selective Parity Refitting: a *reused* policy whose recent streaming
      accuracy falls below `parity_threshold` is refit on the recent buffer, so
      a regime whose label semantics changed is not served indefinitely by a
      stale checkpoint.

The reference online micro-learner and dynamic decision threshold require a
per-sample binary stream and do not apply to the aggregated-window multi-class
protocols used here; they are not applied (documented in the report).
Deterministic and seed-parameterised.
"""

import time
import numpy as np

from rapt_9b import RAPTSystem, RegimeCheckpoint
from models_9b import create_base_ensemble


class RAPTEnhancedSystem(RAPTSystem):
    def __init__(self, seed=42, novelty_refit_n=1500, parity_threshold=0.5,
                 parity_window=3, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.novelty_refit_n = novelty_refit_n
        self.parity_threshold = parity_threshold
        self.parity_window = parity_window
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
            X = np.array(X_buffer)[-self.novelty_refit_n:]
            y = np.array(y_buffer)[-self.novelty_refit_n:]
            if len(np.unique(y)) < 2:
                X, y = np.array(X_buffer), np.array(y_buffer)
            ens.fit(X, y)
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
            self.cumulative_cpu_time += cpu
        return cpu
