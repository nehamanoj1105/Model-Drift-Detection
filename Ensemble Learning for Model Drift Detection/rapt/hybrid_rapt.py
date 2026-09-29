"""
================================================================================
RAPT STAGE 3 — TWO-TIER HYBRID RAPT ARCHITECTURE
================================================================================
Combines:
  1. Tier 1 (Always-On Online Micro-Learner): River HoeffdingAdaptiveTreeClassifier
     updated per-sample to immediately adapt to within-window zero-day attacks.
  2. Tier 2 (Autonomous Regime Repository Macro-Engine): Caches specialized
     tri-model ensembles (RF+ET+GB) and executes zero-retrain continuous policy
     transfer w_t = Sum_k s_k w_k during recurring operational regimes.
  3. Dynamic Hybrid Blending:
       P_Hybrid(X) = (1 - beta_t) * P_RAPT(X) + beta_t * P_Online(X)
     where beta_t = clip(1.0 - s_max, 0.0, 0.85) shifts reliance to the online
     learner when novelty is high, and defers to repository transfer when s_max >= 0.65.
================================================================================
"""

import time
import numpy as np
from river.tree import HoeffdingAdaptiveTreeClassifier
from repository import AutonomousRegimeRepository


class OnlineStreamLearner:
    """Wrapper around River HoeffdingAdaptiveTreeClassifier for fast incremental learning."""
    def __init__(self, seed=42):
        self.seed = seed
        self.model = HoeffdingAdaptiveTreeClassifier(seed=seed)
        self.fitted_samples = 0

    def warm_start(self, X_init, y_init, sample_limit=2000):
        """Pre-train on reference split."""
        n = min(len(X_init), sample_limit)
        for i in range(n):
            x_dict = {f"f{j}": float(X_init[i, j]) for j in range(X_init.shape[1])}
            self.model.learn_one(x_dict, int(y_init[i]))
        self.fitted_samples += n

    def predict_proba(self, X):
        """Predict probabilities across window."""
        n = len(X)
        probs = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x_dict = {f"f{j}": float(X[i, j]) for j in range(X.shape[1])}
            p_dict = self.model.predict_proba_one(x_dict)
            probs[i] = p_dict.get(1, 0.5)
        return np.clip(probs, 1e-5, 1.0 - 1e-5)

    def learn_window(self, X, y):
        """Incremental update on streaming window."""
        t0 = time.perf_counter()
        for i in range(len(X)):
            x_dict = {f"f{j}": float(X[i, j]) for j in range(X.shape[1])}
            self.model.learn_one(x_dict, int(y[i]))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.fitted_samples += len(X)
        return elapsed_ms


class TwoTierHybridRAPT:
    """
    Two-Tier Hybrid RAPT Architecture.
    Blends Autonomous Regime Repository (Tier 2) with Online Micro-Learner (Tier 1).
    """
    def __init__(self, max_capacity=8, gamma=15.0, novelty_threshold=0.65, seed=42):
        self.repository = AutonomousRegimeRepository(
            max_capacity=max_capacity, gamma=gamma, novelty_threshold=novelty_threshold
        )
        self.online_learner = OnlineStreamLearner(seed=seed)
        self.seed = seed
        self.gamma = gamma
        self.novelty_threshold = novelty_threshold

    def warm_start(self, base_fp, base_models, weights, X_init_sample, y_init_sample, window_id=0):
        """Initialize both repository and online learner on baseline reference."""
        self.repository.insert_regime(base_fp, base_models, weights, window_id=window_id, name="Initial_Baseline")
        self.online_learner.warm_start(X_init_sample, y_init_sample)

    def predict(self, X_win, fp, current_window_id=None):
        """
        Compute dynamic hybrid prediction:
          beta_t = clip(1.0 - s_max, 0.0, 0.85)
          P_Hybrid = (1 - beta_t) * P_RAPT + beta_t * P_Online
        """
        weights_dict, max_sim, closest_id = self.repository.compute_similarity_weights(fp)
        p_rapt = self.repository.predict_synthesized_proba(X_win, weights_dict, current_window_id=current_window_id)
        p_online = self.online_learner.predict_proba(X_win)

        # Dynamic reliance parameter beta
        if max_sim >= self.novelty_threshold:
            # High confidence in stored regime -> low reliance on online layer (0.0 to 0.15)
            beta = float(np.clip(0.15 * (1.0 - max_sim) / (1.0 - self.novelty_threshold + 1e-6), 0.0, 0.15))
        else:
            # Novel or transitional state -> elevate online reliance (0.15 to 1.0)
            beta = float(np.clip(0.15 + 0.85 * (self.novelty_threshold - max_sim) / (self.novelty_threshold + 1e-6), 0.15, 1.0))

        p_hybrid = (1.0 - beta) * p_rapt + beta * p_online
        p_hybrid = np.clip(p_hybrid, 1e-5, 1.0 - 1e-5)

        return p_hybrid, p_rapt, p_online, beta, max_sim, closest_id, weights_dict

    def update_online_learner(self, X_win, y_win):
        """Update Tier 1 online model with ground-truth streaming feedback."""
        return self.online_learner.learn_window(X_win, y_win)

    def adapt_macro_repository(self, fp, new_models, weights, window_id, name="Regime"):
        """Insert new specialized regime into Tier 2 repository."""
        return self.repository.insert_regime(fp, new_models, weights, window_id=window_id, name=name)
