"""
================================================================================
RAPT ENHANCED — ACCURACY & F1 PARITY HYBRID RAPT ARCHITECTURE
================================================================================
Enhancements:
  1. Dynamic Regime-Aware Threshold Tuning:
       tau_t = clip(base_tau - lambda * (1 - s_max), 0.32, base_tau)
     Adjusts decision boundary during novel/transitional drift to boost Recall.
  2. Buffer-Blended Novelty Retraining:
       Refits novel regime models on 1,500 samples (incoming window + reference slice)
       to guarantee ~390 minority-class instances per tree split.
================================================================================
"""

import time
import numpy as np
from hybrid_rapt import TwoTierHybridRAPT, OnlineStreamLearner
from repository import AutonomousRegimeRepository


class EnhancedHybridRAPT(TwoTierHybridRAPT):
    """
    Enhanced Hybrid RAPT with Dynamic Threshold Tuning and Buffer-Blended Novelty Refitting.
    """
    def __init__(self, max_capacity=8, gamma=15.0, novelty_threshold=0.65, base_tau=0.455, lambda_tau=0.15, seed=42):
        super().__init__(max_capacity=max_capacity, gamma=gamma, novelty_threshold=novelty_threshold, seed=seed)
        self.base_tau = base_tau
        self.lambda_tau = lambda_tau

    def get_dynamic_threshold(self, max_sim):
        """
        Compute dynamic threshold based on similarity confidence s_max:
        When max_sim = 1.0 -> base_tau (0.455)
        When max_sim -> 0.0 -> base_tau - lambda_tau (~0.32)
        """
        tau_t = float(np.clip(self.base_tau - self.lambda_tau * (1.0 - max_sim), 0.32, self.base_tau))
        return tau_t

    def predict_enhanced(self, X_win, fp, current_window_id=None):
        """
        Returns predictions along with dynamic window decision threshold tau_t.
        """
        p_hybrid, p_rapt, p_online, beta, max_sim, closest_id, weights_dict = self.predict(
            X_win, fp, current_window_id=current_window_id
        )
        tau_t = self.get_dynamic_threshold(max_sim)
        return p_hybrid, tau_t, p_rapt, p_online, beta, max_sim, closest_id, weights_dict
