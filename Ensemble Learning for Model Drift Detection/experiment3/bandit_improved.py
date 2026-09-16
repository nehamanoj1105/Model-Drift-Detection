"""
================================================================================
EXPERIMENT 3 (IMPROVED) — DISCOUNTED UCB BANDIT (D-UCB)
================================================================================
Arms (within Enhanced UCB1 Adaptive Ensemble — Approach 3):
  Arm 0 = RF          (Random Forest)
  Arm 1 = ET          (Extra Trees)
  Arm 2 = Ensemble    (RF + ET + GB rank-weighted soft voting)

Mathematical Formulation:
  - Discounted-UCB (Garivier & Moulines, 2008) for non-stationary streaming drift:
      N_k(t) = gamma * N_k(t-1) + I(a_t = k)
      X_k(t) = (gamma * X_k(t-1) + r_t * I(a_t = k)) / N_k(t)
      UCB_k(t) = X_k(t) + c * sqrt(ln(sum_j N_j(t)) / N_k(t))

  - Dual-Objective Balanced Reward:
      Reward = 0.5 * F1 + 0.5 * Accuracy - lambda * normalized_cost
================================================================================
"""

import numpy as np
from config import ARM_NAMES, N_ARMS, REF_CPU_TIME


class DiscountedUCBBandit:
    """
    Discounted Upper Confidence Bound (D-UCB) multi-armed bandit
    calibrated for streaming drift environments.
    """

    def __init__(self, n_arms=N_ARMS, gamma_d=0.90, c=0.25,
                 lambda_cost=0.03, ref_cpu_time=REF_CPU_TIME,
                 arm_names=None):
        self.n_arms = n_arms
        self.gamma_d = gamma_d
        self.c = c
        self.lambda_cost = lambda_cost
        self.ref_cpu_time = ref_cpu_time
        self.arm_names = arm_names or list(ARM_NAMES)

        self.counts = np.zeros(n_arms, dtype=float)
        self.cumulative_rewards = np.zeros(n_arms, dtype=float)
        self.avg_rewards = np.zeros(n_arms, dtype=float)
        self.ucb_scores = np.full(n_arms, float('inf'))
        self.total_selections = 0
        self.selection_history = []
        self.reward_history = []

    def select_arm(self):
        """
        Initial phase: explore each arm until count >= 0.8.
        Subsequent decisions: argmax(UCB_k).
        """
        for arm in range(self.n_arms):
            if self.counts[arm] < 0.8:
                return arm

        total_n = np.sum(self.counts)
        for arm in range(self.n_arms):
            n_k = max(self.counts[arm], 1e-4)
            avg_r = self.cumulative_rewards[arm] / n_k
            bonus = self.c * np.sqrt(np.log(max(total_n, 1.0)) / n_k)
            self.ucb_scores[arm] = avg_r + bonus

        return int(np.argmax(self.ucb_scores))

    def compute_reward(self, f1, accuracy, cpu_time):
        """
        Dual-objective reward balancing F1 and Accuracy minus resource penalty:
          Reward = 0.5 * F1 + 0.5 * Accuracy - lambda * normalized_cost
        """
        if self.ref_cpu_time > 0:
            normalized_cost = min(1.0, max(0.0, cpu_time / self.ref_cpu_time))
        else:
            normalized_cost = 0.0

        reward = 0.5 * f1 + 0.5 * accuracy - self.lambda_cost * normalized_cost
        return float(reward), float(normalized_cost)

    def update(self, arm, reward):
        """
        Discount historical observations by gamma_d and add current observation.
        """
        self.total_selections += 1

        # Exponential decay of historical arm statistics
        self.counts *= self.gamma_d
        self.cumulative_rewards *= self.gamma_d

        self.counts[arm] += 1.0
        self.cumulative_rewards[arm] += reward
        self.avg_rewards[arm] = self.cumulative_rewards[arm] / max(self.counts[arm], 1e-4)

        self.selection_history.append(arm)
        self.reward_history.append(reward)

        # Update UCB values
        total_n = np.sum(self.counts)
        for a in range(self.n_arms):
            n_k = max(self.counts[a], 1e-4)
            avg_r = self.cumulative_rewards[a] / n_k
            bonus = self.c * np.sqrt(np.log(max(total_n, 1.0)) / n_k)
            self.ucb_scores[a] = avg_r + bonus

    def reset(self):
        """Reset bandit state."""
        self.counts = np.zeros(self.n_arms, dtype=float)
        self.cumulative_rewards = np.zeros(self.n_arms, dtype=float)
        self.avg_rewards = np.zeros(self.n_arms, dtype=float)
        self.ucb_scores = np.full(self.n_arms, float('inf'))
        self.total_selections = 0
        self.selection_history = []
        self.reward_history = []
