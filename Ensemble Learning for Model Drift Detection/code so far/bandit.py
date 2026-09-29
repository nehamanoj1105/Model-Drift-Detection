"""
================================================================================
EXPERIMENT 3 — UCB1 MULTI-ARMED BANDIT
================================================================================
Arms (within UCB1 Adaptive Ensemble — Approach 3):
  Arm 0 = RF          (Random Forest)
  Arm 1 = ET          (Extra Trees)
  Arm 2 = Ensemble    (RF + ET + GB diversity-weighted combination)

Standard UCB1 formula:
    UCB_k = mean_reward_k + c * sqrt(ln(t) / N_k)

Cost-aware reward:
    Reward = F1 - lambda * min(1.0, cpu_time / ref_cpu_time)
================================================================================
"""

import numpy as np
from config import ARM_NAMES, N_ARMS, C_EXPLORATION, LAMBDA_COST, REF_CPU_TIME


class UCB1Bandit:
    """Standard UCB1 multi-armed bandit for online model selection."""

    def __init__(self, n_arms=N_ARMS, c=C_EXPLORATION,
                 lambda_cost=LAMBDA_COST, ref_cpu_time=REF_CPU_TIME,
                 arm_names=None):
        self.n_arms = n_arms
        self.c = c
        self.lambda_cost = lambda_cost
        self.ref_cpu_time = ref_cpu_time
        self.arm_names = arm_names or list(ARM_NAMES)

        self.counts = np.zeros(n_arms, dtype=int)
        self.cumulative_rewards = np.zeros(n_arms, dtype=float)
        self.avg_rewards = np.zeros(n_arms, dtype=float)
        self.ucb_scores = np.full(n_arms, float('inf'))
        self.total_selections = 0
        self.selection_history = []
        self.reward_history = []

    def select_arm(self):
        """
        Initial phase: explore each arm at least once.
        Subsequent decisions: UCB_k = mean_reward_k + c * sqrt(ln(t) / N_k).
        """
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        t = self.total_selections
        for arm in range(self.n_arms):
            bonus = self.c * np.sqrt(np.log(max(t, 1)) / self.counts[arm])
            self.ucb_scores[arm] = self.avg_rewards[arm] + bonus

        return int(np.argmax(self.ucb_scores))

    def compute_reward(self, f1, cpu_time):
        """
        Cost-aware reward: Reward = F1 - lambda * normalized_cost
        """
        if self.ref_cpu_time > 0:
            normalized_cost = min(1.0, max(0.0, cpu_time / self.ref_cpu_time))
        else:
            normalized_cost = 0.0
        reward = f1 - self.lambda_cost * normalized_cost
        return float(reward), float(normalized_cost)

    def update(self, arm, reward):
        """Update arm statistics after receiving reward."""
        self.total_selections += 1
        self.counts[arm] += 1
        self.cumulative_rewards[arm] += reward
        self.avg_rewards[arm] = self.cumulative_rewards[arm] / self.counts[arm]
        self.selection_history.append(arm)
        self.reward_history.append(reward)

        # Update UCB values
        t = self.total_selections
        for a in range(self.n_arms):
            if self.counts[a] > 0:
                bonus = self.c * np.sqrt(np.log(max(t, 1)) / self.counts[a])
                self.ucb_scores[a] = self.avg_rewards[a] + bonus
            else:
                self.ucb_scores[a] = float('inf')

    def reset(self):
        """Reset bandit state."""
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.cumulative_rewards = np.zeros(self.n_arms, dtype=float)
        self.avg_rewards = np.zeros(self.n_arms, dtype=float)
        self.ucb_scores = np.full(self.n_arms, float('inf'))
        self.total_selections = 0
        self.selection_history = []
        self.reward_history = []


class RandomSelector:
    """Uniform random arm selector for benchmarking."""
    def __init__(self, seed=42, n_arms=N_ARMS):
        self.rng = np.random.RandomState(seed + 888)
        self.n_arms = n_arms

    def select_arm(self):
        return int(self.rng.randint(0, self.n_arms))
