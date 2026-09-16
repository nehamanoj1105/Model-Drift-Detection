"""
================================================================================
EXPERIMENT 3 — UCB1 MULTI-ARMED BANDIT & REWARD FUNCTION
================================================================================
Arms (Candidate Base Models):
- Arm 0: RandomForest (RF)
- Arm 1: ExtraTrees (ET)
- Arm 2: GradientBoosting (GB)
No SGDClassifier or other models.

UCB1 Formulation:
    UCB_i = x_bar_i + c * sqrt(ln(t) / n_i)
where:
    i       = model/arm
    x_bar_i = estimated average reward of arm i
    n_i     = number of times arm i was selected
    t       = total number of decisions made across all arms
    c       = exploration coefficient (default c=1.0)

Initial Exploration:
    Every arm is pulled at least once initially before applying the UCB rule.

Cost-Aware Reward:
    Reward = F1 - beta * normalized_computational_cost
    where beta = 0.05 by default.
================================================================================
"""

import numpy as np


class UCB1Bandit:
    """
    Multi-Armed Bandit using UCB1 for online model selection under distribution drift.
    """

    def __init__(self, n_arms=3, c=1.0, ref_cpu_time=0.5, arm_names=None, beta=0.05):
        self.n_arms = n_arms
        self.c = c
        self.ref_cpu_time = ref_cpu_time
        self.beta = beta
        self.arm_names = arm_names or ['RandomForest', 'ExtraTrees', 'GradientBoosting']
        
        self.counts = np.zeros(n_arms, dtype=int)
        self.cumulative_rewards = np.zeros(n_arms, dtype=float)
        self.avg_rewards = np.zeros(n_arms, dtype=float)
        self.ucb_scores = np.full(n_arms, float('inf'))
        self.total_selections = 0
        self.selection_history = []
        self.reward_history = []

    def select_arm(self):
        """
        Select model arm.
        Initial phase explores each arm once.
        Subsequent decisions balance exploitation and exploration using UCB1:
        UCB_i = x_bar_i + c * sqrt(ln(t) / n_i)
        """
        # Initial exploration: pull each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB1 decision rule
        t = self.total_selections
        for arm in range(self.n_arms):
            bonus = self.c * np.sqrt(np.log(max(t, 1)) / self.counts[arm])
            self.ucb_scores[arm] = self.avg_rewards[arm] + bonus

        return int(np.argmax(self.ucb_scores))

    def compute_reward(self, f1, cpu_time, wall_time=0.0, retrained=False, lambda_val=None):
        """
        Cost-aware reward:
        Reward = F1 - beta * Cost
        where Cost = min(1.0, max(0.0, cpu_time / ref_cpu_time))
        """
        beta = self.beta if lambda_val is None else lambda_val
        normalized_cost = min(1.0, max(0.0, cpu_time / self.ref_cpu_time)) if self.ref_cpu_time > 0 else 0.0
        reward = f1 - beta * normalized_cost
        return float(reward), float(normalized_cost)

    def update(self, arm, reward):
        """Update arm statistics and UCB values after receiving reward."""
        self.total_selections += 1
        self.counts[arm] += 1
        self.cumulative_rewards[arm] += reward
        self.avg_rewards[arm] = self.cumulative_rewards[arm] / self.counts[arm]
        self.selection_history.append(arm)
        self.reward_history.append(reward)

        # Update UCB scores
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
    """Random arm selector baseline for regret benchmarking."""
    def __init__(self, seed=42, n_arms=3):
        self.rng = np.random.RandomState(seed + 888)
        self.n_arms = n_arms

    def select_arm(self):
        return int(self.rng.randint(0, self.n_arms))
