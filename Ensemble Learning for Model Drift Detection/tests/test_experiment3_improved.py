"""
Unit Tests for Improved Experiment 3 Components:
  - DiscountedUCBBandit (D-UCB) with non-stationary discounting and dual reward
  - ImprovedAdaptiveEnsemble with rank-weighted soft voting and joint utility tracking
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiment3')))

from bandit_improved import DiscountedUCBBandit
from ensemble_improved import ImprovedAdaptiveEnsemble


def test_discounted_ucb_bandit():
    bandit = DiscountedUCBBandit(n_arms=3, gamma_d=0.90, c=0.25, lambda_cost=0.03)
    assert bandit.n_arms == 3
    assert bandit.gamma_d == 0.90

    # Arm exploration
    a0 = bandit.select_arm()
    r0, _ = bandit.compute_reward(f1=0.75, accuracy=0.82, cpu_time=0.05)
    bandit.update(a0, r0)

    a1 = bandit.select_arm()
    r1, _ = bandit.compute_reward(f1=0.78, accuracy=0.85, cpu_time=0.05)
    bandit.update(a1, r1)

    a2 = bandit.select_arm()
    r2, _ = bandit.compute_reward(f1=0.80, accuracy=0.88, cpu_time=0.10)
    bandit.update(a2, r2)

    assert set([a0, a1, a2]) == {0, 1, 2}

    # Verify discounting took place
    assert bandit.counts[a0] < 1.0  # Discounted twice by 0.90
    assert bandit.total_selections == 3

    # Subsequent selection
    next_arm = bandit.select_arm()
    assert next_arm in [0, 1, 2]


def test_improved_adaptive_ensemble():
    rng = np.random.RandomState(42)
    X_train = rng.normal(size=(200, 4))
    y_train = (rng.uniform(size=200) < 0.3).astype(int)

    ens = ImprovedAdaptiveEnsemble(seed=42, threshold=0.455)
    ens.fit_initial(X_train, y_train)

    X_test = rng.normal(size=(50, 4))
    y_test = (rng.uniform(size=50) < 0.3).astype(int)

    pred, prob, comp_preds, comp_probs = ens.predict(X_test)
    assert len(pred) == 50
    assert len(prob) == 50
    assert set(comp_preds.keys()) == {'RF', 'ET', 'GB'}
    assert all(0.0 <= p <= 1.0 for p in prob)

    # Check rank weights updating
    comp_m, weights, diff, div_m = ens.update_weights(y_test, comp_preds, comp_probs)
    assert np.isclose(sum(weights.values()), 1.0)
    assert set(weights.values()) == {0.18, 0.32, 0.50}
