"""
================================================================================
EXPERIMENT 3 — MODEL DEFINITIONS & ARCHITECTURES
================================================================================
Factory functions for base classifier models with complementary hyperparameters:
  1. Random Forest (RF):      50 trees, max_depth=7, min_samples_split=4
  2. Extra Trees (ET):        50 trees, max_depth=7, min_samples_split=6
  3. Gradient Boosting (GB):  50 trees, max_depth=4, lr=0.08, subsample=0.85

Baseline wrappers:
  - create_frozen_rf(seed)    -> RF trained once on baseline data, never retrained
  - create_retrained_rf(seed) -> RF retrained on all cumulative data at every window
================================================================================
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from config import BASE_MODEL_PARAMS


def create_frozen_rf(seed):
    """Approach 1: Frozen Random Forest — trained once on initial 2,000 samples."""
    p = BASE_MODEL_PARAMS['RandomForest']
    return RandomForestClassifier(random_state=seed, **p)


def create_retrained_rf(seed):
    """Approach 2: Continuously Retrained Random Forest."""
    p = BASE_MODEL_PARAMS['RandomForest']
    return RandomForestClassifier(random_state=seed, **p)


def create_candidate_models(seed):
    """
    Create the candidate models for the UCB1 Adaptive Ensemble with
    complementary hyperparameters to maximize model diversity.
    Returns:
      {'RandomForest': RF, 'ExtraTrees': ET, 'GradientBoosting': GB}
    """
    p_rf = BASE_MODEL_PARAMS['RandomForest']
    p_et = BASE_MODEL_PARAMS['ExtraTrees']
    p_gb = BASE_MODEL_PARAMS['GradientBoosting']
    return {
        'RandomForest': RandomForestClassifier(random_state=seed, **p_rf),
        'ExtraTrees': ExtraTreesClassifier(random_state=seed, **p_et),
        'GradientBoosting': GradientBoostingClassifier(random_state=seed, **p_gb),
    }


# Backward compatibility alias
create_base_models = create_candidate_models
