"""
================================================================================
EXPERIMENT 5 — MODEL DEFINITIONS & FACTORIES
================================================================================
Factory functions for base classifier models and ensemble creation:
  1. Random Forest (RF):      50 trees, max_depth=7, min_samples_split=4
  2. Extra Trees (ET):        50 trees, max_depth=7, min_samples_split=6
  3. Gradient Boosting (GB):  50 trees, max_depth=4, lr=0.08, subsample=0.85
================================================================================
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from config import BASE_MODEL_PARAMS


def create_candidate_models(seed):
    """
    Create the three base candidate models for the heterogeneous ensemble:
      - RandomForestClassifier
      - ExtraTreesClassifier
      - GradientBoostingClassifier
    """
    p_rf = BASE_MODEL_PARAMS['RandomForest']
    p_et = BASE_MODEL_PARAMS['ExtraTrees']
    p_gb = BASE_MODEL_PARAMS['GradientBoosting']
    return {
        'RandomForest': RandomForestClassifier(random_state=seed, **p_rf),
        'ExtraTrees': ExtraTreesClassifier(random_state=seed, **p_et),
        'GradientBoosting': GradientBoostingClassifier(random_state=seed, **p_gb),
    }


def create_frozen_rf(seed):
    """Auxiliary baseline: Single Frozen Random Forest trained on initial 20,000 samples."""
    p = BASE_MODEL_PARAMS['RandomForest']
    return RandomForestClassifier(random_state=seed, **p)
