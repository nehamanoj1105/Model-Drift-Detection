"""
================================================================================
EXPERIMENT 3 — MODEL DEFINITIONS & ARCHITECTURES
================================================================================
Defines base candidate models and baseline models:
1. Random Forest (RF): 50 trees, max_depth=7
2. Extra Trees (ET): 50 trees, max_depth=7
3. Gradient Boosting (GB): 50 trees, max_depth=4
No SGDClassifier or other external architectures.
================================================================================
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)


def create_frozen_rf(seed):
    """Strategy 1 Baseline: Frozen Random Forest trained once on initial data."""
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        random_state=seed,
        n_jobs=-1
    )


def create_retrained_rf(seed):
    """Strategy 2 Baseline: Continuously Retrained Random Forest."""
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        random_state=seed,
        n_jobs=-1
    )


def create_base_models(seed):
    """
    Instantiate the three base candidate models for Experiment 3:
    1. RandomForest
    2. ExtraTrees
    3. GradientBoosting
    """
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=50,
            max_depth=7,
            random_state=seed,
            n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=50,
            max_depth=7,
            random_state=seed,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=seed
        )
    }


def create_candidate_models(seed):
    """Alias for create_base_models for backwards compatibility with tests."""
    return create_base_models(seed)


class StaticEnsemble:
    """
    Static Heterogeneous Ensemble baseline from Experiment 2.
    Averages predicted probabilities of RF, ET, and GB without online adaptation.
    """
    def __init__(self, seed):
        self.seed = seed
        self.models = create_base_models(seed)
        self.model_names = ['RandomForest', 'ExtraTrees', 'GradientBoosting']

    def fit(self, X, y):
        for name in self.model_names:
            self.models[name].fit(X, y)
        return self

    def predict_proba(self, X):
        probs = [self.models[name].predict_proba(X)[:, 1] for name in self.model_names]
        avg_p = np.mean(probs, axis=0)
        return np.column_stack([1.0 - avg_p, avg_p])

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)
