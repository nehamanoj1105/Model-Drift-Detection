"""
================================================================================
EXPERIMENT 8 — MODEL DEFINITIONS & PARTIAL ADAPTATION SUPPORT
================================================================================
Defines base model architectures (RF, ET, GB) with support for:
  - FULL retraining on historical buffer
  - PARTIAL bounded retraining (e.g. restricted window samples or reduced estimators)
  - KEEP (no-op)
================================================================================
"""

import copy
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

BASE_MODEL_SPECS = {
    'RF': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 50, 'max_depth': 7, 'min_samples_split': 4, 'n_jobs': -1},
        'horizon': 3500,
    },
    'ET': {
        'class': ExtraTreesClassifier,
        'params': {'n_estimators': 50, 'max_depth': 7, 'min_samples_split': 6, 'n_jobs': -1},
        'horizon': 2500,
    },
    'GB': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.85},
        'horizon': 1800,
    },
}


def create_model_instance(model_key, seed):
    """Create a fresh model instance with the specified seed."""
    spec = BASE_MODEL_SPECS[model_key]
    cls = spec['class']
    params = copy.deepcopy(spec['params'])
    params['random_state'] = seed
    return cls(**params)


def create_ensemble_models(seed):
    """
    Returns a dictionary of component models:
      {'RF': RandomForestClassifier, 'ET': ExtraTreesClassifier, 'GB': GradientBoostingClassifier}
    """
    return {
        'RF': create_model_instance('RF', seed),
        'ET': create_model_instance('ET', seed),
        'GB': create_model_instance('GB', seed),
    }


def fit_model_with_action(model_key, current_model, action, buffer_X, buffer_y, seed, partial_sample_ratio=0.35):
    """
    Retrains a component model based on the requested action ('KEEP', 'PARTIAL', 'FULL').
    
    FAIRNESS GUARANTEES:
      - FULL retrains on full horizon buffer up to legal point t.
      - PARTIAL retrains on bounded recent subset of legal buffer (sample_ratio of horizon buffer)
        with bounded estimators (35% of n_estimators).
      - KEEP leaves current_model untouched.
      - Never accesses future data.
    """
    if action == 'KEEP':
        return current_model, 0, 0  # model, samples_used, estimators_used

    spec = BASE_MODEL_SPECS[model_key]
    horizon = spec['horizon']
    
    # Extract legal historical data up to horizon limit
    all_X = np.array(buffer_X)[-horizon:]
    all_y = np.array(buffer_y)[-horizon:]
    
    if action == 'FULL':
        # Check single-class fallback
        if len(np.unique(all_y)) < 2:
            return current_model, len(all_X), spec['params']['n_estimators']
        try:
            new_model = create_model_instance(model_key, seed)
            new_model.fit(all_X, all_y)
            n_samples = len(all_X)
            n_est = spec['params']['n_estimators']
            return new_model, n_samples, n_est
        except Exception:
            return current_model, len(all_X), spec['params']['n_estimators']

    elif action == 'PARTIAL':
        n_partial_samples = max(100, int(len(all_X) * partial_sample_ratio))
        sub_X = all_X[-n_partial_samples:]
        sub_y = all_y[-n_partial_samples:]
        
        # Expand if single-class in partial slice
        if len(np.unique(sub_y)) < 2 and len(all_y) > len(sub_y):
            sub_X = all_X
            sub_y = all_y
            n_partial_samples = len(all_X)

        if len(np.unique(sub_y)) < 2:
            return current_model, n_partial_samples, int(spec['params']['n_estimators'] * partial_sample_ratio)

        params = copy.deepcopy(spec['params'])
        params['random_state'] = seed
        params['n_estimators'] = max(10, int(params['n_estimators'] * partial_sample_ratio))
        
        try:
            cls = spec['class']
            new_model = cls(**params)
            new_model.fit(sub_X, sub_y)
            return new_model, n_partial_samples, params['n_estimators']
        except Exception:
            return current_model, n_partial_samples, params['n_estimators']

    else:
        raise ValueError(f"Unknown adaptation action: {action}")
