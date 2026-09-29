"""
Base Models Module for Exp 9 — Heterogeneous Ensemble
Random Forest + Extra Trees classifiers with fixed hyperparameter configurations.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class HeterogeneousEnsemble:
    """
    Lightweight heterogeneous ensemble combining RandomForest and ExtraTrees.
    Supports soft-voting probability predictions and individual model weight updates.
    """
    def __init__(self, seed=42, n_estimators=50, max_depth=7, weights=None):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.weights = weights if weights is not None else [0.5, 0.5]
        
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=1
        )
        self.et = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=1
        )
        self.is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        """Fit both base classifiers on training data."""
        self.classes_ = np.unique(y)
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Compute weighted soft-voting probability distribution over classes."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted.")
            
        p_rf = self.rf.predict_proba(X)
        p_et = self.et.predict_proba(X)
        
        # Ensure class shape alignment
        w1, w2 = self.weights[0], self.weights[1]
        sum_w = w1 + w2
        if sum_w <= 0:
            w1, w2 = 0.5, 0.5
            sum_w = 1.0
        w1_norm = w1 / sum_w
        w2_norm = w2 / sum_w
        
        probs = w1_norm * p_rf + w2_norm * p_et
        return probs

    def predict(self, X):
        """Predict class labels with highest soft-voting probability."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_num_trees(self):
        """Return total number of decision trees in the ensemble."""
        return len(self.rf.estimators_) + len(self.et.estimators_)

def create_base_ensemble(seed=42, n_estimators=50, max_depth=7):
    """Factory helper to instantiate a clean base ensemble."""
    return HeterogeneousEnsemble(seed=seed, n_estimators=n_estimators, max_depth=max_depth)
