"""
Base Models Module for the REVISED Experiment 9A.

Identical heterogeneous ensemble architecture to the existing 9A/9B harness
(RandomForest + ExtraTrees, 50 trees each, max_depth=7, n_jobs=1, soft voting).
The only addition is a class-anchor guard used uniformly by every adaptive
model in this experiment: because the revised 9A stream is a per-sample,
6-class stream, a small stratified slice of the initial training prefix is
always appended to any retraining buffer so that every model can represent all
classes. This guard is applied identically to all adaptive methods and is
documented in the report; it prevents degenerate single-class refits without
favouring any model.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


class HeterogeneousEnsemble:
    """Lightweight heterogeneous ensemble (RandomForest + ExtraTrees, soft voting)."""

    def __init__(self, seed=42, n_estimators=50, max_depth=7, weights=None):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.weights = weights if weights is not None else [0.5, 0.5]

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.seed, n_jobs=1,
        )
        self.et = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.seed, n_jobs=1,
        )
        self.is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted.")
        p_rf = self.rf.predict_proba(X)
        p_et = self.et.predict_proba(X)

        w1, w2 = self.weights[0], self.weights[1]
        sum_w = w1 + w2
        if sum_w <= 0:
            w1, w2, sum_w = 0.5, 0.5, 1.0
        return (w1 / sum_w) * p_rf + (w2 / sum_w) * p_et

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_num_trees(self):
        return len(self.rf.estimators_) + len(self.et.estimators_)


def create_base_ensemble(seed=42, n_estimators=50, max_depth=7):
    return HeterogeneousEnsemble(seed=seed, n_estimators=n_estimators, max_depth=max_depth)


def make_class_anchor(X_init, y_init, per_class=15, seed=0):
    """Stratified slice of the initial training prefix (per_class samples/class).

    Returns (anchor_X, anchor_y) or (None, None) if the prefix is empty.
    """
    if X_init is None or len(X_init) == 0:
        return None, None
    X_init = np.asarray(X_init)
    y_init = np.asarray(y_init)
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y_init):
        c_idx = np.where(y_init == c)[0]
        take = min(per_class, len(c_idx))
        idx.extend(rng.choice(c_idx, size=take, replace=False).tolist())
    idx = np.array(sorted(idx))
    return X_init[idx], y_init[idx]


def make_train_buffer(recent_X, recent_y, anchor_X, anchor_y, cap):
    """Recent buffer (capped) concatenated with the fixed class anchor.

    Applied identically to every adaptive model.
    """
    recent_X = np.asarray(recent_X) if len(recent_X) else np.empty((0, 0))
    recent_y = np.asarray(recent_y) if len(recent_y) else np.empty((0,), dtype=int)
    if len(recent_X) > cap:
        recent_X = recent_X[-cap:]
        recent_y = recent_y[-cap:]
    if anchor_X is not None and len(anchor_X):
        if recent_X.size == 0:
            return anchor_X.copy(), anchor_y.copy()
        return np.vstack([recent_X, anchor_X]), np.concatenate([recent_y, anchor_y])
    return recent_X, recent_y
