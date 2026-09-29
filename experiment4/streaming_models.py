"""
================================================================================
EXPERIMENT 4 — STREAMING RIVER MODELS & WRAPPERS
================================================================================
Implements streaming classifier wrappers for:
  1. Adaptive Random Forest (ARFClassifier):
     Continuous per-sample updates with localized embedded drift detectors.
  2. Streaming Random Patches (SRPClassifier):
     Streaming bagging + random subspace ensemble with drift tracking.
  3. Online Learner Wrapper (HoeffdingAdaptiveTreeClassifier / LogisticRegression):
     Ultra-low-latency learner used for the Two-Tier Hybrid Ensemble.
================================================================================
"""

import pickle
import numpy as np
from config import KPI_COLS, ARF_N_MODELS, SRP_N_MODELS, DECISION_THRESHOLD


class RiverStreamingEnsembleWrapper:
    """
    Uniform wrapper around River's streaming ensemble classifiers (ARF, SRP).
    Operates strictly on per-sample streams:
      - predict(X): prequential test step
      - learn(X, y): prequential train step
    """

    def __init__(self, model_type='ARF', seed=42, threshold=DECISION_THRESHOLD, feature_names=None):
        self.model_type = model_type.upper()
        self.seed = seed
        self.threshold = threshold
        self.feature_names = feature_names or list(KPI_COLS)
        self.classifier = self._init_classifier()
        self.samples_learned = 0

    def _init_classifier(self):
        if 'ARF' in self.model_type:
            from river.forest import ARFClassifier
            return ARFClassifier(n_models=ARF_N_MODELS, seed=self.seed)
        elif 'SRP' in self.model_type:
            from river.ensemble import SRPClassifier
            return SRPClassifier(n_models=SRP_N_MODELS, seed=self.seed)
        else:
            raise ValueError(f"Unknown streaming ensemble model type: {self.model_type}")

    def _row_to_dict(self, row):
        return {self.feature_names[i]: float(row[i]) for i in range(len(self.feature_names))}

    def fit_initial(self, X_train, y_train):
        """Warm up streaming ensemble on reference training data (or slice)."""
        # To avoid excessive warm-up CPU, warm up on up to 5,000 samples
        warm_slice_len = min(len(X_train), 5000)
        X_sub = X_train[-warm_slice_len:]
        y_sub = y_train[-warm_slice_len:]

        for i in range(len(X_sub)):
            x_dict = self._row_to_dict(X_sub[i])
            self.classifier.learn_one(x_dict, int(y_sub[i]))
        self.samples_learned += len(X_sub)

    def predict(self, X):
        """Predict binary labels and positive class probabilities for batch X."""
        probs = []
        for i in range(len(X)):
            x_dict = self._row_to_dict(X[i])
            p_dict = self.classifier.predict_proba_one(x_dict)
            p_pos = p_dict.get(1, p_dict.get(True, 0.5)) if p_dict else 0.5
            probs.append(float(np.clip(p_pos, 1e-5, 1.0 - 1e-5)))

        probs = np.array(probs, dtype=float)
        preds = (probs >= self.threshold).astype(int)
        return preds, probs

    def learn(self, X, y):
        """Update streaming ensemble on batch X, y sample by sample."""
        for i in range(len(X)):
            x_dict = self._row_to_dict(X[i])
            self.classifier.learn_one(x_dict, int(y[i]))
        self.samples_learned += len(X)
        return len(X)


class OnlineLearnerWrapper:
    """
    Ultra-lightweight streaming learner for the Two-Tier Hybrid Ensemble.
    Uses HoeffdingAdaptiveTreeClassifier or LogisticRegression pipeline.
    """

    def __init__(self, learner_type='HAT', seed=42, feature_names=None):
        self.learner_type = learner_type.upper()
        self.seed = seed
        self.feature_names = feature_names or list(KPI_COLS)
        self.learner = self._init_learner()

    def _init_learner(self):
        if 'HAT' in self.learner_type:
            from river.tree import HoeffdingAdaptiveTreeClassifier
            return HoeffdingAdaptiveTreeClassifier(seed=self.seed)
        elif 'LR' in self.learner_type:
            from river.linear_model import LogisticRegression
            from river.preprocessing import StandardScaler
            from river import compose
            return compose.Pipeline(StandardScaler(), LogisticRegression())
        else:
            from river.tree import HoeffdingAdaptiveTreeClassifier
            return HoeffdingAdaptiveTreeClassifier(seed=self.seed)

    def _row_to_dict(self, row):
        return {self.feature_names[i]: float(row[i]) for i in range(len(self.feature_names))}

    def fit_initial(self, X_train, y_train):
        """Fit on initial reference samples."""
        warm_slice_len = min(len(X_train), 5000)
        X_sub = X_train[-warm_slice_len:]
        y_sub = y_train[-warm_slice_len:]

        for i in range(len(X_sub)):
            x_dict = self._row_to_dict(X_sub[i])
            self.learner.learn_one(x_dict, int(y_sub[i]))

    def predict_proba(self, X):
        """Return positive class probabilities for batch X."""
        probs = []
        for i in range(len(X)):
            x_dict = self._row_to_dict(X[i])
            p_dict = self.learner.predict_proba_one(x_dict)
            p_pos = p_dict.get(1, p_dict.get(True, 0.5)) if p_dict else 0.5
            probs.append(float(np.clip(p_pos, 1e-5, 1.0 - 1e-5)))
        return np.array(probs, dtype=float)

    def learn(self, X, y):
        """Update learner incrementally per sample."""
        for i in range(len(X)):
            x_dict = self._row_to_dict(X[i])
            self.learner.learn_one(x_dict, int(y[i]))
