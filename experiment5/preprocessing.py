"""
================================================================================
EXPERIMENT 5 — LEAKAGE-FREE PREPROCESSING PIPELINE
================================================================================
Implements rigorous feature transformations for ToN_IoT telemetry:
  - StandardScaler fitted STRICTLY on the initial 20,000 training samples.
  - Consistent transformation applied to streaming windows without refitting.
  - Complete protection against future data leakage.
================================================================================
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


class Preprocessor:
    """
    Standardizes features using statistics computed strictly from the initial
    stationary training partition (samples 0 to 19,999).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_means = None
        self.feature_scales = None
        self.n_samples_seen = 0

    def fit(self, X_train):
        """
        Fit scaler parameters exclusively on initial training samples.
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
        self.feature_means = np.copy(self.scaler.mean_)
        self.feature_scales = np.copy(self.scaler.scale_)
        self.n_samples_seen = len(X_train)
        return self

    def transform(self, X):
        """
        Transform any sample/window using the frozen training statistics.
        Does NOT update internal statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training partition before transform.")
        return self.scaler.transform(X)

    def fit_transform(self, X_train):
        """Fit on training data and return transformed training array."""
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, filepath):
        """Serialize preprocessor to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_means': self.feature_means,
            'feature_scales': self.feature_scales,
            'n_samples_seen': self.n_samples_seen
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """Load serialized preprocessor."""
        data = joblib.load(filepath)
        inst = cls()
        inst.scaler = data['scaler']
        inst.is_fitted = data['is_fitted']
        inst.feature_means = data['feature_means']
        inst.feature_scales = data['feature_scales']
        inst.n_samples_seen = data['n_samples_seen']
        return inst
