"""
Preprocessing and Feature Extraction Pipeline for Exp 9B
Enforces strict anti-leakage feature isolation and standard scaling fitted ONLY on available historical data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'mean_latency',
    'median_latency',
    'std_latency',
    'p90_latency',
    'p95_latency',
    'max_latency',
    'packet_loss_rate',
    'delivery_rate',
    'mean_interarrival_time',
    'std_interarrival_time',
    'packet_count',
    'effective_throughput'
]

class StreamingPreprocessor:
    """
    StandardScaler wrapper ensuring zero look-ahead bias.
    Scaler parameters (mean, std) are fitted ONLY on initial training prefix.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_initial(self, df_init):
        """Fit scaler on initial historical training prefix."""
        X_raw = df_init[FEATURE_COLS].values
        self.scaler.fit(X_raw)
        self.is_fitted = True
        return self.scaler.transform(X_raw)

    def transform(self, df_win):
        """Transform window telemetry features using frozen initial scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scaler is not fitted.")
        X_raw = df_win[FEATURE_COLS].values
        return self.scaler.transform(X_raw)

def get_feature_names():
    """Return explicit list of predictive feature column names."""
    return list(FEATURE_COLS)
