"""
================================================================================
EXPERIMENT 3 — PREQUENTIAL EVALUATION MODULE
================================================================================
Implements strict Test-Then-Train protocol:
1. Model states are fixed before evaluating on current streaming window.
2. Out-of-sample predictions and metrics calculated.
3. Rewards calculated without leaking future labels.
4. UCB1 bandit statistics updated.
5. Next model chosen.
6. Retraining/adaptation allowed only after predictions/rewards are computed.
================================================================================
"""

import numpy as np
from metrics import evaluate_predictions, evaluate_model
from resource_monitor import measure_execution


def get_model_positive_proba(model, X):
    """Safely obtain positive-class probability P(Y=1|X)."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        if probs.shape[1] > 1:
            return probs[:, 1]
        else:
            return probs[:, 0]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        # Apply sigmoid
        return 1.0 / (1.0 + np.exp(-scores))
    else:
        return model.predict(X).astype(float)


def evaluate_window_prequential(model, X_window, y_window):
    """
    Evaluate a model on a streaming window under strict out-of-sample test-then-train protocol.
    Returns:
    - metrics: dict of accuracy, precision, recall, f1, auc
    - y_pred: binary predictions
    - y_prob: predicted positive probabilities
    - res_profile: psutil and execution timing profile
    """
    def _infer():
        y_prob = get_model_positive_proba(model, X_window)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred, y_prob

    (y_pred, y_prob), res_profile = measure_execution(_infer)
    metrics = evaluate_predictions(y_window, y_pred, y_prob)
    return metrics, y_pred, y_prob, res_profile
