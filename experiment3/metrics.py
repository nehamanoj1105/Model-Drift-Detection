"""
================================================================================
EXPERIMENT 3 — CLASSIFICATION & EVALUATION METRICS
================================================================================
Calculates classification performance metrics:
- F1 Score
- Accuracy
- Precision
- Recall
- ROC AUC
Handles edge cases (single-class windows, zero divisions) robustly.
================================================================================
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive classification metrics for binary QoS violation.
    y_true: true binary labels (0/1)
    y_pred: predicted binary labels (0/1)
    y_prob: predicted probabilities for positive class P(Y=1)
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    auc = 0.5
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_prob))
            else:
                auc = float(acc)
        except Exception:
            auc = 0.5
            
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }


def evaluate_model(model, X, y_true):
    """
    Run prediction and probability estimation on X, then compute all metrics.
    """
    y_pred = model.predict(X)
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if probs.shape[1] > 1:
                y_prob = probs[:, 1]
            else:
                y_prob = probs[:, 0]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X)
        else:
            y_prob = y_pred.astype(float)
    except Exception:
        y_prob = y_pred.astype(float)
        
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    return metrics, y_pred, y_prob
