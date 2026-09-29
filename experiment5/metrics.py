"""
================================================================================
EXPERIMENT 5 — PERFORMANCE METRICS & RECOVERY EVALUATOR
================================================================================
Computes comprehensive binary classification, ranking, and recovery metrics:
  - F1, Macro-F1, Accuracy, Precision, Recall, Balanced Accuracy
  - ROC-AUC, PR-AUC
  - Confusion Matrix (TN, FP, FN, TP)
================================================================================
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute full suite of classification and diagnostic metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion matrix elements
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    else:
        # Edge case: single class in evaluation chunk
        if y_true[0] == 0:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            fn, tp = 0, 0
        else:
            fn = int(np.sum(y_pred == 0))
            tp = int(np.sum(y_pred == 1))
            tn, fp = 0, 0

    # ROC AUC & PR AUC
    roc_auc = 0.5
    pr_auc = float(np.mean(y_true))
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                pr_auc = float(average_precision_score(y_true, y_prob))
            else:
                roc_auc = 0.5
        except Exception:
            pass

    return {
        'f1': f1,
        'macro_f1': macro_f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'balanced_accuracy': bal_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }
