"""Pooled and per-window metrics for the revalidation.

Per-window metrics (the paper's current metric) are computed by
`window_metrics` in the existing evaluation_9a.py and are kept here for
comparison. Pooled metrics concatenate y_true / y_pred over ALL evaluation
windows of a (dataset, method, seed) run before scoring, which is the standard
way to report macro-F1 on an imbalanced stream.

Also returns the full confusion matrix and one-vs-rest TP/FP/FN/TN so nothing
has to be reconstructed from accuracy + macro precision/recall again.
"""
import json

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix, average_precision_score,
)


def _per_class_prf(y_true, y_pred, labels):
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return p, r, f


def pooled_metrics(y_true, y_pred, n_classes):
    """All pooled metrics for one run."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    p, r, f = _per_class_prf(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # one-vs-rest TP/FP/FN/TN summed over classes (macro view)
    tp = int(np.trace(cm))
    total = int(cm.sum())
    fp = int(cm.sum(axis=0).sum() - tp)
    fn = int(cm.sum(axis=1).sum() - tp)
    tn = int(total - tp - fp - fn)
    out = {
        "pooled_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "pooled_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pooled_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cm_json": json.dumps(cm.tolist()),
        "ovr_tp": tp, "ovr_fp": fp, "ovr_fn": fn, "ovr_tn": tn,
        "n_eval_samples": total,
    }
    for c in labels:
        out[f"class{c}_precision"] = float(p[c])
        out[f"class{c}_recall"] = float(r[c])
        out[f"class{c}_f1"] = float(f[c])
    return out


def pooled_binary_attack_metrics(y_true, y_pred, attack_label=1, proba=None,
                                 attack_free_mask=None):
    """Attack-class metrics for the binary UGR'16 stream.

    `attack_free_mask` marks evaluation samples that lie in windows where the
    true window label is attack-free; false positives inside those windows are
    reported separately.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    a = int(attack_label)
    tp = int(np.sum((y_true == a) & (y_pred == a)))
    fp = int(np.sum((y_true != a) & (y_pred == a)))
    fn = int(np.sum((y_true == a) & (y_pred != a)))
    tn = int(np.sum((y_true != a) & (y_pred != a)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "attack_tp": tp, "attack_fp": fp, "attack_fn": fn, "attack_tn": tn,
        "attack_precision": prec, "attack_recall": rec, "attack_f1": f1,
    }
    n_af = int(np.sum(y_true != a))
    out["attack_free_minutes"] = n_af
    out["fp_per_1000_attack_free"] = (1000.0 * fp / n_af) if n_af else 0.0
    if attack_free_mask is not None:
        mask = np.asarray(attack_free_mask)
        out["fp_in_attack_free_windows"] = int(np.sum((y_pred == a) & mask))
    if proba is not None:
        proba = np.asarray(proba)
        out["pr_auc_attack"] = float(average_precision_score((y_true == a).astype(int), proba))
    return out


def window_confusion(y_true, y_pred, n_classes):
    """Per-window confusion counts for the raw CSV."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    tp = int(np.trace(cm))
    total = int(cm.sum())
    fp = int(cm.sum(axis=0).sum() - tp)
    fn = int(cm.sum(axis=1).sum() - tp)
    tn = int(total - tp - fp - fn)
    return {"cm_json": json.dumps(cm.tolist()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_samples": total}
