"""
EXPERIMENT 4 — EVALUATION METRICS
"""
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score
)

def compute_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = acc
    roc_auc = 0.5
    pr_auc = 0.0
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            else:
                roc_auc = 0.5
        except Exception:
            roc_auc = 0.5
        try:
            if len(np.unique(y_true)) > 1:
                p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = float(auc(r_curve, p_curve))
            else:
                pr_auc = float(np.mean(y_true))
        except Exception:
            pr_auc = 0.0
    return {
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'balanced_accuracy': bal_acc
    }


def extract_drift_episodes(drift_schedule):
    """
    Extract contiguous drift episodes: runs of consecutive windows sharing
    the same drift_type != 'none'.
    Returns list of dicts: {'drift_type', 'start_window', 'end_window', 'windows', 'length'}.
    """
    episodes = []
    cur_ep = None

    for w, item in enumerate(drift_schedule):
        dtype = item.get('drift_type', 'none')
        if dtype != 'none':
            if cur_ep is None or cur_ep['drift_type'] != dtype:
                if cur_ep is not None:
                    episodes.append(cur_ep)
                cur_ep = {
                    'drift_type': dtype,
                    'start_window': w,
                    'end_window': w,
                    'windows': [w],
                    'length': 1
                }
            else:
                cur_ep['end_window'] = w
                cur_ep['windows'].append(w)
                cur_ep['length'] += 1
        else:
            if cur_ep is not None:
                episodes.append(cur_ep)
                cur_ep = None

    if cur_ep is not None:
        episodes.append(cur_ep)

    return episodes


def compute_detector_quality(drift_schedule, detector_fired_windows, detector_name, seed):
    """
    Compute detector precision, recall, F1, and episode-level detection latency.
    - Ground truth: drift_type != 'none'
    - Predictions: detector_fired_windows (list of bool for each window)
    """
    n_windows = min(len(drift_schedule), len(detector_fired_windows))
    drift_schedule = drift_schedule[:n_windows]
    y_true = [1 if drift_schedule[w].get('drift_type', 'none') != 'none' else 0 for w in range(n_windows)]
    y_pred = [1 if bool(detector_fired_windows[w]) else 0 for w in range(n_windows)]


    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Contiguous drift episodes
    episodes = extract_drift_episodes(drift_schedule)
    episode_latencies = []
    episode_records = []

    for ep_idx, ep in enumerate(episodes):
        fired_in_ep = [w for w in ep['windows'] if detector_fired_windows[w]]
        if fired_in_ep:
            latency = fired_in_ep[0] - ep['start_window']
            detected = True
        else:
            latency = ep['length']  # Penalty latency = episode duration
            detected = False

        episode_latencies.append(latency)
        episode_records.append({
            'seed': seed,
            'detector': detector_name,
            'episode_idx': ep_idx,
            'drift_type': ep['drift_type'],
            'start_window': ep['start_window'],
            'end_window': ep['end_window'],
            'length': ep['length'],
            'detected': detected,
            'latency': latency,
        })

    detected_only = [rec['latency'] for rec in episode_records if rec['detected']]
    mean_lat = float(np.mean(detected_only)) if detected_only else float('nan')
    std_lat = float(np.std(detected_only)) if len(detected_only) > 1 else 0.0
    median_lat = float(np.median(detected_only)) if detected_only else float('nan')
    episode_detection_rate = float(len(detected_only) / len(episodes)) if episodes else 0.0
    window_trigger_rate = float((tp + fp) / n_windows) if n_windows > 0 else 0.0

    summary = {
        'seed': seed,
        'detector': detector_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'mean_latency': mean_lat,
        'std_latency': std_lat,
        'median_latency': median_lat,
        'window_trigger_rate': window_trigger_rate,
        'episode_detection_rate': episode_detection_rate,
        'detection_rate': episode_detection_rate,  # backward compatibility alias
        'total_episodes': len(episodes),
        'detected_episodes': len(detected_only),
    }

    return summary, episode_records


