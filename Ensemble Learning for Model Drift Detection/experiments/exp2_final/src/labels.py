"""
src/labels.py - Fast vectorized paired bootstrap label assignment.

For each (decision, candidate) pair:
  Paired bootstrap (200 resamples) of dF1 = macroF1(c) - macroF1(Local)
  over windows in I_t = [t, t+K).
  
  Positive if 90% lower bound > 0 (lb > 0 at eps=0)
  Negative if 90% upper bound < 0 (ub < 0)
  Else neutral.
  
  Sensitivity: fixed margin eps in {0, 0.005, 0.01, 0.02}
    With eps: positive if lb > eps, negative if ub < -eps.
    Default eps = 0.01.

OPTIMIZATION: Fully vectorized matrix operations over numpy arrays.
"""

import numpy as np
from sklearn.metrics import f1_score
from typing import Tuple, List, Dict


def _fast_macro_f1_bootstrapped(
    y_true: np.ndarray,
    y_cand: np.ndarray,
    y_local: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """
    Vectorized paired bootstrap dF1 calculation.
    Returns array of shape (n_boot,) containing dF1 for each resample.
    """
    n = len(y_true)
    if n == 0:
        return np.zeros(n_boot)
    
    rng = np.random.default_rng(seed)
    
    # Generate all bootstrap resample index matrix at once: shape (n_boot, n)
    idx_mat = rng.integers(0, n, size=(n_boot, n))
    
    y_t_b = y_true[idx_mat]   # shape (n_boot, n)
    y_c_b = y_cand[idx_mat]   # shape (n_boot, n)
    y_l_b = y_local[idx_mat]  # shape (n_boot, n)
    
    # Get unique classes present in y_true
    classes = np.unique(y_true)
    if len(classes) < 2:
        return np.zeros(n_boot)
    
    # Vectorized F1 computation per class across all n_boot rows simultaneously
    f1_cand_per_class = []
    f1_local_per_class = []
    
    for c in classes:
        # Candidate
        tp_c = np.sum((y_t_b == c) & (y_c_b == c), axis=1)
        fp_c = np.sum((y_t_b != c) & (y_c_b == c), axis=1)
        fn_c = np.sum((y_t_b == c) & (y_c_b != c), axis=1)
        denom_c = 2 * tp_c + fp_c + fn_c
        f1_c = np.where(denom_c > 0, (2.0 * tp_c) / np.maximum(denom_c, 1e-12), 0.0)
        f1_cand_per_class.append(f1_c)
        
        # Local
        tp_l = np.sum((y_t_b == c) & (y_l_b == c), axis=1)
        fp_l = np.sum((y_t_b != c) & (y_l_b == c), axis=1)
        fn_l = np.sum((y_t_b == c) & (y_l_b != c), axis=1)
        denom_l = 2 * tp_l + fp_l + fn_l
        f1_l = np.where(denom_l > 0, (2.0 * tp_l) / np.maximum(denom_l, 1e-12), 0.0)
        f1_local_per_class.append(f1_l)
        
    macro_f1_cand = np.mean(f1_cand_per_class, axis=0)  # shape (n_boot,)
    macro_f1_local = np.mean(f1_local_per_class, axis=0)  # shape (n_boot,)
    
    return macro_f1_cand - macro_f1_local


def assign_label(
    y_true: np.ndarray,
    y_pred_cand: np.ndarray,
    y_pred_local: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.10,
    eps: float = 0.01,
    seed: int = 0,
) -> Dict:
    """
    Assign transfer label for a (decision, candidate) pair.
    
    eps=0: positive if lb > 0, negative if ub < 0.
    eps>0: positive if lb > eps, negative if ub < -eps.
    
    Returns dict with label, lb, ub, mean_df1, direct_df1.
    """
    n = len(y_true)
    if n == 0:
        return {'label': 'neutral', 'lb': 0.0, 'ub': 0.0, 'mean_df1': 0.0, 'direct_df1': 0.0, 'eps': eps}
    
    classes = np.unique(y_true)
    if len(classes) < 2:
        return {'label': 'neutral', 'lb': 0.0, 'ub': 0.0, 'mean_df1': 0.0, 'direct_df1': 0.0, 'eps': eps}
    
    f1_cand_direct = float(f1_score(y_true, y_pred_cand, average='macro', zero_division=0))
    f1_local_direct = float(f1_score(y_true, y_pred_local, average='macro', zero_division=0))
    direct_df1 = f1_cand_direct - f1_local_direct
    
    boot_df1s = _fast_macro_f1_bootstrapped(
        y_true, y_pred_cand, y_pred_local, n_boot=n_boot, seed=seed
    )
    
    lb = float(np.percentile(boot_df1s, alpha * 100 / 2))
    ub = float(np.percentile(boot_df1s, 100 - alpha * 100 / 2))
    mean_df1 = float(np.mean(boot_df1s))
    
    if lb > eps:
        label = 'positive'
    elif ub < -eps:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'label': label,
        'lb': lb,
        'ub': ub,
        'mean_df1': mean_df1,
        'direct_df1': direct_df1,
        'eps': eps,
        'n_boot': n_boot,
    }


def split_half_agreement(
    labels_odd: List[str],
    labels_even: List[str],
) -> Tuple[float, float]:
    """Compute raw agreement and Cohen's kappa between odd and even half labels."""
    assert len(labels_odd) == len(labels_even)
    n = len(labels_odd)
    if n == 0:
        return 0.0, 0.0
    
    match = sum(1 for a, b in zip(labels_odd, labels_even) if a == b)
    p_obs = match / n
    
    unique_labels = list(set(labels_odd + labels_even))
    p_e = 0.0
    for lbl in unique_labels:
        p_a = labels_odd.count(lbl) / n
        p_b = labels_even.count(lbl) / n
        p_e += p_a * p_b
    
    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_obs - p_e) / (1.0 - p_e)
    
    return float(p_obs), float(kappa)


def assign_labels_batch(
    y_true: np.ndarray,          # shape (K,)
    Y_cands: np.ndarray,         # shape (n_cands, K)
    y_local: np.ndarray,         # shape (K,)
    n_boot: int = 200,
    alpha: float = 0.10,
    eps: float = 0.01,
    seed: int = 0,
) -> List[Dict]:
    """
    Compute paired bootstrap dF1 for all n_cands candidates simultaneously in vectorized numpy.
    Returns list of dicts with label, lb, ub, mean_df1, direct_df1 for each candidate.
    """
    n_cands, K = Y_cands.shape
    if K == 0 or n_cands == 0:
        return [{'label': 'neutral', 'lb': 0.0, 'ub': 0.0, 'mean_df1': 0.0, 'direct_df1': 0.0, 'eps': eps} for _ in range(n_cands)]
    
    classes = np.unique(y_true)
    if len(classes) < 2:
        return [{'label': 'neutral', 'lb': 0.0, 'ub': 0.0, 'mean_df1': 0.0, 'direct_df1': 0.0, 'eps': eps} for _ in range(n_cands)]
    
    f1_local_direct = float(f1_score(y_true, y_local, average='macro', zero_division=0))
    direct_df1s = []
    for c_i in range(n_cands):
        f1_c = float(f1_score(y_true, Y_cands[c_i], average='macro', zero_division=0))
        direct_df1s.append(f1_c - f1_local_direct)
    
    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, K, size=(n_boot, K))
    
    y_t_b = y_true[idx_mat]
    y_l_b = y_local[idx_mat]
    Y_cands_b = Y_cands[:, idx_mat]
    
    f1_local_boot_list = []
    for c in classes:
        tp_l = np.sum((y_t_b == c) & (y_l_b == c), axis=1)
        fp_l = np.sum((y_t_b != c) & (y_l_b == c), axis=1)
        fn_l = np.sum((y_t_b == c) & (y_l_b != c), axis=1)
        denom_l = 2 * tp_l + fp_l + fn_l
        f1_l = np.where(denom_l > 0, (2.0 * tp_l) / np.maximum(denom_l, 1e-12), 0.0)
        f1_local_boot_list.append(f1_l)
    macro_f1_local_boot = np.mean(f1_local_boot_list, axis=0)
    
    results = []
    for c_i in range(n_cands):
        y_c_b = Y_cands_b[c_i]
        f1_cand_boot_list = []
        for c in classes:
            tp_c = np.sum((y_t_b == c) & (y_c_b == c), axis=1)
            fp_c = np.sum((y_t_b != c) & (y_c_b == c), axis=1)
            fn_c = np.sum((y_t_b == c) & (y_c_b != c), axis=1)
            denom_c = 2 * tp_c + fp_c + fn_c
            f1_c = np.where(denom_c > 0, (2.0 * tp_c) / np.maximum(denom_c, 1e-12), 0.0)
            f1_cand_boot_list.append(f1_c)
        macro_f1_cand_boot = np.mean(f1_cand_boot_list, axis=0)
        
        boot_df1s = macro_f1_cand_boot - macro_f1_local_boot
        
        lb = float(np.percentile(boot_df1s, alpha * 100 / 2))
        ub = float(np.percentile(boot_df1s, 100 - alpha * 100 / 2))
        mean_df1 = float(np.mean(boot_df1s))
        
        if lb > eps:
            label = 'positive'
        elif ub < -eps:
            label = 'negative'
        else:
            label = 'neutral'
            
        results.append({
            'label': label,
            'lb': lb,
            'ub': ub,
            'mean_df1': mean_df1,
            'direct_df1': direct_df1s[c_i],
            'eps': eps,
            'n_boot': n_boot,
        })
        
    return results


def macro_f1_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 from arrays of true labels and predictions."""
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


