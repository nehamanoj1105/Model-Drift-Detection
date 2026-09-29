"""
Similarity calculation and Softmax Similarity-Weighted candidate policy blending.
"""

import numpy as np

def compute_softmax_similarity_weights(sim_scores: dict, temperature: float = 1.0) -> dict:
    """
    Computes softmax normalized weights over candidate similarity scores.
    
    Args:
        sim_scores: dict mapping candidate_id -> similarity_score (e.g. 1.0 - Wasserstein/MMD distance)
        temperature: softmax temperature scaling factor
        
    Returns:
        dict mapping candidate_id -> weight (summing to 1.0)
    """
    if not sim_scores:
        return {}
    keys = list(sim_scores.keys())
    scores = np.array([sim_scores[k] for k in keys], dtype=np.float64)
    # Numerical stability shift
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    weights = exp_scores / np.sum(exp_scores)
    return {k: float(w) for k, w in zip(keys, weights)}

def predict_softmax_weighted(candidates: list, sim_scores: dict, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Predicts by soft-weighting probability outputs of candidate models according to softmax similarity weights.
    """
    weights = compute_softmax_similarity_weights(sim_scores, temperature=temperature)
    n_samples = len(X)
    accum_probs = None
    
    for idx_cand, cand in enumerate(candidates):
        r_id = getattr(cand, 'regime_id', idx_cand)
        w = weights.get(r_id, weights.get(idx_cand, 0.0))
        p_cand = cand.predict_proba(X) if hasattr(cand, 'predict_proba') else None
        if p_cand is None:
            # Fallback if hard predictions only
            preds = cand.predict(X)
            p_cand = np.zeros((n_samples, 2))
            for i, val in enumerate(preds):
                p_cand[i, int(val)] = 1.0
                
        if accum_probs is None:
            accum_probs = w * p_cand
        else:
            accum_probs += w * p_cand
            
    return np.argmax(accum_probs, axis=1)
