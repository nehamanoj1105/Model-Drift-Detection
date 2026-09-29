"""
Transfer Episode Constructor and Oracle for Experiment 2 Corrected.

Generates genuine transfer episodes at each regime transition with:
  - A fixed per-target baseline (same for all candidates)
  - delta_f1 = transfer_f1 - baseline_f1
  - Oracle = max(transfer_f1 across all candidates)
  - Oracle assertion: oracle_f1 >= similarity_only_f1

Label convention:
  positive : delta_f1 > +0.005
  neutral  : |delta_f1| <= 0.005
  negative : delta_f1 < -0.005
"""

import copy
import numpy as np
from sklearn.metrics import f1_score
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from checkpoint_manager import RegimeCheckpointExp2, CheckpointPool

DELTA_POSITIVE = +0.005
DELTA_NEGATIVE = -0.005

# Minimum windows of target data required before evaluating transfer outcomes
MIN_TARGET_EVAL_WINDOWS = 10


def evaluate_candidate_on_target(
    ckpt: 'RegimeCheckpointExp2',
    X_target: np.ndarray,
    y_target: np.ndarray,
) -> float:
    """
    Evaluate a candidate checkpoint on target-regime windows.

    The checkpoint model is used AS-IS (no additional training).
    This is the "cold transfer" performance.

    Returns F1 macro score.
    """
    if len(X_target) == 0:
        return 0.0
    try:
        y_pred = ckpt.predict(X_target)
        return float(f1_score(y_target, y_pred, average='macro', zero_division=0))
    except Exception:
        return 0.0


def compute_baseline_f1(
    baseline_ckpt: 'RegimeCheckpointExp2',
    X_target: np.ndarray,
    y_target: np.ndarray,
) -> float:
    """
    Compute fixed baseline F1 (Frozen checkpoint evaluated on target windows).
    Same baseline is used for ALL candidates at this target episode.
    """
    return evaluate_candidate_on_target(baseline_ckpt, X_target, y_target)


def build_transfer_episode(
    episode_id: int,
    source_regime: str,
    target_regime: str,
    candidates: list,          # list of RegimeCheckpointExp2
    X_target: np.ndarray,
    y_target: np.ndarray,
    baseline_ckpt: 'RegimeCheckpointExp2',
    similarity_scores: dict,   # regime_id -> float
) -> dict:
    """
    Build a full transfer episode record for one (source, target) transition.

    Parameters
    ----------
    episode_id : int
    source_regime, target_regime : str
    candidates : list of checkpoints available before this episode
    X_target, y_target : target regime windows (pre-transition windows, for evaluation)
    baseline_ckpt : frozen checkpoint used as performance baseline
    similarity_scores : regime_id -> float (pre-computed similarity)

    Returns
    -------
    dict with per-candidate records and oracle summary
    """
    if len(X_target) < MIN_TARGET_EVAL_WINDOWS:
        return None   # Not enough target data to evaluate transfers yet

    baseline_f1 = compute_baseline_f1(baseline_ckpt, X_target, y_target)

    candidate_records = []
    for ckpt in candidates:
        transfer_f1 = evaluate_candidate_on_target(ckpt, X_target, y_target)
        delta_f1 = transfer_f1 - baseline_f1
        if delta_f1 > DELTA_POSITIVE:
            label = 'positive'
        elif delta_f1 < DELTA_NEGATIVE:
            label = 'negative'
        else:
            label = 'neutral'

        sim = similarity_scores.get(ckpt.regime_id, 0.0)
        candidate_records.append({
            'episode_id': episode_id,
            'source_regime': ckpt.regime_id,
            'target_regime': target_regime,
            'baseline_f1': baseline_f1,
            'transfer_f1': transfer_f1,
            'delta_f1': delta_f1,
            'transfer_label': label,
            'similarity': sim,
            'source_historical_f1': ckpt.historical_f1,
            'source_age': episode_id - ckpt.creation_episode,
            'source_reuse_count': ckpt.reuse_count,
            'source_transfer_success_rate': ckpt.transfer_success_rate,
            'source_negative_transfer_rate': ckpt.negative_transfer_rate,
            'source_training_sample_count': ckpt.training_sample_count,
            'n_target_eval_windows': len(X_target),
        })

    if not candidate_records:
        return None

    # Oracle: best actual candidate
    oracle_f1 = max(r['transfer_f1'] for r in candidate_records)
    oracle_delta = oracle_f1 - baseline_f1

    # Similarity-only: candidate with highest similarity
    best_sim_regime = max(similarity_scores, key=similarity_scores.get) if similarity_scores else None
    sim_only_record = next(
        (r for r in candidate_records if r['source_regime'] == best_sim_regime),
        candidate_records[0]
    )
    sim_only_f1 = sim_only_record['transfer_f1']

    # Oracle assertion: must be >= similarity-only
    if oracle_f1 < sim_only_f1 - 1e-8:
        raise AssertionError(
            f"Oracle F1 ({oracle_f1:.4f}) < Similarity-Only F1 ({sim_only_f1:.4f}) "
            f"at episode {episode_id}. Oracle implementation is incorrect."
        )

    return {
        'episode_id': episode_id,
        'source_regime': source_regime,   # transition-initiating regime
        'target_regime': target_regime,
        'n_candidates': len(candidate_records),
        'baseline_f1': baseline_f1,
        'oracle_f1': oracle_f1,
        'oracle_delta': oracle_delta,
        'oracle_label': 'positive' if oracle_delta > DELTA_POSITIVE else (
            'negative' if oracle_delta < DELTA_NEGATIVE else 'neutral'),
        'sim_only_f1': sim_only_f1,
        'candidates': candidate_records,
    }


def select_best_candidate_by_similarity(
    candidates: list,
    X_source_windows_dict: dict,
    X_target_windows: np.ndarray,
) -> tuple:
    """
    Similarity-Only selection: argmax similarity(source, target).

    Returns (best_ckpt, similarity_scores_dict)
    """
    from features import compute_distributional_similarity
    scores = {}
    for ckpt in candidates:
        X_src = X_source_windows_dict.get(ckpt.regime_id)
        if X_src is None or len(X_src) == 0:
            scores[ckpt.regime_id] = 0.0
            continue
        scores[ckpt.regime_id] = compute_distributional_similarity(X_src, X_target_windows)
    if not candidates:
        return None, scores
    best = max(candidates, key=lambda c: scores.get(c.regime_id, 0.0))
    return best, scores


def select_best_candidate_by_probability(
    candidates: list,
    prob_estimator,
    X_source_windows_dict: dict,
    X_target_windows: np.ndarray,
    current_episode: int,
) -> tuple:
    """
    Probability-Guided selection: argmax P(positive transfer | features).

    Returns (best_ckpt, best_probability, all_probabilities_dict)
    """
    from features import compute_source_target_features
    probs = {}
    feats = {}
    for ckpt in candidates:
        X_src = X_source_windows_dict.get(ckpt.regime_id)
        if X_src is None or len(X_src) == 0:
            probs[ckpt.regime_id] = 0.5
            feats[ckpt.regime_id] = None
            continue
        feat_vec = compute_source_target_features(
            source_ckpt=ckpt,
            X_source_windows=X_src,
            X_target_windows=X_target_windows,
            current_episode=current_episode,
        )
        p = prob_estimator.predict_proba_positive(feat_vec, current_episode)
        probs[ckpt.regime_id] = p
        feats[ckpt.regime_id] = feat_vec

    if not candidates:
        return None, 0.0, probs, feats
    best = max(candidates, key=lambda c: probs.get(c.regime_id, 0.0))
    best_prob = probs.get(best.regime_id, 0.0)
    return best, best_prob, probs, feats


def select_by_historical_reliability(candidates: list) -> 'RegimeCheckpointExp2':
    """Historical Reliability: select by highest historical transfer success rate."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.transfer_success_rate)


def select_random(candidates: list, rng: np.random.Generator) -> 'RegimeCheckpointExp2':
    """Random historical: select uniformly at random."""
    if not candidates:
        return None
    idx = rng.integers(0, len(candidates))
    return candidates[idx]


def check_transfer_label_distribution(all_candidate_records: list) -> dict:
    """
    Check that both positive and negative transfer examples exist in the record.
    Used as Sanity Check #5.

    Parameters
    ----------
    all_candidate_records : flat list of per-candidate dicts (from episode['candidates'])

    Returns
    -------
    dict with counts and pass/fail
    """
    if not all_candidate_records:
        return {'positive': 0, 'neutral': 0, 'negative': 0, 'has_positive': False,
                'has_negative': False, 'pass': False}
    pos = sum(1 for r in all_candidate_records if r['transfer_label'] == 'positive')
    neu = sum(1 for r in all_candidate_records if r['transfer_label'] == 'neutral')
    neg = sum(1 for r in all_candidate_records if r['transfer_label'] == 'negative')
    return {
        'positive': pos,
        'neutral': neu,
        'negative': neg,
        'has_positive': pos > 0,
        'has_negative': neg > 0,
        'pass': pos > 0 and neg > 0,
    }
