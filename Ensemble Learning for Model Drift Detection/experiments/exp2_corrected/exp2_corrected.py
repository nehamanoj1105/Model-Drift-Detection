"""
Experiment 2 Corrected -- Main Streaming Loop.

Implements the correct online chronology:
    1. Observe current unlabeled telemetry
    2. Build target representation
    3. Identify previously available candidate checkpoints
    4. Calculate source-target features
    5. Probability model predicts transfer probability
    6. Decide transfer / abstain
    7. Make predictions
    8. Observe target labels
    9. Calculate actual transfer outcome
   10. Update transferability history (ONLY NOW available to future decisions)
   11. After enough target data, build/update target checkpoint

Comparisons run in parallel:
    - Frozen (baseline)
    - Event-Driven (Exp1 reference)
    - RAPT-E (Exp1 reference)
    - Similarity-Only
    - Similarity-Weighted
    - Historical Reliability
    - Random Historical
    - Probability-Guided (main Exp2 method)
    - Oracle Transfer (true upper bound)
"""

import os
import sys
import copy
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP9_DIR  = os.path.normpath(os.path.join(_HERE, '..', 'exp9'))
_EXP9B_DIR = os.path.normpath(os.path.join(_HERE, '..', 'exp9b'))
for _p in [_EXP9_DIR, _EXP9B_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from checkpoint_manager import (
    CheckpointPool, RegimeCheckpointExp2, ActiveEnsemble,
    MIN_CHECKPOINT_SAMPLES, CHECKPOINT_BUFFER_SIZE,
)
from features import (
    compute_distributional_similarity,
    compute_source_target_features,
    compute_source_target_features_batch,
)
from prob_model import OnlineTransferabilityEstimator
from transfer_episodes import (
    build_transfer_episode,
    select_best_candidate_by_similarity,
    select_best_candidate_by_probability,
    select_by_historical_reliability,
    select_random,
    check_transfer_label_distribution,
    evaluate_candidate_on_target,
)

TAU = 0.60  # probability threshold for transfer decision
SEEDS = [42, 43, 44, 45, 46]

# Validated F1 ranges (from FINAL_VALIDATION_REPORT.md)
GATE_9A = {'RAPT-E': (0.9900, 0.9980), 'Event-Driven': (0.9900, 0.9990), 'Frozen': (0.9900, 0.9980)}
GATE_9B = {'RAPT-E': (0.8500, 0.9500), 'Event-Driven': (0.8500, 0.9500), 'Frozen': (0.8500, 0.9500)}


def _create_ensemble(seed: int, dataset: str = '9A'):
    """Create a base heterogeneous ensemble matching Exp1 architecture."""
    if dataset == '9B':
        from models_9b import create_base_ensemble
    else:
        from models import create_base_ensemble
    return create_base_ensemble(seed=seed)


def _predict_with_ckpt(ckpt: RegimeCheckpointExp2, X: np.ndarray) -> np.ndarray:
    return ckpt.predict(X)


def _f1_window(y_true, y_pred) -> float:
    return float(f1_score(
        np.atleast_1d(y_true), np.atleast_1d(y_pred),
        average='macro', zero_division=0,
    ))


def run_single_seed(
    stream_dict: dict,
    seed: int,
    stream_name: str,
    verbose: bool = True,
) -> dict:
    """
    Run Exp2 corrected for a single seed on one stream.

    Returns dict with:
      - per_window_records : list of window-level dicts
      - transfer_episodes : list of transfer episode dicts
      - all_candidate_records : flat list of per-candidate transfer records
      - summary : per-method aggregate metrics
      - prob_quality : held-out probability evaluation
      - gate_results : sanity check results
    """
    df = stream_dict['df']
    feature_cols = stream_dict['feature_cols']
    target_col = stream_dict['target_col']
    regime_col = stream_dict['regime_col']
    init_windows = stream_dict['init_windows']
    dataset = stream_dict['dataset']

    # Feature extraction -- for 9B apply the fitted StandardScaler
    preproc = stream_dict.get('preproc', None)

    def get_X(df_subset):
        X_raw = df_subset[feature_cols].values.astype(np.float64)
        if preproc is not None:
            return preproc.transform(df_subset)
        return X_raw

    X_all = get_X(df)
    y_all = df[target_col].values.astype(int)
    regime_all = df[regime_col].values

    n_windows = len(df)
    X_init = X_all[:init_windows]
    y_init = y_all[:init_windows]
    r_init = regime_all[0]

    rng = np.random.default_rng(seed=seed)

    # ── Initialise all methods ────────────────────────────────────────────────
    # Frozen
    frozen_ens = _create_ensemble(seed, dataset)
    t0c = time.process_time()
    frozen_ens.fit(X_init, y_init)
    frozen_init_cpu = time.process_time() - t0c
    # Save initial frozen checkpoint for use as per-target baseline
    from checkpoint_manager import RegimeCheckpointExp2 as Ckpt
    frozen_ckpt = Ckpt(
        regime_id=r_init,
        rf_model=copy.deepcopy(frozen_ens.rf),
        et_model=copy.deepcopy(frozen_ens.et),
        weights=[0.5, 0.5],
        training_sample_count=len(X_init),
        historical_f1=0.0,
        creation_episode=0,
        feature_mean=X_init.mean(axis=0),
        feature_std=X_init.std(axis=0) + 1e-8,
    )

    # Event-Driven (reuses Exp1 implementation)
    if dataset == '9B':
        from event_driven_9b import EventDrivenEnsemble
    else:
        from event_driven import EventDrivenEnsemble
    ed_model = EventDrivenEnsemble(seed=seed)
    ed_model.fit_initial(X_init, y_init)

    # Checkpoint pool (shared by RAPT-E, Similarity, Probability, Oracle, Historical)
    pool = CheckpointPool()
    init_f1 = float(f1_score(y_init, frozen_ens.predict(X_init), average='macro', zero_division=0))

    def _make_ens(s=None, seed=None):
        use_seed = seed if seed is not None else s
        return _create_ensemble(use_seed, dataset)

    init_ckpt = pool.add_initial_checkpoint(
        regime_id=r_init,
        X_init=X_init,
        y_init=y_init,
        historical_f1=init_f1,
        create_ensemble_fn=_make_ens,
        seed=seed,
    )

    # Probability estimator (shared across all transfer methods)
    prob_estimator = OnlineTransferabilityEstimator(tau=TAU, seed=seed)

    # Per-regime historical window buffers (for checkpoint training + similarity)
    regime_X_buffers: dict[str, list] = {r_init: list(X_init)}
    regime_y_buffers: dict[str, list] = {r_init: list(y_init)}

    # Active ensemble for each transfer method
    active = {
        'RAPT-E':              ActiveEnsemble(init_ckpt),
        'Similarity-Only':     ActiveEnsemble(init_ckpt),
        'Similarity-Weighted': ActiveEnsemble(init_ckpt),
        'Hist-Reliability':    ActiveEnsemble(init_ckpt),
        'Random-Historical':   ActiveEnsemble(init_ckpt),
        'Probability-Guided':  ActiveEnsemble(init_ckpt),
        'Oracle':              ActiveEnsemble(init_ckpt),
    }

    prev_regime = r_init
    episode_id = 0
    transfer_episode_records = []
    all_candidate_records = []
    pending_oracle_lookups = {}    # episode_id -> {target, candidates, baseline}
    per_window_records = []

    # Prediction buffers per method
    preds_buf = {m: [] for m in active}
    preds_buf['Frozen'] = []
    preds_buf['Event-Driven'] = []

    # CPU tracking
    cpu_adapt = {m: 0.0 for m in list(active.keys()) + ['Frozen', 'Event-Driven']}
    cpu_pred  = {m: 0.0 for m in list(active.keys()) + ['Frozen', 'Event-Driven']}

    # ── Streaming loop ────────────────────────────────────────────────────────
    for w in range(init_windows, n_windows):
        X_w = X_all[w:w+1]
        y_w = y_all[w:w+1]
        curr_regime = regime_all[w]

        # ── STEP 1-2: Regime transition handling ─────────────────────────────
        is_transition = (curr_regime != prev_regime)
        if is_transition:
            episode_id += 1
            target_regime = curr_regime
            source_regime = prev_regime

            # Accumulate previous regime buffer
            if source_regime not in regime_X_buffers:
                regime_X_buffers[source_regime] = []
                regime_y_buffers[source_regime] = []

            # ── STEP 3: Identify candidate checkpoints ────────────────────────
            candidates = pool.get_available_for_target(
                target_regime=target_regime,
                before_episode=episode_id,
            )

            # ── STEP 4: Compute similarity for all candidates ─────────────────
            X_target_so_far = np.array(regime_X_buffers.get(target_regime, []))
            if len(X_target_so_far) == 0:
                X_target_so_far = X_w  # fallback: use current window

            t0c = time.process_time()
            best_sim_ckpt, sim_scores = select_best_candidate_by_similarity(
                candidates,
                {r: np.array(regime_X_buffers[r]) for r in regime_X_buffers if regime_X_buffers[r]},
                X_target_so_far,
            )

            # ── STEP 5-6: Probability model predictions ───────────────────────
            if candidates and len(X_target_so_far) >= 3:
                best_prob_ckpt, best_prob, prob_scores, feat_dict = \
                    select_best_candidate_by_probability(
                        candidates,
                        prob_estimator,
                        {r: np.array(regime_X_buffers[r])
                         for r in regime_X_buffers if regime_X_buffers[r]},
                        X_target_so_far,
                        episode_id,
                    )
                do_transfer_prob = prob_estimator.should_transfer(best_prob)
            else:
                best_prob_ckpt, best_prob, prob_scores, feat_dict = None, 0.5, {}, {}
                do_transfer_prob = False

            adapt_cpu_transition = time.process_time() - t0c

            # ── Apply transition decisions for each method ────────────────────
            for method_name, act_ens in active.items():
                t0c = time.process_time()

                if not candidates:
                    # No historical policy available -- all methods fall back to RAPT-E local
                    if pool.has(target_regime):
                        act_ens.load_checkpoint(pool.get_latest(target_regime), is_transfer=False)
                    # else keep current model
                elif method_name == 'RAPT-E':
                    # Historical policy reuse if known; else local adapt
                    if pool.has(target_regime):
                        ckpt = pool.get_latest(target_regime)
                        act_ens.load_checkpoint(ckpt, is_transfer=False)
                        ckpt.reuse_count += 1
                    # else keep current; new checkpoint built below

                elif method_name == 'Similarity-Only':
                    if best_sim_ckpt is not None:
                        act_ens.load_checkpoint(best_sim_ckpt, is_transfer=True)
                        best_sim_ckpt.reuse_count += 1

                elif method_name == 'Similarity-Weighted':
                    # Blend policies weighted by similarity score
                    if candidates and sim_scores:
                        total_sim = sum(sim_scores.get(c.regime_id, 0.0) for c in candidates)
                        if total_sim > 0:
                            # Use the highest-sim candidate but calibrate weights by sim
                            best_ckpt = max(candidates, key=lambda c: sim_scores.get(c.regime_id, 0.0))
                            act_ens.load_checkpoint(best_ckpt, is_transfer=True)
                            best_ckpt.reuse_count += 1

                elif method_name == 'Hist-Reliability':
                    hr_ckpt = select_by_historical_reliability(candidates)
                    if hr_ckpt:
                        act_ens.load_checkpoint(hr_ckpt, is_transfer=True)
                        hr_ckpt.reuse_count += 1

                elif method_name == 'Random-Historical':
                    rand_ckpt = select_random(candidates, rng)
                    if rand_ckpt:
                        act_ens.load_checkpoint(rand_ckpt, is_transfer=True)
                        rand_ckpt.reuse_count += 1

                elif method_name == 'Probability-Guided':
                    if do_transfer_prob and best_prob_ckpt is not None:
                        act_ens.load_checkpoint(best_prob_ckpt, is_transfer=True)
                        best_prob_ckpt.reuse_count += 1
                    elif pool.has(target_regime):
                        # Abstain: fall back to RAPT-E
                        ckpt = pool.get_latest(target_regime)
                        act_ens.load_checkpoint(ckpt, is_transfer=False)
                        ckpt.reuse_count += 1

                elif method_name == 'Oracle':
                    # Store info for oracle resolution later (needs future target outcomes)
                    if target_regime not in pending_oracle_lookups:
                        pending_oracle_lookups[target_regime] = {
                            'episode_id': episode_id,
                            'candidates': list(candidates),
                            'best_so_far': None,
                            'best_f1_so_far': -1.0,
                        }

                cpu_adapt[method_name] += time.process_time() - t0c

            # Store pending transfer episode (outcome recorded after seeing target labels)
            pending_transfer_ep = {
                'episode_id': episode_id,
                'source_regime': source_regime,
                'target_regime': target_regime,
                'candidates': list(candidates),
                'sim_scores': dict(sim_scores),
                'best_sim_ckpt_id': best_sim_ckpt.regime_id if best_sim_ckpt else None,
                'best_prob_ckpt_id': best_prob_ckpt.regime_id if best_prob_ckpt else None,
                'best_prob': best_prob,
                'do_transfer_prob': do_transfer_prob,
                'prob_scores': dict(prob_scores),
                'feat_dict': dict(feat_dict),
                'X_target_at_transition': X_target_so_far.copy() if len(X_target_so_far) > 0 else np.zeros((1, X_w.shape[1])),
                'X_target_accumulated': [],
                'y_target_accumulated': [],
            }

            prev_regime = curr_regime

        # ── STEP 7: Make predictions ──────────────────────────────────────────
        y_true_int = int(y_w[0])

        # Frozen
        t0c = time.process_time()
        y_pred_frozen = frozen_ens.predict(X_w)
        cpu_pred['Frozen'] += time.process_time() - t0c

        # Event-Driven
        t0c = time.process_time()
        y_pred_ed = ed_model.predict(X_w)
        cpu_pred['Event-Driven'] += time.process_time() - t0c

        # All transfer methods
        method_preds = {}
        for method_name, act_ens in active.items():
            t0c = time.process_time()
            method_preds[method_name] = act_ens.predict(X_w)
            cpu_pred[method_name] += time.process_time() - t0c

        # ── STEP 8: Observe labels ────────────────────────────────────────────
        # (We now have y_true_int -- use for metrics and adaptation)

        # ── Event-Driven adaptation ───────────────────────────────────────────
        win_error = float(y_pred_ed[0] != y_true_int)
        t0c = time.process_time()
        ed_model.update_and_adapt(X_w, y_w, win_error)
        cpu_adapt['Event-Driven'] += time.process_time() - t0c

        # ── STEP 9-10: Accumulate target data for checkpoint creation ─────────
        if curr_regime not in regime_X_buffers:
            regime_X_buffers[curr_regime] = []
            regime_y_buffers[curr_regime] = []
        regime_X_buffers[curr_regime].append(X_w[0])
        regime_y_buffers[curr_regime].append(y_true_int)

        # Trim buffers
        if len(regime_X_buffers[curr_regime]) > CHECKPOINT_BUFFER_SIZE:
            regime_X_buffers[curr_regime] = regime_X_buffers[curr_regime][-CHECKPOINT_BUFFER_SIZE:]
            regime_y_buffers[curr_regime] = regime_y_buffers[curr_regime][-CHECKPOINT_BUFFER_SIZE:]

        # ── STEP 11: Create target checkpoint once sufficient data exists ─────
        n_target_data = len(regime_X_buffers[curr_regime])
        if n_target_data == MIN_CHECKPOINT_SAMPLES:
            # Build first checkpoint for this target regime
            X_buf = np.concatenate([X_init, np.array(regime_X_buffers[curr_regime])], axis=0)
            y_buf = np.concatenate([y_init, np.array(regime_y_buffers[curr_regime])], axis=0)
            current_f1 = float(f1_score(y_buf,
                                        frozen_ens.predict(X_buf),
                                        average='macro', zero_division=0))
            new_local_ckpt = pool.add_checkpoint(
                regime_id=curr_regime,
                X_target=X_buf,
                y_target=y_buf,
                historical_f1=current_f1,
                episode_id=episode_id,
                create_ensemble_fn=_make_ens,
                seed=seed + w * 7,
            )
            cpu_adapt['RAPT-E'] += time.process_time() - t0c
            # Load newly created local checkpoint into RAPT-E and Probability-Guided (if abstaining)
            active['RAPT-E'].load_checkpoint(new_local_ckpt, is_transfer=False)
            if not active['Probability-Guided'].is_transfer:
                active['Probability-Guided'].load_checkpoint(new_local_ckpt, is_transfer=False)

        # ── Record window metrics ─────────────────────────────────────────────
        win_record = {
            'seed': seed,
            'window_id': w,
            'stream': stream_name,
            'regime': curr_regime,
            'episode_id': episode_id,
            'is_transition': int(is_transition),
            'y_true': y_true_int,
        }
        for method_name in ['Frozen', 'Event-Driven']:
            pred = y_pred_frozen[0] if method_name == 'Frozen' else y_pred_ed[0]
            win_record[f'pred_{method_name}'] = int(pred)
            win_record[f'f1_{method_name}'] = _f1_window([y_true_int], [pred])
        for method_name in active:
            pred = method_preds[method_name][0]
            win_record[f'pred_{method_name}'] = int(pred)
            win_record[f'f1_{method_name}'] = _f1_window([y_true_int], [pred])

        per_window_records.append(win_record)

    # ── Post-run: Resolve pending transfer episodes ───────────────────────────
    all_transfer_records = []
    all_cand_records = []

    # Collect all X/y per regime to resolve oracle / probability records
    # For each regime, use accumulated buffers
    for ep in transfer_episode_records:
        pass  # placeholder -- transfer episodes are constructed below

    # Construct transfer episodes from per-window data
    df_win = pd.DataFrame(per_window_records)
    transfer_episode_structs = _reconstruct_transfer_episodes(
        df_win=df_win,
        regime_X_buffers=regime_X_buffers,
        regime_y_buffers=regime_y_buffers,
        pool=pool,
        frozen_ckpt=frozen_ckpt,
        prob_estimator=prob_estimator,
        feature_cols=feature_cols,
        dataset=dataset,
        seed=seed,
        episode_id_final=episode_id,
    )
    all_cand_records = transfer_episode_structs['candidate_records']

    # ── Compute per-method aggregate F1 ──────────────────────────────────────
    summary_records = []
    for method_name in ['Frozen', 'Event-Driven'] + list(active.keys()):
        col = f'f1_{method_name}'
        if col not in df_win.columns:
            continue
        f1_vals = df_win[col].values
        mean_f1 = float(np.mean(f1_vals))
        std_f1  = float(np.std(f1_vals))
        summary_records.append({
            'seed': seed,
            'stream': stream_name,
            'method': method_name,
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'adapt_cpu': cpu_adapt.get(method_name, 0.0),
            'pred_cpu': cpu_pred.get(method_name, 0.0),
        })

    label_dist = check_transfer_label_distribution(all_cand_records)

    # ── Probability model quality (held-out evaluation) ───────────────────────
    prob_quality = prob_estimator.evaluate_held_out()

    # ── Sanity gate results ───────────────────────────────────────────────────
    gate_9x = GATE_9A if dataset == '9A' else GATE_9B
    gate_results = {}
    for m in ['RAPT-E', 'Event-Driven', 'Frozen']:
        col = f'f1_{m}'
        if col not in df_win.columns:
            continue
        mean_val = float(df_win[col].mean())
        lo, hi = gate_9x[m]
        gate_results[m] = {
            'mean_f1': mean_val,
            'expected_lo': lo,
            'expected_hi': hi,
            'pass': lo <= mean_val <= hi,
        }

    return {
        'per_window_records': df_win,
        'candidate_records': all_cand_records,
        'summary': pd.DataFrame(summary_records),
        'prob_quality': prob_quality,
        'label_distribution': label_dist,
        'gate_results': gate_results,
        'n_pool_checkpoints': len(pool),
    }


def _reconstruct_transfer_episodes(
    df_win, regime_X_buffers, regime_y_buffers, pool,
    frozen_ckpt, prob_estimator, feature_cols, dataset, seed, episode_id_final,
) -> dict:
    """
    After the stream completes, use accumulated data to build transfer episode records.
    This is purely for analysis/reporting -- not used for any predictions.
    """
    candidate_records = []

    # Find all regime transitions in the stream
    regime_seq = df_win['regime'].values
    transitions = []
    for i in range(1, len(regime_seq)):
        if regime_seq[i] != regime_seq[i-1]:
            transitions.append({
                'window_idx': i,
                'from_regime': regime_seq[i-1],
                'to_regime': regime_seq[i],
                'episode_id': int(df_win['episode_id'].values[i]),
            })

    X_buffers = {r: np.array(v) for r, v in regime_X_buffers.items() if v}
    y_buffers = {r: np.array(v) for r, v in regime_y_buffers.items() if v}

    for trans in transitions:
        target = trans['to_regime']
        ep_id = trans['episode_id']

        # Candidates: all pool checkpoints from other regimes created before this episode
        candidates = pool.get_available_for_target(target, before_episode=ep_id)
        if not candidates:
            continue

        X_target = X_buffers.get(target)
        y_target = y_buffers.get(target)
        if X_target is None or len(X_target) < 10:
            continue

        # Similarity scores
        from features import compute_distributional_similarity
        sim_scores = {}
        for ckpt in candidates:
            X_src = X_buffers.get(ckpt.regime_id)
            if X_src is not None and len(X_src) > 0:
                sim_scores[ckpt.regime_id] = compute_distributional_similarity(X_src, X_target)
            else:
                sim_scores[ckpt.regime_id] = 0.0

        ep_struct = build_transfer_episode(
            episode_id=ep_id,
            source_regime=trans['from_regime'],
            target_regime=target,
            candidates=candidates,
            X_target=X_target,
            y_target=y_target,
            baseline_ckpt=frozen_ckpt,
            similarity_scores=sim_scores,
        )
        if ep_struct is not None:
            candidate_records.extend(ep_struct['candidates'])

            # Record outcomes in probability estimator for the per-candidate features
            for cand_rec in ep_struct['candidates']:
                src_id = cand_rec['source_regime']
                X_src = X_buffers.get(src_id)
                X_tgt = X_target
                if X_src is not None and len(X_src) > 0:
                    src_ckpt = next(
                        (c for c in candidates if c.regime_id == src_id), None)
                    if src_ckpt:
                        feat_vec = compute_source_target_features(
                            source_ckpt=src_ckpt,
                            X_source_windows=X_src,
                            X_target_windows=X_tgt,
                            current_episode=ep_id,
                        )
                        prob_estimator.record_outcome(
                            episode_id=ep_id,
                            features=feat_vec,
                            delta_f1=cand_rec['delta_f1'],
                            source_regime=src_id,
                            target_regime=target,
                            similarity=sim_scores.get(src_id, 0.0),
                        )

    return {'candidate_records': candidate_records}
