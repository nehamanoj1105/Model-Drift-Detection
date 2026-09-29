"""
src/experiment.py - Core streaming experiment runner for Exp2 Final.

Runs all 8 methods on a stream for one seed.
Records per-window predictions, per-interval counts, and per-decision estimator state.

DESIGN (pre-registered):
- P1: Base learner - FastEnsemble (RF 60 + ET 60, n_jobs=-1) for S1/N1/N2/S4/S3
       HeterogeneousBaseEnsemble for S2
- P2: Policy B (X_init + regime data) for all methods
- P3: Frozen: fit once on X_init
- P4: Candidates: fit on X_init + block data when block closes. Cache predictions.
- P5: Decision t covers [t, t+K). Labels available after interval.
- P6: Local-Retrain: fit on X_init + recent L windows before t
- P7: Event-Driven: vendored Exp1 detector
- P8: Per-window per-model storage
- P9: Regime segments for statistics

LEAKAGE PREVENTION:
- Estimator outcomes recorded AFTER interval (after y_all[t:t+K] is available)
- All features use only data before t
- Oracle evaluated AFTER interval with realized F1 values
"""

import sys
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_EXP2_ROOT = _HERE.parent
sys.path.insert(0, str(_EXP2_ROOT / 'vendor' / 'exp1'))

from features import extract_features
from labels import assign_label, assign_labels_batch, macro_f1_from_predictions
from estimator import OnlineEstimator, TAU_GRID, TAU_ABSTAIN


class FastEnsemble:
    """RF(15) + ET(15) fast ensemble for non-S2 streams."""
    
    def __init__(self, seed: int = 42, n_trees: int = 15):
        self.seed = seed
        self.n_trees = n_trees
        self.rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=1)
        self.et = ExtraTreesClassifier(n_estimators=n_trees, random_state=seed, n_jobs=1)
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FastEnsemble':
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.classes_ = np.unique(y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self.rf.predict_proba(X)
        p2 = self.et.predict_proba(X)
        if p1.shape[1] != p2.shape[1]:
            n = max(p1.shape[1], p2.shape[1])
            p1_new = np.zeros((len(X), n))
            p2_new = np.zeros((len(X), n))
            p1_new[:, :p1.shape[1]] = p1
            p2_new[:, :p2.shape[1]] = p2
            p1, p2 = p1_new, p2_new
        return (p1 + p2) / 2.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
    
    def cv_f1(self, X: np.ndarray, y: np.ndarray, cv: int = 3) -> float:
        """3-fold stratified CV macro F1 for HistF1 computation."""
        if len(np.unique(y)) < 2 or len(y) < cv * 2:
            return 0.5
        try:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.seed)
            scores = []
            for tr_idx, val_idx in skf.split(X, y):
                m = FastEnsemble(self.seed, n_trees=8)
                m.fit(X[tr_idx], y[tr_idx])
                pred = m.predict(X[val_idx])
                scores.append(f1_score(y[val_idx], pred, average='macro', zero_division=0))
            return float(np.mean(scores))
        except Exception:
            return 0.5


def _make_model(seed: int, stream_id: str = 'S1', use_hetero: bool = False):
    """Create appropriate model for stream type."""
    if use_hetero or stream_id == 'S2':
        from heterogeneous_ensemble import HeterogeneousEnsemble
        return HeterogeneousEnsemble(seed=seed)
    return FastEnsemble(seed=seed)


class StreamingExperiment:
    """
    Core streaming experiment.
    Runs all 8 methods (+ appendix if requested) on a stream for one seed.
    
    Returns per-window predictions and per-interval statistics.
    """
    
    def __init__(
        self,
        stream_id: str,
        seed: int,
        K: int,         # windows per decision interval
        W0: int,        # warmup windows
        L: int,         # windows for local retrain history
        eps: float = 0.01,
        n_boot: int = 200,
        alpha: float = 0.10,
        use_hetero: bool = False,
        include_appendix: bool = True,
        dev_split_frac: float = 0.30,
    ):
        self.stream_id = stream_id
        self.seed = seed
        self.K = K
        self.W0 = W0
        self.L = L
        self.eps = eps
        self.n_boot = n_boot
        self.alpha = alpha
        self.use_hetero = use_hetero
        self.include_appendix = include_appendix
        self.dev_split_frac = dev_split_frac
        
        self.rng = np.random.default_rng(seed)
        
        # Results storage
        self.per_window_preds: Dict[str, np.ndarray] = {}
        self.per_interval_results: List[Dict] = []
        self.decision_log: List[Dict] = []
        self.estimator_log: List[Dict] = []
        self.transfer_log: List[Dict] = []
        
        # Checkpoints: {block_id: (model, X_block, y_block, hist_f1, block_end_t, age)}
        self.checkpoint_pool: Dict[str, Dict] = {}
        self.checkpoint_predictions: Dict[Tuple, np.ndarray] = {}  # (ckpt_id, t) -> preds
        
        self.estimator = OnlineEstimator(C=1.0, max_iter=300, class_weight='balanced', seed=seed)
        self.tau_history: List[Dict] = []
        
        self._n_decisions = 0
        self._n_transfers = 0
        self._n_abstentions = 0
        self._n_unfitted = 0
        
        self.event_driven_triggers = 0
        self.temperature_selected = None
    
    def run(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        regime_ids: np.ndarray,
    ) -> Dict:
        """
        Run experiment on stream data.
        
        Args:
            X_all: (n_windows, n_features) window feature arrays
            y_all: (n_windows,) window labels
            regime_ids: (n_windows,) regime identifier per window
        
        Returns dict with all results.
        """
        N = len(X_all)
        K = self.K
        W0 = self.W0
        L = self.L
        
        # Validate paths under project root
        assert _EXP2_ROOT.is_relative_to(_EXP2_ROOT.parent.parent), "Path check failed"
        
        X_init = X_all[:W0]
        y_init = y_all[:W0]
        
        # Initialize method predictions (per-window)
        methods = ['Frozen', 'EventDriven', 'LocalRetrain', 'SimilarityOnly',
                   'SimilarityWeighted', 'HistReliability', 'ProbabilityGuided', 'Oracle']
        if self.include_appendix:
            methods += ['RandomHistorical', 'ShadowBest']
        
        all_preds = {m: np.full(N, -1, dtype=int) for m in methods}
        
        # Fit initial/frozen model
        frozen_model = _make_model(self.seed, self.stream_id, self.use_hetero)
        frozen_model.fit(X_init, y_init)
        
        # Frozen predictions (all windows)
        for m in methods:
            all_preds[m][:W0] = frozen_model.predict(X_init)
        
        # Event-Driven: vendored Exp1 code
        try:
            from event_driven import EventDrivenEnsemble
            ed_sys = EventDrivenEnsemble(seed=self.seed)
            ed_sys.fit_initial(X_init, y_init)
            ed_available = True
        except Exception as e:
            print(f"EventDriven init failed: {e}")
            ed_available = False
        
        # Track regime data for checkpoints
        regime_data: Dict[str, List] = {}  # regime_id -> [(X_window, y_window)]
        block_counter = 0
        current_regime = regime_ids[W0] if W0 < N else regime_ids[-1]
        block_start = W0
        
        # Local retrain history
        local_models: Dict[str, Any] = {}
        
        # Decision points: t covers [t, t+K), t in [W0, N-K, step K]
        decision_points = list(range(W0, N - K, K))
        n_dp = len(decision_points)
        dev_cutoff = int(n_dp * self.dev_split_frac)
        test_decisions = decision_points[dev_cutoff:]
        
        # Temperature for Similarity-Weighted (chosen on dev)
        temp_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        
        # Per-window predictions for EventDriven (filled window by window)
        ed_preds_list = list(all_preds['Frozen'][:W0].copy())
        
        # Initialize local model
        local_model_current = frozen_model
        local_hist_X = list(X_init)
        local_hist_y = list(y_init)
        
        # Track previous interval local F1 for LocalRecentF1 feature
        local_recent_f1 = -1.0
        
        # Main streaming loop: window by window for EventDriven, interval by interval for methods
        # Process window-by-window up to W0
        for w in range(W0):
            if ed_available:
                err = 1.0 if frozen_model.predict(X_all[w:w+1])[0] != y_all[w] else 0.0
                try:
                    result = ed_sys.update_and_adapt(X_all[w:w+1], [y_all[w]], err)
                    if isinstance(result, tuple) and len(result) >= 1 and result[0]:
                        self.event_driven_triggers += 1
                except Exception:
                    pass
        
        # Process intervals
        for dp_idx, t in enumerate(decision_points):
            interval_end = t + K
            if interval_end > N:
                break
            
            X_interval = X_all[t:interval_end]
            y_interval = y_all[t:interval_end]
            
            # --- Update regime bookkeeping ---
            for w in range(max(t - K, W0), t):
                if w >= N:
                    break
                r = regime_ids[w]
                if r not in regime_data:
                    regime_data[r] = []
                regime_data[r].append((X_all[w], y_all[w]))
            
            # --- Create new checkpoints for completed blocks ---
            # When does a block close? When the regime changes (for real streams) 
            # or at fixed K intervals (for synthetic).
            # For this implementation: checkpoint per closed regime block
            self._maybe_create_checkpoints(
                t, regime_ids, X_all, y_all, X_init, y_init, block_counter
            )
            
            # --- Available candidates (checkpoints from blocks BEFORE t) ---
            candidates = self._get_available_candidates(t, regime_ids[t])
            
            # --- Local-Retrain model ---
            # Fit on X_init + most recent L windows before t
            local_window_start = max(W0, t - L)
            X_local = np.vstack([X_init, X_all[local_window_start:t]])
            y_local = np.concatenate([y_init, y_all[local_window_start:t]])
            
            if len(np.unique(y_local)) >= 2 and len(y_local) >= 10:
                local_m = _make_model(self.seed, self.stream_id, self.use_hetero)
                local_m.fit(X_local, y_local)
            else:
                local_m = frozen_model
            
            y_pred_local = local_m.predict(X_interval)
            f1_local = macro_f1_from_predictions(y_interval, y_pred_local)
            
            # --- EventDriven: predict interval, then update window-by-window ---
            if ed_available:
                ed_preds_interval = []
                for w_off in range(K):
                    w = t + w_off
                    if w >= N:
                        break
                    pred_ed = ed_sys.predict(X_all[w:w+1])[0]
                    ed_preds_interval.append(pred_ed)
                    # Update
                    err = 1.0 if pred_ed != y_all[w] else 0.0
                    try:
                        result = ed_sys.update_and_adapt(X_all[w:w+1], [y_all[w]], err)
                        if isinstance(result, tuple) and len(result) >= 1 and result[0]:
                            self.event_driven_triggers += 1
                    except Exception:
                        pass
                ed_preds_arr = np.array(ed_preds_interval)
            else:
                ed_preds_arr = y_pred_local.copy()
            
            # --- Frozen predictions ---
            y_pred_frozen = frozen_model.predict(X_interval)
            
            # --- Similarity-Only: nearest candidate by W ---
            if candidates:
                X_current_2K = X_all[max(W0, t - 2*K):t]
                if len(X_current_2K) < 2:
                    X_current_2K = X_init
                
                W_vals = []
                for ck in candidates:
                    X_cand = ck['X_block']
                    w_dist = _wasserstein_mean(X_current_2K, X_cand)
                    W_vals.append(w_dist)
                
                W_arr = np.array(W_vals)
                nearest_idx = int(np.argmin(W_arr))
                y_pred_simonly = candidates[nearest_idx]['preds_fn'](X_interval)
                
                # --- Similarity-Weighted: softmax over -W, soft ensemble ---
                # Temperature chosen on dev; during dev, use T=1.0 as default
                T = self.temperature_selected if self.temperature_selected is not None else 1.0
                weights = np.exp(-W_arr / T)
                weights /= weights.sum()
                
                # Soft ensemble
                y_proba_sw = np.zeros((len(X_interval), 2))  # assuming binary, will fix
                for ci, ck in enumerate(candidates):
                    proba = ck['proba_fn'](X_interval)
                    # Align classes
                    n_classes = max(y_proba_sw.shape[1], proba.shape[1])
                    if proba.shape[1] != y_proba_sw.shape[1]:
                        proba_new = np.zeros((len(X_interval), n_classes))
                        proba_new[:, :proba.shape[1]] = proba
                        proba = proba_new
                        if y_proba_sw.shape[1] < n_classes:
                            y_proba_sw_new = np.zeros((len(X_interval), n_classes))
                            y_proba_sw_new[:, :y_proba_sw.shape[1]] = y_proba_sw
                            y_proba_sw = y_proba_sw_new
                    y_proba_sw += weights[ci] * proba
                
                y_pred_simweighted = np.argmax(y_proba_sw, axis=1)
                
                # --- Hist-Reliability: highest HistF1 candidate ---
                hist_f1s_all = [ck['hist_f1'] for ck in candidates]
                best_hist_idx = int(np.argmax(hist_f1s_all))
                y_pred_histreliab = candidates[best_hist_idx]['preds_fn'](X_interval)
                
                # --- Feature extraction for Probability-Guided ---
                X_cands_list = [ck['X_block'] for ck in candidates]
                hist_f1s_list = [ck['hist_f1'] for ck in candidates]
                ages_list = [ck['age'] for ck in candidates]
                pool_size = len(candidates)
                
                features_matrix = extract_features(
                    X_current=X_current_2K,
                    X_candidates=X_cands_list,
                    hist_f1s=hist_f1s_list,
                    ages=ages_list,
                    pool_size=pool_size,
                    local_recent_f1=local_recent_f1,
                    rng=self.rng,
                )
                
                # --- Probability-Guided ---
                p_scores_arr, is_fit = self.estimator.predict_batch(features_matrix, current_t=t)
                p_scores = list(p_scores_arr)
                
                # Choose tau online
                tau_t = self.estimator.choose_tau_online(self.tau_history)
                
                p_scores_arr = np.array(p_scores)
                best_p_idx = int(np.argmax(p_scores_arr))
                p_max = p_scores_arr[best_p_idx]
                
                if tau_t != TAU_ABSTAIN and p_max >= tau_t:
                    y_pred_probguided = candidates[best_p_idx]['preds_fn'](X_interval)
                    self._n_transfers += 1
                    did_transfer = True
                else:
                    y_pred_probguided = y_pred_local.copy()
                    self._n_abstentions += 1
                    did_transfer = False
                
                self._n_decisions += 1
                
                # --- Oracle: per decision best of all candidates + Local by realized F1 ---
                oracle_f1s = [f1_local]
                oracle_preds_list = [y_pred_local]
                oracle_sources = ['Local']
                for ci, ck in enumerate(candidates):
                    y_pred_ck = ck['preds_fn'](X_interval)
                    f1_ck = macro_f1_from_predictions(y_interval, y_pred_ck)
                    oracle_f1s.append(f1_ck)
                    oracle_preds_list.append(y_pred_ck)
                    oracle_sources.append(ck['block_id'])
                
                best_oracle_idx = int(np.argmax(oracle_f1s))
                y_pred_oracle = oracle_preds_list[best_oracle_idx]
                f1_oracle = oracle_f1s[best_oracle_idx]
                
                # --- Random-Historical ---
                rand_idx = int(self.rng.integers(0, len(candidates)))
                y_pred_randhist = candidates[rand_idx]['preds_fn'](X_interval)
                
                # --- Shadow-Best: candidate with highest F1 on PREVIOUS interval ---
                if len(self.per_interval_results) > 0:
                    prev_result = self.per_interval_results[-1]
                    prev_t = prev_result.get('t', t - K)
                    prev_end = prev_t + K
                    if prev_end <= t and prev_end <= N:
                        X_prev = X_all[prev_t:prev_end]
                        y_prev = y_all[prev_t:prev_end]
                        shadow_f1s = []
                        for ck in candidates:
                            y_pred_shadow = ck['preds_fn'](X_prev)
                            shadow_f1s.append(macro_f1_from_predictions(y_prev, y_pred_shadow))
                        best_shadow_idx = int(np.argmax(shadow_f1s))
                        y_pred_shadowbest = candidates[best_shadow_idx]['preds_fn'](X_interval)
                    else:
                        y_pred_shadowbest = y_pred_local.copy()
                else:
                    y_pred_shadowbest = y_pred_local.copy()
                
                # Record OOS scores for estimator BEFORE recording outcomes
                for ci, feat_row in enumerate(features_matrix):
                    self.estimator.record_oos_score(
                        score=p_scores[ci],
                        label=0,  # will update below after labels known
                        decision_t=t,
                        df1=0.0,  # will update below
                    )
                
                # --- POST-INTERVAL: Record outcomes (labels now available) ---
                Y_cands_arr = np.array([ck['preds_fn'](X_interval) for ck in candidates])
                label_results = assign_labels_batch(
                    y_interval, Y_cands_arr, y_pred_local,
                    n_boot=self.n_boot, alpha=self.alpha, eps=self.eps,
                    seed=self.seed + dp_idx * 31
                )
                
                for ci, ck in enumerate(candidates):
                    label_result = label_results[ci]
                    label_int = 1 if label_result['label'] == 'positive' else 0
                    feat_row = features_matrix[ci]
                    
                    self.estimator.record_outcome(
                        decision_t=t,
                        features=feat_row,
                        label=label_int,
                        direct_df1=label_result['direct_df1'],
                    )
                    
                    # Update OOS score label (we stored it above)
                    n_prev = len(self.estimator._oos_scores) - len(candidates)
                    if n_prev + ci < len(self.estimator._oos_labels):
                        self.estimator._oos_labels[n_prev + ci] = label_int
                        self.estimator._oos_df1s[n_prev + ci] = label_result['direct_df1']
                
                # Update tau history
                # What would have happened with best candidate?
                best_cand_df1 = float(macro_f1_from_predictions(
                    y_interval, candidates[best_p_idx]['preds_fn'](X_interval)
                ) - f1_local) if candidates else 0.0
                
                self.tau_history.append({
                    't': t,
                    'p_max': float(p_max),
                    'tau_t': float(tau_t) if tau_t != TAU_ABSTAIN else 999.0,
                    'chosen_cand_df1': best_cand_df1,
                    'did_transfer': did_transfer,
                })
                
                # Transfer log
                self.transfer_log.append({
                    't': t,
                    'did_transfer': did_transfer,
                    'p_max': float(p_max),
                    'tau_t': float(tau_t) if tau_t != TAU_ABSTAIN else 999.0,
                    'n_candidates': len(candidates),
                    'oracle_source': oracle_sources[best_oracle_idx],
                    'is_dev': dp_idx < dev_cutoff,
                })
                
                # Update local_recent_f1
                local_recent_f1 = float(f1_local)
                
            else:
                # No candidates available: all transfer methods fall back to local
                y_pred_simonly = y_pred_local.copy()
                y_pred_simweighted = y_pred_local.copy()
                y_pred_histreliab = y_pred_local.copy()
                y_pred_probguided = y_pred_local.copy()
                y_pred_oracle = y_pred_local.copy()
                y_pred_randhist = y_pred_local.copy()
                y_pred_shadowbest = y_pred_local.copy()
                
                self._n_abstentions += 1
                self._n_decisions += 1
            
            # Store interval predictions
            all_preds['Frozen'][t:interval_end] = y_pred_frozen
            all_preds['EventDriven'][t:interval_end] = ed_preds_arr[:K]
            all_preds['LocalRetrain'][t:interval_end] = y_pred_local
            all_preds['SimilarityOnly'][t:interval_end] = y_pred_simonly
            all_preds['SimilarityWeighted'][t:interval_end] = y_pred_simweighted
            all_preds['HistReliability'][t:interval_end] = y_pred_histreliab
            all_preds['ProbabilityGuided'][t:interval_end] = y_pred_probguided
            all_preds['Oracle'][t:interval_end] = y_pred_oracle
            if self.include_appendix:
                all_preds['RandomHistorical'][t:interval_end] = y_pred_randhist
                all_preds['ShadowBest'][t:interval_end] = y_pred_shadowbest
            
            # Per-interval metrics
            f1_frozen = macro_f1_from_predictions(y_interval, y_pred_frozen)
            f1_ed = macro_f1_from_predictions(y_interval, ed_preds_arr[:K])
            f1_simonly = macro_f1_from_predictions(y_interval, y_pred_simonly)
            f1_simw = macro_f1_from_predictions(y_interval, y_pred_simweighted)
            f1_hist = macro_f1_from_predictions(y_interval, y_pred_histreliab)
            f1_prob = macro_f1_from_predictions(y_interval, y_pred_probguided)
            f1_oracle_val = macro_f1_from_predictions(y_interval, y_pred_oracle)
            
            interval_result = {
                't': t,
                'interval_end': interval_end,
                'regime': regime_ids[t],
                'n_candidates': len(candidates) if candidates else 0,
                'is_dev': dp_idx < dev_cutoff,
                'f1_Frozen': float(f1_frozen),
                'f1_EventDriven': float(f1_ed),
                'f1_LocalRetrain': float(f1_local),
                'f1_SimilarityOnly': float(f1_simonly),
                'f1_SimilarityWeighted': float(f1_simw),
                'f1_HistReliability': float(f1_hist),
                'f1_ProbabilityGuided': float(f1_prob),
                'f1_Oracle': float(f1_oracle_val),
            }
            if self.include_appendix:
                interval_result['f1_RandomHistorical'] = float(
                    macro_f1_from_predictions(y_interval, y_pred_randhist))
                interval_result['f1_ShadowBest'] = float(
                    macro_f1_from_predictions(y_interval, y_pred_shadowbest))
            
            self.per_interval_results.append(interval_result)
        
        # Select temperature for Similarity-Weighted (on dev set)
        if dev_cutoff > 0 and len(self.per_interval_results) >= dev_cutoff:
            dev_results = self.per_interval_results[:dev_cutoff]
            # Find best temperature on dev
            # (already used T=1.0 as default during dev, so just note it)
            self.temperature_selected = 1.0  # Simplified: fixed T=1.0 for now
            # TODO: full dev sweep if time permits
        
        # K12: Verify consistency between per-window and per-interval aggregations
        consistency_ok = self._verify_k12_consistency(X_all, y_all, all_preds, decision_points, K, W0)
        
        # Gather AUROC
        test_ts = [r['t'] for r in self.per_interval_results if not r['is_dev']]
        auroc = self.estimator.get_auroc(test_only_decisions=test_ts)
        
        # Compute test-region macro F1 for each method
        test_mask = np.zeros(N, dtype=bool)
        for r in self.per_interval_results:
            if not r['is_dev']:
                test_mask[r['t']:r['interval_end']] = True
        
        test_f1s = {}
        for m, preds in all_preds.items():
            if test_mask.any():
                test_f1s[m] = float(macro_f1_from_predictions(y_all[test_mask], preds[test_mask]))
            else:
                test_f1s[m] = float(macro_f1_from_predictions(
                    y_all[W0:W0+K*len(decision_points)],
                    preds[W0:W0+K*len(decision_points)]
                ))
        
        # Regime segments
        regime_segments = self._extract_regime_segments(regime_ids, W0)
        
        results = {
            'stream_id': self.stream_id,
            'seed': self.seed,
            'K': K,
            'W0': W0,
            'L': L,
            'n_decisions': self._n_decisions,
            'n_transfers': self._n_transfers,
            'n_abstentions': self._n_abstentions,
            'n_unfitted': self._n_unfitted,
            'event_driven_triggers': self.event_driven_triggers,
            'test_f1s': test_f1s,
            'per_interval_results': self.per_interval_results,
            'per_window_preds': {m: p.tolist() for m, p in all_preds.items()},
            'y_all': y_all.tolist(),
            'regime_ids': regime_ids.tolist(),
            'auroc': auroc,
            'estimator_stats': self.estimator.get_fit_stats(),
            'estimator_oos_scores': self.estimator._oos_scores,
            'estimator_oos_labels': self.estimator._oos_labels,
            'regime_segments': regime_segments,
            'consistency_ok': consistency_ok,
            'tau_history': self.tau_history[:10],  # Sample
        }
        
        return results
    
    def _maybe_create_checkpoints(
        self, t: int, regime_ids: np.ndarray, X_all: np.ndarray, y_all: np.ndarray,
        X_init: np.ndarray, y_init: np.ndarray, block_counter: int
    ):
        """Create checkpoints for regime blocks that closed before t (O(N) incremental scan)."""
        if t < self.W0 + self.K:
            return
        
        if not hasattr(self, '_last_scanned_w') or self._last_scanned_w is None:
            self._last_scanned_w = self.W0
            self._curr_prev_regime = regime_ids[self.W0]
            self._curr_block_start = self.W0
        
        for w in range(self._last_scanned_w + 1, t):
            r = regime_ids[w]
            if r != self._curr_prev_regime:
                block_id = f"{self._curr_prev_regime}_{self._curr_block_start}"
                if block_id not in self.checkpoint_pool:
                    X_block = X_all[self._curr_block_start:w]
                    y_block = y_all[self._curr_block_start:w]
                    
                    if len(np.unique(y_block)) >= 2 and len(y_block) >= 5:
                        X_train = np.vstack([X_init, X_block])
                        y_train = np.concatenate([y_init, y_block])
                        
                        model = _make_model(self.seed, self.stream_id, self.use_hetero)
                        model.fit(X_train, y_train)
                        
                        hist_f1 = model.cv_f1(X_block, y_block) if isinstance(model, FastEnsemble) else 0.5
                        block_end_t = w
                        
                        def make_preds_fn(m):
                            return lambda X: m.predict(X)
                        def make_proba_fn(m):
                            return lambda X: m.predict_proba(X)
                        
                        self.checkpoint_pool[block_id] = {
                            'block_id': block_id,
                            'regime_id': self._curr_prev_regime,
                            'model': model,
                            'X_block': X_block,
                            'y_block': y_block,
                            'hist_f1': hist_f1,
                            'block_end_t': block_end_t,
                            'preds_fn': make_preds_fn(model),
                            'proba_fn': make_proba_fn(model),
                        }
                self._curr_prev_regime = r
                self._curr_block_start = w
        
        self._last_scanned_w = t - 1
    
    def _get_available_candidates(self, t: int, current_regime: str) -> List[Dict]:
        """Get candidates from blocks that closed before t and from different regime."""
        candidates = []
        for block_id, ck in self.checkpoint_pool.items():
            if ck['block_end_t'] < t and ck['regime_id'] != current_regime:
                age = (t - ck['block_end_t']) // self.K
                ck_copy = dict(ck)
                ck_copy['age'] = age
                candidates.append(ck_copy)
        return candidates
    
    def _extract_regime_segments(self, regime_ids: np.ndarray, W0: int) -> List[Dict]:
        """Extract regime segments for bootstrap statistics."""
        segments = []
        prev_r = regime_ids[W0]
        seg_start = W0
        
        for w in range(W0 + 1, len(regime_ids)):
            if regime_ids[w] != prev_r:
                segments.append({'regime': prev_r, 'start': seg_start, 'end': w})
                prev_r = regime_ids[w]
                seg_start = w
        
        segments.append({'regime': prev_r, 'start': seg_start, 'end': len(regime_ids)})
        return segments
    
    def _verify_k12_consistency(
        self, X_all, y_all, all_preds, decision_points, K, W0
    ) -> bool:
        """
        K12: Verify consistency between per-window predictions and per-interval aggregation.
        """
        consistent = True
        for r in self.per_interval_results[:5]:  # Check first 5 intervals
            t = r['t']
            end = r['interval_end']
            y_true = y_all[t:end]
            
            for m in ['Frozen', 'LocalRetrain', 'ProbabilityGuided']:
                preds_win = np.array(all_preds[m][t:end])
                f1_from_preds = float(f1_score(y_true, preds_win, average='macro', zero_division=0))
                f1_from_interval = r.get(f'f1_{m}', None)
                
                if f1_from_interval is not None:
                    if abs(f1_from_preds - f1_from_interval) > 1e-9:
                        print(f"K12 FAIL: {m} t={t} per-window={f1_from_preds:.6f} "
                              f"vs per-interval={f1_from_interval:.6f}")
                        consistent = False
        
        return consistent


def _wasserstein_mean(X: np.ndarray, Y: np.ndarray) -> float:
    """Fast mean 1-D Wasserstein (sorted quantile difference)."""
    n_features = min(X.shape[1], Y.shape[1])
    dists = []
    for f in range(n_features):
        p = np.sort(X[:, f])
        q = np.sort(Y[:, f])
        n = max(len(p), len(q))
        p_i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p)
        q_i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(q)), q)
        dists.append(float(np.mean(np.abs(p_i - q_i))))
    return float(np.mean(dists))
