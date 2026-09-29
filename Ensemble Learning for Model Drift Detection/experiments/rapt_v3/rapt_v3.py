"""
RAPT-v3 Controller Module — Low-Cost Champion–Challenger Adaptation
Extends RAPT-E with a principled, zero-retraining Champion-Challenger transfer mechanism.

Core Mechanism:
  1. Retrieve historical regime checkpoint -> CHAMPION policy.
  2. Form small, past-only post-transition telemetry buffer (N_buffer in {50, 100, 250}).
  3. Compute base model EWMA performance scores and momentum-blended weights -> CHALLENGER policy.
  4. Compare Champion vs Challenger on adaptation buffer.
  5. Select Challenger ONLY if score_challenger > score_champion, else retain Champion.
  6. Zero RF/ET tree refitting during champion-challenger adaptation.
"""

import copy
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score

class HeterogeneousEnsembleV3:
    """
    Lightweight heterogeneous ensemble combining RandomForest and ExtraTrees.
    Supports weighted soft-voting. Base classifiers are fixed during weight adaptation.
    """
    def __init__(self, seed=42, n_estimators=50, max_depth=7, weights=None):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.weights = list(weights) if weights is not None else [0.5, 0.5]
        
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=1
        )
        self.et = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=1
        )
        self.is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        """Fit both base classifiers on training data."""
        self.classes_ = np.unique(y)
        self.rf.fit(X, y)
        self.et.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Compute weighted soft-voting probability distribution over classes."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted.")
            
        p_rf = self.rf.predict_proba(X)
        p_et = self.et.predict_proba(X)
        
        # Align class prediction shapes if needed
        if len(self.rf.classes_) != len(self.classes_):
            p_rf_full = np.zeros((X.shape[0], len(self.classes_)))
            for idx, c in enumerate(self.rf.classes_):
                c_pos = np.where(self.classes_ == c)[0][0]
                p_rf_full[:, c_pos] = p_rf[:, idx]
            p_rf = p_rf_full

        if len(self.et.classes_) != len(self.classes_):
            p_et_full = np.zeros((X.shape[0], len(self.classes_)))
            for idx, c in enumerate(self.et.classes_):
                c_pos = np.where(self.classes_ == c)[0][0]
                p_et_full[:, c_pos] = p_et[:, idx]
            p_et = p_et_full
            
        w1, w2 = self.weights[0], self.weights[1]
        sum_w = w1 + w2
        if sum_w <= 0:
            w1, w2 = 0.5, 0.5
            sum_w = 1.0
        w1_norm = w1 / sum_w
        w2_norm = w2 / sum_w
        
        probs = w1_norm * p_rf + w2_norm * p_et
        return probs

    def predict(self, X):
        """Predict class labels with highest soft-voting probability."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_num_trees(self):
        """Return total number of decision trees in ensemble."""
        return len(self.rf.estimators_) + len(self.et.estimators_)

def create_base_ensemble_v3(seed=42, n_estimators=50, max_depth=7):
    return HeterogeneousEnsembleV3(seed=seed, n_estimators=n_estimators, max_depth=max_depth)


class RegimeCheckpointV3:
    """
    Checkpoint container for a stored regime policy in RAPT-v3.
    """
    def __init__(self, regime_id, ensemble, weights, window_id):
        self.regime_id = regime_id
        self.ensemble = copy.deepcopy(ensemble)
        self.weights = list(weights)
        self.created_window = window_id
        self.last_accessed_window = window_id
        self.observation_count = 1
        self.historical_score = 1.0


class RAPTv3System:
    """
    RAPT-v3 Controller implementing Low-Cost Champion-Challenger Transfer Adaptation.
    """
    def __init__(self, seed=42, variant='RAPT-v3-50', beta=0.90, alpha=0.10, buffer_size=50, confidence_gate=0.99):
        self.seed = seed
        self.variant = variant
        self.beta = beta
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.confidence_gate = confidence_gate
        
        self.repository = {}  # regime_id -> RegimeCheckpointV3
        self.current_regime_id = None
        self.active_ensemble = None
        
        # Statistics & Metrics
        self.reused_policy_count = 0
        self.created_policy_count = 0
        self.champion_selections = 0
        self.challenger_selections = 0
        self.cumulative_cpu_time = 0.0
        self.cumulative_wall_time = 0.0
        
        # Sub-component CPU tracking
        self.checkpoint_lookup_cpu = 0.0
        self.buffer_eval_cpu = 0.0
        self.weight_calc_cpu = 0.0
        self.challenger_selection_cpu = 0.0
        
        # Internal EWMA performance tracker
        self.ewma_scores = [0.5, 0.5]
        self.trace_records = []

    def fit_initial(self, regime_id, X_init, y_init, window_id=0):
        """Initial training on first regime segment."""
        t_start_cpu = time.process_time()
        t_start_wall = time.perf_counter()
        
        self.active_ensemble = create_base_ensemble_v3(seed=self.seed)
        self.active_ensemble.fit(X_init, y_init)
        
        cpu_spent = time.process_time() - t_start_cpu
        wall_spent = time.perf_counter() - t_start_wall
        
        self.cumulative_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        
        ckpt = RegimeCheckpointV3(regime_id, self.active_ensemble, [0.5, 0.5], window_id)
        
        # Compute baseline score on initial training data
        preds = self.active_ensemble.predict(X_init)
        ckpt.historical_score = float(accuracy_score(y_init, preds))
        
        self.repository[regime_id] = ckpt
        self.current_regime_id = regime_id
        self.created_policy_count += 1
        
        return cpu_spent, wall_spent

    def _eval_policy(self, ensemble, X, y):
        """Helper to compute macro F1 or accuracy on buffer."""
        preds = ensemble.predict(X)
        if len(np.unique(y)) > 1:
            return float(f1_score(y, preds, average='macro', zero_division=0))
        return float(accuracy_score(y, preds))

    def handle_regime_transition(self, new_regime_id, window_id, X_buffer=None, y_buffer=None, dataset_label="9B", seed=42):
        """
        Invoked on regime transition boundary.
        Executes Champion-Challenger evaluation if new_regime_id is in repository.
        """
        t_start_cpu = time.process_time()
        t_start_wall = time.perf_counter()
        
        reused = False
        selected_policy = 'new_regime'
        
        w_champion = [0.5, 0.5]
        w_recent = [0.5, 0.5]
        w_challenger = [0.5, 0.5]
        score_champion = 0.0
        score_challenger = 0.0
        recent_scores = [0.5, 0.5]
        
        if new_regime_id in self.repository:
            # 1. RETRIEVE CHAMPION POLICY
            t0_lookup = time.process_time()
            ckpt = self.repository[new_regime_id]
            ckpt.last_accessed_window = window_id
            ckpt.observation_count += 1
            reused = True
            self.reused_policy_count += 1
            
            champion_ensemble = copy.deepcopy(ckpt.ensemble)
            w_champion = list(ckpt.weights)
            champion_ensemble.weights = list(w_champion)
            self.checkpoint_lookup_cpu += (time.process_time() - t0_lookup)
            
            if self.variant in ['RAPT-E', 'v1']:
                # Standard RAPT-E (RAPT-v1 reference): no champion-challenger adaptation
                self.active_ensemble = champion_ensemble
                selected_policy = 'champion'
                self.champion_selections += 1
                score_champion = ckpt.historical_score
                score_challenger = ckpt.historical_score
            else:
                # Champion-Challenger Transfer Adaptation (RAPT-v3)
                if X_buffer is not None and y_buffer is not None and len(X_buffer) >= 10:
                    t0_buf = time.process_time()
                    X_sub = np.array(X_buffer[-self.buffer_size:])
                    y_sub = np.array(y_buffer[-self.buffer_size:])
                    self.buffer_eval_cpu += (time.process_time() - t0_buf)
                    
                    # A. Evaluate Champion on adaptation buffer
                    t0_sel = time.process_time()
                    score_champion = self._eval_policy(champion_ensemble, X_sub, y_sub)
                    self.challenger_selection_cpu += (time.process_time() - t0_sel)
                    
                    # B. Check confidence gate ratio C
                    confidence_ratio = score_champion / (ckpt.historical_score + 1e-9)
                    
                    if confidence_ratio >= self.confidence_gate:
                        # Confidence gate passed -> Pure Champion reuse
                        self.active_ensemble = champion_ensemble
                        selected_policy = 'champion'
                        self.champion_selections += 1
                        score_challenger = score_champion
                        w_challenger = list(w_champion)
                    else:
                        # C. Construct Challenger via Weight Adaptation (Zero Tree Refitting!)
                        t0_w = time.process_time()
                        p_rf = champion_ensemble.rf.predict_proba(X_sub)
                        p_et = champion_ensemble.et.predict_proba(X_sub)
                        
                        acc_rf = float(accuracy_score(y_sub, np.argmax(p_rf, axis=1)))
                        acc_et = float(accuracy_score(y_sub, np.argmax(p_et, axis=1)))
                        recent_scores = [acc_rf, acc_et]
                        
                        # EWMA Score Update
                        self.ewma_scores[0] = self.beta * self.ewma_scores[0] + (1.0 - self.beta) * acc_rf
                        self.ewma_scores[1] = self.beta * self.ewma_scores[1] + (1.0 - self.beta) * acc_et
                        
                        sum_s = self.ewma_scores[0] + self.ewma_scores[1]
                        if sum_s > 0:
                            w_recent = [self.ewma_scores[0] / sum_s, self.ewma_scores[1] / sum_s]
                        else:
                            w_recent = [0.5, 0.5]
                            
                        # Momentum Blend Weight Update
                        w_challenger = [
                            (1.0 - self.alpha) * w_champion[0] + self.alpha * w_recent[0],
                            (1.0 - self.alpha) * w_champion[1] + self.alpha * w_recent[1]
                        ]
                        
                        challenger_ensemble = copy.deepcopy(champion_ensemble)
                        challenger_ensemble.weights = list(w_challenger)
                        self.weight_calc_cpu += (time.process_time() - t0_w)
                        
                        # D. Champion vs Challenger Selection
                        t0_sel = time.process_time()
                        score_challenger = self._eval_policy(challenger_ensemble, X_sub, y_sub)
                        
                        if score_challenger > score_champion:
                            self.active_ensemble = challenger_ensemble
                            selected_policy = 'challenger'
                            self.challenger_selections += 1
                        else:
                            self.active_ensemble = champion_ensemble
                            selected_policy = 'champion'
                            self.champion_selections += 1
                        self.challenger_selection_cpu += (time.process_time() - t0_sel)
                else:
                    self.active_ensemble = champion_ensemble
                    selected_policy = 'champion'
                    self.champion_selections += 1
                    score_champion = ckpt.historical_score
                    score_challenger = ckpt.historical_score
                    
        else:
            # NEW REGIME: Train new policy on full available historical buffer
            self.created_policy_count += 1
            self.active_ensemble = create_base_ensemble_v3(seed=self.seed + window_id * 19)
            if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
                X_full = np.array(X_buffer)
                y_full = np.array(y_buffer)
                self.active_ensemble.fit(X_full, y_full)
                score_hist = self._eval_policy(self.active_ensemble, X_full, y_full)
            else:
                score_hist = 1.0
                
            ckpt = RegimeCheckpointV3(new_regime_id, self.active_ensemble, [0.5, 0.5], window_id)
            ckpt.historical_score = score_hist
            self.repository[new_regime_id] = ckpt
            selected_policy = 'new_regime'
            
        self.current_regime_id = new_regime_id
        
        cpu_spent = time.process_time() - t_start_cpu
        wall_spent = time.perf_counter() - t_start_wall
        self.cumulative_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        
        # Log diagnostic trace for champion_challenger_trace.csv
        if reused:
            self.trace_records.append({
                'dataset': dataset_label,
                'seed': seed,
                'variant': self.variant,
                'window_id': window_id,
                'regime_id': new_regime_id,
                'buffer_size': self.buffer_size if X_buffer is not None else 0,
                'champion_w_rf': w_champion[0],
                'champion_w_et': w_champion[1],
                'recent_score_rf': recent_scores[0],
                'recent_score_et': recent_scores[1],
                'challenger_w_rf': w_challenger[0],
                'challenger_w_et': w_challenger[1],
                'champion_score': score_champion,
                'challenger_score': score_challenger,
                'selected_policy': selected_policy,
                'adaptation_cpu': cpu_spent
            })
            
        return reused, cpu_spent, selected_policy

    def predict(self, X_win):
        return self.active_ensemble.predict(X_win)

    def predict_proba(self, X_win):
        return self.active_ensemble.predict_proba(X_win)
