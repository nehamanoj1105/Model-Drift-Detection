"""
RAPT-v2 Fixed System Module — Low-Cost Performance Improvement (Corrected Architecture)
Fixes initial window prefix misalignment and buffer truncation bugs from RAPT-v2 audit.
Implements Regime-Aware Policy Transfer with modular cheap adaptation mechanisms:
  - Mechanism A: Exponentially Weighted Moving Average (EWMA) Ensemble Reweighting
  - Mechanism B: Momentum-Based Policy Weight Adaptation
  - Mechanism C: Confidence-Gated Adaptation
  - Mechanism D: Cheap Champion/Challenger Validation Check
  - Mechanism E: Bounded Adaptation Buffer (N_buffer=50 for weight adaptation)
  - Mechanism F: Performance-Triggered Escalation / Fallback Monitor
  - Mechanism G: Hierarchical Compute-Aware Adaptation Hierarchy (Level 0..3)
  - Mechanism H: Historical Policy Freshness / Trust Score Tracking
"""

import copy
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class HeterogeneousEnsembleV2Fixed:
    """
    Lightweight heterogeneous ensemble combining RandomForest and ExtraTrees.
    Supports weighted soft-voting, individual model refitting, and partial updates.
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

    def fit_partial(self, X, y, model_idx=0):
        """Refit only ONE base estimator (0: RF, 1: ET) for Level 2 partial updates."""
        self.classes_ = np.unique(y)
        if model_idx == 0:
            self.rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed + 101,
                n_jobs=1
            )
            self.rf.fit(X, y)
        else:
            self.et = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed + 102,
                n_jobs=1
            )
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

def create_base_ensemble_v2_fixed(seed=42, n_estimators=50, max_depth=7):
    return HeterogeneousEnsembleV2Fixed(seed=seed, n_estimators=n_estimators, max_depth=max_depth)


class RegimeCheckpointV2Fixed:
    """
    Enhanced Checkpoint Container for RAPT-v2 Fixed.
    """
    def __init__(self, regime_id, ensemble, weights, window_id):
        self.regime_id = regime_id
        self.ensemble = copy.deepcopy(ensemble)
        self.weights = list(weights)
        self.created_window = window_id
        self.last_accessed_window = window_id
        self.observation_count = 1
        
        # Mechanism H: Trust and Freshness Score
        self.successful_transfers = 0
        self.failed_transfers = 0
        
    def get_trust_score(self):
        return (self.successful_transfers + 1.0) / (self.successful_transfers + self.failed_transfers + 2.0)


class RAPTv2SystemFixed:
    """
    RAPT-v2 Fixed Controller supporting modular low-cost adaptation mechanisms:
      - variant: 'v1', 'RAPT-A', 'RAPT-B', 'RAPT-C', 'RAPT-D', 'RAPT-E'
      - beta: EWMA decay factor (default 0.9)
      - alpha: Momentum blend factor (default 0.2)
      - confidence_k: Error threshold multiplier (default 1.5)
      - buffer_capacity: Max recent telemetry windows for weight adaptation (default 50)
      - patience: Number of consecutive degraded windows before fallback escalation (default 5)
    """
    def __init__(self, seed=42, variant='RAPT-E', beta=0.9, alpha=0.2, confidence_k=1.5, buffer_capacity=50, patience=5):
        self.seed = seed
        self.variant = variant
        self.beta = beta
        self.alpha = alpha
        self.confidence_k = confidence_k
        self.buffer_capacity = buffer_capacity
        self.patience = patience
        
        self.repository = {}  # regime_id -> RegimeCheckpointV2Fixed
        self.current_regime_id = None
        self.active_ensemble = None
        
        # Performance & Hierarchy tracking
        self.reused_policy_count = 0
        self.created_policy_count = 0
        self.cumulative_cpu_time = 0.0
        self.cumulative_wall_time = 0.0
        
        # Hierarchy counts (Mechanism G)
        self.level_counts = {
            'level_0_pure_reuse': 0,
            'level_1_weight_adapt': 0,
            'level_2_partial_update': 0,
            'level_3_full_retrain': 0
        }
        
        # Rolling performance monitor for fallback (Mechanism F)
        self.recent_errors = []
        self.consecutive_failures = 0
        self.ewma_scores = [0.5, 0.5] # Model 0 (RF), Model 1 (ET)

    def fit_initial(self, regime_id, X_init, y_init, window_id=0):
        """Fit clean initial ensemble on prefix and save first checkpoint."""
        t_start_cpu = time.process_time()
        t_start_wall = time.perf_counter()
        
        self.active_ensemble = create_base_ensemble_v2_fixed(seed=self.seed)
        self.active_ensemble.fit(X_init, y_init)
        
        cpu_spent = time.process_time() - t_start_cpu
        wall_spent = time.perf_counter() - t_start_wall
        
        self.cumulative_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        
        ckpt = RegimeCheckpointV2Fixed(regime_id, self.active_ensemble, [0.5, 0.5], window_id)
        self.repository[regime_id] = ckpt
        self.current_regime_id = regime_id
        self.created_policy_count += 1
        self.level_counts['level_3_full_retrain'] += 1
        
        return cpu_spent, wall_spent

    def handle_regime_transition(self, new_regime_id, window_id, X_buffer=None, y_buffer=None):
        """
        Invoked when regime boundary occurs.
        Restores historical policy checkpoint or trains new checkpoint on full X_buffer (up to 500 samples).
        """
        t_start_cpu = time.process_time()
        
        cpu_spent = 0.0
        reused = False
        level_used = 'level_0_pure_reuse'
        
        if new_regime_id in self.repository:
            ckpt = self.repository[new_regime_id]
            ckpt.last_accessed_window = window_id
            ckpt.observation_count += 1
            reused = True
            self.reused_policy_count += 1
            
            # Restore base models and weights
            self.active_ensemble = copy.deepcopy(ckpt.ensemble)
            hist_weights = list(ckpt.weights)
            
            if self.variant == 'v1':
                # Original RAPT: policy transfer + optional 50-sample calibration
                if X_buffer is not None and y_buffer is not None and len(X_buffer) >= 20:
                    X_sub = np.array(X_buffer[-50:])
                    y_sub = np.array(y_buffer[-50:])
                    
                    p_rf = self.active_ensemble.rf.predict_proba(X_sub)
                    p_et = self.active_ensemble.et.predict_proba(X_sub)
                    
                    acc_rf = np.mean(np.argmax(p_rf, axis=1) == y_sub)
                    acc_et = np.mean(np.argmax(p_et, axis=1) == y_sub)
                    
                    sum_acc = acc_rf + acc_et
                    if sum_acc > 0:
                        self.active_ensemble.weights = [acc_rf / sum_acc, acc_et / sum_acc]
                    else:
                        self.active_ensemble.weights = hist_weights
                else:
                    self.active_ensemble.weights = hist_weights
                level_used = 'level_0_pure_reuse'
                
            elif self.variant in ['RAPT-A', 'RAPT-B', 'RAPT-C', 'RAPT-D', 'RAPT-E']:
                # Mechanism A: EWMA Model Performance Scoring on Recent Buffer (50 samples)
                if X_buffer is not None and y_buffer is not None and len(X_buffer) >= 10:
                    X_sub = np.array(X_buffer[-self.buffer_capacity:])
                    y_sub = np.array(y_buffer[-self.buffer_capacity:])
                    
                    p_rf = self.active_ensemble.rf.predict_proba(X_sub)
                    p_et = self.active_ensemble.et.predict_proba(X_sub)
                    
                    acc_rf = np.mean(np.argmax(p_rf, axis=1) == y_sub)
                    acc_et = np.mean(np.argmax(p_et, axis=1) == y_sub)
                    
                    # Update EWMA scores (Mechanism A)
                    self.ewma_scores[0] = self.beta * self.ewma_scores[0] + (1 - self.beta) * acc_rf
                    self.ewma_scores[1] = self.beta * self.ewma_scores[1] + (1 - self.beta) * acc_et
                    
                    sum_s = self.ewma_scores[0] + self.ewma_scores[1]
                    w_recent = [self.ewma_scores[0] / sum_s, self.ewma_scores[1] / sum_s] if sum_s > 0 else [0.5, 0.5]
                    
                    # Mechanism B: Momentum Weight Update
                    if self.variant in ['RAPT-B', 'RAPT-C', 'RAPT-D', 'RAPT-E']:
                        w_new = [
                            (1 - self.alpha) * hist_weights[0] + self.alpha * w_recent[0],
                            (1 - self.alpha) * hist_weights[1] + self.alpha * w_recent[1]
                        ]
                    else:
                        w_new = w_recent
                        
                    # Mechanism C & D: Confidence Gate & Champion/Challenger Validation
                    should_update = True
                    if self.variant in ['RAPT-C', 'RAPT-D', 'RAPT-E']:
                        recent_err = 1.0 - np.mean([acc_rf, acc_et])
                        if len(self.recent_errors) >= 5:
                            mu_e = np.mean(self.recent_errors)
                            sigma_e = np.std(self.recent_errors)
                            if recent_err <= (mu_e + self.confidence_k * max(sigma_e, 0.05)):
                                should_update = False  # Pure reuse works well!
                                
                    if should_update:
                        self.active_ensemble.weights = w_new
                        level_used = 'level_1_weight_adapt'
                    else:
                        self.active_ensemble.weights = hist_weights
                        level_used = 'level_0_pure_reuse'
                else:
                    self.active_ensemble.weights = hist_weights
                    level_used = 'level_0_pure_reuse'
                    
        else:
            # NEW REGIME: Train new checkpoint on FULL historical buffer (up to 500 samples!)
            self.created_policy_count += 1
            self.active_ensemble = create_base_ensemble_v2_fixed(seed=self.seed + window_id * 19)
            if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
                X_full = np.array(X_buffer) # Full historical buffer!
                y_full = np.array(y_buffer)
                self.active_ensemble.fit(X_full, y_full)
                
            ckpt = RegimeCheckpointV2Fixed(new_regime_id, self.active_ensemble, list(self.active_ensemble.weights), window_id)
            self.repository[new_regime_id] = ckpt
            level_used = 'level_3_full_retrain'
            
        self.current_regime_id = new_regime_id
        self.level_counts[level_used] += 1
        
        cpu_spent = time.process_time() - t_start_cpu
        self.cumulative_cpu_time += cpu_spent
        
        return reused, cpu_spent, level_used

    def check_and_escalate_fallback(self, X_buffer, y_buffer, recent_error):
        """
        Mechanisms F & G: Performance-Triggered Escalation / Fallback Retraining.
        """
        if self.variant not in ['RAPT-D', 'RAPT-E']:
            return False, 'level_0_pure_reuse', 0.0
            
        self.recent_errors.append(recent_error)
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)
            
        escalate = False
        if len(self.recent_errors) >= 5:
            mu_e = np.mean(self.recent_errors)
            sigma_e = np.std(self.recent_errors)
            if recent_error > (mu_e + 2.0 * max(sigma_e, 0.05)):
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
            
        if self.consecutive_failures >= self.patience:
            t0_cpu = time.process_time()
            X_sub = np.array(X_buffer[-self.buffer_capacity:])
            y_sub = np.array(y_buffer[-self.buffer_capacity:])
            
            p_rf = self.active_ensemble.rf.predict_proba(X_sub)
            p_et = self.active_ensemble.et.predict_proba(X_sub)
            acc_rf = np.mean(np.argmax(p_rf, axis=1) == y_sub)
            acc_et = np.mean(np.argmax(p_et, axis=1) == y_sub)
            
            weakest_idx = 0 if acc_rf <= acc_et else 1
            
            # Step 1 / Level 2: Partial Model Refit (weakest model only)
            self.active_ensemble.fit_partial(X_sub, y_sub, model_idx=weakest_idx)
            
            # Recheck error
            p_new = self.active_ensemble.predict(X_sub)
            new_err = 1.0 - np.mean(p_new == y_sub)
            
            if new_err > (np.mean(self.recent_errors) + 0.1):
                # Step 2 / Level 3: Full Ensemble Retrain
                X_full = np.array(X_buffer)
                y_full = np.array(y_buffer)
                self.active_ensemble.fit(X_full, y_full)
                level_used = 'level_3_full_retrain'
            else:
                level_used = 'level_2_partial_update'
                
            self.consecutive_failures = 0
            self.recent_errors = [new_err]
            
            cpu_spent = time.process_time() - t0_cpu
            self.cumulative_cpu_time += cpu_spent
            self.level_counts[level_used] += 1
            
            if self.current_regime_id in self.repository:
                self.repository[self.current_regime_id].failed_transfers += 1
                
            return True, level_used, cpu_spent
            
        return False, 'level_0_pure_reuse', 0.0

    def predict(self, X_win):
        return self.active_ensemble.predict(X_win)

    def predict_proba(self, X_win):
        return self.active_ensemble.predict_proba(X_win)
