"""
RAPT (Regime-Aware Policy Transfer) System Module for Exp 9 (Experiment 1 Version)
Isolates historical policy reuse vs event-driven retraining.
"""

import copy
import time
import numpy as np
from models import create_base_ensemble

class RegimeCheckpoint:
    """
    Checkpoint container for a stored regime policy.
    """
    def __init__(self, regime_id, ensemble, weights, window_id):
        self.regime_id = regime_id
        # Deep copy ensemble model state to avoid pointer mutation
        self.ensemble = copy.deepcopy(ensemble)
        self.weights = list(weights)
        self.created_window = window_id
        self.last_accessed_window = window_id
        self.observation_count = 1

class RAPTSystem:
    """
    RAPT Policy-Transfer Controller for Experiment 1.
    
    Ablation Modes:
      - 'full': Transfer checkpointed base models + learned ensemble weights.
      - 'no_weights' (Ablation D): Keep checkpointed base models, reset ensemble weights to uniform [0.5, 0.5].
      - 'weights_only' (Ablation E): Reuse learned ensemble weights, but reset base models to fresh state.
    """
    def __init__(self, seed=42, mode='full', enable_calibration=True):
        self.seed = seed
        self.mode = mode
        self.enable_calibration = enable_calibration
        
        self.repository = {}  # regime_id -> RegimeCheckpoint
        self.current_regime_id = None
        self.active_ensemble = None
        
        self.reused_policy_count = 0
        self.created_policy_count = 0
        self.cumulative_cpu_time = 0.0
        self.cumulative_wall_time = 0.0
        self.trees_trained_count = 0
        self.trees_reused_count = 0
        
    def fit_initial(self, regime_id, X_init, y_init, window_id=0):
        """Initial training on first regime segment."""
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()
        
        self.active_ensemble = create_base_ensemble(seed=self.seed)
        self.active_ensemble.fit(X_init, y_init)
        
        t_end_cpu = time.process_time()
        t_end_wall = time.perf_counter()
        
        cpu_spent = t_end_cpu - t_start_cpu
        wall_spent = t_end_wall - t_start_wall
        
        self.cumulative_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        self.trees_trained_count += self.active_ensemble.get_num_trees()
        
        # Save checkpoint
        checkpoint = RegimeCheckpoint(
            regime_id=regime_id,
            ensemble=self.active_ensemble,
            weights=[0.5, 0.5],
            window_id=window_id
        )
        self.repository[regime_id] = checkpoint
        self.current_regime_id = regime_id
        self.created_policy_count += 1
        
        return cpu_spent, wall_spent

    def handle_regime_transition(self, new_regime_id, window_id, X_buffer=None, y_buffer=None):
        """
        Invoked when a regime boundary is encountered.
        Checks if a historical policy checkpoint exists for new_regime_id.
        """
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()
        
        cpu_spent = 0.0
        wall_spent = 0.0
        reused = False
        
        if new_regime_id in self.repository:
            # RETRIEVE HISTORICAL CHECKPOINT
            ckpt = self.repository[new_regime_id]
            ckpt.last_accessed_window = window_id
            ckpt.observation_count += 1
            reused = True
            self.reused_policy_count += 1
            
            if self.mode == 'full':
                # Full policy transfer: models + weights
                self.active_ensemble = copy.deepcopy(ckpt.ensemble)
                self.active_ensemble.weights = list(ckpt.weights)
                self.trees_reused_count += self.active_ensemble.get_num_trees()
                
            elif self.mode == 'no_weights':
                # Ablation D: Keep models, reset weights to uniform
                self.active_ensemble = copy.deepcopy(ckpt.ensemble)
                self.active_ensemble.weights = [0.5, 0.5]
                self.trees_reused_count += self.active_ensemble.get_num_trees()
                
            elif self.mode == 'weights_only':
                # Ablation E: Fresh models, reuse learned weights
                self.active_ensemble = create_base_ensemble(seed=self.seed + window_id * 17)
                if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
                    self.active_ensemble.fit(np.array(X_buffer), np.array(y_buffer))
                    self.trees_trained_count += self.active_ensemble.get_num_trees()
                self.active_ensemble.weights = list(ckpt.weights)
                
            # Optional lightweight calibration / reweighting
            if self.enable_calibration and X_buffer is not None and len(X_buffer) >= 20:
                # Calibrate soft-voting weights based on recent performance
                p_rf = self.active_ensemble.rf.predict_proba(np.array(X_buffer[-50:]))
                p_et = self.active_ensemble.et.predict_proba(np.array(X_buffer[-50:]))
                y_sub = np.array(y_buffer[-50:])
                
                # Accuracy-proportional weight update
                acc_rf = np.mean(np.argmax(p_rf, axis=1) == y_sub)
                acc_et = np.mean(np.argmax(p_et, axis=1) == y_sub)
                
                sum_acc = acc_rf + acc_et
                if sum_acc > 0:
                    new_w = [acc_rf / sum_acc, acc_et / sum_acc]
                    self.active_ensemble.weights = new_w
                    
        else:
            # NEW REGIME: Train new policy from historical buffer
            self.created_policy_count += 1
            self.active_ensemble = create_base_ensemble(seed=self.seed + window_id * 19)
            if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
                self.active_ensemble.fit(np.array(X_buffer), np.array(y_buffer))
                self.trees_trained_count += self.active_ensemble.get_num_trees()
                
            checkpoint = RegimeCheckpoint(
                regime_id=new_regime_id,
                ensemble=self.active_ensemble,
                weights=list(self.active_ensemble.weights),
                window_id=window_id
            )
            self.repository[new_regime_id] = checkpoint
            
        self.current_regime_id = new_regime_id
        
        t_end_cpu = time.process_time()
        t_end_wall = time.perf_counter()
        
        cpu_spent = t_end_cpu - t_start_cpu
        wall_spent = t_end_wall - t_start_wall
        
        self.cumulative_cpu_time += cpu_spent
        self.cumulative_wall_time += wall_spent
        
        return reused, cpu_spent, wall_spent

    def predict(self, X_win):
        """Predict class labels for current window."""
        return self.active_ensemble.predict(X_win)

    def predict_proba(self, X_win):
        """Predict class probabilities for current window."""
        return self.active_ensemble.predict_proba(X_win)
