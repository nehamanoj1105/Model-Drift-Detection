"""
================================================================================
RAPT POLICY TRANSFER — SOFT INTERPOLATION VS. HARD NEAREST RETRIEVAL
================================================================================
Implements:
  1. Base candidate model creation (RF, ET, GB matching Experiment 4 parameters)
  2. Stored regime policies for Regime 1 and Regime 2
  3. Transfer strategies:
     - Soft Interpolation: Blends stored ensemble policies proportional to similarity
     - Hard Nearest Retrieval: Selects the single closest stored policy as-is
  4. Prequential streaming loop with dual-trigger adaptation tracking
================================================================================
"""

import sys
import os
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure local rapt package has precedence for config import
_rapt_dir = os.path.dirname(os.path.abspath(__file__))
if _rapt_dir not in sys.path:
    sys.path.insert(0, _rapt_dir)

try:
    from .config import (
        BASE_MODEL_PARAMS,
        DECISION_THRESHOLD,
        DRIFT_PERF_DROP_THRESHOLD,
        BASELINE_EXPECTED_F1,
        BUFFER_CAP,
        WINDOW_SIZE
    )
except ImportError:
    from config import (
        BASE_MODEL_PARAMS,
        DECISION_THRESHOLD,
        DRIFT_PERF_DROP_THRESHOLD,
        BASELINE_EXPECTED_F1,
        BUFFER_CAP,
        WINDOW_SIZE
    )

# Reference experiment4 for exact resource monitor (appended, not prepended)
exp4_dir = os.path.abspath(os.path.join(_rapt_dir, '..', '..', 'experiment4'))
if exp4_dir not in sys.path:
    sys.path.append(exp4_dir)

from resource_monitor import measure_execution


def create_candidate_models(seed):
    """
    Factory function producing the 3 base models matching Experiment 4.
    """
    p_rf = dict(BASE_MODEL_PARAMS['RandomForest'])
    p_et = dict(BASE_MODEL_PARAMS['ExtraTrees'])
    p_gb = dict(BASE_MODEL_PARAMS['GradientBoosting'])
    return {
        'RandomForest': RandomForestClassifier(random_state=seed, **p_rf),
        'ExtraTrees': ExtraTreesClassifier(random_state=seed, **p_et),
        'GradientBoosting': GradientBoostingClassifier(random_state=seed, **p_gb),
    }


class StoredRegimeRepository:
    """
    Maintains stored expert models and policies for known physical regimes R1 and R2.
    """
    def __init__(self, seed):
        self.seed = seed
        self.models_r1 = create_candidate_models(seed)
        self.models_r2 = create_candidate_models(seed)
        self.is_fitted = False

    def train_regimes(self, X_r1, y_r1, X_r2, y_r2):
        """Fit specialized models on offline reference data for R1 and R2."""
        for m in self.models_r1.values():
            m.fit(X_r1, y_r1)
        for m in self.models_r2.values():
            m.fit(X_r2, y_r2)
        self.is_fitted = True

    def predict_regime_proba(self, regime_id, X):
        """Average soft voting probability from a specific regime expert ensemble."""
        models = self.models_r1 if regime_id == 1 else self.models_r2
        probs = [m.predict_proba(X)[:, 1] for m in models.values()]
        return np.mean(probs, axis=0)


def evaluate_streaming_transfer(X_stream, y_stream, alpha, strategy, repository, seed):
    """
    Execute streaming prequential evaluation over 10 windows (5,000 samples) of D_alpha.
    
    Parameters:
      - X_stream, y_stream: Telemetry stream for D_alpha
      - alpha: float in [0.0, 1.0]
      - strategy: 'soft' or 'hard'
      - repository: StoredRegimeRepository
      - seed: random seed for reproducibility
      
    Returns:
      - dict with prequential metrics, adaptation cost, and per-window records
    """
    n_samples = len(X_stream)
    n_windows = n_samples // WINDOW_SIZE
    
    # 1. Compute Policy Weights based on Similarity to R1 (1 - alpha) vs R2 (alpha)
    if strategy == 'soft':
        w1 = 1.0 - alpha
        w2 = alpha
    elif strategy == 'hard':
        # Hard nearest-regime retrieval: strictly pick closest stored regime
        if alpha <= 0.5:
            w1 = 1.0
            w2 = 0.0
        else:
            w1 = 0.0
            w2 = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    buf_X = []
    buf_y = []
    retrain_events = 0
    samples_adapted = 0
    cumulative_cpu_time = 0.0
    cumulative_wall_time = 0.0
    
    window_f1_list = []
    window_acc_list = []
    window_prec_list = []
    window_rec_list = []
    window_auc_list = []
    
    for w in range(n_windows):
        idx_s = w * WINDOW_SIZE
        idx_e = idx_s + WINDOW_SIZE
        X_win = X_stream[idx_s:idx_e]
        y_win = y_stream[idx_s:idx_e]
        
        # A. Prequential Inference using transferred policy
        p1 = repository.predict_regime_proba(1, X_win)
        p2 = repository.predict_regime_proba(2, X_win)
        p_win = w1 * p1 + w2 * p2
        
        pred_win = (p_win >= DECISION_THRESHOLD).astype(int)
        
        f1 = float(f1_score(y_win, pred_win, zero_division=0))
        acc = float(accuracy_score(y_win, pred_win))
        prec = float(precision_score(y_win, pred_win, zero_division=0))
        rec = float(recall_score(y_win, pred_win, zero_division=0))
        try:
            auc_score = float(roc_auc_score(y_win, p_win)) if len(np.unique(y_win)) > 1 else 0.5
        except Exception:
            auc_score = 0.5
            
        window_f1_list.append(f1)
        window_acc_list.append(acc)
        window_prec_list.append(prec)
        window_rec_list.append(rec)
        window_auc_list.append(auc_score)
        
        # B. Sliding Buffer Update
        buf_X.extend(X_win)
        buf_y.extend(y_win)
        if len(buf_X) > BUFFER_CAP:
            buf_X = buf_X[-BUFFER_CAP:]
            buf_y = buf_y[-BUFFER_CAP:]
            
        # C. Event-Driven Adaptation Trigger Check
        # Trigger adaptation if window F1 degrades by >= 0.12 below expected baseline
        perf_drop = BASELINE_EXPECTED_F1 - f1
        if perf_drop >= DRIFT_PERF_DROP_THRESHOLD:
            retrain_events += 1
            train_X = np.array(buf_X[-2000:])
            train_y = np.array(buf_y[-2000:])
            samples_adapted += len(train_X)
            
            def _adapt():
                adapted_models = create_candidate_models(seed + w * 7)
                for m in adapted_models.values():
                    m.fit(train_X, train_y)
                return adapted_models
                
            _, res = measure_execution(_adapt)
            cumulative_cpu_time += res['total_cpu_time']
            cumulative_wall_time += res['wall_clock_time']
            
    return {
        'seed': seed,
        'alpha': alpha,
        'strategy': strategy,
        'f1_mean': float(np.mean(window_f1_list)),
        'accuracy_mean': float(np.mean(window_acc_list)),
        'precision_mean': float(np.mean(window_prec_list)),
        'recall_mean': float(np.mean(window_rec_list)),
        'auc_mean': float(np.mean(window_auc_list)),
        'retrain_events': retrain_events,
        'retrain_rate': float(retrain_events / n_windows),
        'samples_adapted': samples_adapted,
        'adaptation_cpu_time': float(cumulative_cpu_time),
        'adaptation_wall_time': float(cumulative_wall_time),
        'window_f1_scores': window_f1_list,
    }
