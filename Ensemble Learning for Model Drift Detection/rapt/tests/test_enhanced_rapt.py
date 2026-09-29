"""
Unit tests for EnhancedHybridRAPT class and Dynamic Threshold Tuning.
"""

import numpy as np
import pytest
import sys
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_hybrid_rapt import EnhancedHybridRAPT


def test_dynamic_threshold_bounds():
    model = EnhancedHybridRAPT(base_tau=0.455, lambda_tau=0.15)
    
    # When similarity is 1.0 (known regime), threshold should be exactly base_tau (0.455)
    tau_known = model.get_dynamic_threshold(1.0)
    assert abs(tau_known - 0.455) < 1e-6
    
    # When similarity is 0.0 (novel regime), threshold should drop by lambda_tau (0.305 clipped to 0.32)
    tau_novel = model.get_dynamic_threshold(0.0)
    assert 0.30 <= tau_novel <= 0.35
    
    # Monotonicity check
    tau_mid = model.get_dynamic_threshold(0.5)
    assert tau_novel <= tau_mid <= tau_known


def test_enhanced_hybrid_predict_structure():
    model = EnhancedHybridRAPT(seed=42)
    X_init = np.random.randn(100, 3)
    y_init = np.random.randint(0, 2, 100)
    
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_init, y_init)
    et = ExtraTreesClassifier(n_estimators=5, random_state=42).fit(X_init, y_init)
    gb = GradientBoostingClassifier(n_estimators=5, random_state=42).fit(X_init, y_init)
    
    base_models = {'RandomForest': rf, 'ExtraTrees': et, 'GradientBoosting': gb}
    fp_dummy = np.array([0.1, 0.05, 0.02], dtype=np.float32)
    
    model.warm_start(fp_dummy, base_models, [1/3, 1/3, 1/3], X_init, y_init, window_id=0)
    
    X_win = np.random.randn(10, 3)
    p_hybrid, tau_t, p_rapt, p_online, beta, max_sim, closest_id, weights_dict = model.predict_enhanced(X_win, fp_dummy)
    
    assert len(p_hybrid) == 10
    assert 0.30 <= tau_t <= 0.50
    assert 0.0 <= beta <= 1.0
    assert 0.0 <= max_sim <= 1.0
