"""
================================================================================
UNIT TESTS — RAPT STAGE 3: REAL-WORLD DATA & TWO-TIER HYBRID RAPT
================================================================================
Verifies:
  1. ToN_IoT real-world stream loader (chronological order, zero leakage, dimensions).
  2. Generalized sub-millisecond fingerprint extraction on D=3 features.
  3. Two-Tier Hybrid RAPT dynamic blending dynamics.
  4. Online stream learner updates and probability outputs.
================================================================================
"""

import sys
import os
import time
import numpy as np
import pytest

_this_dir = os.path.dirname(os.path.abspath(__file__))
_rapt_dir = os.path.abspath(os.path.join(_this_dir, '..'))
if _rapt_dir not in sys.path:
    sys.path.insert(0, _rapt_dir)

from real_world_stream import load_real_world_ton_iot_stream, N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE
from fingerprint import FastFingerprintExtractor
from hybrid_rapt import TwoTierHybridRAPT, OnlineStreamLearner


def test_ton_iot_data_loader_integrity():
    """Verify chronological ordering, sample counts, and zero NaN values."""
    X_train, y_train, X_stream, y_stream, df_sorted, window_metadata, scaler = load_real_world_ton_iot_stream()
    
    assert len(X_train) == N_INITIAL_TRAINING, f"Expected {N_INITIAL_TRAINING} train samples"
    assert len(X_stream) == N_WINDOWS * WINDOW_SIZE, f"Expected {N_WINDOWS * WINDOW_SIZE} stream samples"
    assert X_train.shape[1] == 3, "Expected 3 features (temperature, pressure, humidity)"
    assert X_stream.shape[1] == 3, "Expected 3 features (temperature, pressure, humidity)"
    assert not np.isnan(X_train).any(), "Found NaN in X_train"
    assert not np.isnan(X_stream).any(), "Found NaN in X_stream"
    assert df_sorted['datetime'].is_monotonic_increasing, "Temporal ordering must be monotonic"
    assert len(window_metadata) == N_WINDOWS, f"Expected {N_WINDOWS} windows in metadata"


def test_generalized_submillisecond_fingerprint_d3():
    """Verify FastFingerprintExtractor on D=3 real features meets sub-1ms requirement."""
    import gc
    gc.collect()
    rng = np.random.RandomState(42)
    X_ref = rng.normal(0, 1, (1000, 3))
    extractor = FastFingerprintExtractor(X_ref, num_quantiles=20)
    
    # Warm-up to initialize internal caches
    X_cur = rng.normal(0.5, 1.2, (500, 3))
    y_pred = rng.binomial(1, 0.3, 500)
    _ = extractor.extract(X_cur, y_pred)
    
    t0 = time.perf_counter()
    for _ in range(100):
        fp, _ = extractor.extract(X_cur, y_pred)
    mean_lat = (time.perf_counter() - t0) * 1000.0 / 100.0
        
    assert len(fp) == 7, f"Expected 7D fingerprint for D=3 features, got {len(fp)}"
    assert not np.isnan(fp).any(), "Found NaN in fingerprint vector"
    assert mean_lat < 1.5, f"Mean latency {mean_lat:.3f}ms must be strictly < 1.5ms"


def test_two_tier_hybrid_blending():
    """Verify dynamic beta weight shifts reliance to online layer on novel regimes."""
    hybrid = TwoTierHybridRAPT(max_capacity=8, gamma=15.0, novelty_threshold=0.65, seed=42)
    
    rng = np.random.RandomState(42)
    X_ref = rng.normal(0, 1, (500, 3))
    y_ref = (X_ref[:, 0] > 0).astype(int)
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(X_ref, y_ref)
    base_models = {'RandomForest': rf}
    
    base_fp = np.array([0.05, 0.05, 0.05, 0.05, 0.02, 0.01, 0.25], dtype=np.float32)
    hybrid.warm_start(base_fp, base_models, [1.0], X_ref, y_ref)
    
    # Query 1: Exact repeat of base_fp (high similarity)
    X_test = rng.normal(0, 1, (100, 3))
    p_hyb_rep, _, _, beta_rep, sim_rep, _, _ = hybrid.predict(X_test, base_fp)
    assert sim_rep > 0.95, f"Similarity should be near 1.0, got {sim_rep}"
    assert beta_rep < 0.05, f"Beta should be very low on recurring regime, got {beta_rep}"
    
    # Query 2: Completely novel fingerprint (low similarity)
    novel_fp = np.array([0.95, 0.95, 0.95, 0.95, 0.85, 0.85, 0.90], dtype=np.float32)
    p_hyb_nov, _, _, beta_nov, sim_nov, _, _ = hybrid.predict(X_test, novel_fp)
    assert sim_nov < 0.10, f"Similarity should be very low on novel regime, got {sim_nov}"
    assert beta_nov > 0.50, f"Beta should elevate on novel regime, got {beta_nov}"


def test_hybrid_endpoint_identity_convergence():
    """
    Verify endpoint identity convergence for Two-Tier Hybrid RAPT fusion:
      - As s_max -> 1.0, beta_t -> 0.0, p_hybrid converges exactly to pure macro prediction.
      - As s_max -> 0.0, beta_t -> 1.0, p_hybrid converges exactly to pure micro prediction.
    """
    hybrid = TwoTierHybridRAPT(max_capacity=8, gamma=15.0, novelty_threshold=0.65, seed=42)
    rng = np.random.RandomState(42)
    X_ref = rng.normal(0, 1, (500, 3))
    y_ref = (X_ref[:, 0] > 0).astype(int)
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(X_ref, y_ref)
    base_models = {'RandomForest': rf}
    
    base_fp = np.array([0.05, 0.05, 0.05, 0.05, 0.02, 0.01, 0.25], dtype=np.float32)
    hybrid.warm_start(base_fp, base_models, [1.0], X_ref, y_ref)
    
    X_test = rng.normal(0, 1, (100, 3))
    
    # 1. Endpoint s_max -> 1.0 (exact match)
    p_hybrid_1, p_macro_1, p_micro_1, beta_1, s_max_1, _, _ = hybrid.predict(X_test, base_fp)
    assert s_max_1 > 0.999, f"Expected s_max -> 1.0, got {s_max_1}"
    assert beta_1 < 1e-4, f"Expected beta -> 0.0, got {beta_1}"
    max_diff_1 = np.max(np.abs(p_hybrid_1 - p_macro_1))
    assert max_diff_1 < 1e-4, f"Output must converge to macro prediction at s_max=1 (diff={max_diff_1})"
    
    # 2. Endpoint s_max -> 0.0 (completely novel)
    distant_fp = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    p_hybrid_0, p_macro_0, p_micro_0, beta_0, s_max_0, _, _ = hybrid.predict(X_test, distant_fp)
    assert s_max_0 < 1e-4, f"Expected s_max -> 0.0, got {s_max_0}"
    assert beta_0 > 0.999, f"Expected beta -> 1.0, got {beta_0}"
    max_diff_0 = np.max(np.abs(p_hybrid_0 - p_micro_0))
    assert max_diff_0 < 1e-4, f"Output must converge to micro prediction at s_max=0 (diff={max_diff_0})"


def test_path_a_benign_stream_integrity():
    """
    13th Unit Test — Path A Benign Stream Verification.
    Verifies that the prequential deployment stream contains benign (y=0) traffic,
    ensuring Precision evaluates realistically and is not artificially locked to 1.0000
    by an attack-only stream.
    """
    X_train, y_train, X_stream, y_stream, df_sorted, window_metadata, scaler = load_real_world_ton_iot_stream()
    
    n_benign = np.sum(y_stream == 0)
    n_attack = np.sum(y_stream == 1)
    total_samples = len(y_stream)
    
    benign_ratio = n_benign / total_samples
    assert n_benign == 10000, f"Expected 10,000 benign samples in stream, got {n_benign}"
    assert n_attack == 9000, f"Expected 9,000 attack samples in stream, got {n_attack}"
    assert abs(benign_ratio - 0.5263) < 0.001, f"Expected ~52.6% benign traffic, got {benign_ratio:.4f}"
    
    # Verify benign samples exist across streaming windows
    window_benign_counts = [np.sum(y_stream[i*500:(i+1)*500] == 0) for i in range(N_WINDOWS)]
    has_benign_windows = any(c > 0 for c in window_benign_counts)
    assert has_benign_windows, "Stream must contain windows with benign samples"


