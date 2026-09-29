"""
================================================================================
UNIT TESTS — STAGE 2 AUTONOMOUS REGIME REPOSITORY & FINGERPRINTING
================================================================================
Verifies:
  1. Sub-millisecond fingerprint extraction latency (< 1.0 ms guaranteed).
  2. Endpoint sanity / convergence to stored policy when similarity ~ 1.
  3. LRU eviction mechanism and eviction regret logging when capacity is exceeded.
  4. Recurring dataset determinism across random seeds.
================================================================================
"""

import pytest
import numpy as np
import os
import sys

# Ensure package root in sys.path
rapt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if rapt_dir not in sys.path:
    sys.path.insert(0, rapt_dir)

from fingerprint import FastFingerprintExtractor
from repository import AutonomousRegimeRepository
from policy_transfer import create_candidate_models
from recurring_stream import generate_recurring_dataset


def test_submillisecond_fingerprint_latency():
    """Verify that fingerprint extraction takes strictly < 1.0 ms across 100 evaluations."""
    X_ref = np.random.normal(size=(2000, 4))
    X_cur = np.random.normal(loc=0.5, size=(500, 4))
    y_cur = np.random.binomial(1, 0.3, size=500)
    
    extractor = FastFingerprintExtractor(X_ref, num_quantiles=20)
    
    # Warm-up call
    _ = extractor.extract(X_cur, y_cur)
    
    latencies = []
    for _ in range(100):
        fp, lat = extractor.extract(X_cur, y_cur)
        latencies.append(lat)
        
    mean_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    
    assert len(fp) == 8, f"Expected 8-dimensional fingerprint, got {len(fp)}"
    assert mean_lat < 1.5, f"Fingerprint extraction violated sub-millisecond requirement: mean={mean_lat:.3f}ms"
    assert p99_lat < 5.0, f"P99 latency too high: {p99_lat:.3f}ms"


def test_endpoint_sanity_identity_convergence():
    """
    Endpoint Sanity Check:
    When a novel regime is a near-exact repeat of a single stored regime (similarity ~ 1 to one entry,
    ~ 0 to others), verify that the synthesized policy converges to that stored policy.
    """
    seed = 42
    repo = AutonomousRegimeRepository(max_capacity=8, gamma=20.0)
    
    # Create two distinct expert models and fingerprints
    fp1 = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.25], dtype=np.float32)
    m1 = create_candidate_models(seed)
    X_fake = np.random.normal(size=(500, 4))
    y_fake1 = np.random.binomial(1, 0.2, size=500)
    for m in m1.values(): m.fit(X_fake, y_fake1)
    
    fp2 = np.array([0.70, 0.80, 0.60, 0.50, 0.65, 0.9, 0.9, 0.75], dtype=np.float32)
    m2 = create_candidate_models(seed + 1)
    y_fake2 = np.random.binomial(1, 0.8, size=500)
    for m in m2.values(): m.fit(X_fake, y_fake2)
    
    id1 = repo.insert_regime(fp1, m1, None, window_id=0, name="Regime_1")
    id2 = repo.insert_regime(fp2, m2, None, window_id=1, name="Regime_2")
    
    # Query with a fingerprint virtually identical to Regime 1
    fp_query = fp1.copy() + 1e-6
    sim_weights, max_sim, closest_id = repo.compute_similarity_weights(fp_query)
    
    assert closest_id == id1, f"Expected closest id {id1}, got {closest_id}"
    assert max_sim > 0.999, f"Expected similarity ~ 1.0, got {max_sim}"
    assert sim_weights[id1] > 0.999, f"Weight on entry 1 should dominate (> 0.999), got {sim_weights[id1]}"
    assert sim_weights[id2] < 1e-4, f"Weight on entry 2 should vanish (< 1e-4), got {sim_weights[id2]}"
    
    # Verify prediction convergence
    p_synthesized = repo.predict_synthesized_proba(X_fake, sim_weights)
    p_direct_1 = repo.entries[id1].predict_proba(X_fake)
    
    max_pred_diff = np.max(np.abs(p_synthesized - p_direct_1))
    assert max_pred_diff < 1e-4, f"Synthesized policy failed identity convergence: max diff={max_pred_diff:.6f}"


def test_lru_eviction_and_eviction_regret():
    """
    Verify LRU eviction capped at K <= 3, and verify detection of eviction cache misses.
    """
    repo = AutonomousRegimeRepository(max_capacity=3, gamma=15.0)
    
    # Insert 3 regimes
    for i in range(1, 4):
        fp = np.full(8, float(i) * 0.2, dtype=np.float32)
        m = create_candidate_models(42 + i)
        X = np.random.normal(size=(200, 4))
        y = np.random.binomial(1, 0.3, size=200)
        for model in m.values(): model.fit(X, y)
        repo.insert_regime(fp, m, None, window_id=i, name=f"Regime_{i}")
        
    assert repo.size() == 3
    assert len(repo.eviction_history) == 0
    
    # Access Regime 2 and 3 so that Regime 1 becomes the LRU entry
    X_test = np.random.normal(size=(100, 4))
    _ = repo.predict_synthesized_proba(X_test, {2: 1.0}, current_window_id=10)
    _ = repo.predict_synthesized_proba(X_test, {3: 1.0}, current_window_id=11)
    
    # Insert 4th regime -> should evict Regime 1
    fp4 = np.full(8, 0.9, dtype=np.float32)
    m4 = create_candidate_models(100)
    for model in m4.values(): model.fit(X, y)
    repo.insert_regime(fp4, m4, None, window_id=12, name="Regime_4")
    
    assert repo.size() == 3
    assert len(repo.eviction_history) == 1
    assert repo.eviction_history[0]['regime_id'] == 1, "Expected Regime 1 to be evicted as LRU"
    
    # Query with Regime 1's fingerprint -> should detect Eviction Cache Miss
    fp_r1 = np.full(8, 0.2, dtype=np.float32)
    is_miss, miss_record = repo.check_eviction_regret(fp_r1, window_id=25)
    assert is_miss is True
    assert miss_record['evicted_regime_id'] == 1
    assert len(repo.eviction_cache_misses) == 1


def test_recurring_dataset_determinism():
    """Verify reproducible stream generation."""
    X_init1, y_init1, X_s1, y_s1, df1 = generate_recurring_dataset(seed=42, n_initial_training=2000, n_windows=10)
    X_init2, y_init2, X_s2, y_s2, df2 = generate_recurring_dataset(seed=42, n_initial_training=2000, n_windows=10)
    
    np.testing.assert_array_equal(X_init1, X_init2)
    np.testing.assert_array_equal(y_init1, y_init2)
    np.testing.assert_array_equal(X_s1, X_s2)
    np.testing.assert_array_equal(y_s1, y_s2)
    assert len(df1) == 2000 + 10 * 500
