"""
================================================================================
UNIT TESTS — RAPT PIPELINE INTEGRITY & ENDPOINT PARITY
================================================================================
Verifies:
  1. Data generation determinism and linear parameter interpolation.
  2. Endpoint parity: Soft Interpolation == Hard Retrieval at alpha = 0.0 and alpha = 1.0.
  3. Prequential streaming evaluation metric consistency.
================================================================================
"""

import pytest
import numpy as np
import os
import sys

# Add project root to sys.path
rapt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if rapt_dir not in sys.path:
    sys.path.insert(0, rapt_dir)

from config import (
    R1_COVARIATE_MAG, R2_COVARIATE_MAG,
    R1_CONCEPT_SHIFTS, R2_CONCEPT_SHIFTS
)
from data_generation import (
    get_interpolated_regime_parameters,
    generate_regime_telemetry
)
from policy_transfer import (
    StoredRegimeRepository,
    evaluate_streaming_transfer
)


def test_parameter_convexity():
    """Verify that regime parameters interpolate strictly linearly."""
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cov_mag, concept = get_interpolated_regime_parameters(alpha)
        expected_cov = (1.0 - alpha) * R1_COVARIATE_MAG + alpha * R2_COVARIATE_MAG
        assert np.isclose(cov_mag, expected_cov), f"Covariate magnitude mismatch at alpha={alpha}"
        
        for k in R2_CONCEPT_SHIFTS:
            exp_shift = (1.0 - alpha) * R1_CONCEPT_SHIFTS[k] + alpha * R2_CONCEPT_SHIFTS[k]
            assert np.isclose(concept[k], exp_shift), f"Concept shift mismatch for {k} at alpha={alpha}"


def test_data_generation_determinism():
    """Verify identical seeds produce identical telemetry."""
    X1, y1, df1 = generate_regime_telemetry(1000, seed=42, alpha=0.3)
    X2, y2, df2 = generate_regime_telemetry(1000, seed=42, alpha=0.3)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert len(df1) == 1000


def test_endpoint_parity_alpha_0():
    """At alpha = 0.0, Soft and Hard retrieval must yield identical metrics."""
    seed = 42
    X_r1, y_r1, _ = generate_regime_telemetry(1000, seed=seed+10, alpha=0.0)
    X_r2, y_r2, _ = generate_regime_telemetry(1000, seed=seed+20, alpha=1.0)
    
    repo = StoredRegimeRepository(seed)
    repo.train_regimes(X_r1, y_r1, X_r2, y_r2)
    
    X_stream, y_stream, _ = generate_regime_telemetry(1500, seed=seed+100, alpha=0.0)
    
    res_soft = evaluate_streaming_transfer(X_stream, y_stream, alpha=0.0, strategy='soft', repository=repo, seed=seed)
    res_hard = evaluate_streaming_transfer(X_stream, y_stream, alpha=0.0, strategy='hard', repository=repo, seed=seed)
    
    assert np.isclose(res_soft['f1_mean'], res_hard['f1_mean'], atol=1e-7)
    assert np.isclose(res_soft['accuracy_mean'], res_hard['accuracy_mean'], atol=1e-7)
    assert res_soft['retrain_events'] == res_hard['retrain_events']


def test_endpoint_parity_alpha_1():
    """At alpha = 1.0, Soft and Hard retrieval must yield identical metrics."""
    seed = 42
    X_r1, y_r1, _ = generate_regime_telemetry(1000, seed=seed+10, alpha=0.0)
    X_r2, y_r2, _ = generate_regime_telemetry(1000, seed=seed+20, alpha=1.0)
    
    repo = StoredRegimeRepository(seed)
    repo.train_regimes(X_r1, y_r1, X_r2, y_r2)
    
    X_stream, y_stream, _ = generate_regime_telemetry(1500, seed=seed+100, alpha=1.0)
    
    res_soft = evaluate_streaming_transfer(X_stream, y_stream, alpha=1.0, strategy='soft', repository=repo, seed=seed)
    res_hard = evaluate_streaming_transfer(X_stream, y_stream, alpha=1.0, strategy='hard', repository=repo, seed=seed)
    
    assert np.isclose(res_soft['f1_mean'], res_hard['f1_mean'], atol=1e-7)
    assert np.isclose(res_soft['accuracy_mean'], res_hard['accuracy_mean'], atol=1e-7)
    assert res_soft['retrain_events'] == res_hard['retrain_events']
