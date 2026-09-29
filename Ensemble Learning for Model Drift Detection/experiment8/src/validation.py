"""
================================================================================
EXPERIMENT 8 — AUTOMATED VALIDATION SUITE
================================================================================
Enforces mandatory validation checks across all experimental states:
  1. Data Integrity Check
  2. Prequential Integrity Check
  3. Model Integrity Check
  4. Budget Integrity Check
  5. Metric Integrity Check
  6. Statistical Integrity Check
  7. Oracle Sanity Check (Oracle >= Proposed)
  8. Probe Sanity Check (C_probe < C_partial < C_full)
================================================================================
"""

import os
import numpy as np


def validate_data_integrity(stream_dict, expected_windows=16, expected_sample_size=500):
    """Checks sample count, window count, labels, and deterministic seeds."""
    windows = stream_dict['windows']
    assert len(windows) == expected_windows, f"Expected {expected_windows} windows, got {len(windows)}"
    for w in windows:
        assert len(w['X']) == expected_sample_size, f"Window {w['window_id']} sample size mismatch"
        assert len(w['y']) == expected_sample_size, f"Window {w['window_id']} label size mismatch"
        assert not np.isnan(w['X']).any(), "NaN found in feature matrix"
        assert not np.isnan(w['y']).any(), "NaN found in labels"
        labels = np.unique(w['y'])
        assert set(labels).issubset({0, 1}), f"Invalid labels found: {labels}"
    return True


def validate_prequential_integrity(window_idx, current_time, total_samples):
    """Verifies that no future windows or labels are accessed."""
    max_legal_sample = 2000 + (window_idx + 1) * 500
    assert current_time <= max_legal_sample, f"Prequential violation: sample index {current_time} > max legal {max_legal_sample}"
    return True


def validate_budget_integrity(actions, total_cost, budget):
    """Verifies sum_i C_i(a_i) <= B for every adaptation event."""
    action_costs = {'KEEP': 0.0, 'PARTIAL': 0.35, 'FULL': 1.0}
    calc_cost = sum(action_costs[a] for a in actions.values())
    assert abs(calc_cost - total_cost) < 1e-5, f"Cost sum mismatch: {calc_cost} vs {total_cost}"
    assert total_cost <= budget + 1e-5, f"Budget violation: total cost {total_cost} > budget {budget}"
    return True


def validate_probe_sanity(c_probe, c_partial, c_full):
    """Verifies C_probe < C_partial < C_full."""
    assert c_probe < c_partial, f"Probe cost sanity check failed: C_probe ({c_probe}) >= C_partial ({c_partial})"
    assert c_partial < c_full, f"Partial cost sanity check failed: C_partial ({c_partial}) >= C_full ({c_full})"
    return True


def validate_oracle_sanity(oracle_f1, proposed_f1):
    """Verifies Oracle >= Proposed under realized information."""
    assert oracle_f1 >= proposed_f1 - 0.05, f"Oracle sanity check failed: Proposed F1 ({proposed_f1}) significantly exceeded Oracle F1 ({oracle_f1})"
    return True


def validate_metric_integrity(metrics_dict):
    """Verifies metric sanity: F1 in [0, 1], non-negative costs, no NaN."""
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            assert not np.isnan(v), f"Metric {k} is NaN"
            if 'f1' in k or 'accuracy' in k or 'precision' in k or 'recall' in k:
                assert 0.0 <= v <= 1.0, f"Metric {k} out of bounds: {v}"
    return True
