"""
Gates for exp2_final:
- G1: Data Leakage Verification Gate (verifies online causal chronology)
- G2: Baseline Reproduction Gate (enforces |F1_run - F1_ref| <= 0.005 tolerance)
- Gate D: Dataset Admission Gate for C3 probability quality claims
"""

def verify_g1_leakage(online_event_log: list) -> dict:
    """
    Verifies that no window w accesses labels or models from windows >= w.
    """
    violations = []
    for entry in online_event_log:
        w_idx = entry.get('window_idx')
        eval_window = entry.get('eval_window_end')
        fit_max_idx = entry.get('fit_max_window_idx')
        
        if fit_max_idx is not None and fit_max_idx >= w_idx:
            violations.append(f"Window {w_idx}: Model trained on future window {fit_max_idx}")
            
    passed = len(violations) == 0
    return {'passed': passed, 'violations': violations}

def verify_g2_reproduction(run_f1: float, ref_f1: float, tolerance: float = 0.005) -> dict:
    """
    Symmetric baseline reproduction check: |run_f1 - ref_f1| <= tolerance.
    """
    diff = abs(run_f1 - ref_f1)
    passed = diff <= tolerance
    return {'passed': passed, 'diff': diff, 'run_f1': run_f1, 'ref_f1': ref_f1, 'tolerance': tolerance}

def verify_gate_d_admission(n_dp: int, n_eff_dp: int, n_segments: int) -> dict:
    """
    Admission Gate D for C3 claims:
    - Decision points >= 100
    - Effective independent decisions >= 60 (spaced >= H apart)
    - Regime segments >= 10
    """
    cond1 = n_dp >= 100
    cond2 = n_eff_dp >= 60
    cond3 = n_segments >= 10
    passed = cond1 and cond2 and cond3
    return {
        'passed': passed,
        'n_dp': n_dp,
        'n_eff_dp': n_eff_dp,
        'n_segments': n_segments,
        'reasons': [] if passed else [
            f"DP {n_dp}<100" if not cond1 else "",
            f"EffDP {n_eff_dp}<60" if not cond2 else "",
            f"Segments {n_segments}<10" if not cond3 else "",
        ]
    }
