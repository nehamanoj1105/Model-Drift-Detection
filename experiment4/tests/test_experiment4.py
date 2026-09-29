"""
================================================================================
UNIT TESTS FOR EXPERIMENT 4 v3
================================================================================
Validates:
  1. 50,000-sample dataset integrity, dimensions, column schema, label balance
  2. Balanced 60-window drift schedule distribution
  3. Base model factories and ensemble operations (fit, predict, adapt)
  4. Dual-trigger drift detector logic (covariate Wasserstein & concept drop)
  5. River drift detectors (ADWIN, DDM, EDDM, Page-Hinkley) via uniform wrapper
  6. Memory measurement: model_footprint_bytes and resource monitoring
  7. Detector quality metrics: Precision, Recall, F1, Episode-level Latency, and trigger rates
  8. Validation hyperparameter tuning across all 5 detectors
  9. Warm-Start Ensemble tree addition and pruning
  10. Component-Selective Ensemble selective retraining
  11. Two-Tier Hybrid Ensemble dynamic online weight adaptation
  12. River streaming wrappers (ARF, SRP)
  13. Bounded buffer cap enforcement
================================================================================
"""

import sys
import os
import pytest
import numpy as np

# Add experiment4 to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_DEPLOYMENT_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, KPI_COLS, TARGET_COL, SEEDS,
    BUFFER_CAP, APPROACH_NAMES
)
from data_generation import generate_experiment_dataset, generate_drift_schedule
from models import create_candidate_models, create_frozen_rf, create_warm_start_models
from ensemble import (
    HeterogeneousAdaptiveEnsemble,
    WarmStartEnsemble,
    ComponentSelectiveEnsemble,
    TwoTierHybridEnsemble
)
from streaming_models import RiverStreamingEnsembleWrapper, OnlineLearnerWrapper
from drift import (
    compute_drift_metrics,
    DualTriggerDriftDetector,
    create_drift_detector,
    tune_detector_hyperparameters
)
from resource_monitor import measure_execution, measure_model_footprint
from metrics import compute_metrics, compute_detector_quality, extract_drift_episodes


def test_dataset_generation_dimensions():
    """Verify that dataset generates exactly 50,000 samples with no NaNs."""
    df, schedule = generate_experiment_dataset(seed=42)
    assert len(df) == N_TOTAL_SAMPLES, f"Expected {N_TOTAL_SAMPLES}, got {len(df)}"
    assert len(df) == 50_000
    assert not df[KPI_COLS].isna().any().any(), "Dataset contains NaN features"
    assert not df[TARGET_COL].isna().any(), "Dataset contains NaN targets"

    # Check QoS violation rate
    violation_rate = df[TARGET_COL].mean()
    assert 0.15 <= violation_rate <= 0.45, f"Unexpected violation rate: {violation_rate:.2%}"


def test_drift_schedule_balance():
    """Verify that the 60-window schedule is perfectly balanced."""
    schedule = generate_drift_schedule(seed=42)
    assert len(schedule) == N_WINDOWS, f"Expected {N_WINDOWS} windows, got {len(schedule)}"
    assert schedule[0]['drift_type'] == 'none', "Window 0 must be stationary baseline"

    types = [cfg['drift_type'] for cfg in schedule]
    severities = [cfg['severity'] for cfg in schedule if cfg['drift_type'] != 'none']
    transitions = [cfg['transition'] for cfg in schedule if cfg['drift_type'] != 'none']

    assert types.count('none') == 15, f"Expected 15 none, got {types.count('none')}"
    assert types.count('covariate') == 15, f"Expected 15 covariate, got {types.count('covariate')}"
    assert types.count('concept') == 15, f"Expected 15 concept, got {types.count('concept')}"
    assert types.count('mixed') == 15, f"Expected 15 mixed, got {types.count('mixed')}"

    assert severities.count('mild') == 15, f"Expected 15 mild, got {severities.count('mild')}"
    assert severities.count('moderate') == 15, f"Expected 15 moderate, got {severities.count('moderate')}"
    assert severities.count('severe') == 15, f"Expected 15 severe, got {severities.count('severe')}"

    assert transitions.count('sudden') == 15
    assert transitions.count('gradual') == 15
    assert transitions.count('recovery') == 15


def test_candidate_models_creation():
    """Verify that RF, ET, GB candidate models are properly created with/without warm_start."""
    models = create_candidate_models(seed=42)
    assert 'RandomForest' in models
    assert 'ExtraTrees' in models
    assert 'GradientBoosting' in models
    assert not models['RandomForest'].warm_start

    ws_models = create_warm_start_models(seed=42)
    assert ws_models['RandomForest'].warm_start
    assert ws_models['ExtraTrees'].warm_start
    assert ws_models['GradientBoosting'].warm_start


def test_heterogeneous_ensemble_operations():
    """Verify fit, predict, update_weights, and adapt on small synthetic batch."""
    rng = np.random.RandomState(42)
    X_train = rng.normal(0, 1, (200, 4))
    y_train = (rng.uniform(0, 1, 200) < 0.3).astype(int)

    ens = HeterogeneousAdaptiveEnsemble(seed=42)
    ens.fit_initial(X_train, y_train)

    X_test = rng.normal(0, 1, (50, 4))
    y_test = (rng.uniform(0, 1, 50) < 0.3).astype(int)

    pred, prob, comp_preds, comp_probs = ens.predict(X_test)
    assert len(pred) == 50
    assert len(prob) == 50
    assert set(comp_preds.keys()) == {'RF', 'ET', 'GB'}
    assert set(comp_probs.keys()) == {'RF', 'ET', 'GB'}

    # Verify metrics
    mets = compute_metrics(y_test, pred, prob)
    assert 'f1' in mets
    assert 'accuracy' in mets

    # Verify weight update
    comp_mets, weights, spread, div = ens.update_weights(y_test, comp_preds, comp_probs)
    assert np.isclose(sum(weights.values()), 1.0)

    # Verify adaptation
    retrained, res, samples = ens.adapt(X_test, y_test)
    assert all(retrained.values())
    assert samples > 0


def test_dual_trigger_drift_detector():
    """Verify dual-trigger drift detection for both covariate and concept drift."""
    rng = np.random.RandomState(42)
    X_ref = rng.normal(10, 2, (500, 4))

    detector = DualTriggerDriftDetector(wasserstein_threshold=0.12, perf_drop_threshold=0.12, initial_baseline_f1=0.85)

    # 1. Stationary window with high F1 -> No drift
    X_stat = rng.normal(10, 2, (500, 4))
    dec1 = detector.check_drift(X_ref, X_stat, current_f1=0.84)
    assert not dec1['drift_detected'], "Expected no drift on stationary data"

    # 2. Covariate drift -> Large feature shift
    X_shifted = rng.normal(30, 5, (500, 4))
    dec2 = detector.check_drift(X_ref, X_shifted, current_f1=0.84)
    assert dec2['drift_detected'], "Expected drift detected on covariate shift"
    assert dec2['covariate_drift'], "Expected covariate drift trigger"

    # 3. Concept drift -> Stationary features but sharp performance collapse
    dec3 = detector.check_drift(X_ref, X_stat, current_f1=0.50)
    assert dec3['drift_detected'], "Expected drift detected on F1 collapse"
    assert dec3['concept_drift'], "Expected concept drift trigger"


def test_river_drift_detectors():
    """Verify standard river drift detectors instantiate and detect shifts."""
    for det_name in ['ADWIN', 'DDM', 'EDDM', 'Page-Hinkley']:
        det = create_drift_detector(det_name)
        assert hasattr(det, 'check_drift_window')
        assert hasattr(det, 'warm_start')

        # Warm start on low error rate
        y_true_warm = [0] * 200
        y_pred_warm = [0] * 200
        det.warm_start(y_true_warm, y_pred_warm)

        # Feed 100% errors to force detection
        y_true_err = [1] * 300
        y_pred_err = [0] * 300
        res = det.check_drift_window(y_true_err, y_pred_err)
        assert res['drift_detected'], f"Expected {det_name} to detect shift on error burst"


def test_model_footprint_and_memory_tracking():
    """Verify honest model footprint measurement and non-contaminating resource metrics."""
    frozen_rf = create_frozen_rf(seed=42)
    X = np.random.randn(200, 4)
    y = np.random.randint(0, 2, 200)
    frozen_rf.fit(X, y)

    ens = HeterogeneousAdaptiveEnsemble(seed=42)
    ens.fit_initial(X, y)

    sz_frozen = measure_model_footprint(frozen_rf)
    sz_ens = measure_model_footprint(ens)

    assert sz_frozen > 10_000, f"Frozen RF footprint too small: {sz_frozen} bytes"
    assert sz_ens > sz_frozen, f"Expected ensemble footprint ({sz_ens}) > single RF ({sz_frozen})"

    # Verify resource monitoring returns clean CPU time
    result, res = measure_execution(lambda: sum(i for i in range(50_000)))
    assert 'total_cpu_time' in res
    assert 'incremental_rss_delta_mb' in res
    assert 'process_rss_mb' in res


def test_detector_quality_metrics():
    """Verify precision, recall, F1, episode latency, and window trigger rate computation."""
    schedule = [
        {'drift_type': 'none'},
        {'drift_type': 'none'},
        {'drift_type': 'concept'},
        {'drift_type': 'concept'},
        {'drift_type': 'concept'},
        {'drift_type': 'none'},
        {'drift_type': 'none'},
        {'drift_type': 'covariate'},
        {'drift_type': 'covariate'},
    ]

    episodes = extract_drift_episodes(schedule)
    assert len(episodes) == 2

    # Mock detector firing at window 2 and window 8
    fired_windows = [False, False, True, False, False, False, False, False, True]
    summary, ep_recs = compute_detector_quality(schedule, fired_windows, 'TestDetector', seed=42)

    assert summary['tp'] == 2
    assert summary['fp'] == 0
    assert summary['precision'] == 1.0
    assert summary['window_trigger_rate'] == pytest.approx(2.0 / 9.0)
    assert summary['episode_detection_rate'] == 1.0
    assert summary['detection_rate'] == 1.0
    assert summary['mean_latency'] == 0.5


def test_detector_hyperparameter_tuning():
    """Verify validation split tuning runs without error and returns all 5 configurations."""
    rng = np.random.RandomState(42)
    X_val = rng.randn(1000, 4)
    y_val = (rng.uniform(0, 1, 1000) > 0.7).astype(int)

    ens = HeterogeneousAdaptiveEnsemble(42)
    ens.fit_initial(X_val[:500], y_val[:500])

    tuned = tune_detector_hyperparameters(X_val, y_val, ens, seed=42)
    assert 'Custom Dual-Trigger' in tuned
    assert 'ADWIN' in tuned
    assert 'DDM' in tuned
    assert 'EDDM' in tuned
    assert 'Page-Hinkley' in tuned


def test_warm_start_ensemble_operations():
    """Verify warm start ensemble adds new estimators and prunes oldest estimators."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 4)
    y = (rng.uniform(0, 1, 500) > 0.7).astype(int)

    ws = WarmStartEnsemble(42, trees_per_event=20)
    ws.fit_initial(X, y)

    # Base models have 50 trees
    assert len(ws.models['RF'].estimators_) == 50
    assert len(ws.models['ET'].estimators_) == 50

    # Adapt on new window: adds 20 and prunes 20, maintaining 50
    X_new = rng.randn(200, 4)
    y_new = (rng.uniform(0, 1, 200) > 0.7).astype(int)
    ret, res, samples = ws.adapt(X_new, y_new)

    assert len(ws.models['RF'].estimators_) == 50
    assert len(ws.models['ET'].estimators_) == 50
    assert samples == len(X_new) * 3


def test_component_selective_ensemble():
    """Verify component-selective ensemble retrains only degraded components."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 4)
    y = (rng.uniform(0, 1, 500) > 0.7).astype(int)

    cs = ComponentSelectiveEnsemble(42, comp_drop_threshold=0.08)
    cs.fit_initial(X, y)

    # Force RF performance drop in comp_metrics
    comp_mets = {
        'RF': {'f1': 0.50},  # dropped from 0.85 (>0.08 drop)
        'ET': {'f1': 0.84},  # healthy
        'GB': {'f1': 0.83},  # healthy
    }
    ret, res, samples, sel = cs.adapt_selective(X[:100], y[:100], latest_comp_metrics=comp_mets)
    assert 'RF' in sel
    assert 'ET' not in sel
    assert 'GB' not in sel
    assert ret['RF'] is True
    assert ret['ET'] is False


def test_two_tier_hybrid_ensemble():
    """Verify Two-Tier Hybrid Ensemble dynamic online weight transitions."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 4)
    y = (rng.uniform(0, 1, 500) > 0.7).astype(int)

    tt = TwoTierHybridEnsemble(42)
    tt.fit_initial(X, y)

    # Initial online weight is baseline 0.10
    assert tt.online_weight == 0.10

    # Drift fired -> online weight spikes to 0.40
    tt.adapt(X[:100], y[:100], drift_fired=True)
    assert tt.online_weight == 0.40

    # Drift not fired -> online weight decays toward 0.10
    tt.adapt(X[:100], y[:100], drift_fired=False)
    assert tt.online_weight < 0.40


def test_streaming_river_ensembles():
    """Verify ARF and SRP wrappers predict and learn without exception."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = (rng.uniform(0, 1, 100) > 0.7).astype(int)

    arf = RiverStreamingEnsembleWrapper('ARF', seed=42)
    arf.fit_initial(X, y)
    pred, prob = arf.predict(X[:20])
    assert len(pred) == 20
    assert len(prob) == 20

    srp = RiverStreamingEnsembleWrapper('SRP', seed=42)
    srp.fit_initial(X, y)
    pred, prob = srp.predict(X[:20])
    assert len(pred) == 20


def test_bounded_buffer_cap():
    """Verify buffer cap is strictly enforced across streaming windows."""
    ens = HeterogeneousAdaptiveEnsemble(seed=42, buffer_cap=1000)
    X_init = np.random.randn(800, 4)
    y_init = np.random.randint(0, 2, 800)
    ens.fit_initial(X_init, y_init)

    # Add 500 samples -> total would be 1300 without cap
    X_add = np.random.randn(500, 4)
    y_add = np.random.randint(0, 2, 500)
    ens.adapt(X_add, y_add)

    assert len(ens.buffer_X) == 1000
    assert len(ens.buffer_y) == 1000
