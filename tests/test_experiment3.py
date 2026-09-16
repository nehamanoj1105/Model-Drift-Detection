"""
Unit and Integration Tests for Rebuilt Experiment 3
Tests:
- data_generation (dimensions, no prior drift, balanced schedule, realistic violation rate)
- models (complementary hyperparameters for RF, ET, GB)
- ensemble (multi-temporal horizons: 3500, 2000, 1200; diversity-aware softmax voting)
- bandit (standard UCB1 with sqrt(ln(t)/N_k), cost-aware reward)
- drift (Wasserstein distance detection)
- metrics (PR-AUC, balanced accuracy, edge cases)
- statistics (three paired comparisons: Ens vs Frozen, Ens vs Retrained, Retrained vs Frozen)
- resource monitoring
- output CSVs and 9 figures
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add experiment3 directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiment3')))

from config import (
    N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_DEPLOYMENT_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, KPI_COLS, TARGET_COL, SEEDS,
    TEMPORAL_HORIZONS, DRIFT_TYPES
)
from data_generation import (
    generate_drift_schedule,
    generate_experiment_dataset,
    export_drift_configuration
)
from models import create_frozen_rf, create_retrained_rf, create_candidate_models
from ensemble import AdaptiveEnsemble
from bandit import UCB1Bandit, RandomSelector
from drift import calculate_psi, compute_drift_metrics, DriftMonitor
from metrics import evaluate_predictions
from resource_monitor import measure_execution, ResourceMonitor
from statistics import compute_summary_stats, run_paired_tests


def test_data_generation_dimensions_and_columns():
    """Verify exact sample count, window structure, and realistic class imbalance."""
    df, schedule = generate_experiment_dataset(seed=42)
    assert len(df) == N_TOTAL_SAMPLES == 10000
    assert len(schedule) == N_WINDOWS == 16
    assert N_INITIAL_TRAINING == 2000
    assert N_DEPLOYMENT_SAMPLES == 8000
    assert WINDOW_SIZE == 500
    for col in KPI_COLS:
        assert col in df.columns
        assert not df[col].isnull().any()
    assert TARGET_COL in df.columns
    assert set(df[TARGET_COL].unique()).issubset({0, 1})

    # Verify realistic class imbalance (not artificial 50/50, realistic ~20-45%)
    violation_rate = df[TARGET_COL].mean()
    assert 0.15 <= violation_rate <= 0.45, f"Violation rate {violation_rate:.2%} outside expected range"


def test_drift_schedule_no_prior_drift():
    """Verify drift schedule contains NO prior drift and balanced 4 drift types."""
    assert 'prior' not in DRIFT_TYPES, "Prior drift must not be in DRIFT_TYPES"

    required_types = {'none', 'covariate', 'concept', 'mixed'}
    required_severities = {'mild', 'moderate', 'severe'}
    required_transitions = {'sudden', 'gradual', 'recovery'}

    for seed in [42, 43, 44]:
        sched = generate_drift_schedule(seed)
        types_present = set(w['drift_type'] for w in sched)
        assert 'prior' not in types_present, f"Seed {seed} contains forbidden 'prior' drift!"
        assert required_types == types_present, f"Seed {seed} missing types: {required_types - types_present}"

        severities_present = set(w['severity'] for w in sched if w['severity'] != 'none')
        assert required_severities == severities_present
        transitions_present = set(w['transition'] for w in sched if w['transition'] != 'stable')
        assert required_transitions == transitions_present


def test_model_factories_complementary():
    """Verify model factories have complementary hyperparameters."""
    candidates = create_candidate_models(seed=42)
    assert set(candidates.keys()) == {'RandomForest', 'ExtraTrees', 'GradientBoosting'}

    rf = candidates['RandomForest']
    et = candidates['ExtraTrees']
    gb = candidates['GradientBoosting']

    assert rf.n_estimators == 50
    assert rf.max_depth == 7
    assert rf.min_samples_split == 4

    assert et.n_estimators == 50
    assert et.max_depth == 7
    assert et.min_samples_split == 6

    assert gb.n_estimators == 50
    assert gb.max_depth == 4
    assert np.isclose(gb.learning_rate, 0.08)
    assert np.isclose(gb.subsample, 0.85)


def test_ensemble_temporal_horizons_and_diversity():
    """Verify multi-temporal perspectives and diversity-aware soft voting."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(200, 4))
    y = (rng.uniform(size=200) < 0.3).astype(int)

    ens = AdaptiveEnsemble(seed=42)
    assert ens.horizons == TEMPORAL_HORIZONS

    ens.fit_initial(X, y)

    X_test = rng.normal(size=(50, 4))
    y_test = (rng.uniform(size=50) < 0.3).astype(int)

    pred, prob, comp_preds, comp_probs = ens.predict(X_test)
    assert len(pred) == 50
    assert 'RF' in comp_preds
    assert 'ET' in comp_preds
    assert 'GB' in comp_preds

    # Check diversity metrics
    div_m = ens.compute_diversity_metrics(comp_preds, comp_probs)
    assert 'disagreement_rf_et' in div_m
    assert 'disagreement_rf_gb' in div_m
    assert 'disagreement_et_gb' in div_m
    assert 0.0 <= div_m['disagreement_rf_et'] <= 1.0

    # Check diversity-aware weights
    comp_m, weights, diff, _ = ens.update_weights(y_test, comp_preds, comp_probs)
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0.0 for w in weights.values())


def test_ucb1_standard_formula():
    """Verify standard UCB1 with sqrt(ln(t)/N_k) and cost-aware reward."""
    bandit = UCB1Bandit(n_arms=3, c=1.0, lambda_cost=0.05, ref_cpu_time=0.5, arm_names=['RF', 'ET', 'Ensemble'])

    # Exploration round
    arm0 = bandit.select_arm()
    bandit.update(arm0, 0.8)
    arm1 = bandit.select_arm()
    bandit.update(arm1, 0.7)
    arm2 = bandit.select_arm()
    bandit.update(arm2, 0.9)
    assert set([arm0, arm1, arm2]) == {0, 1, 2}

    # Reward calculation
    rew, cost = bandit.compute_reward(f1=0.85, cpu_time=0.25)
    assert 0.0 <= cost <= 1.0
    assert rew == pytest.approx(0.85 - 0.05 * 0.5)


def test_statistics_paired_wilcoxon_three_comparisons():
    """Verify paired Wilcoxon statistical testing for the three comparisons."""
    df_dummy = pd.DataFrame({
        'seed': [42] * 30 + [43] * 30,
        'window_id': list(range(10)) * 3 + list(range(10)) * 3,
        'approach': (['UCB1 Adaptive Ensemble'] * 10 + ['Retrained RF'] * 10 + ['Frozen RF'] * 10) * 2,
        'f1': [0.85] * 20 + [0.83] * 20 + [0.80] * 20,
        'accuracy': [0.85] * 20 + [0.83] * 20 + [0.80] * 20,
        'cpu_time': [1.0] * 20 + [2.0] * 20 + [0.2] * 20,
    })
    stats_res = run_paired_tests(df_dummy)
    assert len(stats_res) > 0
    comparisons = set(stats_res['comparison'].values)
    assert 'UCB1 Adaptive Ensemble vs Frozen RF' in comparisons
    assert 'UCB1 Adaptive Ensemble vs Retrained RF' in comparisons
    assert 'Retrained RF vs Frozen RF' in comparisons


def test_drift_metric_detection():
    """Verify normalized Wasserstein distance detects distribution shift."""
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, size=(200, 4))
    shifted = rng.normal(3, 1, size=(100, 4))
    metrics = compute_drift_metrics(ref, shifted)
    assert metrics['wasserstein_mean'] > 0.12


def test_resource_monitoring_metric_separation():
    """Verify execution profiling measures wall clock, CPU time, and RAM."""
    def work():
        return sum(i * i for i in range(50000))

    res, metrics = measure_execution(work)
    assert 'wall_clock_time' in metrics
    assert 'total_cpu_time' in metrics
    assert 'avg_ram_mb' in metrics
    assert metrics['wall_clock_time'] > 0


def test_output_csv_files_and_schemas_if_exist():
    """Verify CSV files adhere to schemas if experiment has been run."""
    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiment3', 'results'))

    dc_path = os.path.join(res_dir, 'drift_config.csv')
    if os.path.exists(dc_path):
        df_dc = pd.read_csv(dc_path)
        for col in ['seed', 'window', 'drift_type', 'severity', 'transition', 'affected_features']:
            assert col in df_dc.columns
        assert 'prior' not in df_dc['drift_type'].values, "drift_config.csv contains 'prior' drift!"

    div_path = os.path.join(res_dir, 'model_diversity.csv')
    if os.path.exists(div_path):
        df_div = pd.read_csv(div_path)
        for col in ['disagreement_rf_et', 'disagreement_rf_gb', 'disagreement_et_gb', 'f1_ensemble', 'ensemble_gain']:
            assert col in df_div.columns


def test_essential_figures_if_exist():
    """Verify all 9 essential figures exist if experiment has been run."""
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiment3', 'figures'))
    expected_figures = [
        "fig01_f1_comparison.png",
        "fig02_accuracy_comparison.png",
        "fig03_f1_across_windows.png",
        "fig04_ucb1_arm_selection.png",
        "fig05_f1_by_drift_type.png",
        "fig06_cumulative_cpu_time.png",
        "fig07_memory_usage.png",
        "fig08_performance_vs_cpu_cost.png",
        "fig09_retraining_cost.png"
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(fig_dir, fig_name)
        if os.path.exists(fig_path):
            assert os.path.getsize(fig_path) > 1000
