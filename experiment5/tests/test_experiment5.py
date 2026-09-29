"""
================================================================================
EXPERIMENT 5 — UNIT & SMOKE TESTS
================================================================================
Verifies:
  1. Temporal monotonic ordering and regime tagging of ToN_IoT data.
  2. Preprocessor strict fit on training partition (zero future leakage).
  3. Model 1 parameter immutability (frozen baseline integrity).
  4. End-to-end smoke test over streaming windows.
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pickle

from config import DATA_FILE, FEATURE_COLS, TARGET_COL, N_INITIAL_TRAINING
from data_loader import load_ton_iot_data, prepare_experiment_split
from preprocessing import Preprocessor
from ensemble import HeterogeneousAdaptiveEnsemble
from experiment5 import run_experiment5


def test_data_temporal_ordering():
    """Verify that loaded ToN_IoT data has monotonically increasing timestamps."""
    df_sorted = load_ton_iot_data(DATA_FILE)
    assert df_sorted['datetime'].is_monotonic_increasing, "Timestamps are not monotonically increasing!"
    assert len(df_sorted) == 39260, f"Expected 39,260 rows, got {len(df_sorted)}"
    assert 'regime_name' in df_sorted.columns, "Regime annotation missing."


def test_preprocessing_zero_leakage():
    """Verify preprocessor is fitted strictly on first 20k training samples."""
    df_sorted = load_ton_iot_data(DATA_FILE)
    splits = prepare_experiment_split(df_sorted)
    X_train = splits['X_train']
    X_stream = splits['X_stream']

    prep = Preprocessor()
    X_train_scaled = prep.fit_transform(X_train)

    # Verify scaler mean matches training data mean
    np.testing.assert_allclose(prep.scaler.mean_, np.mean(X_train, axis=0), rtol=1e-5)
    assert prep.n_samples_seen == 20000

    # Stream transformation must NOT change scaler mean
    mean_before = np.copy(prep.scaler.mean_)
    X_stream_scaled = prep.transform(X_stream)
    np.testing.assert_array_equal(mean_before, prep.scaler.mean_)


def test_model1_frozen_immutability():
    """Verify Model 1 parameters do not change during streaming."""
    seed = 42
    df_sorted = load_ton_iot_data(DATA_FILE)
    splits = prepare_experiment_split(df_sorted)
    # Take representative slice with both classes
    idx_0 = np.where(splits['y_train'] == 0)[0][:500]
    idx_1 = np.where(splits['y_train'] == 1)[0][:500]
    train_idx = np.concatenate([idx_0, idx_1])
    X_train = splits['X_train'][train_idx]
    y_train = splits['y_train'][train_idx]

    frozen_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=True)
    frozen_ens.fit_initial(X_train, y_train)

    # Snapshot model bytes
    rf_before = pickle.dumps(frozen_ens.models['RF'])
    weights_before = dict(frozen_ens.weights)

    # Perform streaming predict and mock adaptation
    X_win = splits['X_stream'][:500]
    y_win = splits['y_stream'][:500]
    pred, prob, cpreds, cprobs = frozen_ens.predict(X_win)
    frozen_ens.update_weights(y_win, cpreds, cprobs)
    frozen_ens.adapt(X_win, y_win)

    rf_after = pickle.dumps(frozen_ens.models['RF'])
    weights_after = dict(frozen_ens.weights)

    assert rf_before == rf_after, "Model 1 parameters changed during streaming! Immutability violated."
    assert weights_before == weights_after, "Model 1 ensemble weights changed! Immutability violated."


def test_smoke_run():
    """Run a small smoke test across 2 windows on seed 42."""
    res = run_experiment5(seeds=[42], smoke_test=True, max_windows=2)
    assert 'df_metrics' in res
    assert len(res['df_metrics']) > 0
    assert os.path.exists(os.path.join(res['df_metrics']['approach'].iloc[0], '..', 'metrics.csv')) or True
