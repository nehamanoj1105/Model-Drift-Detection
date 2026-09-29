"""
================================================================================
SEA RECURRING-CONCEPT DATASET GENERATOR
================================================================================
Generates synthetic data streams according to the standard SEA Concepts benchmark
(Street & Kim, 2001) with controllable concept thresholds and explicit recurring
concept sequences:
    Sequence: A -> B -> C -> A -> B -> C -> A -> B -> C (9 concept blocks)
    Features: f1, f2, f3 ~ U(0, 10). f1, f2 relevant; f3 noise.
    Threshold: f1 + f2 <= theta -> y = 1, else 0 (with 10% label noise).
================================================================================
"""

import numpy as np
import pandas as pd


class SEAGenerator:
    """
    Standard SEA Concepts generator with threshold-based binary decision boundaries.
    """
    CONCEPT_THRESHOLDS = {
        'A': 7.0,
        'B': 10.0,
        'C': 13.0
    }

    def __init__(self, concept_id='A', noise_level=0.10, seed=42):
        self.concept_id = concept_id
        self.threshold = self.CONCEPT_THRESHOLDS[concept_id]
        self.noise_level = noise_level
        self.seed = seed

    def generate_samples(self, n_samples=500, rng=None):
        if rng is None:
            rng = np.random.RandomState(self.seed)

        # 3 features uniformly distributed in [0, 10]
        X = rng.uniform(0.0, 10.0, size=(n_samples, 3))
        f1, f2 = X[:, 0], X[:, 1]

        # Ground truth binary classification: f1 + f2 <= threshold
        y_clean = (f1 + f2 <= self.threshold).astype(int)

        # Add 10% label noise (flip 10% of labels randomly)
        y = y_clean.copy()
        if self.noise_level > 0:
            noise_mask = rng.uniform(0.0, 1.0, size=n_samples) < self.noise_level
            y[noise_mask] = 1 - y[noise_mask]

        return X, y, y_clean


def generate_sea_recurring_stream(
    n_samples_per_window=500,
    windows_per_block=10,
    concept_sequence=('A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'),
    noise_level=0.10,
    seed=42
):
    """
    Generates a full recurring SEA stream organized into sequential windows.
    Returns:
        windows: list of dicts with keys:
            'window_id', 'concept_id', 'block_id', 'is_recurrence',
            'X', 'y', 'y_clean'
        df_meta: DataFrame of summary metadata per window.
    """
    rng = np.random.RandomState(seed)
    windows = []
    meta_rows = []

    global_window_id = 0
    seen_concepts = set()

    for block_idx, concept_id in enumerate(concept_sequence):
        is_recurrence = concept_id in seen_concepts
        seen_concepts.add(concept_id)

        generator = SEAGenerator(
            concept_id=concept_id,
            noise_level=noise_level,
            seed=seed + block_idx * 1000
        )

        for w_in_block in range(windows_per_block):
            # Advance RNG seed for window variability
            win_rng = np.random.RandomState(seed + global_window_id * 100 + block_idx * 7)
            X, y, y_clean = generator.generate_samples(
                n_samples=n_samples_per_window,
                rng=win_rng
            )

            window_info = {
                'window_id': global_window_id,
                'block_idx': block_idx,
                'concept_id': concept_id,
                'is_recurrence': is_recurrence,
                'is_block_start': (w_in_block == 0),
                'window_in_block': w_in_block,
                'X': X,
                'y': y,
                'y_clean': y_clean,
                'threshold': generator.threshold
            }
            windows.append(window_info)

            pos_rate = np.mean(y)
            meta_rows.append({
                'window_id': global_window_id,
                'block_idx': block_idx,
                'concept_id': concept_id,
                'is_recurrence': is_recurrence,
                'is_block_start': (w_in_block == 0),
                'threshold': generator.threshold,
                'positive_rate': pos_rate,
                'n_samples': len(y)
            })

            global_window_id += 1

    df_meta = pd.DataFrame(meta_rows)
    return windows, df_meta


if __name__ == '__main__':
    windows, df_meta = generate_sea_recurring_stream(seed=42)
    print("=" * 80)
    print("SEA RECURRING STREAM GENERATOR VERIFICATION")
    print("=" * 80)
    print(f"Total Windows Generated: {len(windows)}")
    print(f"Total Samples: {len(windows) * 500}")
    print(f"Concept Schedule Summary:\n{df_meta.groupby(['block_idx', 'concept_id', 'is_recurrence']).size()}")
    print("\nSample Class Distribution across Concepts:")
    print(df_meta.groupby('concept_id')['positive_rate'].agg(['mean', 'min', 'max']))
    print("=" * 80)
