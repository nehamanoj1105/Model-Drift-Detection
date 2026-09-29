"""
src/streams.py - Stream generators and loaders for Exp2 Final.

All streams return (X_all, y_all, regime_ids, config_hash).
X_all: shape (n_windows, n_features), each row is per-window mean features.
y_all: shape (n_windows,), integer class labels.
regime_ids: shape (n_windows,), regime identifier per window.

IMPORTANT: No leakage. Stream config is fixed before any run.
"""

import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

_HERE = Path(__file__).resolve().parent
_EXP2_ROOT = _HERE.parent
_EXPERIMENTS = _EXP2_ROOT.parent

# Verify we're under project root
_PROJECT_ROOT = _EXPERIMENTS.parent
assert _EXP2_ROOT.is_relative_to(_PROJECT_ROOT), f"exp2_final not under project root: {_EXP2_ROOT}"


def _hash_config(cfg: dict) -> str:
    """SHA-256 of a JSON-serialized config dict (sorted keys)."""
    data = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def generate_s1_stream(seed: int, variant: str = 'S1_Main', cfg: dict = None) -> tuple:
    """
    Generate S1/N1/N2 stream.
    
    Returns: (X_all, y_all, regime_ids, config_hash)
    X_all: (n_windows, n_features) - each row is per-window mean features
    y_all: (n_windows,) - majority label per window
    regime_ids: (n_windows,) - regime/segment identifier
    config_hash: str - SHA-256 of config
    
    Window = window_size instances. X_all[i] = mean of instances in window i.
    y_all[i] = majority class in window i.
    
    Regime ID format: f"seg_{seg_idx}_{concept_name}"
    """
    if cfg is None:
        cfg = _load_stream_config()
    
    scfg = cfg[variant[:2]] if variant[:2] in cfg else cfg.get('S1', cfg.get(variant, {}))
    # Find config by variant name
    for key, val in cfg.items():
        if val.get('variant') == variant:
            scfg = val
            break
    
    rng = np.random.default_rng(seed)
    
    n_windows = scfg.get('n_windows', 6000)
    window_size = scfg.get('window_size', 20)
    n_features = scfg.get('n_features', 10)
    n_segments = scfg.get('n_segments', 60)
    windows_per_segment = scfg.get('windows_per_segment', 100)
    n_concepts = scfg.get('n_concepts', 4)
    s_cov = scfg.get('s_cov', 1.5)
    boundary_scale = scfg.get('boundary_scale', 1.0)
    label_noise = scfg.get('label_noise', 0.05)
    pure_noise_labels = scfg.get('pure_noise_labels', False) or scfg.get('label_noise_type') == 'pure_noise'
    fresh_concepts = scfg.get('fresh_concepts', False) or scfg.get('concept_type') == 'fresh_unique'
    partial_alphas = scfg.get('partial_alphas', [0.75, 0.50, 0.25])
    seg_type_probs = scfg.get('segment_types', {
        'exact_recurrence': 0.60,
        'partial_recurrence': 0.20,
        'decoy': 0.20
    })
    cycle = scfg.get('cycle', ['C0', 'C1', 'C2', 'C3'])
    
    n_instances = n_windows * window_size
    
    # Define base concepts (fixed by config, seed only affects instance sampling)
    concept_rng = np.random.default_rng(0)  # Concept definitions fixed
    n_base = n_concepts if not fresh_concepts else n_segments
    concept_means = concept_rng.standard_normal((n_base, n_features)) * s_cov
    concept_weights = concept_rng.standard_normal((n_base, n_features)) * boundary_scale
    concept_biases = concept_rng.standard_normal(n_base) * 0.5
    
    # Plan segments
    segment_concepts = []  # (concept_idx_for_cov, concept_idx_for_label, alpha, seg_type_name)
    
    if fresh_concepts:
        for i in range(n_segments):
            # Each segment is a fresh concept
            segment_concepts.append((i, i, 1.0, 'fresh'))
        # Regime IDs: each segment is unique
        segment_regime_ids = [f'seg_{i}_fresh_{i}' for i in range(n_segments)]
    else:
        seg_type_keys = list(seg_type_probs.keys())
        seg_type_cumprob = np.cumsum([seg_type_probs[k] for k in seg_type_keys])
        
        for i in range(n_segments):
            # Cycle defines which base concept is "primary"
            primary_concept_idx = i % len(cycle)
            primary_concept_name = cycle[primary_concept_idx]
            primary_idx = int(primary_concept_name[1:])  # C0->0, C1->1, etc.
            
            # Sample segment type
            u = rng.uniform()
            if u < seg_type_cumprob[0]:  # exact_recurrence
                segment_concepts.append((primary_idx, primary_idx, 1.0, 'exact'))
            elif u < seg_type_cumprob[1]:  # partial_recurrence
                other_idx = rng.integers(0, n_concepts)
                alpha = rng.choice(partial_alphas)
                segment_concepts.append((primary_idx, primary_idx, alpha, 'partial'))
            else:  # decoy
                decoy_label_idx = rng.integers(0, n_concepts)
                while decoy_label_idx == primary_idx:
                    decoy_label_idx = rng.integers(0, n_concepts)
                # Covariates follow primary, labels follow decoy (nearest-by-sim is wrong)
                segment_concepts.append((primary_idx, decoy_label_idx, 1.0, 'decoy'))
        
        segment_regime_ids = [f'seg_{i}_{cycle[i % len(cycle)]}' for i in range(n_segments)]
    
    # Generate instances
    X_instances = np.zeros((n_instances, n_features))
    y_instances = np.zeros(n_instances, dtype=int)
    
    for seg_idx, (cov_concept, label_concept, alpha, seg_type) in enumerate(segment_concepts):
        start = seg_idx * windows_per_segment * window_size
        end = start + windows_per_segment * window_size
        n_inst = end - start
        
        # Covariate distribution
        mu_cov = concept_means[cov_concept]
        X_seg = rng.standard_normal((n_inst, n_features)) + mu_cov
        
        if fresh_concepts:
            # Simple: just use the concept features
            y_logits = X_seg @ concept_weights[label_concept] + concept_biases[label_concept]
        elif seg_type == 'partial':
            # Blended boundary
            other_concept = (cov_concept + 1) % n_concepts
            w_blended = alpha * concept_weights[label_concept] + (1 - alpha) * concept_weights[other_concept]
            b_blended = alpha * concept_biases[label_concept] + (1 - alpha) * concept_biases[other_concept]
            y_logits = X_seg @ w_blended + b_blended
        else:
            y_logits = X_seg @ concept_weights[label_concept] + concept_biases[label_concept]
        
        y_seg = (y_logits > 0).astype(int)
        
        X_instances[start:end] = X_seg
        y_instances[start:end] = y_seg
    
    # Apply label noise
    if not pure_noise_labels and label_noise > 0:
        noise_mask = rng.uniform(size=n_instances) < label_noise
        y_instances[noise_mask] = 1 - y_instances[noise_mask]
    
    # Pure noise override
    if pure_noise_labels:
        y_instances = rng.integers(0, 2, size=n_instances)
    
    # Aggregate into windows
    X_windows = np.zeros((n_windows, n_features))
    y_windows = np.zeros(n_windows, dtype=int)
    regime_ids = []
    
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        X_windows[w] = X_instances[start:end].mean(axis=0)
        # Majority vote for label with random tie-break
        c0 = np.sum(y_instances[start:end] == 0)
        c1 = np.sum(y_instances[start:end] == 1)
        if c1 > c0:
            y_windows[w] = 1
        elif c0 > c1:
            y_windows[w] = 0
        else:
            y_windows[w] = int(rng.integers(0, 2))
        
        # Regime ID based on segment
        seg_idx = w // windows_per_segment
        if seg_idx < len(segment_regime_ids):
            regime_ids.append(segment_regime_ids[seg_idx])
        else:
            regime_ids.append(f'seg_{seg_idx}')
    
    if pure_noise_labels:
        y_windows = rng.integers(0, 2, size=n_windows)
        
    regime_ids = np.array(regime_ids)
    
    # Config hash
    hash_cfg = {
        'variant': variant,
        'n_windows': n_windows,
        'window_size': window_size,
        'n_features': n_features,
        'n_segments': n_segments,
        'windows_per_segment': windows_per_segment,
        'n_concepts': n_concepts,
        's_cov': s_cov,
        'boundary_scale': boundary_scale,
        'label_noise': label_noise,
        'pure_noise_labels': pure_noise_labels,
        'fresh_concepts': fresh_concepts,
    }
    config_hash = _hash_config(hash_cfg)
    
    return X_windows, y_windows, regime_ids, config_hash


def load_s2_stream() -> tuple:
    """
    Load S2: 9B 5G NR Latency stream using Exp1 protocol.
    n_init = 99 windows (rows), StreamingPreprocessor fit on first 99 rows.
    
    Returns: (X_all, y_all, regime_ids, data_hash)
    Each row in the CSV corresponds to one 'window' (already pre-aggregated).
    """
    csv_path = _EXPERIMENTS / 'exp9b' / 'data' / 'processed_exp9b_stream.csv'
    assert csv_path.exists(), f"9B CSV not found: {csv_path}"
    
    data_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    
    df = pd.read_csv(csv_path)
    assert len(df) == 499, f"Expected 499 rows, got {len(df)}"
    
    feature_cols = [
        'mean_latency', 'median_latency', 'std_latency', 'p90_latency',
        'p95_latency', 'max_latency', 'packet_loss_rate', 'delivery_rate',
        'mean_interarrival_time', 'std_interarrival_time', 'packet_count',
        'effective_throughput'
    ]
    
    # Use available features
    available_feat = [c for c in feature_cols if c in df.columns]
    if not available_feat:
        # Try alternative feature names
        available_feat = [c for c in df.columns if c not in ['qos_target', 'regime_id', 'regime_letter', 'window_id']][:12]
    
    target_col = 'qos_target' if 'qos_target' in df.columns else 'label'
    regime_col = 'regime_id' if 'regime_id' in df.columns else ('regime_letter' if 'regime_letter' in df.columns else None)
    
    # Apply StreamingPreprocessor (Exp1 protocol)
    sys.path.insert(0, str(_EXP2_ROOT / 'vendor' / 'exp1'))
    from streaming_preprocessor import StreamingPreprocessor
    
    n_init = 99
    preproc = StreamingPreprocessor()
    df_init = df.iloc[:n_init]
    preproc.fit_initial(df_init[available_feat] if hasattr(preproc, 'fit_initial') else df_init)
    
    X_raw = df[available_feat].values.astype(np.float64)
    # Transform all rows
    X_all = np.zeros_like(X_raw)
    for i in range(len(df)):
        row_df = df.iloc[[i]][available_feat]
        X_all[i] = preproc.transform(row_df).flatten()
    
    y_all = df[target_col].values.astype(int) if target_col in df.columns else np.zeros(len(df), dtype=int)
    
    if regime_col and regime_col in df.columns:
        regime_ids = df[regime_col].values.astype(str)
    else:
        regime_ids = np.array([f'regime_{i // 50}' for i in range(len(df))])
    
    return X_all, y_all, regime_ids, data_hash, available_feat, target_col, regime_col, n_init


def load_s4_elec2() -> tuple:
    """
    Load S4: Electricity Elec2 dataset via river.datasets.Elec2.
    Caches locally. Returns (X_all, y_all, regime_ids, data_hash).
    window_size=50, block_windows=20, W0=20 windows.
    Each window = 50 instances aggregated to mean features.
    """
    cache_path = _EXP2_ROOT / 'data' / 'elec2_cache.npz'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        X_windows = data['X_windows']
        y_windows = data['y_windows']
        regime_ids = data['regime_ids']
        data_hash = str(data['data_hash'])
        print(f"Loaded Elec2 from cache: {cache_path}, SHA-256: {data_hash[:16]}...")
        return X_windows, y_windows, regime_ids, data_hash
    
    try:
        from river.datasets import Elec2
        dataset = Elec2()
        
        rows = []
        labels = []
        for x, y in dataset:
            rows.append(list(x.values()))
            labels.append(int(y))
        
        X_raw = np.array(rows, dtype=np.float64)
        y_raw = np.array(labels, dtype=int)
        
    except Exception as e:
        print(f"Could not load Elec2 via river: {e}")
        # Try loading from scikit-multiflow or other source
        raise RuntimeError(f"Could not load Elec2: {e}")
    
    # Compute data hash before windowing
    h = hashlib.sha256(X_raw.tobytes() + y_raw.tobytes()).hexdigest()
    
    window_size = 50
    block_windows = 20
    n_raw = len(X_raw)
    n_windows = n_raw // window_size
    
    X_windows = np.zeros((n_windows, X_raw.shape[1]))
    y_windows = np.zeros(n_windows, dtype=int)
    
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        X_windows[w] = X_raw[start:end].mean(axis=0)
        counts = np.bincount(y_raw[start:end], minlength=2)
        y_windows[w] = int(np.argmax(counts))
    
    # Assign regime IDs: each block of block_windows windows is one regime
    block_ids = np.arange(n_windows) // block_windows
    regime_ids = np.array([f'block_{b}' for b in block_ids])
    
    np.savez(cache_path, X_windows=X_windows, y_windows=y_windows,
             regime_ids=regime_ids, data_hash=np.array(h))
    
    print(f"Elec2 loaded: {n_raw} instances -> {n_windows} windows, SHA-256: {h[:16]}...")
    return X_windows, y_windows, regime_ids, h


def load_s3_insects(insects_path: str) -> tuple:
    """
    Load S3: INSECTS incremental-reoccurring dataset.
    Returns None if path is empty or file doesn't exist.
    """
    if not insects_path or not Path(insects_path).exists():
        print(f"INSECTS_PATH empty or not found: '{insects_path}'. Skipping S3.")
        return None
    
    df = pd.read_csv(insects_path)
    feature_cols = [c for c in df.columns if c not in ['class', 'label', 'regime', 'target']]
    target_col = 'class' if 'class' in df.columns else ('label' if 'label' in df.columns else df.columns[-1])
    
    le = LabelEncoder()
    y_raw = le.fit_transform(df[target_col].values)
    X_raw = df[feature_cols].values.astype(np.float64)
    
    h = hashlib.sha256(X_raw.tobytes() + y_raw.tobytes()).hexdigest()
    
    window_size = 50
    block_windows = 20
    n_raw = len(X_raw)
    n_windows = n_raw // window_size
    
    X_windows = np.zeros((n_windows, X_raw.shape[1]))
    y_windows = np.zeros(n_windows, dtype=int)
    
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        X_windows[w] = X_raw[start:end].mean(axis=0)
        counts = np.bincount(y_raw[start:end], minlength=len(np.unique(y_raw)))
        y_windows[w] = int(np.argmax(counts))
    
    block_ids = np.arange(n_windows) // block_windows
    regime_ids = np.array([f'block_{b}' for b in block_ids])
    
    print(f"INSECTS loaded: {n_raw} instances -> {n_windows} windows, SHA-256: {h[:16]}...")
    return X_windows, y_windows, regime_ids, h


def _load_stream_config() -> dict:
    """Load stream config from config/stream_config.json."""
    cfg_path = _EXP2_ROOT / 'config' / 'stream_config.json'
    return json.loads(cfg_path.read_text(encoding='utf-8'))


def _load_thresholds() -> dict:
    """Load thresholds config from config/thresholds.json."""
    cfg_path = _EXP2_ROOT / 'config' / 'thresholds.json'
    return json.loads(cfg_path.read_text(encoding='utf-8'))
