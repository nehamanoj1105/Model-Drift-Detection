"""
Data Loaders for exp2_final: S1 (Main, N1, N2), S2 (9B_Original), Ref 9A (9A_Original), S4 (Elec2).
"""

import os
import sys
import hashlib
import json
import numpy as np
import pandas as pd
from river.datasets import Elec2

def load_s1_stream(seed=42, n_segments=40, segment_length=50, n_features=10, shift_scale=0.6, variant="S1_Main"):
    rng = np.random.default_rng(seed)
    n_windows = n_segments * segment_length
    
    config_dict = {
        'seed': seed,
        'n_segments': n_segments,
        'segment_length': segment_length,
        'n_features': n_features,
        'shift_scale': shift_scale,
        'variant': variant,
    }
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode('utf-8')).hexdigest()
    
    if variant == "N1_Noise":
        X = rng.normal(loc=0.0, scale=1.0, size=(n_windows, n_features))
        y = rng.integers(0, 2, size=n_windows)
        regime_list = [f"Segment_{s:02d}" for s in range(n_segments) for _ in range(segment_length)]
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(n_features)])
        df['target'] = y
        df['regime'] = regime_list
        return df, config_hash
        
    elif variant == "N2_FreshConcepts":
        X_list, y_list, regime_list = [], [], []
        for s in range(n_segments):
            W_s = rng.normal(loc=0.0, scale=1.0, size=n_features)
            X_seg = rng.normal(loc=0.0, scale=1.0, size=(segment_length, n_features))
            logits = np.dot(X_seg, W_s) + rng.normal(0, 0.3, size=segment_length)
            y_seg = (logits > 0).astype(int)
            X_list.append(X_seg)
            y_list.append(y_seg)
            regime_list.extend([f"Segment_{s:02d}_Unique"] * segment_length)
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(n_features)])
        df['target'] = y
        df['regime'] = regime_list
        return df, config_hash

    # S1_Main
    base_W = rng.normal(loc=1.0, scale=0.5, size=n_features)
    concept_W = {
        'C0': base_W,
        'C1': base_W + rng.normal(0, shift_scale, size=n_features),
        'C2': base_W + rng.normal(0, shift_scale, size=n_features),
        'C3': base_W + rng.normal(0, shift_scale, size=n_features),
    }
    concept_keys = ['C0', 'C1', 'C2', 'C3']
    
    X_list, y_list, regime_list = [], [], []
    for s in range(n_segments):
        c_name = concept_keys[s % 4]
        W = concept_W[c_name]
        X_seg = rng.normal(loc=0.0, scale=1.0, size=(segment_length, n_features))
        logits = np.dot(X_seg, W) + rng.normal(0, 0.3, size=segment_length)
        y_seg = (logits > 0).astype(int)
        
        X_list.append(X_seg)
        y_list.append(y_seg)
        regime_list.extend([f"Segment_{s:02d}_{c_name}"] * segment_length)
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(n_features)])
    df['target'] = y
    df['regime'] = regime_list
    return df, config_hash

def load_s2_stream(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_s4_stream(block_size=1000):
    elec_data = list(Elec2())
    df = pd.DataFrame([dict(x, target=int(y)) for x, y in elec_data])
    df['regime_block'] = [f"Block_{i // block_size:02d}" for i in range(len(df))]
    return df
