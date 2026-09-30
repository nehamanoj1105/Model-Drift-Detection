"""
Stream construction for the REVISED Experiment 9A.

Dataset (provenance)
--------------------
INSECTS "incremental_reoccurring_balanced" (Souza, Reis, Maletzke & Batista,
2020, "Challenges in Benchmarking Stream Learning Algorithms with Real-world
Data"; distributed by the USP DS repository and shipped with the `river`
library as `river.datasets.Insects`).

* 79,986 sequential samples, 33 numeric features, 6 balanced classes.
* Real-world optical-sensor recordings of flying insects collected over time.
* The variant is explicitly designed for RECURRING concept drift: a subset of
  species dominates for a period, then other species dominate, and earlier
  species compositions RECUR later in the stream.

Why this dataset (see report Section 1 for the full criterion matrix)
---------------------------------------------------------------------
The original 9A 5G-campus stream had three regimes (A/B/C) whose target
distributions were almost identical (~33/33/33 GOOD/DEGRADED/BAD), so no
regime-specific policy existed to transfer and RAPT could not differentiate
itself from Full Retraining. INSECTS provides genuine, naturally recurring
regimes with distinct joint distributions P(X, Y), which is precisely the
structure RAPT's policy-reuse mechanism is designed to exploit.

Streaming protocol (per-sample, windowed)
-----------------------------------------
This is a genuine labelled-instance stream, so we use the natural prequential
protocol: each window is WINDOW_SIZE=500 consecutive samples, every sample in
the window is predicted and scored, and window metrics are computed over the
500 per-sample predictions. This differs from 9A/9B telemetry (where a window
is one aggregated measurement with one label); the difference is documented in
the report. Window size (500) and the initial-training fraction are held
consistent with the rest of the experiment family.

Regime construction (natural, not synthetic)
--------------------------------------------
The dataset's own design partitions the stream into ~5,000-sample segments
that separate the recurring species compositions. We therefore group every
WINDOWS_PER_REGIME_BLOCK=10 consecutive windows (5,000 samples, matching the
dataset's native segment granularity) into one regime block and assign the
block's majority insect species as its regime id. Consecutive blocks sharing a
majority species merge into a single regime visit; when that species recurs
later the regime id recurs, producing genuine regime recurrence. The
construction is deterministic, uses only temporal ordering, and never inspects
future windows to assign a regime label.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd

from exp9a_config import (
    WINDOW_SIZE, EXP9A_DATA_DIR, REVISED_STREAM_CSV, RAW_DIR,
    INITIAL_TRAIN_WINDOWS,
)

# 10 windows x 500 samples = 5,000 samples per regime block (native segment).
WINDOWS_PER_REGIME_BLOCK = 10

_CACHE_CANDIDATES = [
    os.path.expanduser("~/river_data/Insects/incremental_reoccurring_balanced.csv"),
    os.path.expanduser("~/.river_data_cache/Insects/incremental_reoccurring_balanced.csv"),
]


def _find_cached_csv():
    for p in _CACHE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load_insects():
    """Load INSECTS incremental_reoccurring_balanced from the local river cache
    (preferred, avoids re-download) or via river if the cache is absent."""
    cached = _find_cached_csv()
    if cached is not None:
        df = pd.read_csv(cached, header=None)
        print(f"Loaded INSECTS from cache: {cached}")
    else:
        from river.datasets import Insects
        rows = []
        for x, y in Insects("incremental_reoccurring_balanced"):
            rows.append(list(x.values()) + [y])
        df = pd.DataFrame(rows)
        print("Loaded INSECTS via river.datasets.Insects")

    n_features = df.shape[1] - 1
    df.columns = [f"f{i}" for i in range(n_features)] + ["species"]
    df["species"] = df["species"].astype(str)
    return df


def build_stream(save=True):
    """Build the revised 9A recurring-regime stream. Returns (df_stream, stream_def)."""
    os.makedirs(EXP9A_DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    df = _load_insects()
    feature_cols = [c for c in df.columns if c != "species"]

    data_hash = hashlib.sha256(
        df[feature_cols].values.astype(np.float64).tobytes()
        + "".join(df["species"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()

    n_samples = len(df)
    n_windows = n_samples // WINDOW_SIZE
    n_full_blocks = n_windows // WINDOWS_PER_REGIME_BLOCK
    keep_samples = n_full_blocks * WINDOWS_PER_REGIME_BLOCK * WINDOW_SIZE

    df = df.iloc[:keep_samples].reset_index(drop=True)

    # --- Per-sample window and block indices ---------------------------------
    sample_idx = np.arange(len(df))
    df["window_id"] = sample_idx // WINDOW_SIZE
    df["block_index"] = df["window_id"] // WINDOWS_PER_REGIME_BLOCK

    # --- Regime id = majority species of the block ---------------------------
    block_majority = (
        df.groupby("block_index")["species"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    df["regime_id"] = df["block_index"].map(lambda b: f"regime_{block_majority[b]}")

    # --- Regime visits (merge consecutive blocks with same majority species) -
    blocks = sorted(df["block_index"].unique())
    visits = []
    cur, start, prev_b = None, None, None
    for b in blocks:
        reg = block_majority[b]
        if reg != cur:
            if cur is not None:
                visits.append((cur, start, prev_b))
            cur, start = reg, b
        prev_b = b
    visits.append((cur, start, prev_b))

    visit_start_blocks = {s for (_, s, _) in visits}
    df["is_transition"] = df["block_index"].isin(visit_start_blocks).astype(int)

    # --- Chronological ordering guarantee ------------------------------------
    assert (df["window_id"].values == np.arange(len(df)) // WINDOW_SIZE).all()

    segment_info = [
        {
            "visit_index": vi,
            "regime_id": f"regime_{reg}",
            "start_block": int(s),
            "end_block": int(e),
            "start_window": int(s * WINDOWS_PER_REGIME_BLOCK),
            "end_window": int((e + 1) * WINDOWS_PER_REGIME_BLOCK - 1),
        }
        for vi, (reg, s, e) in enumerate(visits)
    ]

    stream_def = {
        "dataset": "INSECTS incremental_reoccurring_balanced",
        "source": "river.datasets.Insects (USP DS repository; Souza et al. 2020)",
        "data_hash": data_hash,
        "n_samples": int(len(df)),
        "n_features": len(feature_cols),
        "n_classes": int(df["species"].nunique()),
        "classes": sorted(df["species"].unique().tolist()),
        "feature_columns": feature_cols,
        "window_size": WINDOW_SIZE,
        "windows_per_regime_block": WINDOWS_PER_REGIME_BLOCK,
        "total_windows": int(n_full_blocks * WINDOWS_PER_REGIME_BLOCK),
        "initial_train_windows": INITIAL_TRAIN_WINDOWS,
        "streaming_eval_windows": int(n_full_blocks * WINDOWS_PER_REGIME_BLOCK - INITIAL_TRAIN_WINDOWS),
        "regime_sequence": [f"regime_{r}" for (r, _, _) in visits],
        "segments": segment_info,
    }

    if save:
        df.to_csv(REVISED_STREAM_CSV, index=False)
        with open(os.path.join(RAW_DIR, "stream_definition_9a.json"), "w") as f:
            json.dump(stream_def, f, indent=2)
        print(f"Saved revised 9A stream: {REVISED_STREAM_CSV} ({len(df)} samples, "
              f"{stream_def['total_windows']} windows)")
        print(f"  regimes: {stream_def['regime_sequence']}")
        print(f"  initial_train_windows={INITIAL_TRAIN_WINDOWS}, "
              f"streaming={stream_def['streaming_eval_windows']}")

    return df, stream_def


if __name__ == "__main__":
    build_stream()
