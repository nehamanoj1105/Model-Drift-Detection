"""Adapter that exposes the committed 5G NR latency stream in the same
sample-level schema as the three_dataset_load streams (window_id, regime_id, y,
feature columns). Reads experiments/exp9b/data/processed_exp9b_stream.csv
(499 windows of 500 packets, one feature row per window)."""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
STREAM_CSV = os.path.join(ROOT, "experiments", "exp9b", "data",
                          "processed_exp9b_stream.csv")

FEATURE_COLS = [
    "mean_latency", "median_latency", "std_latency", "p90_latency",
    "p95_latency", "max_latency", "packet_loss_rate", "delivery_rate",
    "mean_interarrival_time", "std_interarrival_time", "packet_count",
    "effective_throughput",
]


def load_9b_stream(initial_frac=0.20):
    df = pd.read_csv(STREAM_CSV)
    df = df.sort_values("window_id").reset_index(drop=True)
    n_windows = int(df["window_id"].max()) + 1
    n_init = max(1, int(round(n_windows * initial_frac)))

    out = pd.DataFrame({
        "window_id": df["window_id"].values.astype(int),
        "regime_id": df["regime_id"].astype(str).values,
        "y": df["qos_target"].values.astype(int),
    })
    for c in FEATURE_COLS:
        out[c] = df[c].values.astype(np.float64)
    out["_t0"] = np.arange(len(out))

    sd = {
        "dataset": "5G NR Latency",
        "slug": "5g_nr",
        "source": STREAM_CSV,
        "target_desc": "3-class next-window p90-delay QoS",
        "regime_desc": "scenario tuple (UE/SCS/CAP/SINR/bitrate)",
        "window_size": 1,                    # one row per window in the CSV
        "n_raw_samples": int(len(df)),
        "n_usable_samples": int(len(out)),
        "total_windows": int(n_windows),
        "initial_train_windows": int(n_init),
        "feature_columns": FEATURE_COLS,
        "class_map": {0: "GOOD", 1: "DEGRADED", 2: "BAD"},
        "regime_sequence": df["regime_id"].astype(str).tolist(),
        "n_classes": int(df["qos_target"].nunique()),
    }
    return out, sd
