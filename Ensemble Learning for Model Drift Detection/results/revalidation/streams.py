"""Build the four evaluation streams in the common schema.

5G Campus, UGR'16, NordicDat use the existing 9A loaders unchanged.
5G NR uses the existing 9B processed stream (window-level rows).
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
EXP9B = os.path.join(ROOT, "experiments", "exp9b")
sys.path.insert(0, EXP9A)

from three_dataset_load import LOADERS, build_windows  # noqa: E402

DATASETS_9A = ["5G Campus QoS", "UGR'16", "NordicDat"]
SLUG = {"5G Campus QoS": "5g_campus", "UGR'16": "ugr16",
        "NordicDat": "nordicdat", "5G NR": "5g_nr"}


def load_9a(name, mode="full"):
    data = LOADERS[name]()
    maxs = 100_000 if mode == "pilot" else None
    stream, sd = build_windows(data, max_samples=maxs)
    return stream, sd


def load_5g_nr():
    """5G NR from the existing 9B processed stream (window-level rows)."""
    csv = os.path.join(EXP9B, "data", "processed_exp9b_stream.csv")
    with open(os.path.join(EXP9B, "results", "stream_definition.json")) as f:
        sd9b = json.load(f)
    df = pd.read_csv(csv)
    feats = ["mean_latency", "median_latency", "std_latency", "p90_latency",
             "p95_latency", "max_latency", "packet_loss_rate", "delivery_rate",
             "mean_interarrival_time", "std_interarrival_time", "packet_count",
             "effective_throughput"]
    stream = pd.DataFrame({
        "window_id": df["window_id"].values,
        "regime_id": df["regime_id"].astype(str).values,
        "y": df["qos_target"].astype(int).values,
    })
    for c in feats:
        stream[c] = df[c].values
    sd = {
        "dataset": "5G NR", "slug": "5g_nr",
        "window_size": 1,
        "n_raw_samples": int(len(df)),
        "n_usable_samples": int(len(df)),
        "total_windows": int(len(df)),
        "initial_train_windows": int(sd9b["initial_train_windows"]),
        "feature_columns": feats,
        "n_classes": int(df["qos_target"].nunique()),
        "regime_sequence": df.groupby("window_id")["regime_id"].first().tolist(),
        "source": csv,
    }
    return stream, sd


def get_stream(name, mode="full"):
    if name == "5G NR":
        return load_5g_nr()
    return load_9a(name, mode)


ALL_DATASETS = ["5G Campus QoS", "UGR'16", "NordicDat", "5G NR"]
