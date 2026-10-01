"""A1: pooled metrics + per-window confusion counts, all four streams.

Runs the five paper models + four detectors on Campus, UGR'16, NordicDat, 5G NR
for seeds 42-46, writes per-window raw CSVs with confusion counts and pooled
per-(dataset,method,seed) metrics.

Usage:
    python run_a1.py [--datasets 5G Campus QoS UGR'16 ...] [--seeds 42 43]
                     [--check-repro]
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RAW, OUT, SEEDS, write_csv, write_raw_csv, sha256, CONFIG_FROZEN  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402

PAPER_METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT",
                 "RAPT-Enhanced", "ADWIN", "EDD", "Page-Hinkley", "EDMA"]


def aggregate(pooled):
    num = [c for c in pooled.columns if pooled[c].dtype.kind in "fi"
           and c not in ("seed",)]
    g = pooled.groupby("method")[num].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--check-repro", action="store_true")
    args = ap.parse_args()

    print("config_frozen sha256:", sha256(CONFIG_FROZEN))
    all_pw, all_pooled, all_sum = [], [], []
    for ds in args.datasets:
        stream, sd = get_stream(ds, args.mode)
        slug = SLUG[ds]
        print(f"\n### {ds} windows={sd['total_windows']} init={sd['initial_train_windows']} "
              f"classes={sd['n_classes']} window_size={sd['window_size']}", flush=True)
        for seed in args.seeds:
            t0 = time.perf_counter()
            pw, sm, pooled, _ = R.run_seed_v2(stream, sd, seed, PAPER_METHODS)
            pw.insert(0, "dataset", slug)
            pooled.insert(0, "dataset", slug)
            sm.insert(0, "dataset", slug)
            all_pw.append(pw); all_pooled.append(pooled); all_sum.append(sm)
            print(f"   seed {seed}: {time.perf_counter()-t0:.1f}s", flush=True)

    pw = pd.concat(all_pw, ignore_index=True)
    pooled = pd.concat(all_pooled, ignore_index=True)
    summ = pd.concat(all_sum, ignore_index=True)
    write_raw_csv(pw, "A1_per_window_all_streams.csv")
    write_csv(pooled, "A1_pooled_all_streams.csv")
    write_csv(summ, "A1_summary_all_streams.csv")

    agg = pooled.groupby(["dataset", "method"]).agg(
        pooled_macro_f1_mean=("pooled_macro_f1", "mean"),
        pooled_macro_f1_sd=("pooled_macro_f1", "std"),
        pooled_accuracy_mean=("pooled_accuracy", "mean"),
        pooled_accuracy_sd=("pooled_accuracy", "std"),
    ).reset_index()
    write_csv(agg, "A1_pooled_agg.csv")
    print("\n=== pooled macro-F1 (mean +/- sd over seeds) ===")
    for ds in args.datasets:
        s = agg[agg.dataset == SLUG[ds]]
        print(f"-- {ds}")
        for _, r in s.iterrows():
            print(f"   {r['method']:16s} {r['pooled_macro_f1_mean']:.4f} +/- {r['pooled_macro_f1_sd']:.4f}")

    # ---- reproduction cross-check vs existing 9A summary ----
    if args.check_repro:
        print("\n=== reproduction check vs results/experiment_9a_three ===")
        raw9a = os.path.join(os.path.dirname(HERE), "..", "experiments", "..",
                             "results", "experiment_9a_three", "raw")
        raw9a = os.path.abspath(os.path.join(HERE, "..", "..", "results",
                                             "experiment_9a_three", "raw"))
        for ds in ["5G Campus QoS", "UGR'16", "NordicDat"]:
            slug = SLUG[ds]
            if ds not in args.datasets:
                continue
            old = pd.read_csv(os.path.join(raw9a, f"summary_{slug}_full.csv"))
            old = old[old.seed.isin(args.seeds)]
            new = summ[summ.dataset == slug]
            for m in PAPER_METHODS:
                o = old[old.method == m]["macro_f1"].mean()
                n = new[new.method == m]["per_window_macro_f1"].mean()
                if pd.notna(o) and pd.notna(n):
                    flag = "" if abs(o - n) < 0.002 else "  <-- DIFF"
                    print(f"   {ds:16s} {m:16s} old={o:.4f} new={n:.4f}{flag}")


if __name__ == "__main__":
    main()
