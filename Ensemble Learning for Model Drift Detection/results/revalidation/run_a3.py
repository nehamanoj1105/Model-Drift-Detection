"""A3 / A4: cost-matched baselines, no-memory control, and headline config
beyond Campus.

Methods (all on the four streams, seeds 42-46):
  Frozen, Event-Driven, Full Retraining, RAPT, RAPT-Enhanced,
  FR-Cheap, Periodic-Cheap-5, Periodic-Cheap-10, Event-Driven-Cheap,
  RAPT-Cheap (refresh every 5 windows, 20 trees / 300 buffer),
  RAPT-Deferred, RAPT-GateFix.

Writes A3_cost_matched.csv (per seed) and A3_pareto.csv (per stream).
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RAW, SEEDS, write_csv, write_raw_csv, sha256, CONFIG_FROZEN  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced",
           "FR-Cheap", "Periodic-Cheap-5", "Periodic-Cheap-10",
           "Event-Driven-Cheap", "RAPT-Cheap", "RAPT-Deferred", "RAPT-GateFix"]


def pareto(df):
    """Non-dominated on (max macro-F1, min adaptation CPU)."""
    pts = df[["pooled_macro_f1_mean", "adaptation_cpu_sec_mean"]].values
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if (pts[j][0] >= pts[i][0] and pts[j][1] <= pts[i][1]
                    and (pts[j][0] > pts[i][0] or pts[j][1] < pts[i][1])):
                dominated = True
                break
        keep.append(not dominated)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    args = ap.parse_args()
    print("config sha256", sha256(CONFIG_FROZEN))
    pooled_all, summ_all, pw_all = [], [], []
    for ds in args.datasets:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        print(f"\n### {ds}", flush=True)
        for seed in SEEDS:
            t0 = time.perf_counter()
            pw, sm, pooled, _ = R.run_seed_v2(stream, sd, seed, METHODS)
            pw.insert(0, "dataset", slug)
            pooled.insert(0, "dataset", slug)
            sm.insert(0, "dataset", slug)
            pooled_all.append(pooled); summ_all.append(sm); pw_all.append(pw)
            print(f"   seed {seed} {time.perf_counter()-t0:.1f}s", flush=True)
    pooled = pd.concat(pooled_all, ignore_index=True)
    summ = pd.concat(summ_all, ignore_index=True)
    pw = pd.concat(pw_all, ignore_index=True)
    write_raw_csv(pw, "A3_per_window.csv")
    write_csv(pooled, "A3_cost_matched.csv")
    write_csv(summ, "A3_summary.csv")

    agg = pooled.groupby(["dataset", "method"]).agg(
        pooled_macro_f1_mean=("pooled_macro_f1", "mean"),
        pooled_macro_f1_sd=("pooled_macro_f1", "std"),
        pooled_accuracy_mean=("pooled_accuracy", "mean"),
    ).reset_index()
    cpu = summ.groupby(["dataset", "method"]).agg(
        adaptation_cpu_sec_mean=("adaptation_cpu_sec", "mean"),
        total_runtime_sec_mean=("total_runtime_sec", "mean"),
        retrain_events_mean=("retrain_events", "mean"),
        refresh_events_mean=("refresh_events", "mean"),
    ).reset_index()
    agg = agg.merge(cpu, on=["dataset", "method"])
    rows = []
    for ds, g in agg.groupby("dataset"):
        g = g.copy()
        g["pareto"] = pareto(g)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    write_csv(out, "A3_pareto.csv")
    for ds, g in out.groupby("dataset"):
        print(f"\n-- {ds} (pooled F1 / adapt CPU / pareto)")
        for _, r in g.sort_values("pooled_macro_f1_mean", ascending=False).iterrows():
            print(f"   {r['method']:20s} {r['pooled_macro_f1_mean']:.4f} "
                  f"{r['adaptation_cpu_sec_mean']:.3f} {'*' if r['pareto'] else ''}")


if __name__ == "__main__":
    main()
