"""
Driver for the unified final harness.

Usage (from the "Ensemble Learning for Model Drift Detection" directory):

    python experiments/final/run_final_driver.py --mode dev  --streams campus,ugr16,nordicdat,5g_nr
    python experiments/final/run_final_driver.py --mode final --streams all

Seeds default to 42-46 (dev) or 42-71 (final). A `--cfg` JSON overrides any
final_config.DEFAULT_CFG key.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp9a"))

from final_config import (SEEDS_DEV, SEEDS_FINAL, DEFAULT_CFG, RAW_FINAL,
                          CONFIG_DIR, ensure_dirs)
from run_final import _prepare_stream, run_seed

SLUGS = {
    "campus": "5G Campus QoS",
    "ugr16": "UGR'16",
    "nordicdat": "NordicDat",
    "5g_nr": "5G NR Latency",
    "5G Campus QoS": "5G Campus QoS",
    "UGR'16": "UGR'16",
    "NordicDat": "NordicDat",
    "5G NR Latency": "5G NR Latency",
}
OUT_SLUG = {
    "5G Campus QoS": "5g_campus",
    "UGR'16": "ugr16",
    "NordicDat": "nordicdat",
    "5G NR Latency": "5g_nr",
}


def _one(name, out_slug, seed, cfg, mode):
    t0 = time.time()
    stream, sd = _prepare_stream(name)
    df = run_seed(stream, sd, seed, cfg, mode)
    df["dataset"] = name
    print(f"  [{name} seed={seed}] {len(df)} rows in {time.time() - t0:.1f}s", flush=True)
    return df


def run(streams, seeds, cfg, mode, out_tag, n_jobs=4, stream_parallel=False):
    ensure_dirs()
    from joblib import Parallel, delayed
    if stream_parallel:
        jobs = [(SLUGS[s], OUT_SLUG[SLUGS[s]]) for s in streams]
        results = Parallel(n_jobs=len(jobs), verbose=5)(
            delayed(_run_stream)(name, slug, seeds, cfg, mode) for (name, slug) in jobs)
        allw = pd.concat(results, ignore_index=True)
    else:
        jobs = [(SLUGS[s], OUT_SLUG[SLUGS[s]], seed) for s in streams for seed in seeds]
        n_jobs = min(n_jobs, len(jobs))
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_one)(name, slug, seed, cfg, mode) for (name, slug, seed) in jobs)
        allw = pd.concat(results, ignore_index=True)
    for name in set(SLUGS[s] for s in streams):
        slug = OUT_SLUG[name]
        sub = allw[allw["dataset"] == name]
        sub.to_csv(os.path.join(RAW_FINAL, f"per_window_{slug}_{out_tag}.csv"), index=False)
        agg = aggregate(sub)
        agg.to_csv(os.path.join(RAW_FINAL, f"summary_{slug}_{out_tag}.csv"), index=False)
        print(f"[{name}] -> {slug}_{out_tag}: {len(sub)} window rows, {len(agg)} method rows")
    allw.to_csv(os.path.join(RAW_FINAL, f"per_window_all_{out_tag}.csv"), index=False)
    return allw


def _run_stream(name, out_slug, seeds, cfg, mode):
    """Run every seed of one stream sequentially (used with --stream-parallel so
    sklearn stays single-threaded and the four streams share the cores)."""
    stream, sd = _prepare_stream(name)
    frames = []
    for seed in seeds:
        t0 = time.time()
        df = run_seed(stream, sd, seed, cfg, mode)
        df["dataset"] = name
        frames.append(df)
        print(f"  [{name} seed={seed}] {len(df)} rows in {time.time() - t0:.1f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    g = df.groupby("method")
    out = g.agg(
        macro_f1=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        adaptation_cpu_sec=("adaptation_cpu_sec", "mean"),
        prediction_cpu_sec=("prediction_cpu_sec", "mean"),
        total_cpu_sec=("total_cpu_sec", "mean"),
        retrain_events=("is_retrain", "sum"),
        reuse_events=("is_reuse", "sum"),
        n_windows=("window_id", "count"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    if "gate_allow" in df.columns:
        out = out.merge(g["gate_allow"].sum().rename("gate_allow").reset_index(), on="method", how="left")
        out = out.merge(g["gate_block"].sum().rename("gate_block").reset_index(), on="method", how="left")
    # per-seed averages for the event counts
    per_seed = df.groupby(["method", "seed"]).agg(
        r=("is_retrain", "sum"), u=("is_reuse", "sum")).reset_index()
    ev = per_seed.groupby("method").agg(per_seed_retrains=("r", "mean"),
                                        per_seed_reuses=("u", "mean")).reset_index()
    out = out.merge(ev, on="method", how="left")
    return out.sort_values("method").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev", choices=["dev", "final"])
    ap.add_argument("--streams", default="all")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--stream-parallel", action="store_true")
    a = ap.parse_args()

    if a.streams == "all":
        streams = ["campus", "ugr16", "nordicdat", "5g_nr"]
    else:
        streams = a.streams.split(",")
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",")]
    else:
        seeds = SEEDS_DEV if a.mode == "dev" else SEEDS_FINAL
    cfg = dict(DEFAULT_CFG)
    cfg_path = a.cfg
    if not cfg_path and a.mode == "final":
        cand = os.path.join(CONFIG_DIR, "final.yaml")
        if os.path.exists(cand):
            cfg_path = cand
    if cfg_path:
        with open(cfg_path) as f:
            text = f.read()
        try:
            import yaml
            loaded = yaml.safe_load(text) or {}
        except Exception:
            loaded = json.loads(text)
        cfg.update({k: v for k, v in loaded.items() if v is not None})
        print(f"loaded config {cfg_path}", flush=True)
    out_tag = a.tag or a.mode
    t0 = time.time()
    run(streams, seeds, cfg, a.mode, out_tag, n_jobs=a.n_jobs,
        stream_parallel=a.stream_parallel)
    print(f"TOTAL {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
