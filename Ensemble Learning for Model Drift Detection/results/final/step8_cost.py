"""Step 8: cost measurement consistency.

The paper's cost columns mix two pipelines (Step 0.5: UGR'16 Full Retraining
adaptation CPU is 23.63s in Table II but 25.30s under the revalidation protocol).
This step measures cost directly under one frozen protocol, with repeated runs,
and reports median / mean / sd so the cost table has a single provenance.

Protocol: revalidation runner (run_stream_v2.run_seed_v2), OMP/MKL threads = 1,
seeds 42-46, REPEATS repeats per seed. Models: the four headline models plus
RAPT-Enhanced. Streams: 5G Campus and UGR'16 (the two the cost claim rests on).

Outputs
  raw/T8_cost_per_run.csv    one row per (dataset, seed, repeat, method)
  T8_cost_table.csv          median/mean/sd adaptation CPU, runtime, events
  T8_cost_vs_paper.csv       measured vs Table II, per model/stream
"""
import os
import re
import sys
import time
import json
from hashlib import sha256

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")
sys.path.insert(0, REVAL)

from common import SEEDS  # noqa: E402
from streams import get_stream  # noqa: E402
import run_stream_v2 as R  # noqa: E402

REPEATS = 3
MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
STREAMS = ["5G Campus QoS", "UGR'16"]
CFG = os.path.join(REVAL, "config_frozen.json")


def measure():
    out = os.path.join(FINAL, "raw", "T8_cost_per_run.csv")
    if os.path.exists(out):
        print(f"   loaded cached {out}", flush=True)
        return pd.read_csv(out)
    rows = []
    for ds in STREAMS:
        stream, sd = get_stream(ds)
        for rep in range(REPEATS):
            for seed in SEEDS:
                pw, sm, pooled, _ = R.run_seed_v2(stream, sd, seed, MODELS)
                for _, r in sm.iterrows():
                    rows.append({"dataset": ds, "seed": seed, "repeat": rep,
                                 "method": r["method"],
                                 "adaptation_cpu_sec": r["adaptation_cpu_sec"],
                                 "total_runtime_sec": r["total_runtime_sec"],
                                 "retrain_events": r["retrain_events"],
                                 "reuse_events": r["reuse_events"],
                                 "refresh_events": r["refresh_events"]})
            print(f"   {ds} repeat {rep} done", flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    return d


def paper_table():
    tex = os.path.join(ROOT, "paper", "main.tex")
    with open(tex) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "\\label{tab:primary}" in l)
    end = next(i for i, l in enumerate(lines) if i > start and "\\bottomrule" in l)
    rows = []
    cur = None
    for ln in lines[start:end]:
        hm = re.search(r"\\multirow\{5\}\{\*\}\{([^}]+)\}", ln)
        if hm:
            cur = hm.group(1).strip()
            continue
        if not ln.strip().startswith("&") or not cur:
            continue
        fields = [f.strip() for f in ln.split("&")]
        if len(fields) < 8:
            continue
        model = re.sub(r"\\[a-zA-Z]+", "", fields[1]).strip()
        nums = re.findall(r"[0-9]+\.[0-9]+", fields[6])
        rows.append({"dataset_tex": cur, "method": model,
                     "paper_adapt_cpu": float(nums[0]) if nums else np.nan})
    d = pd.DataFrame(rows)
    tex2slug = {"5G Campus": "5G Campus QoS", "UGR'16": "UGR'16",
                "NordicDat": "NordicDat", "5G NR Lat.": "5G NR"}
    d["dataset"] = d["dataset_tex"].map(tex2slug)
    return d


def definition_audit():
    """Is 'adaptation CPU' defined the same way for every model?

    In the A1 summaries Frozen's adaptation_cpu equals its init_cpu (the
    one-time initial fit is counted as adaptation) while Event-Driven, Full
    Retraining and RAPT report adaptation_cpu with the initial fit excluded.
    That makes the Frozen cost column non-comparable.
    """
    s = pd.read_csv(os.path.join(REVAL, "A1_summary_all_streams.csv"))
    g = s.groupby(["dataset", "method"]).agg(
        adapt=("adaptation_cpu_sec", "mean"),
        init=("init_cpu_sec", "mean")).reset_index()
    g["adapt_minus_init"] = g["adapt"] - g["init"]
    g["includes_init_fit"] = (g["adapt_minus_init"].abs() < 1e-9) & (g["init"] > 1e-6)
    g.to_csv(os.path.join(FINAL, "T8_cost_definition_audit.csv"), index=False)
    return g


def main():
    print("Step 8")
    if os.path.exists(CFG):
        print(f"   config sha256 {sha256(open(CFG,'rb').read()).hexdigest()}")
    d = measure()
    t = d.groupby(["dataset", "method"]).agg(
        adapt_cpu_median=("adaptation_cpu_sec", "median"),
        adapt_cpu_mean=("adaptation_cpu_sec", "mean"),
        adapt_cpu_sd=("adaptation_cpu_sec", "std"),
        runtime_median=("total_runtime_sec", "median"),
        runtime_mean=("total_runtime_sec", "mean"),
        retrain_events_mean=("retrain_events", "mean"),
        reuse_events_mean=("reuse_events", "mean"),
        refresh_events_mean=("refresh_events", "mean"),
        n_runs=("adaptation_cpu_sec", "size")).reset_index()
    t.to_csv(os.path.join(FINAL, "T8_cost_table.csv"), index=False)
    p = paper_table()
    m = t.merge(p, on=["dataset", "method"], how="inner")
    m["delta_vs_paper"] = m["adapt_cpu_median"] - m["paper_adapt_cpu"]
    m.to_csv(os.path.join(FINAL, "T8_cost_vs_paper.csv"), index=False)
    da = definition_audit()
    print(t.to_string(index=False))
    print()
    print(m[["dataset", "method", "adapt_cpu_median", "paper_adapt_cpu",
             "delta_vs_paper"]].to_string(index=False))
    print("\ncost-definition audit (includes_init_fit flags non-comparable rows):")
    print(da[da.includes_init_fit].to_string(index=False))


if __name__ == "__main__":
    main()
