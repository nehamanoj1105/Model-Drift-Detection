"""Phase 1 validation harness.

Recomputes every numeric claim in paper/main.tex from raw per-seed outputs.
Prints a machine-readable report; no values are invented.
"""
import csv
import collections
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp9a"))

R9A3 = os.path.join(ROOT, "results", "experiment_9a_three", "raw")
FE = os.path.join(ROOT, "Final_Experiments", "results")

STREAMS = {
    "Campus": "per_window_5g_campus_full.csv",
    "UGR16": "per_window_ugr16_full.csv",
    "Nordic": "per_window_nordicdat_full.csv",
}
SEEDS = ["42", "43", "44", "45", "46"]


def load_pw(name):
    return list(csv.DictReader(open(os.path.join(R9A3, STREAMS[name]))))


def per_seed(rows, method, metric):
    d = collections.defaultdict(list)
    for r in rows:
        if r["method"] == method:
            d[r["seed"]].append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in sorted(d.items())}


def ms(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(ddof=1)


def main():
    print("=" * 78)
    print("STD CONVENTION CHECK")
    rows = load_pw("UGR16")
    v = list(per_seed(rows, "RAPT", "macro_f1").values())
    print(f"  RAPT UGR16 F1 per-seed {[round(x,4) for x in v]}")
    print(f"  mean={np.mean(v):.4f}  ddof=0 {np.std(v):.4f}  ddof=1 {np.std(v,ddof=1):.4f}")
    print(f"  paper 0.8360 +/- 0.0262  ->  ddof=1 MATCH")

    print("=" * 78)
    print("TABLE II (primary) recomputed from per-window means")
    for name in STREAMS:
        rows = load_pw(name)
        methods = ["Frozen", "Event-Driven", "Full Retraining", "RAPT",
                   "RAPT-Enhanced"]
        for m in methods:
            f1 = ms(list(per_seed(rows, m, "macro_f1").values()))
            ac = ms(list(per_seed(rows, m, "accuracy").values()))
            pr = ms(list(per_seed(rows, m, "precision").values()))
            rc = ms(list(per_seed(rows, m, "recall").values()))
            ad = ms(list(per_seed(rows, m, "adaptation_cpu_sec").values()))
            wl = ms(list(per_seed(rows, m, "wall_sec").values()))
            rt = ms(list(per_seed(rows, m, "is_retrain").values()))
            ru = ms(list(per_seed(rows, m, "is_reuse").values()))
            print(f"  {name:7s} {m:15s} F1 {f1[0]:.4f}+/-{f1[1]:.4f} "
                  f"Acc {ac[0]:.4f} Prec {pr[0]:.4f} Rec {rc[0]:.4f} "
                  f"CPU {ad[0]:.4f} Run {wl[0]:.4f} "
                  f"Retr {rt[0]:.1f} Reuse {ru[0]:.1f}")

    print("=" * 78)
    print("CPU REDUCTIONS (RAPT vs Full Retraining)")
    for name in STREAMS:
        rows = load_pw(name)
        fr = ms(list(per_seed(rows, "Full Retraining", "adaptation_cpu_sec").values()))[0]
        rp = ms(list(per_seed(rows, "RAPT", "adaptation_cpu_sec").values()))[0]
        print(f"  {name:7s} FR {fr:.4f} RAPT {rp:.4f} reduction {(rp-fr)/fr*100:.1f}%")

    print("=" * 78)
    print("TABLE III (ablation) from Final_Experiments")
    fm = list(csv.DictReader(open(os.path.join(FE, "tables", "table_final_main.csv"))))
    for r in fm:
        print(f"  {r['Model']:18s} F1 {r['Macro-F1']:18s} CPU {r['Adapt CPU (s)']:18s}")
    print("  -- stats --")
    fs = list(csv.DictReader(open(os.path.join(FE, "tables", "table_final_stats.csv"))))
    for r in fs:
        print(f"  {r['Model']:18s} d={r['mean_delta']:>12s} d_cohen={r['cohens_d']:>10s} "
              f"p={r['wilcoxon_p']:>8s} ci=[{r['ci95_low']}, {r['ci95_high']}]")


if __name__ == "__main__":
    main()
