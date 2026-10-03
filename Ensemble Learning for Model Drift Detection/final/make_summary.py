#!/usr/bin/env python3
"""
make_summary.py — one-screen end-of-run summary: every headline number, plus any
check that failed and any value that could not be computed.

    python make_summary.py --config final.yaml
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))
W = 78


def main():
    with open(os.path.join(HERE, "final.yaml")) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")

    def rd(p):
        fp = os.path.join(out, p)
        return pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()

    prim, stats, cost = rd("table_primary.csv"), rd("stats.csv"), rd("cost_savings.csv")
    det, abl = rd("table_detectors.csv"), rd("table_ablation.csv")
    ds = [d["name"] for d in cfg["datasets"]]

    def g(df, dn, m, c):
        r = df[(df.dataset == dn) & (df.model == m)] if len(df) else pd.DataFrame()
        return float(r[c].iloc[0]) if len(r) else float("nan")

    print("=" * W)
    print("RAPT FINAL EXPERIMENT — SUMMARY".center(W))
    print("=" * W)

    print("\n[MACRO-F1, mean over seeds 42-46]")
    print(f"  {'dataset':<12} {'Frozen':>8} {'EventDrv':>9} {'FullRetr':>9} "
          f"{'RAPT':>8} {'RAPTEnh':>9}")
    for n in ds:
        print(f"  {n:<12} " + " ".join(
            f"{g(prim, n, m, 'macro_f1'):>8.4f}" for m in
            ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]))

    print("\n[ADAPTATION CPU s, mean over seeds]")
    for n in ds:
        print(f"  {n:<12} FullRetr={g(prim, n, 'Full Retraining', 'adapt_cpu_s'):>8.4f}  "
              f"RAPT={g(prim, n, 'RAPT', 'adapt_cpu_s'):>8.4f}  "
              f"saved={100*(g(prim,n,'Full Retraining','adapt_cpu_s')-g(prim,n,'RAPT','adapt_cpu_s'))/max(g(prim,n,'Full Retraining','adapt_cpu_s'),1e-9):>6.1f}%")

    print("\n[PRIMARY TEST A: RAPT vs Full Retraining | test B: block bootstrap]")
    nA = nB = 0
    for n in ds:
        r = stats[(stats.dataset == n) &
                  (stats.comparison == "RAPT vs Full Retraining")] if len(stats) else pd.DataFrame()
        if not len(r):
            continue
        r = r.iloc[0]
        sigA = bool(r["significant"]) and r["delta"] < 0
        sigB = bool(r["p_window"] < 0.05) and r["delta_window"] < 0
        nA += int(sigA); nB += int(sigB)
        print(f"  {n:<12} dF1={r['delta']:+.4f} pA={r['p_seed']:.4f} dz={r['d_z']:+.2f} "
              f"| dWin={r['delta_window']:+.4f} pB={r['p_window']:.4f} "
              f"| belowA={sigA} belowB={sigB}")
    print(f"  RAPT significantly below Full Retraining: A={nA}/4  B={nB}/4 "
          f"(min attainable two-sided p at n=5 is 0.0625)")

    print("\n[DETECTORS: macro-F1 / retrains]")
    for n in ds:
        row = "  " + f"{n:<12}"
        for m in ["ADWIN", "Page-Hinkley", "EDDM", "ECDD-EWMA"]:
            f1 = g(det, n, m, "macro_f1")
            rt = g(det, n, m, "retrains")
            rts = str(int(rt)) if np.isfinite(rt) else "n/a"
            row += f" {m}:{f1:.3f}/{rts}"
        print(row)

    print("\n[UGR'16 attribution]")
    u_rapt = g(prim, "UGR'16", "RAPT", "macro_f1")
    u_enh = g(prim, "UGR'16", "RAPT-Enhanced", "macro_f1")
    u_cheap = g(abl, "UGR'16", "RAPT-Cheap", "macro_f1")
    print(f"  RAPT={u_rapt:.4f}  RAPT-Enhanced={u_enh:.4f}  RAPT-Cheap={u_cheap:.4f}")

    # --- failed checks / uncomputable values -------------------------------
    issues = []
    for p in ["table_primary.csv", "table_ablation.csv", "table_detectors.csv",
              "stats.csv", "cost_savings.csv", "dataset_stats.csv",
              "cpu_breakdown.csv", "refit_timing.csv", "numbers.tex",
              "claims_audit.md", "env.json", "config_used.yaml"]:
        if not os.path.exists(os.path.join(out, p)):
            issues.append(f"MISSING artifact: {p}")
    for name, df in [("table_primary", prim), ("stats", stats), ("detectors", det)]:
        if len(df):
            num = df.select_dtypes(include=[np.number])
            nn = int(num.isna().sum().sum())
            if nn:
                issues.append(f"{name}.csv has {nn} NaN cell(s)")
    lc = os.path.join(out, "leakage_check.txt")
    if os.path.exists(lc):
        if "PASS" not in open(lc).read():
            issues.append("leakage check did not PASS")
    else:
        issues.append("leakage_check.txt missing")
    dc = os.path.join(out, "determinism_check.txt")
    if os.path.exists(dc):
        txt = open(dc).read().strip()
        if txt and "No model" not in txt:
            issues.append("determinism: " + txt.splitlines()[0] + " ...")

    print("\n[CHECKS]")
    if issues:
        for i in issues:
            print("  ! " + i)
    else:
        print("  all expected artifacts present; leakage PASS; no all-NaN tables")
    print("=" * W)


if __name__ == "__main__":
    main()
