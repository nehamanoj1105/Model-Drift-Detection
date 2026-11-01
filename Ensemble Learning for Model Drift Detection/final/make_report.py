#!/usr/bin/env python3
"""
make_report.py — every table, figure, statistic and headline number for the
final RAPT paper, computed from results/final/raw/*.parquet and the other saved
result files. Nothing here is computed inside the experiment loop.

    python make_report.py --config final.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))

from stats_utils import seed_level, block_bootstrap, cost_reduction_ci  # noqa

PRIMARY = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
ABLATION = ["RAPT-Full", "RAPT-Evidence", "RAPT-Floor", "RAPT-Periodic-FullRefit",
            "RAPT-Cheap", "RAPT-Incremental"]
DETECTORS = ["ADWIN", "Page-Hinkley", "EDDM", "ECDD-EWMA"]


def load_raw(cfg):
    raw_dir = os.path.join(PROJECT_DIR, cfg["output_dir"], "final", "raw")
    wins, seeds = {}, {}
    for d in cfg["datasets"]:
        slug = d["slug"]
        wins[d["name"]] = pd.read_parquet(os.path.join(raw_dir, f"per_window_{slug}.parquet"))
        seeds[d["name"]] = pd.read_parquet(os.path.join(raw_dir, f"per_seed_{slug}.parquet"))
    return wins, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final.yaml")
    args = ap.parse_args()
    with open(os.path.join(HERE, args.config)) as f:
        cfg = yaml.safe_load(f)

    out_root = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")
    wins, seeds = load_raw(cfg)
    ds_names = [d["name"] for d in cfg["datasets"]]

    # =====================================================================
    # Determinism check: std across seeds exactly 0.0000 must be reported
    # =====================================================================
    det_lines = []
    for name in ds_names:
        s = seeds[name]
        for model in PRIMARY + ABLATION + DETECTORS:
            sub = s[s["model"] == model]
            if len(sub) < 2:
                continue
            sd = float(sub["macro_f1"].std(ddof=1))
            if sd == 0.0:
                reason = ("deterministic model / seed not applied"
                          if model == "Frozen" or "Incremental" in model
                          else "seed has no effect on this model")
                det_lines.append(f"{name} / {model}: std(macro_f1)=0.0000 across "
                                 f"{len(sub)} seeds -- {reason}")
    with open(os.path.join(out_root, "determinism_check.txt"), "w") as f:
        f.write("\n".join(det_lines) if det_lines else "No model has zero seed variance.")

    # =====================================================================
    # Primary table (mean +/- std over seeds)
    # =====================================================================
    def agg_table(models):
        rows = []
        for name in ds_names:
            s = seeds[name]
            for model in models:
                sub = s[s["model"] == model]
                if sub.empty:
                    continue
                rows.append({
                    "dataset": name, "model": model,
                    "macro_f1": sub["macro_f1"].mean(), "macro_f1_std": sub["macro_f1"].std(ddof=1),
                    "accuracy": sub["accuracy"].mean(), "accuracy_std": sub["accuracy"].std(ddof=1),
                    "precision": sub["precision"].mean(), "precision_std": sub["precision"].std(ddof=1),
                    "recall": sub["recall"].mean(), "recall_std": sub["recall"].std(ddof=1),
                    "adapt_cpu_s": sub["adapt_cpu_s"].mean(), "adapt_cpu_std": sub["adapt_cpu_s"].std(ddof=1),
                    "runtime_s": sub["runtime_s"].mean(), "runtime_std": sub["runtime_s"].std(ddof=1),
                    "retrains": sub["retrains"].mean(), "reuses": sub["reuses"].mean(),
                    "refreshes": sub["refreshes"].mean(),
                    "trees_trained": sub["trees_trained"].mean(),
                    "trees_reused": sub["trees_reused"].mean(),
                    "n_seeds": len(sub),
                })
        return pd.DataFrame(rows)

    prim = agg_table(PRIMARY)
    prim.to_csv(os.path.join(out_root, "table_primary.csv"), index=False)
    abl = agg_table(ABLATION)
    abl.to_csv(os.path.join(out_root, "table_ablation.csv"), index=False)
    det = agg_table(DETECTORS)
    det.to_csv(os.path.join(out_root, "table_detectors.csv"), index=False)

    # =====================================================================
    # Statistics (A seed-level primary, B window-level secondary)
    # =====================================================================
    comps = [("RAPT", "Full Retraining"), ("RAPT-Enhanced", "RAPT"),
             ("RAPT-Enhanced", "Full Retraining"), ("RAPT-Cheap", "Full Retraining"),
             ("RAPT-Cheap", "RAPT-Periodic-FullRefit")]
    stat_rows = []
    below_a = below_b = 0
    for name in ds_names:
        s, w = seeds[name], wins[name]
        for m1, m2 in comps:
            a1 = dict(zip(s[s.model == m1]["seed"], s[s.model == m1]["macro_f1"]))
            a2 = dict(zip(s[s.model == m2]["seed"], s[s.model == m2]["macro_f1"]))
            if not a1 or not a2:
                continue
            sl = seed_level(a1, a2)
            w1 = {k: v.values for k, v in w[w.model == m1].groupby("seed")["window_f1"]}
            w2 = {k: v.values for k, v in w[w.model == m2].groupby("seed")["window_f1"]}
            bb = block_bootstrap(w1, w2, block=cfg["stats"]["bootstrap_block_windows"],
                                 reps=cfg["stats"]["bootstrap_reps"])
            stat_rows.append({"comparison": f"{m1} vs {m2}", "dataset": name,
                              **sl, **bb})
            if m1 == "RAPT" and m2 == "Full Retraining":
                if sl["significant"] and sl["delta"] < 0:
                    below_a += 1
                if bb["p_window"] < 0.05 and bb["delta_window"] < 0:
                    below_b += 1
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(out_root, "stats.csv"), index=False)

    # cost reduction with CI
    cost_rows = []
    for name in ds_names:
        s = seeds[name]
        fr = dict(zip(s[s.model == "Full Retraining"]["seed"],
                      s[s.model == "Full Retraining"]["adapt_cpu_s"]))
        for model in ["RAPT", "RAPT-Enhanced", "RAPT-Cheap"]:
            mm = dict(zip(s[s.model == model]["seed"], s[s.model == model]["adapt_cpu_s"]))
            if not mm:
                continue
            from stats_utils import cost_reduction_ci
            ci = cost_reduction_ci(mm, fr)
            cost_rows.append({"dataset": name, "model": model, **ci})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(os.path.join(out_root, "cost_savings.csv"), index=False)

    # =====================================================================
    # UGR'16 diagnostics
    # =====================================================================
    diag_rows = []
    pe_path = os.path.join(out_root, "rapt_policy_events.csv")
    if os.path.exists(pe_path):
        pe = pd.read_csv(pe_path)
        for name in ds_names:
            for model in ["RAPT", "RAPT-Enhanced"]:
                sub = pe[(pe.dataset == name) & (pe.model == model) &
                         (pe.event == "reuse")]
                if sub.empty:
                    continue
                diag_rows.append({"dataset": name, "model": model,
                                  "n_reuse_events": len(sub),
                                  "mean_stored_origin_f1":
                                      sub["stored_policy_origin_f1"].mean(),
                                  "n_missing": int(sub["stored_policy_origin_f1"].isna().sum())})
    pd.DataFrame(diag_rows).to_csv(os.path.join(out_root, "ugr_staleness.csv"), index=False)

    # =====================================================================
    # numbers.tex
    # =====================================================================
    def num(x, fmt="{:.4f}"):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NaN"
        return fmt.format(x)

    def macro(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"

    lines = ["% Auto-generated by make_report.py. Do not edit.\n"]
    for name in ds_names:
        key = {"5G Campus": "Campus", "UGR'16": "Ugr", "NordicDat": "Nordic",
               "5G NR Lat.": "Nr"}[name]
        s = seeds[name]
        for model, mk in [("Frozen", "Froz"), ("Full Retraining", "Full"),
                          ("RAPT", "Rapt"), ("RAPT-Enhanced", "RaptE"),
                          ("Event-Driven", "Event")]:
            sub = s[s.model == model]
            if sub.empty:
                continue
            lines.append(macro(f"{mk.lower()}{key}F", num(sub["macro_f1"].mean())))
        fr = s[s.model == "Full Retraining"]["adapt_cpu_s"].mean()
        rp = s[s.model == "RAPT"]["adapt_cpu_s"].mean()
        lines.append(macro(f"cpuFull{key}", num(fr, "{:.4f}")))
        lines.append(macro(f"cpuRapt{key}", num(rp, "{:.4f}")))
        if fr > 0:
            lines.append(macro(f"cpuSaved{key}", num((fr - rp) / fr * 100, "{:.1f}")))
        # RAPT-Cheap vs Full Retraining: delta, seed-level p, cost reduction.
        cr = stats_df[(stats_df.dataset == name) &
                   (stats_df.comparison == "RAPT-Cheap vs Full Retraining")]
        if len(cr):
            lines.append(macro(f"deltaCheap{key}", num(cr["delta"].iloc[0], "{:+.4f}")))
            lines.append(macro(f"pCheap{key}", num(cr["p_seed"].iloc[0], "{:.4f}")))
        cc = cost_df[(cost_df.dataset == name) & (cost_df.model == "RAPT-Cheap")]
        if len(cc):
            lines.append(macro(f"cpuSavedCheap{key}", num(cc["reduction_pct"].iloc[0], "{:.1f}")))
        # RAPT vs Full Retraining delta / p.
        rr = stats_df[(stats_df.dataset == name) &
                   (stats_df.comparison == "RAPT vs Full Retraining")]
        if len(rr):
            lines.append(macro(f"deltaRapt{key}", num(rr["delta"].iloc[0], "{:+.4f}")))
            lines.append(macro(f"pRapt{key}", num(rr["p_seed"].iloc[0], "{:.4f}")))
    lines.append(macro("nRaptBelowA", str(below_a)))
    lines.append(macro("nRaptBelowB", str(below_b)))
    with open(os.path.join(out_root, "numbers.tex"), "w") as f:
        f.writelines(lines)

    print(f"stats rows={len(stats_df)}; RAPT below FR: A={below_a}, B={below_b}")
    print("make_report complete.")


if __name__ == "__main__":
    main()
