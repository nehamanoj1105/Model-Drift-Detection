#!/usr/bin/env python3
"""
make_latex.py — ready-to-\\input LaTeX tables from the saved CSVs.

  tab_datasets.tex, tab_primary.tex, tab_ablation.tex, tab_detectors.tex, tab_stats.tex
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

PRIMARY = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
ABLATION = ["RAPT-Full", "RAPT-Evidence", "RAPT-Floor", "RAPT-Periodic-FullRefit",
            "RAPT-Cheap", "RAPT-Incremental"]
DETECTORS = ["ADWIN", "Page-Hinkley", "EDDM", "ECDD-EWMA"]


def fmt(m, s):
    return f"${m:.4f} \\pm {s:.4f}$"


def main():
    with open(os.path.join(HERE, "final.yaml")) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")
    tex = os.path.join(out, "latex")
    os.makedirs(tex, exist_ok=True)
    ds = [d["name"] for d in cfg["datasets"]]
    prim = pd.read_csv(os.path.join(out, "table_primary.csv"))
    abl = pd.read_csv(os.path.join(out, "table_ablation.csv"))
    det = pd.read_csv(os.path.join(out, "table_detectors.csv"))
    stats = pd.read_csv(os.path.join(out, "stats.csv"))
    dstats = pd.read_csv(os.path.join(out, "dataset_stats.csv"))

    def write(name, lines):
        with open(os.path.join(tex, name), "w") as f:
            f.write("\n".join(lines) + "\n")

    # datasets
    L = [r"\begin{table}[t]", r"\caption{Datasets under the final protocol.}",
         r"\label{tab:datasets}", r"\centering", r"\footnotesize",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Samples} & \textbf{Feat.} & \textbf{Win.} & \textbf{Reg.} & \textbf{Recur.} \\",
         r"\midrule"]
    for _, r in dstats.iterrows():
        L.append(f"{r['dataset']} & {int(r['n_samples']):,} & {int(r['n_features'])} & "
                 f"{int(r['total_windows'])} & {int(r['n_regimes'])} & {int(r['n_recurrences'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_datasets.tex", L)

    # primary
    L = [r"\begin{table*}[t]",
         r"\caption{Primary comparison, mean $\pm$ std over seeds 42--46. Bold marks best macro-F1 per stream.}",
         r"\label{tab:primary}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Model} & \textbf{F1} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{CPU (s)} & \textbf{Retr.} & \textbf{Reuse} \\",
         r"\midrule"]
    for n in ds:
        sub = prim[prim.dataset == n]
        best = sub.loc[sub["macro_f1"].idxmax(), "model"]
        for _, r in sub.iterrows():
            f1 = fmt(r["macro_f1"], r["macro_f1_std"])
            if r["model"] == best:
                f1 = r"\mathbf{" + f1 + "}"
            L.append(f"{n} & {r['model']} & {f1} & {fmt(r['accuracy'], r['accuracy_std'])} & "
                     f"{fmt(r['precision'], r['precision_std'])} & {fmt(r['recall'], r['recall_std'])} & "
                     f"{r['adapt_cpu_s']:.4f} & {int(r['retrains'])} & {int(r['reuses'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_primary.tex", L)

    # ablation
    L = [r"\begin{table*}[t]",
         r"\caption{Ablation ladder on all four datasets, mean over seeds 42--46.}",
         r"\label{tab:ablation}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Configuration} & \textbf{F1} & \textbf{CPU (s)} & \textbf{Retr.} & \textbf{Reuse} & \textbf{Refresh} \\",
         r"\midrule"]
    for n in ds:
        for _, r in abl[abl.dataset == n].iterrows():
            L.append(f"{n} & {r['model']} & {fmt(r['macro_f1'], r['macro_f1_std'])} & "
                     f"{r['adapt_cpu_s']:.4f} & {int(r['retrains'])} & {int(r['reuses'])} & "
                     f"{r['refreshes']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_ablation.tex", L)

    # detectors
    L = [r"\begin{table*}[t]",
         r"\caption{Drift detectors in the shared harness, mean over seeds 42--46.}",
         r"\label{tab:detectors}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Detector} & \textbf{F1} & \textbf{CPU (s)} & \textbf{Retrains} \\",
         r"\midrule"]
    for n in ds:
        for _, r in det[det.dataset == n].iterrows():
            L.append(f"{n} & {r['model']} & {fmt(r['macro_f1'], r['macro_f1_std'])} & "
                     f"{r['adapt_cpu_s']:.4f} & {int(r['retrains'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_detectors.tex", L)

    # stats
    L = [r"\begin{table*}[t]",
         r"\caption{Paired comparisons. Test A: seed-level Wilcoxon ($n=5$, min $p=0.0625$) with paired $d_z$ and 95\% $t$-interval. Test B: moving-block bootstrap over windows (block=10).}",
         r"\label{tab:stats}", r"\centering", r"\scriptsize",
         r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"\textbf{Comparison} & \textbf{Dataset} & \textbf{$\Delta$ (seed)} & \textbf{$p_A$} & \textbf{$d_z$} & \textbf{$\Delta$ (win)} & \textbf{$p_B$} \\",
         r"\midrule"]
    for _, r in stats.iterrows():
        L.append(f"{r['comparison']} & {r['dataset']} & {r['delta']:+.4f} & {r['p_seed']:.4f} & "
                 f"{r['d_z']:+.2f} & {r['delta_window']:+.4f} & {r['p_window']:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_stats.tex", L)
    print("LaTeX tables written to", tex)


if __name__ == "__main__":
    main()
