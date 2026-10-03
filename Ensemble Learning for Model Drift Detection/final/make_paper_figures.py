"""Regenerate the five manuscript figures from the single final run.

Every value is read from results/paper_final_run/final (the output of
`run_all.py --config final.yaml`); nothing is hand-entered. Figures are written
as 300 dpi PNG plus vector PDF into both deliverable/figures/ and paper/figures/
so the two manuscripts compile against identical, freshly produced images.

Output names (unchanged, so main.tex needs no edits):
  primary_performance  computational_cost  rapt_ablation
  detector_comparison  cost_tradeoff
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results", "paper_final_run", "final")
OUT_DIRS = [os.path.join(ROOT, "deliverable", "figures"),
            os.path.join(ROOT, "Paper_Final", "figures")]

plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True,
                     "savefig.bbox": "tight"})

MODELS5 = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
SHORT = {"Frozen": "Frozen", "Event-Driven": "Event-Driven",
         "Full Retraining": "Full Retraining", "RAPT": "RAPT",
         "RAPT-Enhanced": "RAPT-Enh"}
COLORS = {"Frozen": "#7f7f7f", "Event-Driven": "#1f77b4",
          "Full Retraining": "#d62728", "RAPT": "#2ca02c",
          "RAPT-Enhanced": "#17becf"}


def _save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + ".png"), dpi=300)
        fig.savefig(os.path.join(d, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def _primary():
    return pd.read_csv(os.path.join(RES, "table_primary.csv"))


def fig_primary():
    """Macro-F1 +/- sd over seeds on the two largest cross-dataset streams."""
    df = _primary()
    streams = [("UGR'16", "UGR'16"), ("NordicDat", "NordicDat")]
    fig, ax = plt.subplots(figsize=(7.9, 2.5))
    x = np.arange(len(streams))
    w = 0.15
    for i, m in enumerate(MODELS5):
        means, errs = [], []
        for _, ds in streams:
            g = df[(df.dataset == ds) & (df.model == m)]
            means.append(float(g.macro_f1.mean()))
            errs.append(float(g.macro_f1_std.mean()))
        ax.bar(x + (i - 2) * w, means, w, yerr=errs, capsize=2.5,
               label=SHORT[m], color=COLORS[m], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in streams])
    ax.set_ylabel("Macro-F1 (mean $\\pm$ sd over seeds)")
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    _save(fig, "primary_performance")


def fig_cost():
    """Adaptation CPU on the two largest cross-dataset streams."""
    df = _primary()
    streams = [("UGR'16", "UGR'16"), ("NordicDat", "NordicDat")]
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    x = np.arange(len(streams))
    w = 0.15
    for i, m in enumerate(MODELS5):
        vals = [float(df[(df.dataset == ds) & (df.model == m)].adapt_cpu_s.mean())
                for _, ds in streams]
        ax.bar(x + (i - 2) * w, vals, w, label=SHORT[m], color=COLORS[m],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in streams])
    ax.set_ylabel("Adaptation CPU (s)")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    _save(fig, "computational_cost")


def fig_ablation():
    """Efficiency-ablation ladder on 5G Campus."""
    df = pd.read_csv(os.path.join(RES, "table_ablation.csv"))
    campus = df[df.dataset == "5G Campus"].set_index("model")
    order = ["RAPT-Full", "RAPT-Evidence", "RAPT-Floor",
             "RAPT-Periodic-FullRefit", "RAPT-Cheap", "RAPT-Incremental"]
    pretty = {"RAPT-Full": "RAPT-Full", "RAPT-Evidence": "RAPT-Evidence",
              "RAPT-Floor": "RAPT-Floor",
              "RAPT-Periodic-FullRefit": "RAPT-Periodic-Full",
              "RAPT-Cheap": "RAPT-Cheap", "RAPT-Incremental": "RAPT-Incremental"}
    prim = _primary()
    fr_row = prim[(prim.dataset == "5G Campus")
                  & (prim.model == "Full Retraining")].iloc[0]
    fr = float(fr_row.macro_f1)
    fr_cpu = float(fr_row.adapt_cpu_s)
    models = ["Full Retrain."] + [pretty[o] for o in order]
    f1 = np.array([fr] + [float(campus.loc[o, "macro_f1"]) for o in order])
    cpu = np.array([fr_cpu] + [float(campus.loc[o, "adapt_cpu_s"]) for o in order])

    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    xs = np.arange(len(models))
    ax.plot(xs, f1, "o-", color="#9467bd", label="Macro-F1")
    ax.set_ylabel("Macro-F1", color="#9467bd")
    ax.set_ylim(0.68, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax2 = ax.twinx()
    ax2.bar(xs, cpu, 0.5, color="#1f77b4", alpha=0.35, label="Adapt CPU")
    ax2.set_ylabel("Adaptation CPU (s)", color="#1f77b4")
    ax2.grid(False)
    ax.set_title("5G Campus — RAPT efficiency ablation")
    _save(fig, "rapt_ablation")


def fig_detectors():
    """Historical detector operating points: macro-F1 vs adaptation CPU."""
    det = pd.read_csv(os.path.join(RES, "table_detectors.csv"))
    prim = _primary()
    fig, ax = plt.subplots(figsize=(7.3, 1.9))
    streams = ["5G Campus", "UGR'16", "NordicDat"]
    dets = ["ADWIN", "Page-Hinkley", "EDDM", "ECDD-EWMA"]
    markers = {"ADWIN": "o", "Page-Hinkley": "s", "EDDM": "^", "ECDD-EWMA": "D"}
    for ds in streams:
        g = det[det.dataset == ds].set_index("model")
        for d in dets:
            ax.scatter(g.loc[d, "adapt_cpu_s"], g.loc[d, "macro_f1"],
                       marker=markers[d], s=55, color="#d62728",
                       edgecolor="black", linewidth=0.4, zorder=3)
        r = prim[(prim.dataset == ds) & (prim.model == "RAPT")].iloc[0]
        ax.scatter(r.adapt_cpu_s, r.macro_f1, marker="*", s=170,
                   color="#2ca02c", edgecolor="black", linewidth=0.5, zorder=4)
        ax.annotate(ds, (g.loc["EDDM", "adapt_cpu_s"], g.loc["EDDM", "macro_f1"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    handles = [plt.Line2D([], [], marker=markers[d], color="#d62728",
                          linestyle="", markeredgecolor="black", label=d)
               for d in dets]
    handles.append(plt.Line2D([], [], marker="*", color="#2ca02c", linestyle="",
                              markeredgecolor="black", markersize=12, label="RAPT"))
    ax.legend(handles=handles, fontsize=8, ncol=5, loc="lower right")
    ax.set_xlabel("Adaptation CPU (s)")
    ax.set_ylabel("Macro-F1")
    _save(fig, "detector_comparison")


def fig_tradeoff():
    """Accuracy vs adaptation cost on 5G Campus, one point per seed."""
    df = pd.read_parquet(os.path.join(RES, "raw", "per_seed_campus.parquet"))
    models = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Full",
              "RAPT-Cheap", "RAPT-Floor", "RAPT-Incremental"]
    colors = {"Frozen": "#7f7f7f", "Event-Driven": "#1f77b4",
              "Full Retraining": "#d62728", "RAPT": "#2ca02c",
              "RAPT-Full": "#9467bd", "RAPT-Cheap": "#ff7f0e",
              "RAPT-Floor": "#8c564b", "RAPT-Incremental": "#e377c2"}
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    for m in models:
        d = df[df.model == m]
        if d.empty:
            continue
        ax.scatter(d["adapt_cpu_s"], d["macro_f1"], s=45,
                   color=colors.get(m, "gray"), label=m, edgecolor="black",
                   linewidth=0.4)
    ax.set_xlabel("Adaptation CPU time (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("5G Campus — accuracy vs adaptation cost (per seed)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "cost_tradeoff")


if __name__ == "__main__":
    fig_primary()
    fig_cost()
    fig_ablation()
    fig_detectors()
    fig_tradeoff()
