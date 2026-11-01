"""Regenerate the five manuscript figures from committed artifacts.

Outputs 300 dpi PNG (matching the originals in paper/figures/) and vector PDF,
into paper/figures/.

Sources (authoritative, committed):
  primary_performance / computational_cost
      results/experiment_9a_three/raw/summary_{ugr16,nordicdat}_full.csv
  rapt_ablation
      Final_Experiments/results/tables/table_final_main.csv
  detector_comparison
      results/experiment_9a_three/raw/summary_{5g_campus,ugr16,nordicdat}_full.csv
  cost_tradeoff
      Final_Experiments/results/raw/summary_full.csv (per-seed points)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW9A = os.path.join(ROOT, "results", "experiment_9a_three", "raw")
FINAL = os.path.join(ROOT, "Final_Experiments", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

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


def _summary(stream):
    return pd.read_csv(os.path.join(RAW9A, f"summary_{stream}_full.csv"))


def _agg(df, cols):
    return df.groupby("method")[cols].agg(["mean", "std"])


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def fig_primary():
    """Macro-F1 on the two largest cross-dataset streams: UGR'16, NordicDat."""
    streams = [("UGR'16", "ugr16"), ("NordicDat", "nordicdat")]
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    x = np.arange(len(streams))
    w = 0.15
    for i, m in enumerate(MODELS5):
        means, errs = [], []
        for _, key in streams:
            g = _summary(key).groupby("method")["macro_f1"]
            means.append(g.mean()[m])
            errs.append(g.std()[m])
        ax.bar(x + (i - 2) * w, means, w, yerr=errs, capsize=2.5,
               label=SHORT[m], color=COLORS[m], edgecolor="black",
               linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in streams])
    ax.set_ylabel("Macro-F1 (mean $\\pm$ sd over seeds)")
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    _save(fig, "primary_performance")


def fig_cost():
    """Adaptation CPU on the two largest streams."""
    streams = [("UGR'16", "ugr16"), ("NordicDat", "nordicdat")]
    fig, ax = plt.subplots(figsize=(7.9, 4.9))
    x = np.arange(len(streams))
    w = 0.15
    for i, m in enumerate(MODELS5):
        vals = [_summary(key).groupby("method")["adaptation_cpu_sec"]
                .mean()[m] for _, key in streams]
        ax.bar(x + (i - 2) * w, vals, w, label=SHORT[m], color=COLORS[m],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in streams])
    ax.set_ylabel("Adaptation CPU (s)")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    _save(fig, "computational_cost")


def fig_ablation():
    """Efficiency-ablation ladder on 5G Campus (Table III)."""
    df = pd.read_csv(os.path.join(FINAL, "tables", "table_final_main.csv"))
    order = ["Full Retraining", "RAPT_FULL", "RAPT_EVIDENCE", "RAPT_FLOOR",
             "RAPT_CHEAP", "RAPT_INCR"]
    pretty = {"Full Retraining": "Full Retrain.", "RAPT_FULL": "RAPT-Full",
              "RAPT_EVIDENCE": "RAPT-Evidence", "RAPT_FLOOR": "RAPT-Floor",
              "RAPT_CHEAP": "RAPT-Cheap", "RAPT_INCR": "RAPT-Incremental"}
    d = df.set_index("Model").loc[order]
    f1 = d["Macro-F1"].str.split(r"\s*\+/-\s*", expand=True, regex=True)[0].astype(float).values
    cpu = d["Adapt CPU (s)"].str.split(r"\s*\+/-\s*", expand=True, regex=True)[0].astype(float).values

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    xs = np.arange(len(order))
    ax.plot(xs, f1, "o-", color="#9467bd", label="Macro-F1")
    ax.set_ylabel("Macro-F1", color="#9467bd")
    ax.set_ylim(0.94, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty[o] for o in order], rotation=15, ha="right")
    ax2 = ax.twinx()
    ax2.bar(xs, cpu, 0.5, color="#1f77b4", alpha=0.35, label="Adapt CPU")
    ax2.set_ylabel("Adaptation CPU (s)", color="#1f77b4")
    ax2.grid(False)
    ax.set_title("5G Campus — RAPT efficiency ablation")
    _save(fig, "rapt_ablation")


def fig_detectors():
    """Historical detector operating points: macro-F1 vs adaptation CPU."""
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    streams = {"5G Campus": "5g_campus", "UGR'16": "ugr16",
               "NordicDat": "nordicdat"}
    dets = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]
    markers = {"ADWIN": "o", "Page-Hinkley": "s", "EDD": "^", "EDMA": "D"}
    for ds, key in streams.items():
        g = _summary(key).groupby("method")[["macro_f1",
                                             "adaptation_cpu_sec"]].mean()
        for det in dets:
            ax.scatter(g.loc[det, "adaptation_cpu_sec"],
                       g.loc[det, "macro_f1"], marker=markers[det], s=55,
                       color="#d62728", edgecolor="black", linewidth=0.4,
                       zorder=3)
        ax.scatter(g.loc["RAPT", "adaptation_cpu_sec"],
                   g.loc["RAPT", "macro_f1"], marker="*", s=170,
                   color="#2ca02c", edgecolor="black", linewidth=0.5,
                   zorder=4)
        ax.annotate(ds, (g.loc["EDD", "adaptation_cpu_sec"],
                         g.loc["EDD", "macro_f1"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    handles = [plt.Line2D([], [], marker=markers[d], color="#d62728",
                          linestyle="", markeredgecolor="black", label=d)
               for d in dets]
    handles.append(plt.Line2D([], [], marker="*", color="#2ca02c",
                              linestyle="", markeredgecolor="black",
                              markersize=12, label="RAPT"))
    ax.legend(handles=handles, fontsize=8, ncol=5, loc="lower right")
    ax.set_xlabel("Adaptation CPU (s)")
    ax.set_ylabel("Macro-F1")
    _save(fig, "detector_comparison")


def fig_tradeoff():
    """Accuracy vs adaptation cost on 5G Campus, one point per seed."""
    df = pd.read_csv(os.path.join(FINAL, "raw", "summary_full.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    models = ["Frozen", "Event-Driven", "Full Retraining", "RAPT_T2",
              "RAPT_FULL", "RAPT_CHEAP", "RAPT_FLOOR", "RAPT_INCR"]
    colors = {"Frozen": "#7f7f7f", "Event-Driven": "#1f77b4",
              "Full Retraining": "#d62728", "RAPT_T2": "#2ca02c",
              "RAPT_FULL": "#9467bd", "RAPT_CHEAP": "#ff7f0e",
              "RAPT_FLOOR": "#8c564b", "RAPT_INCR": "#e377c2"}
    for m in models:
        d = df[df.method == m]
        if d.empty:
            continue
        ax.scatter(d["adaptation_cpu_sec"], d["macro_f1"], s=45,
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
