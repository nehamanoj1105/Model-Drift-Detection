"""Step D: regenerate the manuscript figures from the v2 results.

Figures written to results/final/figures/ (PNG 300 dpi + PDF):
  primary_performance   corrected cross-stream macro-F1, all four streams on one
                        metric (per-window mean), one bar per model
  primary_pooled        the same on the pooled metric, to show the scale shift
  metric_mixing         per-window vs pooled macro-F1 per stream, to expose the
                        mixed scale in the published Table II
  ugr16_streaming       UGR'16 per-window macro-F1 with reuse/retrain markers
  ugr16_rolling         UGR'16 rolling accuracy (window 10), all models
  cost_tradeoff         accuracy vs adaptation CPU, one point per (stream, model)
  5g_nr_refresh         5G NR adaptation CPU and refresh/tree counts for the
                        cheap-refresh anomaly

All inputs are committed v2 CSVs.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

OUT = os.path.join(lib.ROOT, "results", "final", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True,
                     "savefig.bbox": "tight"})

MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
SHORT = {"Frozen": "Frozen", "Event-Driven": "Event-Driven",
         "Full Retraining": "Full Retrain.", "RAPT": "RAPT",
         "RAPT-Enhanced": "RAPT-Enh"}
COLORS = {"Frozen": "#7f7f7f", "Event-Driven": "#1f77b4",
          "Full Retraining": "#d62728", "RAPT": "#2ca02c",
          "RAPT-Enhanced": "#17becf"}
STREAM_ORDER = ["5g_campus", "ugr16", "nordicdat", "5g_nr"]
STREAM_LABEL = {"5g_campus": "5G Campus", "ugr16": "UGR'16",
                "nordicdat": "NordicDat", "5g_nr": "5G NR"}


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def load():
    B = pd.read_csv(os.path.join(HERE, "B_pooled_all_streams.csv"))
    P = pd.read_csv(os.path.join(HERE, "B_perwindow_all_streams.csv"))
    return B, P


def fig_primary(P, metric="per_window_macro_f1", name="primary_performance",
                ylabel="Macro-F1 (per-window mean $\\pm$ sd over seeds)"):
    fig, ax = plt.subplots(figsize=(7.9, 4.2))
    x = np.arange(len(STREAM_ORDER)); w = 0.16
    for i, m in enumerate(MODELS):
        means, errs = [], []
        for s in STREAM_ORDER:
            g = P[(P.dataset == s) & (P.method == m)][metric]
            means.append(g.mean()); errs.append(g.std())
        ax.bar(x + (i - 2) * w, means, w, yerr=errs, capsize=2.5,
               label=SHORT[m], color=COLORS[m], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([STREAM_LABEL[s] for s in STREAM_ORDER])
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    _save(fig, name)


def fig_metric_mixing(P, B):
    fig, ax = plt.subplots(figsize=(7.9, 4.2))
    x = np.arange(len(STREAM_ORDER)); w = 0.16
    pw = P[P.method.isin(MODELS)].groupby(["dataset", "method"])["per_window_macro_f1"].mean()
    po = B[B.method.isin(MODELS)].groupby(["dataset", "method"])["pooled_macro_f1"].mean()
    for i, m in enumerate(MODELS):
        for s in STREAM_ORDER:
            ax.bar(x[STREAM_ORDER.index(s)] + (i - 2) * w, pw.loc[(s, m)], w,
                   color=COLORS[m], edgecolor="black", linewidth=0.4)
            ax.plot(x[STREAM_ORDER.index(s)] + (i - 2) * w, po.loc[(s, m)], "k*",
                    markersize=6, zorder=5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], edgecolor="black",
                             label=SHORT[m]) for m in MODELS]
    handles.append(plt.Line2D([], [], marker="*", color="black", linestyle="",
                              label="pooled"))
    ax.set_xticks(x); ax.set_xticklabels([STREAM_LABEL[s] for s in STREAM_ORDER])
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
    ax.legend(handles=handles, fontsize=8, ncol=6, loc="upper center")
    ax.set_title("Bars: per-window mean (paper's scale). Stars: pooled.")
    _save(fig, "metric_mixing")


def fig_ugr16_streaming(seed=42):
    d = pd.read_csv(os.path.join(lib.RAW, f"B_ugr16_seed{seed}.csv"))
    d = d[d.method.isin(MODELS)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for m in MODELS:
        g = d[d.method == m].sort_values("window_id")
        ax.plot(g.window_id, g.macro_f1, color=COLORS[m], lw=1.2, label=SHORT[m])
    r = d[(d.method == "RAPT")]
    ax.scatter(r[r.is_reuse == 1].window_id, r[r.is_reuse == 1].macro_f1,
               marker="v", s=22, color="#2ca02c", edgecolor="black",
               linewidth=0.3, zorder=6, label="RAPT reuse")
    ax.scatter(r[r.is_retrain == 1].window_id, r[r.is_retrain == 1].macro_f1,
               marker="^", s=26, color="#d62728", edgecolor="black",
               linewidth=0.3, zorder=6, label="RAPT retrain")
    ax.set_xlabel("streaming window"); ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.set_title(f"UGR'16 — per-window macro-F1 (seed {seed})")
    _save(fig, "ugr16_streaming")


def fig_ugr16_rolling(seed=42, w=10):
    d = pd.read_csv(os.path.join(lib.RAW, f"B_ugr16_seed{seed}.csv"))
    d = d[d.method.isin(MODELS)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for m in MODELS:
        g = d[d.method == m].sort_values("window_id")
        ax.plot(g.window_id, g.accuracy.rolling(w, min_periods=1).mean(),
                color=COLORS[m], lw=1.4, label=SHORT[m])
    ax.set_xlabel("streaming window"); ax.set_ylabel(f"Rolling accuracy (w={w})")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, ncol=5, loc="lower right")
    ax.set_title(f"UGR'16 — rolling accuracy (seed {seed})")
    _save(fig, "ugr16_rolling")


def fig_cost_tradeoff(P):
    g = P[P.method.isin(MODELS)].groupby(["dataset", "method"]).agg(
        f1=("per_window_macro_f1", "mean"),
        cpu=("adaptation_cpu_sec", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    marks = {"5g_campus": "o", "ugr16": "s", "nordicdat": "^", "5g_nr": "D"}
    for s in STREAM_ORDER:
        for m in MODELS:
            r = g[(g.dataset == s) & (g.method == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.scatter(r.cpu, r.f1, marker=marks[s], s=60, color=COLORS[m],
                       edgecolor="black", linewidth=0.4, zorder=3)
    hs = [plt.Line2D([], [], marker=marks[s], color="k", linestyle="", label=STREAM_LABEL[s])
          for s in STREAM_ORDER]
    hs += [plt.Line2D([], [], marker="o", color=COLORS[m], linestyle="", label=SHORT[m])
           for m in MODELS]
    ax.legend(handles=hs, fontsize=7, ncol=3, loc="lower right")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Adaptation CPU (s, symlog)"); ax.set_ylabel("Macro-F1 (per-window mean)")
    _save(fig, "cost_tradeoff")


def fig_5g_nr_refresh():
    p = os.path.join(HERE, "A3_5g_nr_table_ii.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.method.isin(["Full Retraining", "RAPT", "RAPT-Cheap"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(d)); w = 0.28
    ax.bar(x - w, d["adaptation_cpu_sec"], w, label="Adaptation CPU (s)",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(x, d["trees_trained"], w, label="Trees trained", color="#ff7f0e",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w, d["refresh_events"], w, label="Refresh events", color="#2ca02c",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(d["method"])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("count / seconds (symlog)")
    ax.legend(fontsize=8)
    ax.set_title("5G NR — cheap periodic refresh fires every 5 samples")
    _save(fig, "5g_nr_refresh")


def main():
    B, P = load()
    fig_primary(P, "per_window_macro_f1", "primary_performance",
                "Macro-F1 (per-window mean $\\pm$ sd over seeds)")
    fig_primary(B, "pooled_macro_f1", "primary_pooled",
                "Macro-F1 (pooled $\\pm$ sd over seeds)")
    fig_metric_mixing(P, B)
    fig_ugr16_streaming()
    fig_ugr16_rolling()
    fig_cost_tradeoff(P)
    fig_5g_nr_refresh()


if __name__ == "__main__":
    main()
