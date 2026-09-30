"""
Cross-dataset comparison: selected 9A dataset (UGR'16) vs existing 9B
(5G NR end-to-end latency QoS).

Produces the eight required figures comparing the same five models across the
two independent recurring-regime streams, plus a combined CSV/LaTeX table.

fig_cross_dataset_accuracy.png        fig_cross_dataset_f1.png
fig_cross_dataset_precision.png       fig_cross_dataset_recall.png
fig_cross_dataset_cpu.png             fig_cross_dataset_runtime.png
fig_cross_dataset_accuracy_cpu.png    fig_cross_dataset_f1_cpu.png
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from three_dataset_config import (
    DATASET_SLUG, MODELS, RAW_DIR, FIGURES_DIR, TABLES_DIR, PROJECT_DIR, ensure_dirs,
)

RESULTS_9B = os.path.join(PROJECT_DIR, "results", "experiment_9b")
SELECTED_9A = "UGR'16"
DS_LABEL = {"9A": "9A UGR'16 (binary anomaly)",
            "9B": "9B 5G latency QoS (3-class)"}
COLORS = {"9A": "#55A868", "9B": "#4C72B0"}
METHODS = MODELS


def _load():
    s9a = pd.read_csv(os.path.join(RAW_DIR, f"summary_{DATASET_SLUG[SELECTED_9A]}_full.csv"))
    s9b = pd.read_csv(os.path.join(RESULTS_9B, "natural_drift", "per_seed.csv"))
    return s9a, s9b


def _metric(s9a, s9b, col, m):
    a = s9a[s9a["method"] == m][col]
    b = s9b[s9b["method"] == m][col]
    return (a.mean(), a.std()), (b.mean(), b.std())


def grouped(s9a, s9b, col, fname, ylabel, log=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(METHODS))
    w = 0.38
    for k, (key, s) in enumerate([("9A", s9a), ("9B", s9b)]):
        means = [s[s["method"] == m][col].mean() for m in METHODS]
        stds = [s[s["method"] == m][col].std() for m in METHODS]
        ax.bar(x + (k - 0.5) * w, means, w, yerr=stds, capsize=3,
               label=DS_LABEL[key], color=COLORS[key], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_title(f"Cross-dataset {ylabel}: 9A vs 9B (5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def scatter(s9a, s9b, xcol, ycol, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for key, s, marker in [("9A", s9a, "s"), ("9B", s9b, "o")]:
        for m in METHODS:
            g = s[s["method"] == m]
            ax.scatter(g[xcol].mean(), g[ycol].mean(), marker=marker, s=75,
                       color=COLORS[key], edgecolor="black", linewidth=0.4)
            ax.annotate(m, (g[xcol].mean(), g[ycol].mean()), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    handles = [plt.Line2D([], [], marker="s", color=COLORS["9A"], linestyle="",
                          label=DS_LABEL["9A"], markersize=8),
               plt.Line2D([], [], marker="o", color=COLORS["9B"], linestyle="",
                          label=DS_LABEL["9B"], markersize=8)]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}: 9A vs 9B")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def main():
    ensure_dirs()
    s9a, s9b = _load()
    grouped(s9a, s9b, "accuracy", "fig_cross_dataset_accuracy.png", "Accuracy")
    grouped(s9a, s9b, "macro_f1", "fig_cross_dataset_f1.png", "Macro-F1")
    grouped(s9a, s9b, "precision", "fig_cross_dataset_precision.png", "Precision")
    grouped(s9a, s9b, "recall", "fig_cross_dataset_recall.png", "Recall")
    grouped(s9a, s9b, "adaptation_cpu_sec", "fig_cross_dataset_cpu.png", "Adaptation CPU (s)")
    grouped(s9a, s9b, "total_cpu_sec", "fig_cross_dataset_runtime.png", "Total runtime (s)", log=True)
    scatter(s9a, s9b, "adaptation_cpu_sec", "accuracy",
            "fig_cross_dataset_accuracy_cpu.png", "Adaptation CPU (s)", "Accuracy")
    scatter(s9a, s9b, "adaptation_cpu_sec", "macro_f1",
            "fig_cross_dataset_f1_cpu.png", "Adaptation CPU (s)", "Macro-F1")

    rows = []
    for key, s in [("9A", s9a), ("9B", s9b)]:
        for m in METHODS:
            g = s[s["method"] == m]
            rows.append({
                "Dataset": DS_LABEL[key], "Model": m,
                "Macro-F1": f"{g['macro_f1'].mean():.4f} +/- {g['macro_f1'].std():.4f}",
                "Accuracy": f"{g['accuracy'].mean():.4f} +/- {g['accuracy'].std():.4f}",
                "Adapt CPU (s)": f"{g['adaptation_cpu_sec'].mean():.3f}",
                "Retrains": f"{g['retrain_events'].mean():.1f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "table_cross_dataset.csv"), index=False)
    with open(os.path.join(TABLES_DIR, "table_cross_dataset.tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True))
    print("Cross-dataset outputs written.")
    print("Figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
