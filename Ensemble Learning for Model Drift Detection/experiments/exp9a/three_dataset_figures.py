"""
Experiment 9A (three telecom datasets) — figures.

Every primary metric figure plots ALL THREE DATASETS together across the five
models. Trade-off, streaming, RAPT-mechanism and drift-detector figures follow.
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
    DATASET_SLUG, MODELS, RAW_DIR, FIGURES_DIR, ensure_dirs,
)
from three_dataset_config import DATASETS as DATASET_LIST

PRIMARY = MODELS
DETECTOR_METHODS = ["ADWIN", "EDD", "Page-Hinkley", "EDMA", "RAPT"]
MODEL_ORDER = MODELS + DETECTOR_METHODS
COLORS = {
    "Frozen": "#4C72B0", "Event-Driven": "#DD8452", "Full Retraining": "#55A868",
    "RAPT": "#C44E52", "RAPT-Enhanced": "#8172B3",
    "ADWIN": "#937860", "EDD": "#DA8BC3", "Page-Hinkley": "#8C8C8C",
    "EDMA": "#CCB974",
}
SLUG = DATASET_SLUG
DS_MARKERS = {"5G Campus QoS": "o", "UGR'16": "s", "NordicDat": "^"}

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid"
              in plt.style.available else "default")


def _load(mode="full"):
    sums = {}
    wins = {}
    for ds in DATASET_LIST:
        s = pd.read_csv(os.path.join(RAW_DIR, f"summary_{SLUG[ds]}_{mode}.csv"))
        w = pd.read_csv(os.path.join(RAW_DIR, f"per_window_{SLUG[ds]}_{mode}.csv"))
        sums[ds] = s
        wins[ds] = w
    return sums, wins


def _metric_by_model(sums, metric):
    rows = {}
    for ds in DATASET_LIST:
        s = sums[ds]
        rows[ds] = [s[s["method"] == m][metric].mean() for m in MODEL_ORDER]
    return rows


# ---------------------------------------------------------------------------
# Primary grouped-bar figures
# ---------------------------------------------------------------------------
def grouped_bars(sums, metric, fname, ylabel, log=False):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_models = len(PRIMARY)
    n_ds = len(DATASET_LIST)
    width = 0.8 / n_ds
    x = np.arange(n_models)
    for k, ds in enumerate(DATASET_LIST):
        s = sums[ds]
        means = [s[s["method"] == m][metric].mean() for m in PRIMARY]
        stds = [s[s["method"] == m][metric].std() for m in PRIMARY]
        ax.bar(x + k * width - 0.4 + width / 2, means, width, yerr=stds,
               capsize=3, label=ds, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PRIMARY, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(title="Dataset", fontsize=9)
    ax.set_title(f"{ylabel} by model — three telecom datasets (5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trade-off scatter
# ---------------------------------------------------------------------------
def tradeoff(sums, xmetric, ymetric, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ds in DATASET_LIST:
        s = sums[ds]
        for m in PRIMARY:
            g = s[s["method"] == m]
            ax.scatter(g[xmetric].mean(), g[ymetric].mean(), marker=DS_MARKERS[ds],
                       s=70, color=COLORS[m], edgecolor="black", linewidth=0.4,
                       label=ds if m == PRIMARY[0] else None)
            ax.annotate(m, (g[xmetric].mean(), g[ymetric].mean()),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles = [plt.Line2D([], [], marker=DS_MARKERS[d], color="gray", linestyle="",
                          markersize=8, label=d) for d in DATASET_LIST]
    ax.legend(handles=handles, fontsize=8, title="Dataset")
    ax.set_title(f"{ylabel} vs {xlabel} (dataset x model)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def streaming(wins, metric, fname, ylabel):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    for ax, ds in zip(axes, DATASET_LIST):
        w = wins[ds]
        for m in PRIMARY:
            g = w[w["method"] == m].sort_values("window_id")
            if g.empty:
                continue
            ax.plot(g["window_id"], g[metric], label=m, color=COLORS[m],
                    linewidth=1.1, alpha=0.9)
        # regime boundaries
        g0 = w[w["method"] == "Frozen"].sort_values("window_id")
        reg = g0["regime_id"].values
        for i in range(1, len(reg)):
            if reg[i] != reg[i - 1]:
                ax.axvline(g0["window_id"].values[i], color="gray", alpha=0.25,
                           linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ds} — streaming {ylabel}")
        ax.legend(fontsize=7, ncol=5)
    axes[-1].set_xlabel("streaming window")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RAPT mechanism per dataset
# ---------------------------------------------------------------------------
def rapt_policy_reuse(wins, ds, fname):
    w = wins[ds]
    g = w[(w["method"] == "RAPT")].sort_values("window_id")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(g["window_id"], g["macro_f1"], color=COLORS["RAPT"], label="F1")
    reuse = g[g["is_reuse"] == 1]
    ax.scatter(reuse["window_id"], reuse["macro_f1"], marker="^", s=45,
               color="#2ca02c", zorder=5, label="reuse event")
    retr = g[g["is_retrain"] == 1]
    ax.scatter(retr["window_id"], retr["macro_f1"], marker="v", s=45,
               color="#d62728", zorder=5, label="retrain event")
    reg = g["regime_id"].values
    for i in range(1, len(reg)):
        if reg[i] != reg[i - 1]:
            ax.axvline(g["window_id"].values[i], color="gray", alpha=0.2, linewidth=0.6)
    ax.set_xlabel("streaming window")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"RAPT policy reuse timeline — {ds}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def rapt_adaptation_cost(wins, ds, fname):
    w = wins[ds]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for m in ["RAPT", "RAPT-Enhanced", "Full Retraining", "Event-Driven"]:
        g = w[w["method"] == m].sort_values("window_id")
        cum = g["adaptation_cpu_sec"].cumsum()
        ax.plot(g["window_id"], cum, label=m, color=COLORS[m], linewidth=1.2)
    ax.set_xlabel("streaming window")
    ax.set_ylabel("cumulative adaptation CPU (s)")
    ax.set_title(f"Chronological adaptation cost — {ds}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def rapt_vs_retraining(sums, ds, fname):
    s = sums[ds]
    methods = ["RAPT", "RAPT-Enhanced", "Full Retraining"]
    metrics = [("macro_f1", "Macro-F1"), ("adaptation_cpu_sec", "Adapt CPU (s)"),
               ("retrain_events", "Retrains"), ("trees_reused", "Reused trees")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (met, lab) in zip(axes, metrics):
        vals = [s[s["method"] == m][met].mean() for m in methods]
        errs = [s[s["method"] == m][met].std() for m in methods]
        ax.bar(methods, vals, yerr=errs, capsize=3,
               color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.4)
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"RAPT vs Full Retraining — {ds}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drift detectors
# ---------------------------------------------------------------------------
def detectors_grouped(sums, metric, fname, ylabel, methods=DETECTOR_METHODS):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(methods)
    width = 0.8 / len(DATASET_LIST)
    x = np.arange(n)
    for k, ds in enumerate(DATASET_LIST):
        s = sums[ds]
        means = [s[s["method"] == m][metric].mean() for m in methods]
        stds = [s[s["method"] == m][metric].std() for m in methods]
        ax.bar(x + k * width - 0.4 + width / 2, means, width, yerr=stds,
               capsize=3, label=ds, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(title="Dataset", fontsize=9)
    ax.set_title(f"{ylabel} — RAPT vs historical drift detectors")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def detector_tradeoff(sums, fname):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ds in DATASET_LIST:
        s = sums[ds]
        for m in DETECTOR_METHODS:
            g = s[s["method"] == m]
            if g.empty:
                continue
            ax.scatter(g["adaptation_cpu_sec"].mean(), g["macro_f1"].mean(),
                       marker=DS_MARKERS[ds], s=70, color=COLORS[m],
                       edgecolor="black", linewidth=0.4)
    handles = [plt.Line2D([], [], marker="o", color=COLORS[m], linestyle="",
                          label=m, markersize=8) for m in DETECTOR_METHODS]
    handles += [plt.Line2D([], [], marker=DS_MARKERS[d], color="gray",
                           linestyle="", label=d, markersize=8) for d in DATASET_LIST]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_xlabel("Adaptation CPU (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Drift detectors: F1 vs adaptation CPU")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def detector_stream(wins, ds, fname):
    w = wins[ds]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for m in DETECTOR_METHODS:
        g = w[w["method"] == m].sort_values("window_id")
        if g.empty:
            continue
        ax.plot(g["window_id"], g["macro_f1"], label=m, color=COLORS[m], linewidth=1.0)
    g0 = w[w["method"] == "ADWIN"].sort_values("window_id")
    det = g0[g0["is_retrain"] == 1]
    ax.scatter(det["window_id"], np.full(len(det), ax.get_ylim()[0]),
               marker="x", color="black", s=25, label="ADWIN adaptation")
    reg = g0["regime_id"].values
    for i in range(1, len(reg)):
        if reg[i] != reg[i - 1]:
            ax.axvline(g0["window_id"].values[i], color="gray", alpha=0.2, linewidth=0.6)
    ax.set_xlabel("streaming window")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"RAPT vs drift detectors — {ds}")
    ax.legend(fontsize=7, ncol=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=200)
    plt.close(fig)


def main(mode="full"):
    ensure_dirs()
    sums, wins = _load(mode)
    for metric, fname, lab, log in [
        ("accuracy", "fig9a_three_dataset_accuracy.png", "Accuracy", False),
        ("macro_f1", "fig9a_three_dataset_macro_f1.png", "Macro-F1", False),
        ("precision", "fig9a_three_dataset_precision.png", "Precision", False),
        ("recall", "fig9a_three_dataset_recall.png", "Recall", False),
        ("adaptation_cpu_sec", "fig9a_three_dataset_adaptation_cpu.png", "Adaptation CPU (s)", False),
        ("total_runtime_sec", "fig9a_three_dataset_runtime.png", "Total runtime (s)", True),
        ("retrain_events", "fig9a_three_dataset_retraining.png", "Retraining events", False),
        ("reuse_events", "fig9a_three_dataset_reuse.png", "RAPT reuse events", False),
        ("trees_reused", "fig9a_three_dataset_reused_trees.png", "Reused trees", False),
        ("trees_trained", "fig9a_three_dataset_new_trees.png", "Newly trained trees", False),
    ]:
        grouped_bars(sums, metric, fname, lab, log=log)

    tradeoff(sums, "adaptation_cpu_sec", "accuracy",
             "fig9a_three_dataset_accuracy_cpu.png", "Adaptation CPU (s)", "Accuracy")
    tradeoff(sums, "adaptation_cpu_sec", "macro_f1",
             "fig9a_three_dataset_f1_cpu.png", "Adaptation CPU (s)", "Macro-F1")
    tradeoff(sums, "total_runtime_sec", "macro_f1",
             "fig9a_three_dataset_f1_runtime.png", "Total runtime (s)", "Macro-F1")

    streaming(wins, "macro_f1", "fig9a_three_dataset_streaming_f1.png", "Macro-F1")
    streaming(wins, "accuracy", "fig9a_three_dataset_streaming_accuracy.png", "Accuracy")

    for ds in DATASET_LIST:
        sl = SLUG[ds]
        rapt_policy_reuse(wins, ds, f"fig9a_rapt_policy_reuse_{sl}.png")
        rapt_adaptation_cost(wins, ds, f"fig9a_rapt_adaptation_cost_{sl}.png")
        rapt_vs_retraining(sums, ds, f"fig9a_rapt_vs_retraining_{sl}.png")
        detector_stream(wins, ds, f"fig9a_detector_stream_{sl}.png")

    detectors_grouped(sums, "macro_f1", "fig9a_rapt_vs_drift_detectors_f1.png", "Macro-F1")
    detectors_grouped(sums, "accuracy", "fig9a_rapt_vs_drift_detectors_accuracy.png", "Accuracy")
    detectors_grouped(sums, "adaptation_cpu_sec", "fig9a_rapt_vs_drift_detectors_cpu.png", "Adaptation CPU (s)")
    detectors_grouped(sums, "total_runtime_sec", "fig9a_rapt_vs_drift_detectors_runtime.png", "Total runtime (s)")
    detectors_grouped(sums, "retrain_events", "fig9a_rapt_vs_drift_detectors_events.png", "Adaptation events")
    detector_tradeoff(sums, "fig9a_rapt_vs_drift_detectors_tradeoff.png")
    print("Figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
