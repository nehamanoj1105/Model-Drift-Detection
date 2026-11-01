"""Figure: UGR'16 rolling accuracy over evaluation windows.

Source data (authoritative, do not substitute):
  results/experiment_9a_three/raw/per_window_ugr16_full.csv
  results/experiment_9a_three/raw/summary_ugr16_full.csv

Rolling convention matches experiments/exp9a/figures_tables_9a.py
(mean over seeds per window_id, then rolling(5, min_periods=1)).
"""
import csv
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PER_WINDOW = os.path.join(ROOT, "results", "experiment_9a_three", "raw",
                          "per_window_ugr16_full.csv")
OUT = os.path.join(HERE, "ugr_rolling_accuracy.png")

MODELS = ["RAPT", "RAPT-Enhanced", "Full Retraining"]
COLORS = {"RAPT": "#1f77b4", "RAPT-Enhanced": "#2ca02c",
          "Full Retraining": "#ff7f0e"}
LABELS = {"Full Retraining": "Full Retraining"}


def load():
    with open(PER_WINDOW) as f:
        rows = list(csv.DictReader(f))
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        acc[r["method"]][int(r["window_id"])].append(float(r["accuracy"]))
    series = {}
    for m in MODELS:
        ws = sorted(acc[m])
        vals = np.array([np.mean(acc[m][w]) for w in ws])
        rolling = np.convolve(vals, np.ones(5) / 5, mode="full")[:len(vals)]
        # min_periods=1: divide by the number of available points at the start
        for i in range(min(4, len(rolling))):
            rolling[i] = vals[: i + 1].mean()
        series[m] = (np.array(ws), vals, rolling)
    return series


def novelty_refit_windows():
    """First window at which each regime appears -> RAPT trains a new policy."""
    with open(PER_WINDOW) as f:
        rows = list(csv.DictReader(f))
    first = {}
    for r in rows:
        if r["method"] == "RAPT" and r["seed"] == "42":
            first.setdefault(r["regime_id"], int(r["window_id"]))
    return sorted(first.values())


def main():
    series = load()
    refits = novelty_refit_windows()

    fig, ax = plt.subplots(figsize=(7.16, 3.1))
    for m in MODELS:
        ws, _, rolling = series[m]
        ax.plot(ws, rolling, label=LABELS.get(m, m), color=COLORS[m],
                linewidth=2.2 if m == "RAPT-Enhanced" else 1.6,
                zorder=3 if m == "RAPT-Enhanced" else 2)

    for i, w in enumerate(refits):
        ax.axvline(w, color="k", alpha=0.18, linewidth=0.7,
                   label="Novelty refit" if i == 0 else None)

    ax.set_xlabel("Evaluation window")
    ax.set_ylabel("Accuracy (rolling mean, 5)")
    ax.set_ylim(0.80, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("saved", OUT)

    for m in MODELS:
        ws, vals, rolling = series[m]
        print(f"  {m:16s} mean acc (streamed) = {vals.mean():.4f}")


if __name__ == "__main__":
    main()
