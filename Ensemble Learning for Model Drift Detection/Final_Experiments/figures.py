"""Final_Experiments figures (publication quality, saved to results/figures)."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RAW_DIR, FIGURES_DIR, MODELS, RAPT_VARIANTS, ensure_dirs

ALL_MODELS = MODELS + RAPT_VARIANTS
COLORS = {
    "Frozen": "#7f7f7f", "Event-Driven": "#1f77b4", "Full Retraining": "#d62728",
    "RAPT_T2": "#2ca02c", "RAPT_T1": "#98df8a", "RAPT_T1_REFIT": "#17becf",
    "RAPT_FULL": "#9467bd", "RAPT_REL_REFIT": "#c5b0d5",
    "RAPT_REFRESH_W5": "#e377c2", "RAPT_REFRESH_W10": "#f7b6d2",
}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def _load(mode="full"):
    w = pd.read_csv(os.path.join(RAW_DIR, f"per_window_{mode}.csv"))
    s = pd.read_csv(os.path.join(RAW_DIR, f"summary_{mode}.csv"))
    return w, s


def fig_f1_bars(s):
    order = [m for m in ALL_MODELS if m in set(s.method)]
    g = s.groupby("method")["macro_f1"].agg(["mean", "std"]).reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(g)), g["mean"], yerr=g["std"], capsize=3,
           color=[COLORS.get(m, "gray") for m in g.index], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (mean $\\pm$ sd over seeds)")
    ax.set_title("5G Campus QoS — predictive performance by model")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(g["mean"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_final_f1_bars.png"))
    plt.close(fig)


def fig_stream(w):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    reg = w[w.method == "Frozen"].groupby("window_id")["regime_id"].first()
    # regime shading
    bounds, prev, start = [], None, reg.index.min()
    for i, r in reg.items():
        if prev is None:
            prev = r
        elif r != prev:
            bounds.append((start, i - 1, prev)); start = i; prev = r
    bounds.append((start, reg.index.max(), prev))
    palette = {"A": "#f2f2f2", "B": "#e6f2ff", "C": "#fff2e6"}
    for lo, hi, r in bounds:
        ax.axvspan(lo - 0.5, hi + 0.5, color=palette.get(r, "#f7f7f7"), zorder=0)
        ax.text((lo + hi) / 2, 1.02, r, ha="center", fontsize=9, color="#555")
        if lo > reg.index.min():
            ax.axvline(lo - 0.5, color="black", ls="--", lw=0.8, alpha=0.6, zorder=1)
    for m in ALL_MODELS:
        if m not in set(w.method):
            continue
        piv = (w[w.method == m].groupby("window_id")["macro_f1"].mean())
        lw = 2.2 if m == "RAPT_FULL" else 1.1
        ax.plot(piv.index, piv.values, label=m, color=COLORS.get(m), lw=lw,
                alpha=1.0 if m in ("RAPT_FULL", "Full Retraining", "Event-Driven", "Frozen") else 0.7)
    ax.set_xlabel("Streaming window")
    ax.set_ylabel("Macro-F1")
    ax.set_title("5G Campus QoS — Macro-F1 over the stream (dashed = regime transition)")
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_final_stream_f1.png"))
    plt.close(fig)


def fig_cost_tradeoff(s):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m in ALL_MODELS:
        d = s[s.method == m]
        if d.empty:
            continue
        ax.scatter(d["adaptation_cpu_sec"], d["macro_f1"], s=45,
                   color=COLORS.get(m), label=m, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Adaptation CPU time (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy vs adaptation cost (per seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_final_cost_tradeoff.png"))
    plt.close(fig)


def fig_reuse_retrain(s):
    g = s.groupby("method")[["reuse_events", "retrains", "trees_trained", "trees_reused"]].mean()
    g = g.reindex([m for m in ALL_MODELS if m in g.index])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(g)); w_ = 0.2
    ax.bar(x - 1.5 * w_, g["reuse_events"], w_, label="Reuse events", color="#2ca02c")
    ax.bar(x - 0.5 * w_, g["retrains"], w_, label="Retrains", color="#d62728")
    ax.bar(x + 0.5 * w_, g["trees_trained"], w_, label="Trees trained", color="#1f77b4")
    ax.bar(x + 1.5 * w_, g["trees_reused"], w_, label="Trees reused", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20, ha="right")
    ax.set_ylabel("Count (mean over seeds)")
    ax.set_title("RAPT adaptation behaviour: reuse vs retraining")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_final_reuse_retrain.png"))
    plt.close(fig)


def fig_regime_f1(w):
    piv = (w.groupby(["method", "regime_id"])["macro_f1"].mean().unstack("regime_id"))
    piv = piv.reindex([m for m in ALL_MODELS if m in piv.index])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    regs = list(piv.columns)
    x = np.arange(len(regs)); w_ = 0.8 / len(piv)
    for i, (m, row) in enumerate(piv.iterrows()):
        ax.bar(x + i * w_ - 0.4, row.values, w_, label=m, color=COLORS.get(m))
    ax.set_xticks(x); ax.set_xticklabels(regs)
    ax.set_xlabel("Regime"); ax.set_ylabel("Macro-F1 (mean over seeds)")
    ax.set_title("Per-regime Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_final_regime_f1.png"))
    plt.close(fig)


def fig_ladder(s):
    """Incremental effect of each mechanism: F1 (left) and adapt CPU (right)."""
    order = [m for m in ["RAPT_T2", "RAPT_T1", "RAPT_FULL", "RAPT_REFRESH_W10",
                         "RAPT_REFRESH_W5"] if m in set(s.method)]
    g = s.groupby("method")[["macro_f1", "adaptation_cpu_sec"]].mean().reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(order, g["macro_f1"], "o-", color="#9467bd")
    axes[0].set_ylabel("Macro-F1"); axes[0].set_title("Mechanism ladder — Macro-F1")
    axes[0].set_ylim(min(0.9, g["macro_f1"].min() - 0.02), 1.0)
    for i, v in enumerate(g["macro_f1"]):
        axes[0].annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 6), fontsize=8)
    axes[1].plot(order, g["adaptation_cpu_sec"], "s-", color="#d62728")
    axes[1].set_ylabel("Adaptation CPU (s)"); axes[1].set_title("Mechanism ladder — adaptation cost")
    for i, v in enumerate(g["adaptation_cpu_sec"]):
        axes[1].annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 6), fontsize=8)
    for a in axes:
        a.set_xticklabels(order, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_final_ladder.png"))
    plt.close(fig)


def main(mode="full"):
    ensure_dirs()
    w, s = _load(mode)
    fig_f1_bars(s); fig_stream(w); fig_cost_tradeoff(s)
    fig_reuse_retrain(s); fig_regime_f1(w); fig_ladder(s)
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
