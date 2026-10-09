"""
Regenerate all paper figures and tables from the saved final results.

Reads results/final/raw/per_window_<slug>_final.csv and
results/final/tables/*.csv; writes PNGs into results/final/figures and
LaTeX/CSV tables into results/final/tables. No numbers are entered by hand.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from final_config import RAW_FINAL, TABLES_FINAL, FIGURES_FINAL, ensure_dirs

SLUGS = ["5g_campus", "ugr16", "nordicdat", "5g_nr"]
LABEL = {"5g_campus": "5G Campus", "ugr16": "UGR'16",
         "nordicdat": "NordicDat", "5g_nr": "5G NR Lat."}
CORE = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced",
        "RAPT-Guard", "RAPT-Anchor", "Frozen+Periodic", "Event-Driven+Reuse"]
EXTRA = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]
COLORS = {"Frozen": "tab:gray", "Event-Driven": "tab:orange",
          "Full Retraining": "tab:blue", "RAPT": "tab:red",
          "RAPT-Enhanced": "tab:green", "RAPT-Guard": "purple",
          "RAPT-Anchor": "black"}


def load(tag="final"):
    return {s: pd.read_csv(os.path.join(RAW_FINAL, f"per_window_{s}_{tag}.csv"))
            for s in SLUGS
            if os.path.exists(os.path.join(RAW_FINAL, f"per_window_{s}_{tag}.csv"))}


def _seed_mean(df, method):
    return df[df["method"] == method].groupby("seed").agg(
        f1=("macro_f1", "mean"), adapt=("adaptation_cpu_sec", "sum"),
        pred=("prediction_cpu_sec", "sum"), total=("total_cpu_sec", "sum"),
        retr=("is_retrain", "sum"), reuse=("is_reuse", "sum"))


def table_primary(frames):
    rows = []
    for s, df in frames.items():
        for m in CORE:
            if m not in set(df["method"]):
                continue
            ps = _seed_mean(df, m)
            rows.append(dict(stream=LABEL[s], method=m,
                             f1=ps["f1"].mean(), f1_std=ps["f1"].std(ddof=1),
                             adapt=ps["adapt"].mean(), pred=ps["pred"].mean(),
                             total=ps["total"].mean(),
                             retr=ps["retr"].mean(), reuse=ps["reuse"].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLES_FINAL, "table_primary.csv"), index=False)
    # LaTeX
    lines = []
    for s in ["5G Campus", "UGR'16", "NordicDat", "5G NR Lat."]:
        sub = t[t["stream"] == s]
        best = sub["f1"].max()
        for _, r in sub.iterrows():
            star = r"$\mathbf{%.4f}$" % r["f1"] if abs(r["f1"] - best) < 1e-12 else "%.4f" % r["f1"]
            lines.append(f"{s} & {r['method']} & ${star} \\pm {r['f1_std']:.4f}$ & "
                         f"{r['adapt']:.4f} & {r['total']:.4f} & "
                         f"{int(round(r['retr']))} & {int(round(r['reuse']))} \\\\")
        lines.append(r"\midrule")
    with open(os.path.join(TABLES_FINAL, "table_primary.tex"), "w") as f:
        f.write("\n".join(lines[:-1]))
    return t


def figure_cost(frames):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(CORE))
    streams = list(frames.keys())
    width = 0.8 / len(streams)
    for i, s in enumerate(streams):
        df = frames[s]
        vals = [_seed_mean(df, m)["total"].mean() if m in set(df["method"]) else np.nan
                for m in CORE]
        ax.bar(x + i * width, vals, width, label=LABEL[s])
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(CORE, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("total CPU (s)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_cost.png"), dpi=160)
    plt.close(fig)


def figure_rolling(frames):
    df = frames["ugr16"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for m in ["RAPT", "RAPT-Enhanced", "RAPT-Guard", "Full Retraining"]:
        if m not in set(df["method"]):
            continue
        g = df[df["method"] == m].groupby("window_id")["macro_f1"].mean().rolling(5).mean()
        ax.plot(g.index, g.values, label=m, color=COLORS.get(m), lw=1.4)
    ax.set_xlabel("window")
    ax.set_ylabel("rolling macro-F1 (5)")
    ax.set_title("UGR'16 rolling macro-F1")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_ugr_rolling.png"), dpi=160)
    plt.close(fig)


def figure_tradeoff(frames):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    marker = {"RAPT": "o", "RAPT-Enhanced": "s", "RAPT-Guard": "^",
              "RAPT-Anchor": "D"}
    for s, df in frames.items():
        base = _seed_mean(df, "Full Retraining")["f1"].mean()
        base_cpu = _seed_mean(df, "Full Retraining")["adapt"].mean()
        for m in ["RAPT", "RAPT-Enhanced", "RAPT-Guard", "RAPT-Anchor"]:
            if m not in set(df["method"]) or base_cpu == 0:
                continue
            ps = _seed_mean(df, m)
            cpu_saved = 100 * (1 - ps["adapt"].mean() / base_cpu)
            ax.scatter(cpu_saved, ps["f1"].mean() - base, marker=marker[m],
                       color=COLORS.get(m), label=f"{m} ({LABEL[s]})")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("adaptation CPU saved vs Full Retraining (%)")
    ax.set_ylabel(r"$\Delta$ macro-F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_tradeoff.png"), dpi=160)
    plt.close(fig)


def table_detectors(frames):
    rows = []
    for s, df in frames.items():
        for m in EXTRA + ["Event-Driven", "RAPT"]:
            if m not in set(df["method"]):
                continue
            ps = _seed_mean(df, m)
            rows.append(dict(stream=LABEL[s], method=m, f1=ps["f1"].mean(),
                             events=ps["retr"].mean(), adapt=ps["adapt"].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLES_FINAL, "table_detectors.csv"), index=False)
    return t


def figure_detectors(frames):
    fig, axes = plt.subplots(1, len(frames), figsize=(11, 3.0), sharex=False)
    if len(frames) == 1:
        axes = [axes]
    names = EXTRA + ["Event-Driven", "RAPT"]
    for ax, (s, df) in zip(axes, frames.items()):
        vals = [_seed_mean(df, m)["retr"].mean() if m in set(df["method"]) else 0
                for m in names]
        ax.bar(range(len(names)), vals, color="tab:purple")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
        ax.set_title(LABEL[s], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("retrain events (per seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_detectors.png"), dpi=160)
    plt.close(fig)


def table_gate(frame_df):
    """Gate outcomes: for each stream, how often the guard allowed a reuse that
    was worse than the prefix model (bad allow) or blocked a reuse that was
    better (bad block). Computed from RAPT-Guard window flags plus a
    counterfactual prefix/stored comparison replayed in the guard harness."""
    p = os.path.join(TABLES_FINAL, "gate_accuracy.csv")
    if os.path.exists(p):
        return pd.read_csv(p)


def figure_primary(frames):
    """Macro-F1 per method and stream, mean over seeds (manuscript Fig. 2)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    names = [m for m in CORE if any(m in set(df["method"]) for df in frames.values())]
    x = np.arange(len(frames)); width = 0.8 / len(names)
    for i, m in enumerate(names):
        vals = []
        for s in frames:
            df = frames[s]
            vals.append(_seed_mean(df, m)["f1"].mean()
                        if m in set(df["method"]) else np.nan)
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([LABEL[s] for s in frames], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_primary.png"), dpi=160)
    plt.close(fig)


def table_ablation():
    """The 5G Campus mechanism ladder is produced by the legacy ablation
    harness (Final_Experiments); we surface its saved table unchanged so the
    manuscript number stays traceable to a results file."""
    p = os.path.join(os.path.dirname(RAW_FINAL), "..", "..",
                     "Final_Experiments", "results", "tables",
                     "table_final_main.csv")
    p = os.path.abspath(p)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def figure_ablation():
    t = table_ablation()
    if t is None:
        return
    order = ["Full Retraining", "RAPT_FULL", "RAPT_EVIDENCE", "RAPT_FLOOR",
             "RAPT_CHEAP", "RAPT_INCR"]
    pretty = {"Full Retraining": "Full Retrain.", "RAPT_FULL": "RAPT-Full",
              "RAPT_EVIDENCE": "RAPT-Evidence", "RAPT_FLOOR": "RAPT-Floor",
              "RAPT_CHEAP": "RAPT-Cheap", "RAPT_INCR": "RAPT-Incremental"}
    d = t.set_index("Model")
    order = [o for o in order if o in d.index]
    d = d.loc[order]
    f1 = d["Macro-F1"].str.split(r"\s*\+/-\s*", expand=True, regex=True)[0].astype(float).values
    cpu = d["Adapt CPU (s)"].str.split(r"\s*\+/-\s*", expand=True, regex=True)[0].astype(float).values
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs = np.arange(len(order))
    ax.plot(xs, f1, "o-", color="#9467bd", label="Macro-F1")
    ax.set_ylabel("macro-F1", color="#9467bd"); ax.set_ylim(0.94, 1.0)
    ax.set_xticks(xs); ax.set_xticklabels([pretty[o] for o in order],
                                          rotation=15, ha="right", fontsize=7)
    ax2 = ax.twinx()
    ax2.bar(xs, cpu, 0.5, color="#1f77b4", alpha=0.35, label="adapt CPU")
    ax2.set_ylabel("adaptation CPU (s)", color="#1f77b4"); ax2.grid(False)
    ax.set_title("5G Campus mechanism ladder")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_ablation.png"), dpi=160)
    plt.close(fig)


def main():
    ensure_dirs()
    frames = load("final")
    if not frames:
        raise SystemExit("no final results yet")
    table_primary(frames)
    figure_primary(frames)
    figure_cost(frames)
    figure_rolling(frames)
    figure_tradeoff(frames)
    table_detectors(frames)
    figure_detectors(frames)
    figure_ablation()
    print("figures/tables written to", FIGURES_FINAL, TABLES_FINAL)


if __name__ == "__main__":
    main()
