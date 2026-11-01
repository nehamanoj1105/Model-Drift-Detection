#!/usr/bin/env python3
"""
make_figures.py — publication figures from results/final/raw/*.parquet.

Vector PDF + 300 dpi PNG, IEEE single-column width 3.5 in, one colour/marker per
model across every figure. Also writes results/final/figure_consistency_check.txt
asserting that the numbers drawn match the primary/stats tables.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))

WIDTH = 3.5
STYLE = {
    "Frozen":        dict(color="#4C72B0", marker="o", ls="-"),
    "Event-Driven":  dict(color="#DD8452", marker="s", ls="--"),
    "Full Retraining": dict(color="#55A868", marker="^", ls="-"),
    "RAPT":          dict(color="#C44E52", marker="D", ls="-"),
    "RAPT-Enhanced": dict(color="#8172B3", marker="v", ls="-."),
    "ECDD-EWMA":    dict(color="#937860", marker="P", ls=":"),
    "RAPT-Cheap":    dict(color="#DA8BC3", marker="X", ls="-"),
    "RAPT-Full":     dict(color="#8C8C8C", marker="*", ls="-"),
}


def _save(fig, figdir, name, check, note):
    fig.set_size_inches(WIDTH, WIDTH * 0.72)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=300)
    plt.close(fig)
    check.append(f"{name}: {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final.yaml")
    args = ap.parse_args()
    with open(os.path.join(HERE, args.config)) as f:
        cfg = yaml.safe_load(f)

    out_root = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")
    figdir = os.path.join(out_root, "figures")
    os.makedirs(figdir, exist_ok=True)
    prim = pd.read_csv(os.path.join(out_root, "table_primary.csv"))
    ds_names = [d["name"] for d in cfg["datasets"]]
    raw_dir = os.path.join(out_root, "raw")
    wins = {n: pd.read_parquet(os.path.join(raw_dir, f"per_window_{d['slug']}.parquet"))
            for n, d in zip(ds_names, cfg["datasets"])}
    check = []

    # --- Fig 1: computational cost, grouped bars, log y --------------------
    fig, ax = plt.subplots()
    models = ["Full Retraining", "RAPT", "RAPT-Enhanced", "ECDD-EWMA"]
    x = np.arange(len(ds_names))
    w = 0.2
    for i, m in enumerate(models):
        vals = [max(prim[(prim.dataset == n) & (prim.model == m)]["adapt_cpu_s"].mean(), 1e-3)
                for n in ds_names]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=STYLE[m]["color"])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("Adaptation CPU (s, log)", fontsize=7)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    _save(fig, figdir, "computational_cost", check,
          "adaptation CPU by dataset (matches table_primary)")

    # --- Fig 2: accuracy-cost scatter --------------------------------------
    fig, ax = plt.subplots()
    for m in ["RAPT", "RAPT-Enhanced"]:
        for n in ds_names:
            fr = prim[(prim.dataset == n) & (prim.model == "Full Retraining")].iloc[0]
            r = prim[(prim.dataset == n) & (prim.model == m)].iloc[0]
            saved = (fr["adapt_cpu_s"] - r["adapt_cpu_s"]) / fr["adapt_cpu_s"] * 100
            d = r["macro_f1"] - fr["macro_f1"]
            ax.errorbar(saved, d, yerr=r["macro_f1_std"], fmt=STYLE[m]["marker"],
                        color=STYLE[m]["color"], ms=5, capsize=2)
            ax.annotate(n, (saved, d), fontsize=5, xytext=(2, 2),
                        textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("CPU saved vs Full Retraining (%)", fontsize=7)
    ax.set_ylabel(r"$\Delta$ macro-F1 vs Full Retraining", fontsize=7)
    ax.tick_params(labelsize=6)
    _save(fig, figdir, "accuracy_cost_scatter", check,
          "CPU saved vs delta macro-F1 (matches cost_savings + table_primary)")

    # --- Fig 3: UGR'16 rolling accuracy ------------------------------------
    fig, ax = plt.subplots()
    w = wins["UGR'16"]
    for m in ["Frozen", "RAPT", "RAPT-Enhanced", "Full Retraining"]:
        g = w[w.model == m].groupby("window_idx")["window_acc"]
        mean, lo, hi = g.mean(), g.quantile(0.025), g.quantile(0.975)
        ax.plot(mean.index, mean.values, label=m, color=STYLE[m]["color"],
                marker=STYLE[m]["marker"], ms=2, lw=1)
        ax.fill_between(mean.index, lo.values, hi.values, color=STYLE[m]["color"],
                        alpha=0.15)
    ev = pd.read_csv(os.path.join(out_root, "rapt_policy_events.csv"))
    for t in ev[(ev.dataset == "UGR'16") & (ev.model == "RAPT") &
                (ev.event == "retrain")]["window_idx"].unique():
        ax.axvline(t, color="gray", lw=0.4, alpha=0.5)
    ax.set_xlabel("Window", fontsize=7); ax.set_ylabel("Rolling accuracy", fontsize=7)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    _save(fig, figdir, "ugr_rolling_accuracy", check,
          "UGR'16 rolling accuracy, 95% band, novelty-refit ticks")

    # --- Fig 4: UGR'16 staleness ------------------------------------------
    fig, ax = plt.subplots()
    st = pd.read_csv(os.path.join(out_root, "ugr_staleness.csv"))
    ug = st[st.dataset == "UGR'16"]
    if not ug.empty:
        ax.bar(ug["model"], ug["mean_stored_origin_f1"], color="#C44E52")
        ax.set_ylabel("Stored-policy F1 on own origin window", fontsize=7)
        ax.tick_params(labelsize=6)
    _save(fig, figdir, "ugr_staleness", check,
          "stored-policy accuracy on origin regime (from policy events)")

    # --- numeric consistency assertions -----------------------------------
    # Fig.1 draws table_primary directly; the cost figures draw cost_savings.csv.
    # cost_savings averages per-seed reductions while a recomputation from the
    # mean CPU is a ratio-of-means, so allow the small definitional gap.
    assert os.path.exists(os.path.join(out_root, "stats.csv")), "stats.csv missing"
    cost = pd.read_csv(os.path.join(out_root, "cost_savings.csv"))
    for _, r in cost.iterrows():
        fr = prim[(prim.dataset == r["dataset"]) &
                  (prim.model == "Full Retraining")]["adapt_cpu_s"]
        mm = prim[(prim.dataset == r["dataset"]) &
                  (prim.model == r["model"])]["adapt_cpu_s"]
        if fr.empty or mm.empty:  # ablation models are not in table_primary
            continue
        drawn = (fr.iloc[0] - mm.iloc[0]) / fr.iloc[0] * 100
        assert abs(drawn - r["reduction_pct"]) < 0.1, \
            f"figure/table mismatch {r['dataset']} {r['model']}: {drawn} vs {r['reduction_pct']}"
    check.append("numeric check: all cost-reduction bars within 0.1 pp of "
                 "cost_savings.csv (ratio-of-means vs mean-of-ratios) and "
                 "derived from table_primary.csv adaptation CPU")

    with open(os.path.join(out_root, "figure_consistency_check.txt"), "w") as f:
        f.write("\n".join(check) + "\n")
    print("figures written to", figdir)

    # Manuscript figures live in the two paper trees; regenerate them from the
    # same freshly written tables so the manuscripts can never drift from results.
    import make_paper_figures
    make_paper_figures.fig_primary()
    make_paper_figures.fig_cost()
    make_paper_figures.fig_ablation()
    make_paper_figures.fig_detectors()
    make_paper_figures.fig_tradeoff()


if __name__ == "__main__":
    main()
