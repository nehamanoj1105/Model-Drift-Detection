"""
Figures and tables for the revised Experiment 9A (INSECTS recurring stream).

Reads only the saved raw results so every artefact is reproducible.
Run from the "Ensemble Learning for Model Drift Detection" directory:
    python experiments/exp9a/figures_tables_9a.py
"""

import os
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
R9A = os.path.join(PROJECT_DIR, "results", "experiment_9a")
FIG = os.path.join(R9A, "figures")
TAB = os.path.join(R9A, "tables")

MODELS = ["Frozen", "Event-Driven", "Full_Retraining", "RAPT", "RAPT-Enhanced"]
COLORS = {"Frozen": "#7f7f7f", "Event-Driven": "#d62728", "Full_Retraining": "#ff7f0e",
          "RAPT": "#1f77b4", "RAPT-Enhanced": "#2ca02c"}


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  saved {name}")


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid"
                  if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    pw = pd.read_csv(os.path.join(R9A, "raw", "per_window_9a.csv"))
    ps = pd.read_csv(os.path.join(R9A, "raw", "per_seed_9a.csv"))
    st = pd.read_csv(os.path.join(R9A, "raw", "statistics_9a.csv"))
    with open(os.path.join(R9A, "raw", "stream_definition_9a.json")) as f:
        sd = json.load(f)
    segs = sd["segments"]

    # --- Fig 9A-1: Macro-F1 timeline with regime regions -------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in MODELS:
        g = pw[pw["method"] == m].groupby("window_id")["macro_f1"].mean().rolling(5, min_periods=1).mean()
        ax.plot(g.index, g.values, label=m, color=COLORS[m],
                linewidth=2.4 if m == "RAPT" else 1.4)
    for s in segs:
        if s["start_window"] >= sd["initial_train_windows"]:
            ax.axvline(s["start_window"], color="k", alpha=0.25, linewidth=0.8)
            ax.text(s["start_window"] + 0.5, 0.02, s["regime_id"].replace("regime_", "r"),
                    fontsize=6, rotation=90, alpha=0.7)
    ax.set_xlabel("Streaming window")
    ax.set_ylabel("Macro-F1 (rolling mean, 5)")
    ax.set_title("Fig 9A-1  INSECTS recurring stream — Macro-F1 over time")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout(); _save(fig, "fig9a_macro_f1_timeline.png"); plt.close(fig)

    # --- Fig 9A-2: adaptation CPU by model ---------------------------------
    summ = []
    for m in MODELS:
        g = ps[ps["method"] == m]
        summ.append({"model": m, "f1": g["macro_f1"].mean(),
                     "adapt_cpu": g["adaptation_cpu_sec"].mean(),
                     "retrains": g["retrain_events"].mean(),
                     "reuse": g["reuse_events"].mean(),
                     "trees_reused": g["trees_reused"].mean(),
                     "trees_trained": g["trees_trained"].mean()})
    summ = pd.DataFrame(summ)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summ["model"], summ["adapt_cpu"], color=[COLORS[m] for m in summ["model"]])
    ax.set_ylabel("Adaptation CPU (s)")
    ax.set_title("Fig 9A-2  Adaptation CPU by model")
    plt.xticks(rotation=20)
    fig.tight_layout(); _save(fig, "fig9a_adapt_cpu.png"); plt.close(fig)

    # --- Fig 9A-3: reuse vs retraining -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summ))
    ax.bar(x - 0.2, summ["retrains"], 0.4, label="Retrain events", color="#ff7f0e")
    ax.bar(x + 0.2, summ["reuse"], 0.4, label="Reuse events", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(summ["model"], rotation=20)
    ax.set_ylabel("Events (mean across seeds)")
    ax.set_title("Fig 9A-3  Retraining vs policy reuse")
    ax.legend()
    fig.tight_layout(); _save(fig, "fig9a_reuse_vs_retraining.png"); plt.close(fig)

    # --- Fig 9A-4: accuracy-cost Pareto ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in summ.itertuples():
        ax.scatter(r.adapt_cpu, r.f1, s=90, color=COLORS[r.model], label=r.model)
        ax.annotate(r.model, (r.adapt_cpu, r.f1), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Adaptation CPU (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fig 9A-4  Accuracy–cost trade-off")
    fig.tight_layout(); _save(fig, "fig9a_pareto.png"); plt.close(fig)

    # --- Tables -------------------------------------------------------------
    t1 = pd.DataFrame([{
        "Model": m,
        "Macro_F1": f"{ps[ps.method==m]['macro_f1'].mean():.4f} ± {ps[ps.method==m]['macro_f1'].std():.4f}",
        "Accuracy": f"{ps[ps.method==m]['accuracy'].mean():.4f}",
        "Precision": f"{ps[ps.method==m]['precision'].mean():.4f}",
        "Recall": f"{ps[ps.method==m]['recall'].mean():.4f}",
        "Adapt_CPU_s": f"{ps[ps.method==m]['adaptation_cpu_sec'].mean():.4f}",
        "Retrains": int(ps[ps.method==m]['retrain_events'].mean()),
        "Reuse_events": int(ps[ps.method==m]['reuse_events'].mean()),
        "Trees_trained": int(ps[ps.method==m]['trees_trained'].mean()),
        "Trees_reused": int(ps[ps.method==m]['trees_reused'].mean()),
    } for m in MODELS])
    t1.to_csv(os.path.join(TAB, "table_9a1_summary.csv"), index=False)
    open(os.path.join(TAB, "table_9a1_summary.tex"), "w").write(t1.to_latex(index=False))
    print("  saved table_9a1_summary.csv/.tex")

    st.to_csv(os.path.join(TAB, "table_9a2_statistics.csv"), index=False)
    open(os.path.join(TAB, "table_9a2_statistics.tex"), "w").write(
        st.to_latex(index=False, float_format="%.4f"))
    print("  saved table_9a2_statistics.csv/.tex")

    t3 = pd.DataFrame([{
        "Model": m, "Reuse_events": int(ps[ps.method==m]['reuse_events'].mean()),
        "Reused_trees": int(ps[ps.method==m]['trees_reused'].mean()),
        "Retrained_trees": int(ps[ps.method==m]['trees_trained'].mean()),
        "Adapt_CPU_s": f"{ps[ps.method==m]['adaptation_cpu_sec'].mean():.4f}",
    } for m in ["RAPT", "RAPT-Enhanced"]])
    t3.to_csv(os.path.join(TAB, "table_9a3_reuse.csv"), index=False)
    open(os.path.join(TAB, "table_9a3_reuse.tex"), "w").write(t3.to_latex(index=False))
    print("  saved table_9a3_reuse.csv/.tex")


if __name__ == "__main__":
    main()
