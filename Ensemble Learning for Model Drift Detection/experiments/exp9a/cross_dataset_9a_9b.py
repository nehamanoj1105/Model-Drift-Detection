"""
Cross-dataset evaluation for Experiments 9A / 9B (final paper pass).

Validates whether the RAPT policy-transfer mechanism behaves consistently across
two independent recurring-regime streams:

  9A — INSECTS incremental-reoccurring (6-class insect species; 150 windows;
       11 recurring regime visits).
  9B — 5G NR end-to-end latency QoS (3-class; 499 windows; regimes A/B/C/D
       recurring as ABCDACBDAB).

Outputs (results/experiment_9a/cross_dataset and results/experiment_9a/{tables,
figures}):
  * cross_dataset_summary.csv        per-dataset per-model aggregate metrics
  * cross_dataset_stats.csv          paired Wilcoxon RAPT(-Enhanced) vs baselines
  * fig_cross_dataset_f1.png         Macro-F1 by model, both datasets
  * fig_cross_dataset_adapt_cpu.png  adaptation CPU by model, both datasets
  * fig_cross_dataset_reuse.png      reuse / retrain events by model, both datasets
  * table_cross_dataset.csv/.tex     combined summary table
  * EXPERIMENT_9A_9B_FINAL_REPORT.md combined report

Run from the "Ensemble Learning for Model Drift Detection" directory:
    python experiments/exp9a/cross_dataset_9a_9b.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_9A = os.path.join(PROJECT_DIR, "results", "experiment_9a")
RESULTS_9B = os.path.join(PROJECT_DIR, "results", "experiment_9b")
CROSS_DIR = os.path.join(RESULTS_9A, "cross_dataset")
FIG_DIR = os.path.join(RESULTS_9A, "figures")
TAB_DIR = os.path.join(RESULTS_9A, "tables")

MODELS = ["Frozen", "Event-Driven", "Full_Retraining", "RAPT", "RAPT-Enhanced"]
# 9A uses underscore, 9B uses spaces for Full Retraining.
MODEL_9B = {"Full_Retraining": "Full Retraining", "RAPT-Enhanced": "RAPT-Enhanced"}

DATASET_LABEL = {
    "9A": "9A INSECTS recurring (6-class)",
    "9B": "9B 5G latency QoS (3-class)",
}


def _load_9a():
    ps = pd.read_csv(os.path.join(RESULTS_9A, "raw", "per_seed_9a.csv"))
    pw = pd.read_csv(os.path.join(RESULTS_9A, "raw", "per_window_9a.csv"))
    st = pd.read_csv(os.path.join(RESULTS_9A, "raw", "statistics_9a.csv"))
    return ps, pw, st


def _load_9b():
    ps = pd.read_csv(os.path.join(RESULTS_9B, "natural_drift", "per_seed.csv"))
    pw = pd.read_csv(os.path.join(RESULTS_9B, "natural_drift", "per_window.csv"))
    return ps, pw


def _aggregate_9a(ps):
    rows = []
    for m in MODELS:
        g = ps[ps["method"] == m]
        rows.append({
            "dataset": "9A", "model": m,
            "macro_f1_mean": g["macro_f1"].mean(), "macro_f1_std": g["macro_f1"].std(),
            "accuracy_mean": g["accuracy"].mean(),
            "adapt_cpu_mean": g["adaptation_cpu_sec"].mean(),
            "retrain_events_mean": g["retrain_events"].mean(),
            "reuse_events_mean": g["reuse_events"].mean(),
            "trees_trained_mean": g["trees_trained"].mean(),
            "trees_reused_mean": g["trees_reused"].mean(),
        })
    return pd.DataFrame(rows)


def _aggregate_9b(ps):
    rows = []
    for m in MODELS:
        name = MODEL_9B.get(m, m)
        g = ps[ps["method"] == name]
        rows.append({
            "dataset": "9B", "model": m,
            "macro_f1_mean": g["macro_f1"].mean(), "macro_f1_std": g["macro_f1"].std(),
            "accuracy_mean": g["accuracy"].mean(),
            "adapt_cpu_mean": g["adaptation_cpu_sec"].mean(),
            "retrain_events_mean": g["retrain_events"].mean(),
            "reuse_events_mean": g["reused_checkpoints"].mean(),
            "trees_trained_mean": g["trees_trained"].mean(),
            "trees_reused_mean": g["trees_reused"].mean(),
        })
    return pd.DataFrame(rows)


def _wilcoxon(df_window, m1, m2, metric="is_correct"):
    from scipy import stats
    a = df_window[df_window["method"] == m1]
    b = df_window[df_window["method"] == m2]
    merged = a.merge(b, on=["seed", "window_id"], suffixes=("_1", "_2"))
    if merged.empty:
        return None
    d = merged[f"{metric}_1"].values.astype(float) - merged[f"{metric}_2"].values.astype(float)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d)) + 1e-12
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = stats.wilcoxon(d, zero_method="pratt")
            p = float(p)
        except Exception:
            p = 1.0
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(2000)]
    return {
        "comparison": f"{m1} vs {m2}", "mean_diff": mean_d,
        "cohen_d": mean_d / std_d, "p_value": p,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "significant": bool(p < 0.05), "n_windows": int(len(d)),
    }


def _stats_for(pw, dataset, method_map=None):
    rows = []
    method_map = method_map or {}
    for m1, m2 in [("RAPT", "Frozen"), ("RAPT", "Event-Driven"),
                   ("RAPT", "Full_Retraining"),
                   ("RAPT-Enhanced", "Frozen"), ("RAPT-Enhanced", "Event-Driven"),
                   ("RAPT-Enhanced", "Full_Retraining"), ("RAPT-Enhanced", "RAPT")]:
        a = method_map.get(m1, m1)
        b = method_map.get(m2, m2)
        r = _wilcoxon(pw, a, b)
        if r is None:
            continue
        r["dataset"] = dataset
        rows.append(r)
    return pd.DataFrame(rows)


def _make_figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid"
                  if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"Frozen": "#7f7f7f", "Event-Driven": "#d62728",
              "Full_Retraining": "#ff7f0e", "RAPT": "#1f77b4",
              "RAPT-Enhanced": "#2ca02c"}
    ds_order = ["9A", "9B"]

    def _grouped(metric, err, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ds_order))
        width = 0.15
        for i, m in enumerate(MODELS):
            vals, errs = [], []
            for ds in ds_order:
                row = summary[(summary["dataset"] == ds) & (summary["model"] == m)]
                vals.append(row[metric].values[0] if len(row) else np.nan)
                errs.append(row[err].values[0] if (err and len(row)) else 0.0)
            ax.bar(x + (i - 2) * width, vals, width, yerr=errs, capsize=3,
                   label=m, color=colors[m])
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL[d] for d in ds_order])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    _grouped("macro_f1_mean", "macro_f1_std", "Macro-F1",
             "fig_cross_dataset_f1.png",
             "Cross-dataset Macro-F1: RAPT policy transfer vs baselines")
    _grouped("adapt_cpu_mean", None, "Adaptation CPU (s)",
             "fig_cross_dataset_adapt_cpu.png",
             "Cross-dataset adaptation CPU")
    _grouped("reuse_events_mean", None, "Policy reuse events",
             "fig_cross_dataset_reuse.png",
             "Cross-dataset RAPT policy reuse events")


def _write_table(summary):
    out = summary.copy()
    out["macro_f1"] = out.apply(lambda r: f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}", axis=1)
    out = out[["dataset", "model", "macro_f1", "accuracy_mean", "adapt_cpu_mean",
               "retrain_events_mean", "reuse_events_mean", "trees_trained_mean",
               "trees_reused_mean"]]
    out.to_csv(os.path.join(TAB_DIR, "table_cross_dataset.csv"), index=False)
    with open(os.path.join(TAB_DIR, "table_cross_dataset.tex"), "w") as f:
        f.write(out.to_latex(index=False, float_format="%.4f"))
    print("  saved table_cross_dataset.csv/.tex")


def main():
    for d in (CROSS_DIR, FIG_DIR, TAB_DIR):
        os.makedirs(d, exist_ok=True)

    ps9a, pw9a, st9a = _load_9a()
    ps9b, pw9b = _load_9b()

    agg9a = _aggregate_9a(ps9a)
    agg9b = _aggregate_9b(ps9b)
    summary = pd.concat([agg9a, agg9b], ignore_index=True)
    summary.to_csv(os.path.join(CROSS_DIR, "cross_dataset_summary.csv"), index=False)

    stats = pd.concat([
        _stats_for(pw9a, "9A"),
        _stats_for(pw9b, "9B", method_map={"Full_Retraining": "Full Retraining"}),
    ], ignore_index=True)
    stats.to_csv(os.path.join(CROSS_DIR, "cross_dataset_stats.csv"), index=False)

    _make_figures(summary)
    _write_table(summary)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
