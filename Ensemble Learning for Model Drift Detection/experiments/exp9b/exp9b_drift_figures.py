"""
Publication-quality figures for Experiment 9B — Drift Severity & Concept Drift.

All 14 required figures are generated from the saved per-window / aggregate
results so every figure is reproducible from raw data. Figures are written to
results/experiment_9b/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

import exp9b_drift_config as CFG

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
DPI = 300

COLORS = {
    "Frozen": "#7f7f7f",
    "Event-Driven": "#d62728",
    "Full Retraining": "#ff7f0e",
    "RAPT": "#1f77b4",
    "RAPT-Enhanced": "#2ca02c",
}
LEVELS = CFG.DRIFT_LEVELS


def _save(fig, name):
    path = os.path.join(CFG.FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}", flush=True)


def _f1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


# --------------------------------------------------------------------------
# Figure 9B-1 — Natural drift timeline
# --------------------------------------------------------------------------
def fig_natural_timeline(natural):
    pw = natural["per_window"]
    segs = natural["segments"]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=DPI)
    for m in ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]:
        df_m = pw[pw["method"] == m].sort_values("window_id")
        roll = df_m.groupby("window_id")["is_correct"].mean().rolling(15, min_periods=1).mean()
        ax.plot(roll.index, roll.values, label=m, color=COLORS.get(m, "#333"), linewidth=2.0)

    ymin, ymax = ax.get_ylim()
    for i, s in enumerate(segs.itertuples()):
        if s.is_transition:
            ax.axvline(s.start_window, color="black", linestyle="--", alpha=0.5, linewidth=1.0)
        ax.axvspan(s.start_window, s.end_window, alpha=0.06,
                   color=plt.cm.tab10(i % 10))
        ax.text((s.start_window + s.end_window) / 2, ymax * 0.99,
                s.regime_id.replace("regime_", ""), ha="center", va="top", fontsize=8)
    ax.set_xlabel("Streaming window index")
    ax.set_ylabel("Rolling accuracy (15-window mean)")
    ax.set_title("Fig 9B-1  Natural recurring drift — Macro-F1 proxy over time with regime regions")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    _save(fig, "fig9b_natural_drift_f1.png")


# --------------------------------------------------------------------------
# Severity curves
# --------------------------------------------------------------------------
def _severity_curve(agg, metric_mean, ylabel, title, fname, agg_rec=None,
                    rec_metric=None, note=None):
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=DPI)
    for m in ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]:
        df_m = agg[agg["method"] == m].sort_values("drift_level")
        if rec_metric is not None and agg_rec is not None:
            df_r = agg_rec[agg_rec["method"] == m].sort_values("drift_level")
            xs = df_r["drift_level"].values
            ys = df_r[f"{rec_metric}_mean"].values
            err = df_r[f"{rec_metric}_std"].fillna(0).values
        else:
            xs = df_m["drift_level"].values
            ys = df_m[f"{metric_mean}_mean"].values
            err = df_m[f"{metric_mean}_std"].fillna(0).values
        ax.errorbar(xs * 100, ys, yerr=err, marker="o", capsize=4,
                    label=m, color=COLORS.get(m, "#333"), linewidth=2.0)
    ax.set_xlabel("Drift severity (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([l * 100 for l in LEVELS])
    ax.legend(frameon=True, fontsize=8)
    if note:
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, style="italic", color="#555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    _save(fig, fname)


def fig_covariate(cov_agg, cov_agg_rec):
    _severity_curve(cov_agg, "macro_f1", "Macro-F1",
                    "Fig 9B-2  Covariate drift severity vs Macro-F1",
                    "fig9b_covariate_severity_f1.png")
    _severity_curve(cov_agg, "adaptation_cpu_sec", "Adaptation CPU (s)",
                    "Fig 9B-3  Covariate drift severity vs adaptation CPU",
                    "fig9b_covariate_severity_cpu.png")
    _severity_curve(cov_agg, "macro_f1", "Recovery time (windows)",
                    "Fig 9B-4  Covariate drift severity vs recovery time",
                    "fig9b_covariate_severity_recovery.png",
                    agg_rec=cov_agg_rec, rec_metric="recovery_windows")


def fig_concept(con_agg, con_agg_rec):
    _severity_curve(con_agg, "macro_f1", "Macro-F1",
                    "Fig 9B-5  Concept drift severity vs Macro-F1",
                    "fig9b_concept_severity_f1.png")
    _severity_curve(con_agg, "accuracy", "Accuracy",
                    "Fig 9B-6  Concept drift severity vs accuracy",
                    "fig9b_concept_severity_accuracy.png")
    _severity_curve(con_agg, "macro_f1", "Recovery time (windows)",
                    "Fig 9B-7  Concept drift severity vs recovery time\n"
                    "(censored: permanent drift, no model recovers)",
                    "fig9b_concept_severity_recovery.png",
                    agg_rec=con_agg_rec, rec_metric="recovery_windows",
                    note="Recovery censored for every model: the concept reversal is "
                         "permanent, so the 95%-of-pre-drift target is unreachable and "
                         "every value equals the 40-window horizon.")


# --------------------------------------------------------------------------
# Figure 9B-8 — F1 degradation: covariate vs concept
# --------------------------------------------------------------------------
def fig_drift_type_comparison(cov_agg, con_agg):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    levels = [l for l in LEVELS]
    x = np.arange(len(levels))
    width = 0.2
    for i, m in enumerate(["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]):
        cv = [cov_agg[(cov_agg["method"] == m) & (cov_agg["drift_level"] == l)]["macro_f1_mean"].mean()
              for l in levels]
        cn = [con_agg[(con_agg["method"] == m) & (con_agg["drift_level"] == l)]["macro_f1_mean"].mean()
              for l in levels]
        lw = 2.6 if m == "RAPT" else 1.2
        ax.plot(x, cv, marker="o", color=COLORS[m], linestyle="-", linewidth=lw,
                alpha=1.0 if m == "RAPT" else 0.7, label=f"{m} (covariate)")
        ax.plot(x, cn, marker="s", color=COLORS[m], linestyle="--", linewidth=lw,
                alpha=1.0 if m == "RAPT" else 0.7, label=f"{m} (concept)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(l*100)}%" for l in levels])
    ax.set_xlabel("Drift severity")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fig 9B-8  Macro-F1 degradation: covariate (solid) vs concept (dashed) drift")
    ax.legend(fontsize=7, ncol=2, frameon=True)
    _save(fig, "fig9b_drift_type_comparison.png")


# --------------------------------------------------------------------------
# Figure 9B-9 — RAPT adaptation timeline
# --------------------------------------------------------------------------
def fig_rapt_timeline(natural):
    pw = natural["per_window"]
    segs = natural["segments"]
    df_r = pw[pw["method"] == "RAPT"].sort_values("window_id")
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=DPI)
    roll = df_r.groupby("window_id")["is_correct"].mean().rolling(15, min_periods=1).mean()
    ax.plot(roll.index, roll.values, color=COLORS["RAPT"], linewidth=2.0, label="RAPT rolling accuracy")

    reuse = df_r[df_r["is_checkpoint_reuse"] == 1]
    retr = df_r[df_r["is_retrain"] == 1]
    ax.scatter(reuse["window_id"], roll.reindex(reuse["window_id"]).values,
               marker="^", s=60, color="green", zorder=5, label="RAPT policy reuse")
    ax.scatter(retr["window_id"], roll.reindex(retr["window_id"]).values,
               marker="v", s=60, color="red", zorder=5, label="RAPT retrain")
    for s in segs[segs["is_transition"] == 1].itertuples():
        ax.axvline(s.start_window, color="black", linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_xlabel("Streaming window index")
    ax.set_ylabel("Rolling accuracy (15-window mean)")
    ax.set_title("Fig 9B-9  RAPT adaptation timeline (reuse / retrain / regime changes)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    _save(fig, "fig9b_rapt_adaptation_timeline.png")


# --------------------------------------------------------------------------
# Figure 9B-10 — Reuse vs retraining
# --------------------------------------------------------------------------
def fig_reuse_vs_retraining(natural, cov_agg, con_agg, rec_agg):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=DPI)
    # Use RAPT only; one group per scenario (natural, covariate@100, concept@100, recurring@100)
    groups = []
    labels = []
    reuse_events, reused_trees, retrained_trees, retrain_events = [], [], [], []

    # natural
    ps = natural["per_seed"]
    r = ps[ps["method"] == "RAPT"]
    labels.append("Natural")
    reuse_events.append(r["reused_checkpoints"].mean())
    reused_trees.append(r["trees_reused"].mean() if "trees_reused" in r else 0)
    retrained_trees.append(r["trees_trained"].mean() if "trees_trained" in r else 0)
    retrain_events.append(r["retrain_events"].mean())

    for name, agg in [("Covariate 100%", cov_agg), ("Concept 100%", con_agg),
                      ("Recurring 100%", rec_agg)]:
        r = agg[(agg["method"] == "RAPT") & (agg["drift_level"] == 1.0)]
        if len(r) == 0:
            continue
        labels.append(name)
        reuse_events.append(r["reused_checkpoints_mean"].mean())
        reused_trees.append(r["trees_reused_mean"].mean() if "trees_reused_mean" in r else 0)
        retrained_trees.append(r["trees_trained_mean"].mean() if "trees_trained_mean" in r else 0)
        retrain_events.append(r["retrain_events_mean"].mean())

    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, retrained_trees, width, label="Newly trained trees", color="#c44e52", edgecolor="black")
    ax.bar(x, reused_trees, width, label="Reused trees", color="#55a868", edgecolor="black")
    ax.bar(x + width, retrain_events, width, label="Retraining events", color="#4c72b0", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Fig 9B-10  RAPT reuse vs retraining across drift scenarios")
    ax.legend(frameon=True, fontsize=8)
    _save(fig, "fig9b_reuse_vs_retraining.png")


# --------------------------------------------------------------------------
# Figure 9B-11 — Accuracy / cost trade-off
# --------------------------------------------------------------------------
def fig_accuracy_cost(cov_agg, con_agg):
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=DPI)
    for agg, marker in [(cov_agg, "o"), (con_agg, "s")]:
        for m in ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]:
            df_m = agg[agg["method"] == m]
            ax.scatter(df_m["adaptation_cpu_sec_mean"], df_m["macro_f1_mean"],
                       marker=marker, s=70, color=COLORS[m], edgecolors="black",
                       alpha=0.8)
    # legend proxies
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[m],
                      markersize=9, label=m) for m in COLORS]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=9, label="covariate"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                       markersize=9, label="concept")]
    ax.set_xlabel("Adaptation CPU (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fig 9B-11  Adaptation efficiency: Macro-F1 vs adaptation CPU")
    ax.legend(handles=handles, fontsize=7, frameon=True, ncol=2)
    _save(fig, "fig9b_accuracy_cost_tradeoff.png")


# --------------------------------------------------------------------------
# Figure 9B-12 — F1 heatmaps
# --------------------------------------------------------------------------
def _heatmap(agg, title, fname):
    methods = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
    data = np.zeros((len(methods), len(LEVELS)))
    for i, m in enumerate(methods):
        for j, l in enumerate(LEVELS):
            v = agg[(agg["method"] == m) & (agg["drift_level"] == l)]["macro_f1_mean"]
            data[i, j] = v.mean() if len(v) else np.nan
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=DPI)
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"{int(l*100)}%" for l in LEVELS])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(LEVELS)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="white" if data[i, j] < 0.6 else "black", fontsize=9)
    ax.set_xlabel("Drift severity")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Macro-F1")
    _save(fig, fname)


def fig_heatmaps(cov_agg, con_agg):
    _heatmap(cov_agg, "Fig 9B-12a  Macro-F1 heatmap — covariate drift",
             "fig9b_covariate_f1_heatmap.png")
    _heatmap(con_agg, "Fig 9B-12b  Macro-F1 heatmap — concept drift",
             "fig9b_concept_f1_heatmap.png")


# --------------------------------------------------------------------------
# Figure 9B-13 — Confusion matrices (RAPT, concept drift)
# --------------------------------------------------------------------------
def fig_confusion(con_pw, n_before=30):
    for level in LEVELS:
        tag = CFG.severity_tag(level)
        df_l = con_pw[(con_pw["drift_level"] == level) & (con_pw["method"] == "RAPT")]
        dp = int(df_l["window_id"][df_l["is_drift_point"] == 1].unique()[0])
        stages = {
            "before": (dp - n_before, dp),
            "immediately after": (dp, dp + 10),
            "after recovery": (dp + 30, dp + 50),
        }
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=DPI)
        for ax, (stage, (lo, hi)) in zip(axes, stages.items()):
            sub = df_l[(df_l["window_id"] >= lo) & (df_l["window_id"] < hi)]
            if len(sub) == 0:
                ax.axis("off")
                continue
            cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1, 2])
            im = ax.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
            ax.set_title(f"{stage}\n(n={len(sub)})", fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["GOOD", "DEGR", "BAD"], fontsize=7)
            ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["GOOD", "DEGR", "BAD"], fontsize=7)
        fig.suptitle(f"Fig 9B-13  RAPT confusion matrices — concept drift severity {tag}%")
        _save(fig, f"fig9b_concept_cm_{tag}.png")


# --------------------------------------------------------------------------
# Figure 9B-14 — Per-regime / per-concept F1
# --------------------------------------------------------------------------
def fig_regime_f1(natural, phase_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=DPI)
    # left: natural per-regime F1
    pw = natural["per_window"]
    ax = axes[0]
    regimes = sorted(pw["regime_id"].unique())
    x = np.arange(len(regimes)); width = 0.2
    for i, m in enumerate(["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]):
        vals = []
        for r in regimes:
            s = pw[(pw["method"] == m) & (pw["regime_id"] == r)]
            vals.append(_f1(s["y_true"], s["y_pred"]) if len(s) else 0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=COLORS[m], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([r.replace("regime_", "") for r in regimes])
    ax.set_ylabel("Macro-F1"); ax.set_title("Per-regime F1 (natural stream)")
    ax.legend(fontsize=7, frameon=True)

    # right: recurring per-phase F1 at 100%
    ax = axes[1]
    df_p = phase_df[phase_df["drift_level"] == 1.0]
    phases = ["A", "B", "A_prime"]
    x = np.arange(len(phases))
    for i, m in enumerate(["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]):
        vals = [df_p[(df_p["phase"] == ph) & (df_p["method"] == m)]["macro_f1"].mean()
                for ph in phases]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=COLORS[m], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "A'"])
    ax.set_ylabel("Macro-F1"); ax.set_title("Per-phase F1 (recurring concept drift, 100%)")
    ax.legend(fontsize=7, frameon=True)
    fig.suptitle("Fig 9B-14  Per-regime / per-concept F1")
    _save(fig, "fig9b_regime_f1.png")


# --------------------------------------------------------------------------
def generate_all(natural, cov_pw, cov_agg, cov_rec, cov_agg_rec,
                 con_pw, con_agg, con_rec, con_agg_rec,
                 rec_pw, rec_agg, rec_rec, rec_agg_rec, phase_df, stats):
    CFG.ensure_dirs()
    print("Generating Experiment 9B drift figures ...", flush=True)
    fig_natural_timeline(natural)
    fig_covariate(cov_agg, cov_agg_rec)
    fig_concept(con_agg, con_agg_rec)
    fig_drift_type_comparison(cov_agg, con_agg)
    fig_rapt_timeline(natural)
    fig_reuse_vs_retraining(natural, cov_agg, con_agg, rec_agg)
    fig_accuracy_cost(cov_agg, con_agg)
    fig_heatmaps(cov_agg, con_agg)
    fig_confusion(con_pw)
    fig_regime_f1(natural, phase_df)
    print("All figures generated.", flush=True)
