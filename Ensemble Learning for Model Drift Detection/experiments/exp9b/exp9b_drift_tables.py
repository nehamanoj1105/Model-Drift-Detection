"""
CSV and LaTeX tables for Experiment 9B — Drift Severity & Concept Drift.

Tables 9B-1 .. 9B-5 are written to results/experiment_9b/tables/ as both .csv
and .tex (booktabs). Every value is derived from the saved per-seed / per-window
results; nothing is fabricated.
"""

import os
import numpy as np
import pandas as pd

import exp9b_drift_config as CFG

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]


def _write(df, name):
    csv_path = os.path.join(CFG.TABLES_DIR, f"{name}.csv")
    tex_path = os.path.join(CFG.TABLES_DIR, f"{name}.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f",
                            caption=name.replace("_", " "),
                            label=f"tab:{name}"))
    print(f"  saved {csv_path} and {tex_path}", flush=True)


def _mean_std(df, by, col):
    g = df.groupby(by)[col]
    return g.mean(), g.std()


def table_1_natural(natural):
    ps = natural["per_seed"]
    rows = []
    for m in METHODS:
        d = ps[ps["method"] == m]
        rows.append({
            "Model": m,
            "Macro_F1": d["macro_f1"].mean(),
            "Macro_F1_std": d["macro_f1"].std(),
            "Accuracy": d["accuracy"].mean(),
            "Precision": d["precision"].mean(),
            "Recall": d["recall"].mean(),
            "Adapt_CPU_s": d["adaptation_cpu_sec"].mean(),
            "Runtime_s": d["total_cpu_sec"].mean(),
            "Retrains": d["retrain_events"].mean(),
            "Reuse_events": d["reused_checkpoints"].mean(),
        })
    _write(pd.DataFrame(rows), "table_9b1_natural_drift")


def _severity_table(agg, name):
    rows = []
    for m in METHODS:
        for l in CFG.DRIFT_LEVELS:
            d = agg[(agg["method"] == m) & (agg["drift_level"] == l)]
            if len(d) == 0:
                continue
            d = d.iloc[0]
            rows.append({
                "Model": m,
                "Severity": f"{int(l*100)}%",
                "F1": round(d["macro_f1_mean"], 4),
                "F1_std": round(d["macro_f1_std"], 4),
                "Accuracy": round(d["accuracy_mean"], 4),
                "Precision": round(d["precision_mean"], 4),
                "Recall": round(d["recall_mean"], 4),
                "Adapt_CPU_s": round(d["adaptation_cpu_sec_mean"], 4),
                "Runtime_s": round(d["total_cpu_sec_mean"], 4),
                "Retrains": round(d["retrain_events_mean"], 2),
            })
    _write(pd.DataFrame(rows), name)


def table_2_covariate(agg):
    _severity_table(agg, "table_9b2_covariate_drift")


def table_3_concept(agg):
    _severity_table(agg, "table_9b3_concept_drift")


def table_4_recovery(cov_rec, con_rec, rec_rec):
    rows = []
    for scenario, rec in [("covariate", cov_rec), ("concept", con_rec),
                          ("recurring", rec_rec)]:
        if rec is None or rec.empty:
            continue
        for m in METHODS:
            for l in CFG.DRIFT_LEVELS:
                d = rec[(rec["method"] == m) & (rec["drift_level"] == l)]
                if len(d) == 0:
                    continue
                rows.append({
                    "Model": m,
                    "Drift_type": scenario,
                    "Severity": f"{int(l*100)}%",
                    "Pre_drift_F1": round(d["pre_drift_f1"].mean(), 4),
                    "Minimum_F1": round(d["min_post_f1"].mean(), 4),
                    "Recovery_F1": round(d["recovery_f1"].mean(), 4),
                    "Recovery_windows": round(d["recovery_windows"].mean(), 2),
                    "Delta_F1": round(d["delta_f1"].mean(), 4),
                })
    _write(pd.DataFrame(rows), "table_9b4_recovery")


def table_5_rapt_reuse(natural, cov_agg, con_agg, rec_agg):
    rows = []
    ps = natural["per_seed"]
    d = ps[ps["method"] == "RAPT"]
    # The original 9B run did not persist per-tree counts, so report them as n/a
    # rather than fabricating values.
    has_trees = "trees_reused" in ps.columns
    rows.append({
        "Scenario": "natural", "Severity": "-",
        "Reuse_events": round(d["reused_checkpoints"].mean(), 2),
        "Reused_trees": round(d["trees_reused"].mean(), 2) if has_trees else "-",
        "Retrained_trees": round(d["trees_trained"].mean(), 2) if has_trees else "-",
        "Adapt_CPU_s": round(d["adaptation_cpu_sec"].mean(), 4),
    })
    for scenario, agg in [("covariate", cov_agg), ("concept", con_agg),
                          ("recurring", rec_agg)]:
        for l in CFG.DRIFT_LEVELS:
            d = agg[(agg["method"] == "RAPT") & (agg["drift_level"] == l)]
            if len(d) == 0:
                continue
            d = d.iloc[0]
            rows.append({
                "Scenario": scenario, "Severity": f"{int(l*100)}%",
                "Reuse_events": round(d["reused_checkpoints_mean"], 2),
                "Reused_trees": round(d.get("trees_reused_mean", 0), 2),
                "Retrained_trees": round(d.get("trees_trained_mean", 0), 2),
                "Adapt_CPU_s": round(d["adaptation_cpu_sec_mean"], 4),
            })
    df = pd.DataFrame(rows)
    # Mixed numeric/string columns are coerced to float; restore the placeholder.
    # (Avoid "n/a" — pandas treats it as NaN on read-back.)
    df = df.fillna("-")
    _write(df, "table_9b5_rapt_reuse")


def generate_all(natural, cov_agg, con_agg, rec_agg, cov_rec, con_rec, rec_rec):
    CFG.ensure_dirs()
    print("Generating Experiment 9B drift tables ...", flush=True)
    table_1_natural(natural)
    table_2_covariate(cov_agg)
    table_3_concept(con_agg)
    table_4_recovery(cov_rec, con_rec, rec_rec)
    table_5_rapt_reuse(natural, cov_agg, con_agg, rec_agg)
    print("All tables generated.", flush=True)
