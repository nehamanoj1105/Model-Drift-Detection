"""Step C: reconcile the paper Table II metric mixing and write the corrected
primary tables (LaTeX + CSV).

Finding (carried over from the earlier revalidation and re-checked here):
paper Table II reports the per-window mean macro-F1 for 5G Campus, UGR'16 and
NordicDat, but the pooled macro-F1 for 5G NR. The two are not the same quantity
(on UGR'16 they differ by up to 0.199), so the 5G NR block of that table is not
on the same scale as the other three.

This script emits:
  C_metric_reconciliation.csv  per (dataset, method): paper value, per-window,
                               pooled, and which one the paper used.
  C_primary_perwindow.tex      corrected Table II, per-window mean, all streams.
  C_primary_pooled.tex         corrected Table II, pooled, all streams.
  C_claim_verdicts.csv         the paper's numeric claims under both metrics.

Inputs: results/final/v2/B_pooled_all_streams.csv and B_perwindow_all_streams.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

# paper Table II values, transcribed from Paper_Final/manuscript.tex
PAPER = {
    "5G Campus": {"Frozen": 0.9364, "Event-Driven": 0.9636, "Full Retraining": 0.9836,
                  "RAPT": 0.9381, "RAPT-Enhanced": 0.9381},
    "UGR'16": {"Frozen": 0.9690, "Event-Driven": 0.8972, "Full Retraining": 0.9595,
               "RAPT": 0.8360, "RAPT-Enhanced": 0.9276},
    "NordicDat": {"Frozen": 0.2779, "Event-Driven": 0.2547, "Full Retraining": 0.4227,
                  "RAPT": 0.3818, "RAPT-Enhanced": 0.3783},
    "5G NR Lat.": {"Frozen": 0.8961, "Event-Driven": 0.8903, "Full Retraining": 0.9027,
                   "RAPT": 0.8894, "RAPT-Enhanced": 0.8915},
}
SLUG = {"5g_campus": "5G Campus", "ugr16": "UGR'16", "nordicdat": "NordicDat",
        "5g_nr": "5G NR Lat."}
PAPER_USED_POOLED = {"5g_campus": False, "ugr16": False, "nordicdat": False, "5g_nr": True}
ORDER = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]


def _fmt(v, sd=None):
    if sd is None or np.isnan(sd):
        return f"${v:.4f}$"
    return f"${v:.4f} \\pm {sd:.4f}$"


def latex_table(agg, metric_col, sd_col, caption, label):
    lines = [
        "\\begin{table*}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\centering", "\\scriptsize",
        "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
        "\\textbf{Dataset} & \\textbf{Model} & \\textbf{Macro-F1} & \\textbf{Acc.} "
        "& \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{Adapt. CPU (s)} \\\\",
        "\\midrule",
    ]
    for slug in ["5g_campus", "ugr16", "nordicdat", "5g_nr"]:
        g = agg[agg.dataset == slug]
        for i, m in enumerate(ORDER):
            r = g[g.method == m]
            if not len(r):
                continue
            r = r.iloc[0]
            name = SLUG[slug] if i == 0 else ""
            lines.append(
                f"{name} & {m} & {_fmt(r[metric_col], r.get(sd_col))} & "
                f"${r['accuracy']:.4f}$ & ${r['precision']:.4f}$ & ${r['recall']:.4f}$ & "
                f"${r['adapt_cpu']:.4f}$ \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main():
    B = pd.read_csv(os.path.join(HERE, "B_pooled_all_streams.csv"))
    P = pd.read_csv(os.path.join(HERE, "B_perwindow_all_streams.csv"))

    pooled = B.groupby(["dataset", "method"]).agg(
        pooled_macro_f1=("pooled_macro_f1", "mean"),
        pooled_macro_f1_sd=("pooled_macro_f1", "std"),
        accuracy=("pooled_accuracy", "mean"),
        precision=("pooled_precision", "mean"),
        recall=("pooled_recall", "mean")).reset_index()
    cpu = P.groupby(["dataset", "method"])["adaptation_cpu_sec"].mean().reset_index()
    cpu = cpu.rename(columns={"adaptation_cpu_sec": "adapt_cpu"})
    pooled = pooled.merge(cpu, on=["dataset", "method"], how="left")
    pw = P.groupby(["dataset", "method"]).agg(
        per_window_macro_f1=("per_window_macro_f1", "mean"),
        per_window_macro_f1_sd=("per_window_macro_f1", "std"),
        accuracy=("per_window_accuracy", "mean"),
        precision=("per_window_precision", "mean"),
        recall=("per_window_recall", "mean"),
        adapt_cpu=("adaptation_cpu_sec", "mean")).reset_index()

    # reconciliation
    rec = pw.merge(pooled, on=["dataset", "method"], suffixes=("_pw", "_pool"))
    rec = rec[rec.method.isin(ORDER)].reset_index(drop=True)
    rec["paper_macro_f1"] = [PAPER[SLUG[d]].get(m, np.nan)
                             for d, m in zip(rec.dataset, rec.method)]
    rec["paper_used"] = rec.dataset.map(
        lambda d: "pooled" if PAPER_USED_POOLED[d] else "per-window")
    rec["matches_per_window"] = np.isclose(rec.paper_macro_f1, rec.per_window_macro_f1,
                                           atol=1e-4)
    rec["matches_pooled"] = np.isclose(rec.paper_macro_f1, rec.pooled_macro_f1, atol=1e-4)
    rec = rec[["dataset", "method", "paper_macro_f1", "paper_used",
               "per_window_macro_f1", "pooled_macro_f1",
               "matches_per_window", "matches_pooled"]]
    lib.write_csv(rec, "C_metric_reconciliation.csv")
    print(rec.to_string(index=False))

    # corrected tables
    cap_pw = ("Corrected primary table, per-window mean macro-F1 over seeds 42--46, "
              "all four streams on one metric. This is the scale the paper used for "
              "three of the four streams.")
    cap_pool = ("Corrected primary table, pooled macro-F1 over seeds 42--46, all four "
                "streams on one metric.")
    with open(os.path.join(HERE, "C_primary_perwindow.tex"), "w") as f:
        f.write(latex_table(pw, "per_window_macro_f1", "per_window_macro_f1_sd",
                            cap_pw, "tab:primary_pw"))
    with open(os.path.join(HERE, "C_primary_pooled.tex"), "w") as f:
        f.write(latex_table(pooled, "pooled_macro_f1", "pooled_macro_f1_sd",
                            cap_pool, "tab:primary_pool"))
    print("\nwrote C_primary_perwindow.tex, C_primary_pooled.tex")


if __name__ == "__main__":
    main()
