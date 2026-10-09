"""
Generate the manuscript's LaTeX table fragments from saved results files.

Every number in the generated .tex is read from results/final/tables/*.csv
(and results/final/summary.json); nothing is typed by hand. Run from the
"Ensemble Learning for Model Drift Detection" directory:

    python paper/make_paper_tables.py
"""

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TABLES = os.path.join(ROOT, "results", "final", "tables")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)

LAB = {"5G Campus": "Campus", "UGR'16": "UGR'16", "NordicDat": "Nordic",
       "5G NR Lat.": "5G NR"}
STREAMS = ["5G Campus", "UGR'16", "NordicDat", "5G NR Lat."]


def _rd(name):
    return pd.read_csv(os.path.join(TABLES, name))


def _w(name, s):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(s.strip() + "\n")
    print("wrote", os.path.join(OUT, name))


def table1():
    t = _rd("table_datasets.csv").set_index("dataset")
    rows = [
        ("Total samples", "raw_samples", "{:,.0f}"),
        ("Usable samples", "usable_samples", "{:,.0f}"),
        ("Features", "features", "{:.0f}"),
        ("Classes", "classes", "{:.0f}"),
        ("Windows", "windows", "{:.0f}"),
        ("Samples/window", "window_size", "{:.0f}"),
        ("Regimes", "regimes", "{:.0f}"),
        ("Initial-train windows", "initial_train_windows", "{:.0f}"),
    ]
    lines = [r"\begin{table}[hbt!]", r"\small",
             r"\caption{Datasets used in evaluations. A window is the adaptation step; "
             r"5G NR is one row per window. Recurrences are revisits of an already-seen "
             r"regime.}", r"\label{tab:datasets}", r"\centering",
             r"\begin{tabular}{@{}lrrrr@{}}",
             r"\toprule",
             r"\textbf{Property} & \textbf{Campus} & \textbf{UGR'16} & "
             r"\textbf{Nordic} & \textbf{5G NR}\\", r"\midrule"]
    for label, col, fmt in rows:
        vals = " & ".join(fmt.format(t.loc[s, col]) for s in STREAMS)
        lines.append(f"{label} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _w("table1_datasets.tex", "\n".join(lines))


def table2():
    t = _rd("table_primary.csv")
    models = ["Frozen", "Event-Driven", "Full Retraining", "RAPT",
              "RAPT-Enhanced", "RAPT-Guard", "RAPT-Anchor"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Primary comparison, mean over 30 seeds. Bold marks the best "
             r"macro-F1 per stream; CPU in seconds, retrains and reuses are counts. "
             r"RAPT-Anchor's total CPU includes its doubled inference.}",
             r"\label{tab:primary}", r"\centering", r"\scriptsize",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Model} & \textbf{Macro-F1} & "
             r"\textbf{Adapt CPU} & \textbf{Total CPU} & \textbf{Retr.} & "
             r"\textbf{Reuse}\\", r"\midrule"]
    for s in STREAMS:
        sub = t[t["stream"] == s].set_index("method")
        best = sub.loc[[m for m in models if m in sub.index], "f1"].max()
        for i, m in enumerate(models):
            r = sub.loc[m]
            f1 = f"{r['f1']:.4f}"
            if abs(r["f1"] - best) < 1e-12:
                f1 = r"$\mathbf{" + f"{r['f1']:.4f}" + "}$"
            pre = (r"\multirow{7}{*}{" + LAB[s] + "}") if i == 0 else ""
            lines.append(
                f"{pre} & {m} & {f1} & {r['adapt']:.3f} & {r['total']:.3f} & "
                f"{int(round(r['retr']))} & {int(round(r['reuse']))} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    _w("table2_primary.tex", "\n".join(lines))


def table3():
    t = _rd("table_detectors.csv")
    rows = ["ADWIN", "Page-Hinkley", "EDD", "EDMA", "Event-Driven", "RAPT"]
    lines = [r"\begin{table}[t]",
             r"\caption{Historical detectors and the reuse methods at the frozen "
             r"operating points, mean over 30 seeds. Each cell is macro-F1 with "
             r"retrain events per seed in parentheses.}", r"\label{tab:detectors}",
             r"\centering", r"\footnotesize",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Campus} & \textbf{UGR'16} & "
             r"\textbf{Nordic} & \textbf{5G NR}\\", r"\midrule"]
    for m in rows:
        cells = []
        for s in STREAMS:
            r = t[(t["stream"] == s) & (t["method"] == m)]
            if len(r) == 0:
                cells += ["--"]
            else:
                cells += [f"{r['f1'].iloc[0]:.3f} ({r['events'].iloc[0]:.1f})"]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _w("table3_detectors.tex", "\n".join(lines))


def table4():
    t = _rd("gate_accuracy.csv").set_index("stream")
    order = ["campus", "ugr16", "nordicdat", "5g_nr"]
    lab = {"campus": "Campus", "ugr16": "UGR'16", "nordicdat": "Nordic",
           "5g_nr": "5G NR"}
    lines = [r"\begin{table}[t]",
             r"\caption{RAPT-Guard decisions at every regime return, three seeds, with "
             r"a two-point tolerance. Bad allow counts a reuse that lost more than two "
             r"macro-F1 points to the prefix policy on the incoming window.}",
             r"\label{tab:gate}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"\textbf{Stream} & \textbf{Returns} & \textbf{Allow} & "
             r"\textbf{Block} & \textbf{Bad allow} & \textbf{Bad block} & "
             r"\textbf{Bad-allow rate}\\", r"\midrule"]
    for s in order:
        r = t.loc[s]
        lines.append(f"{lab[s]} & {int(r['reuse_returns'])} & {int(r['allows'])} & "
                     f"{int(r['blocks'])} & {int(r['bad_allows'])} & "
                     f"{int(r['bad_blocks'])} & {r['bad_allow_rate']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _w("table4_gate.tex", "\n".join(lines))


def table5():
    df = pd.read_csv(os.path.join(ROOT, "results", "final", "dev",
                                  "guard_sweep.csv"))
    g = df.groupby("margin").agg(f1=("macro_f1", "mean"),
                                 allow=("gate_allow", "sum"),
                                 block=("gate_block", "sum")).reset_index()
    g["rate"] = g["allow"] / (g["allow"] + g["block"])
    lines = [r"\begin{table}[t]",
             r"\caption{Development-split guard margin sweep (pooled over four streams, "
             r"five seeds). The frozen margin is $0.00$.}",
             r"\label{tab:guardmargin}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{@{}rrr@{}}", r"\toprule",
             r"\textbf{Margin} & \textbf{Dev macro-F1} & \textbf{Allow rate}\\",
             r"\midrule"]
    for _, r in g.iterrows():
        lines.append(f"{r['margin']:+.2f} & {r['f1']:.4f} & {r['rate']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _w("table5_guard_margin.tex", "\n".join(lines))


if __name__ == "__main__":
    table1(); table2(); table3(); table4(); table5()
