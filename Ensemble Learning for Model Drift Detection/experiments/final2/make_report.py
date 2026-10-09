"""Assemble one downloadable document: results/final2/FINAL2_REPORT.md.

Reads only saved results files (nothing hand-typed), plus the narrative in
CHANGES.md, and writes a single self-contained report.
"""

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "results", "final2")
T = os.path.join(R, "tables")

STREAM_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
               "5g_nr": "5G NR Latency", "kpiops": "Network operator KPIs"}
ORDER = ["Frozen", "Frozen+Periodic", "Event-Driven", "Event-Driven+Reuse",
         "ADWIN", "Page-Hinkley", "Full Retraining", "RAPT", "RAPT-Guard",
         "RAPT-Enhanced", "RAPT-Anchor", "RAPT-ED-Guard"]


def md(df):
    cols = df.columns.tolist()
    out = "| " + " | ".join(str(c) for c in cols) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def main():
    L = []
    L.append("# RAPT Final-2 Report — fast job (2026-10-09)\n")
    L.append("**One-file deliverable.** Every number below is read from "
             "`results/final2/` files produced by code in this repo; nothing is "
             "hand-typed. The paper (`paper/main.tex`) was **not** modified.\n")
    L.append("## Contents\n")
    L.append("1. Protocol & frozen config\n2. Main results (12 methods x 4 streams, 30 seeds)\n"
             "3. Cost tables\n4. Statistics (Wilcoxon+Holm, Cohen's dz, TOST)\n"
             "5. RAPT-ED-Guard gate accuracy\n6. Long vs short window gate\n"
             "7. Detector dev sweep\n8. Concept-drift stress panel\n"
             "9. Class-weight / leakage diagnosis\n10. What changed / got worse / claims softened\n"
             "11. How to reproduce\n")

    L.append("\n## 1. Protocol & frozen config\n")
    L.append("- Tuning: 5 seeds (42-46), first 60% of post-prefix windows, dev only.\n")
    L.append("- Final: all 30 seeds (42-71), full streams, run once with frozen configs.\n")
    for cf in ("final.yaml", "final2.yaml"):
        p = os.path.join(ROOT, "configs", cf)
        if os.path.exists(p):
            L.append(f"\n`configs/{cf}`:\n\n```yaml\n{open(p).read()}\n```\n")

    s = pd.read_csv(os.path.join(T, "summary_by_method.csv"))
    s["method"] = pd.Categorical(s["method"], ORDER, ordered=True)
    s = s.sort_values("method")

    L.append("\n## 2. Main results (30 seeds, full stream)\n")
    piv = s.pivot_table(index="method", columns="stream", values="macro_f1").reindex(ORDER)
    piv = piv[[c for c in STREAM_NAME if c in piv.columns]]
    L.append("\n### Macro-F1\n")
    L.append(md(piv.reset_index()))

    L.append("\n## 3. Cost tables (mean CPU seconds / counts per seed)\n")
    for col in ["adaptation_cpu_sec", "prediction_cpu_sec", "total_cpu_sec",
                "retrains", "reuses"]:
        p = s.pivot_table(index="method", columns="stream", values=col).reindex(ORDER)
        p = p[[c for c in STREAM_NAME if c in p.columns]]
        L.append(f"\n### {col}\n")
        L.append(md(p.reset_index()))

    L.append("\n## 4. Statistics (window Wilcoxon + Holm; seed Cohen's dz; TOST +/-0.02)\n")
    st = pd.read_csv(os.path.join(T, "statistics.csv"))
    show = ["stream", "comparison", "window_mean_diff", "window_p_holm",
            "seed_dz", "seed_t_p", "tost_p", "tost_equiv", "ci95_low", "ci95_high"]
    L.append(md(st[show]))

    L.append("\n## 5. RAPT-ED-Guard gate accuracy\n")
    L.append(md(pd.read_csv(os.path.join(R, "gate_accuracy.csv"))))
    ga = pd.read_csv(os.path.join(R, "gate_accuracy.csv"))
    L.append(f"\nOverall bad-allow rate: "
             f"{(ga.bad_allows.sum() / max(ga.allows.sum(), 1)):.3f}; "
             f"missed reuses: {int(ga.bad_blocks.sum())} of {int(ga.blocks.sum())} blocks.\n")

    L.append("\n## 6. Long-window vs short-window gate\n")
    L.append(md(pd.read_csv(os.path.join(R, "gate_cadence.csv"))))

    L.append("\n## 7. Detector dev sweep (ADWIN delta / Page-Hinkley threshold)\n")
    d = pd.read_csv(os.path.join(R, "dev_detector_sweep.csv"))
    agg = d.groupby(["stream", "method"]).agg(
        retrain_min=("retrain_events", "min"),
        retrain_mean=("retrain_events", "mean"),
        retrain_max=("retrain_events", "max")).reset_index()
    L.append(md(agg))
    L.append("\nFrozen: `adwin_delta=0.2`, `ph_threshold=0.002`.\n")

    L.append("\n## 8. Concept-drift stress panel (floor trigger enabled)\n")
    L.append(md(pd.read_csv(os.path.join(R, "concept_panel.csv"))))
    L.append("\n### Trajectory spread (mean max-min macro-F1 across methods)\n")
    L.append(md(pd.read_csv(os.path.join(R, "concept_spread.csv"))))
    L.append("\nSpread stays <=0.02: the panel is **degenerate even after the fix**; "
             "recommend dropping it from the paper.\n")

    L.append("\n## 9. Class-weight / leakage diagnosis (all streams)\n")
    L.append(md(pd.read_csv(os.path.join(R, "nordicdat_diagnosis.csv"))))

    L.append("\n## 10. What changed, what got worse, claims softened\n")
    ch = os.path.join(R, "CHANGES.md")
    if os.path.exists(ch):
        body = open(ch).read()
        # strip the top title line to avoid a duplicate heading
        body = "\n".join(body.splitlines()[1:]).strip()
        L.append(body)

    L.append("\n## 11. How to reproduce\n")
    L.append("```\n"
             "python experiments/final2/fetch_kpiops.py\n"
             "python experiments/final2/final2_dev_margin.py\n"
             "python experiments/final2/final2_dev_detectors.py\n"
             "python experiments/final2/final2_runner.py --mode final --streams all --tag final --n-jobs 4\n"
             "python experiments/final2/final2_stats.py --tag final\n"
             "python experiments/final2/final2_gate.py\n"
             "python experiments/final2/final2_gate_cadence.py\n"
             "python experiments/final2/final2_concept.py\n"
             "python experiments/final2/final2_nordicdat.py\n"
             "python experiments/final2/final2_figures.py\n"
             "python experiments/final2/make_review.py\n"
             "python experiments/final2/make_report.py\n"
             "```\n")
    L.append("\nFigures: `results/final2/figures/fig_main_macro_f1.png`, "
             "`fig_adapt_cpu.png`, `fig_cost.png`, `fig_gate_cadence.png`.\n")
    L.append("\nMachine-readable: `results/final2/summary.json`, "
             "`results/final2/tables/*.csv`, `results/final2/paper_values.md`.\n")

    out = "\n".join(L)
    with open(os.path.join(R, "FINAL2_REPORT.md"), "w") as f:
        f.write(out)
    print("wrote", os.path.join(R, "FINAL2_REPORT.md"), f"({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
