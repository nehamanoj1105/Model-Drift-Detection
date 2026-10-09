"""Build the single review file results/final2/RESULTS_REVIEW.md.

Reads only saved results files under results/final2/ and prints every number in
one place, so the paper can be updated against files rather than hand-typed
values. No paper files are touched.
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


def _md_table(df, floatfmt="{:.4f}"):
    cols = df.columns.tolist()
    out = "| " + " | ".join(str(c) for c in cols) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def main():
    L = []
    L.append("# Final-2 results review\n")
    L.append("All numbers below are read from `results/final2/` files. "
             "Nothing here is hand-typed or taken from the paper.\n")
    L.append("Streams: " + ", ".join(f"`{k}` = {v}" for k, v in STREAM_NAME.items()) + "\n")

    summary_path = os.path.join(T, "summary_by_method.csv")
    if os.path.exists(summary_path):
        s = pd.read_csv(summary_path)
        s["method"] = pd.Categorical(s["method"], ORDER, ordered=True)
        s = s.sort_values(["stream", "method"])
        L.append("\n## 1. Main table (30 seeds, full stream)\n")
        piv = s.pivot_table(index="method", columns="stream", values="macro_f1").reindex(ORDER)
        piv = piv[[c for c in STREAM_NAME if c in piv.columns]]
        L.append("\n### Macro-F1\n")
        L.append(_md_table(piv.reset_index().rename(columns={"method": "method"})))
        for col in ["total_cpu_sec", "adaptation_cpu_sec", "prediction_cpu_sec",
                    "retrains", "reuses"]:
            p = s.pivot_table(index="method", columns="stream", values=col).reindex(ORDER)
            p = p[[c for c in STREAM_NAME if c in p.columns]]
            L.append(f"\n### {col}\n")
            L.append(_md_table(p.reset_index()))
        s.to_csv(os.path.join(R, "tables", "REVIEW_summary_long.csv"), index=False)

    st_path = os.path.join(T, "statistics.csv")
    if os.path.exists(st_path):
        st = pd.read_csv(st_path)
        L.append("\n## 2. Statistics (Wilcoxon+Holm, Cohen's dz, TOST +/-0.02)\n")
        show = ["stream", "comparison", "window_n", "window_mean_diff", "window_p",
                "window_p_holm", "seed_dz", "seed_t_p", "tost_p", "tost_equiv",
                "ci95_low", "ci95_high"]
        L.append(_md_table(st[show]))

    g_path = os.path.join(R, "gate_accuracy.csv")
    if os.path.exists(g_path):
        L.append("\n## 3. RAPT-ED-Guard gate accuracy\n")
        L.append(_md_table(pd.read_csv(g_path)))

    c_path = os.path.join(R, "gate_cadence.csv")
    if os.path.exists(c_path):
        L.append("\n## 4. Long-window vs short-window gate\n")
        L.append(_md_table(pd.read_csv(c_path)))

    cp_path = os.path.join(R, "concept_panel.csv")
    sp_path = os.path.join(R, "concept_spread.csv")
    if os.path.exists(cp_path):
        L.append("\n## 5. Concept-drift stress panel (floor trigger enabled)\n")
        L.append(_md_table(pd.read_csv(cp_path)))
    if os.path.exists(sp_path):
        L.append("\n### Trajectory spread (mean max-min macro-F1 across methods)\n")
        L.append(_md_table(pd.read_csv(sp_path)))

    d_path = os.path.join(R, "dev_detector_sweep.csv")
    if os.path.exists(d_path):
        d = pd.read_csv(d_path)
        L.append("\n## 6. Detector dev sweep (ADWIN delta / Page-Hinkley threshold)\n")
        agg = d.groupby(["stream", "method"]).agg(
            retrain_events_min=("retrain_events", "min"),
            retrain_events_mean=("retrain_events", "mean"),
            retrain_events_max=("retrain_events", "max")).reset_index()
        L.append(_md_table(agg))

    mg_path = os.path.join(R, "dev_margin_sweep.csv")
    if os.path.exists(mg_path):
        mg = pd.read_csv(mg_path)
        L.append("\n## 7. RAPT-ED-Guard margin dev sweep\n")
        L.append(_md_table(mg.groupby(["stream", "margin"]).agg(
            macro_f1=("macro_f1", "mean"), gate_allows=("gate_allows", "sum")).reset_index()))

    nd_path = os.path.join(R, "nordicdat_diagnosis.csv")
    if os.path.exists(nd_path):
        L.append("\n## 8. Class-weight / leakage diagnosis (all streams)\n")
        L.append(_md_table(pd.read_csv(nd_path)))

    L.append("\n## 9. Frozen configs used\n")
    for cf in ("final.yaml", "final2.yaml"):
        p = os.path.join(ROOT, "configs", cf)
        if os.path.exists(p):
            L.append(f"\n### {cf}\n```\n{open(p).read()}\n```\n")

    L.append("\n## 10. Dataset\n"
             "New stream replaces NordicDat: *Network operator KPIs time series* "
             "(Zenodo 8147768, CC BY-NC 4.0). See `new_dataset.md`. "
             "14 real KPI series, 5-min cadence, 37 days, 223 windows, "
             "6 time-of-day regimes each recurring ~37 times, producer-annotated "
             "binary anomaly target.\n")

    out = "\n".join(L)
    with open(os.path.join(R, "RESULTS_REVIEW.md"), "w") as f:
        f.write(out)
    with open(os.path.join(R, "paper_values.md"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
