"""
Emit every number that appears in the manuscript, read from the saved final
results, so the paper can be value-synced against the code.

Run from the "Ensemble Learning for Model Drift Detection" directory:

    python paper/make_paper_values.py

Writes paper_values.md next to this file. Nothing is typed by hand: every value
comes from results/final/summary.json, results/final/tables/*.csv or the raw
per-seed files.
"""

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FINAL = os.path.join(ROOT, "results", "final")
TABLES = os.path.join(FINAL, "tables")

STREAMS = ["5G Campus", "UGR'16", "NordicDat", "5G NR Lat."]
SHORT = {"5G Campus": "Campus", "UGR'16": "UGR'16", "NordicDat": "Nordic",
         "5G NR Lat.": "5G NR"}
PRIMARY = ["Frozen", "Event-Driven", "Full Retraining", "RAPT",
           "RAPT-Enhanced", "RAPT-Guard", "RAPT-Anchor"]
STAT_STREAM = {"5G Campus": "5g_campus", "UGR'16": "ugr16",
               "NordicDat": "nordicdat", "5G NR Lat.": "5g_nr"}

out = []


def w(line=""):
    out.append(line)


def rd(name):
    return pd.read_csv(os.path.join(TABLES, name))


summary = json.load(open(os.path.join(FINAL, "summary.json")))
sm = pd.DataFrame(summary["summary"])
primary = rd("table_primary.csv")
stats = rd("statistics.csv")
datasets = rd("table_datasets.csv").set_index("dataset")
detectors = rd("table_detectors.csv")
gate = rd("gate_accuracy.csv").set_index("stream")
cadence = rd("gate_cadence.csv")

w("# Paper values (auto-generated from results/final)")
w()
w(f"Seeds: {summary['n_seeds']} (42-71). TOST equivalence margin: "
  f"+/-{summary['tost_margin']} macro-F1.")
w()

w("## Table I - Datasets")
w()
for ds in datasets.index:
    r = datasets.loc[ds]
    w(f"- {ds}: total={int(r['raw_samples'])}, usable={int(r['usable_samples'])}, "
      f"features={int(r['features'])}, classes={int(r['classes'])}, "
      f"windows={int(r['windows'])}, samples/window={int(r['window_size'])}, "
      f"regimes={int(r['regimes'])}, initial_train_windows="
      f"{int(r['initial_train_windows'])}")
w()

w("## Table II - Primary comparison (mean over 30 seeds)")
w()
for st in STREAMS:
    sub = primary[primary["stream"] == st]
    best = sub["f1"].max()
    w(f"### {SHORT[st]}")
    for _, r in sub.iterrows():
        bold = " <-- best F1" if abs(r["f1"] - best) < 1e-12 else ""
        w(f"- {r['method']}: F1={r['f1']:.4f}, adapt={r['adapt']:.3f}s, "
          f"total={r['total']:.3f}s, retrains={r['retr']:.0f}, "
          f"reuses={r['reuse']:.0f}{bold}")
    w()

w("## Table III - Historical detectors (events per seed, F1)")
w()
for _, r in detectors.iterrows():
    w(f"- {r['method']} / {SHORT.get(r['stream'], r['stream'])}: "
      f"events={r['events']:.1f}, F1={r['f1']:.3f}")
w()

w("## Table IV - RAPT-Guard decisions")
w()
for s in gate.index:
    r = gate.loc[s]
    w(f"- {s}: returns={int(r['reuse_returns'])}, allows={int(r['allows'])}, "
      f"blocks={int(r['blocks'])}, bad_allows={int(r['bad_allows'])}, "
      f"bad_blocks={int(r['bad_blocks'])}, bad_allow_rate={r['bad_allow_rate']:.3f}")
w()

w("## Gate cadence (short vs long window)")
w()
w(cadence.to_string(index=False))
w()

w("## Abstract / headline numbers")
w()
for st in STREAMS:
    sub = primary[primary["stream"] == st]
    fr = sub[sub["method"] == "Full Retraining"].iloc[0]
    ra = sub[sub["method"] == "RAPT"].iloc[0]
    gu = sub[sub["method"] == "RAPT-Guard"].iloc[0]
    fo = sub[sub["method"] == "Frozen"].iloc[0]
    red = 100.0 * (fr["adapt"] - ra["adapt"]) / fr["adapt"]
    gred = 100.0 * (fr["adapt"] - gu["adapt"]) / fr["adapt"]
    w(f"- {SHORT[st]}: RAPT adaptation-CPU reduction vs Full Retraining "
      f"= {red:.1f}%; RAPT-Guard = {gred:.1f}%")
    w(f"    Frozen F1={fo['f1']:.4f}, RAPT F1={ra['f1']:.4f}, "
      f"RAPT-Guard F1={gu['f1']:.4f}, Full F1={fr['f1']:.4f}")
w()

w("## Key significance tests (window p, Holm; seed dz)")
w()
pairs = [
    ("5G Campus", "RAPT vs Full Retraining"),
    ("UGR'16", "RAPT vs Frozen"),
    ("UGR'16", "RAPT-Guard vs Frozen"),
    ("UGR'16", "RAPT-Guard vs Event-Driven"),
    ("UGR'16", "RAPT vs Full Retraining"),
    ("NordicDat", "RAPT vs Frozen"),
    ("NordicDat", "RAPT vs Full Retraining"),
    ("5G NR Lat.", "RAPT-Guard vs Full Retraining"),
]
for st, comp in pairs:
    row = stats[(stats["stream"] == STAT_STREAM[st]) &
                (stats["comparison"] == comp)]
    if len(row):
        r = row.iloc[0]
        w(f"- {SHORT[st]} / {comp}: seed_diff={r['seed_mean_diff']:+.4f}, "
          f"dz={r['seed_dz']:+.2f}, window_p_holm={r['window_p_holm']:.2e}, "
          f"seed_t_p={r['seed_t_p']:.2e}")
w()

w("## Efficiency ablation (5G Campus, five seeds, legacy harness)")
w()
abl_path = os.path.join(ROOT, "Final_Experiments", "results", "tables",
                        "table_final_main.csv")
ablation_map = {
    "Full Retraining": "Full Retraining (ref.)",
    "RAPT_FULL": "RAPT-Full",
    "RAPT_EVIDENCE": "RAPT-Evidence",
    "RAPT_FLOOR": "RAPT-Floor",
    "RAPT_CHEAP": "RAPT-Cheap",
    "RAPT_INCR": "RAPT-Incremental",
}
if os.path.exists(abl_path):
    abl = pd.read_csv(abl_path).set_index("Model")
    for src, lab in ablation_map.items():
        if src in abl.index:
            row = abl.loc[src]
            f1 = str(row["Macro-F1"]).split(" +/- ")[0]
            cpu = str(row["Adapt CPU (s)"]).split(" +/- ")[0]
            w(f"- {lab}: F1={f1}, adapt={cpu}s")
    w("- Source: Final_Experiments/results/tables/table_final_main.csv "
      "(five-seed legacy 5G-Campus harness, not results/final).")
    stats_path = os.path.join(ROOT, "Final_Experiments", "results", "tables",
                              "table_final_stats.csv")
    if os.path.exists(stats_path):
        fs = pd.read_csv(stats_path)
        fs = fs[fs["Comparison"].str.contains("vs Full Retraining")]
        for _, r in fs.iterrows():
            w(f"  {r['Model']} vs Full Retraining: delta={r['mean_delta']:+.4f}, "
              f"CI=[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}], "
              f"d={r['cohens_d']:+.2f}, p={r['wilcoxon_p']:.4f}")
else:
    w("- MISSING: table_final_main.csv")
w()

w("## Cross-stream controls (from Independent Re-validation)")
w()
oracle = pd.read_csv(os.path.join(ROOT, "results", "revalidation",
                                  "A2_oracle_diagnostics.csv"))
w("Oracle same-regime revisit (pooled macro-F1, mean over seeds):")
for ds in ["5g_campus", "ugr16", "nordicdat", "5g_nr"]:
    sub = oracle[oracle["dataset"] == ds]
    w(f"- {ds}: {sub['pooled_macro_f1'].mean():.4f} "
      f"+/- {sub['pooled_macro_f1'].std():.4f}")
for lab in ["Frozen+Periodic", "Event-Driven+Reuse"]:
    row = primary[primary["method"] == lab]
    for _, r in row.iterrows():
        w(f"- {lab} on {SHORT[r['stream']]}: F1={r['f1']:.4f}")
delay = pd.read_csv(os.path.join(ROOT, "results", "revalidation",
                                 "A8_label_delay.csv"))
w("Label-delay (pooled macro-F1, delay=1 minus delay=0):")
for ds in delay["dataset"].unique():
    for m in ["RAPT", "Full Retraining"]:
        d0 = delay[(delay["dataset"] == ds) & (delay["method"] == m) &
                   (delay["label_delay"] == 0)]["pooled_macro_f1"]
        d1 = delay[(delay["dataset"] == ds) & (delay["method"] == m) &
                   (delay["label_delay"] == 1)]["pooled_macro_f1"]
        if len(d0) and len(d1):
            w(f"- {ds} / {m}: delta={d1.mean() - d0.mean():+.4f}")
w()

path = os.path.join(HERE, "paper_values.md")
with open(path, "w") as f:
    f.write("\n".join(out) + "\n")
print("wrote", path)
print("\n".join(out))
