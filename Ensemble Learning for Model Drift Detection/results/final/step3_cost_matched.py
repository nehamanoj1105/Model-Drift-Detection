"""Step 3: cost-matched baselines on all four streams (primary metric).

Source: existing revalidation A3 raw per-window CSV. Cost-matched methods hold
the refit budget roughly constant: cheap refits use 20 trees / 300 rows vs the
full 50 trees / 1000 rows.

Outputs
  T3_cost_matched.csv   per (dataset, method): macro-F1 (per-window), pooled,
                        adaptation CPU, runtime, retrains, reuse, refresh
  T3_pareto.csv         non-dominated set on (max F1, min adaptation CPU)
  T3_vs_full.csv        each cheap method vs Full Retraining, paired over seeds
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")
sys.path.insert(0, REVAL)

from common import SEEDS  # noqa: E402

SLUG_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
             "nordicdat": "NordicDat", "5g_nr": "5G NR"}
CHEAP = ["FR-Cheap", "Periodic-Cheap-5", "Periodic-Cheap-10",
         "Event-Driven-Cheap", "RAPT-Cheap"]


def load():
    pw = pd.read_csv(os.path.join(REVAL, "raw", "A3_per_window.csv"))
    sm = pd.read_csv(os.path.join(REVAL, "A3_summary.csv"))
    pooled = pd.read_csv(os.path.join(REVAL, "A3_cost_matched.csv"))
    return pw, sm, pooled


def table(pw, sm, pooled):
    pw_mean = pw.groupby(["dataset", "method", "seed"])["macro_f1"].mean().reset_index()
    rows = []
    for (ds, m), g in pw_mean.groupby(["dataset", "method"]):
        s = sm[(sm.dataset == ds) & (sm.method == m)]
        pl = pooled[(pooled.dataset == ds) & (pooled.method == m)]["pooled_macro_f1"]
        rows.append({
            "dataset": ds, "method": m,
            "macro_f1_mean": g["macro_f1"].mean(), "macro_f1_sd": g["macro_f1"].std(),
            "pooled_macro_f1_mean": pl.mean(),
            "adaptation_cpu_sec_mean": s["adaptation_cpu_sec"].mean(),
            "total_runtime_sec_mean": s["total_runtime_sec"].mean(),
            "retrain_events_mean": s["retrain_events"].mean(),
            "reuse_events_mean": s["reuse_events"].mean(),
            "refresh_events_mean": s["refresh_events"].mean(),
            "trees_trained_mean": s["trees_trained"].mean(),
        })
    return pd.DataFrame(rows)


def pareto(df):
    keep = []
    pts = df[["macro_f1_mean", "adaptation_cpu_sec_mean"]].values
    for i in range(len(pts)):
        dom = any((pts[j][0] >= pts[i][0] and pts[j][1] <= pts[i][1]
                   and (pts[j][0] > pts[i][0] or pts[j][1] < pts[i][1]))
                  for j in range(len(pts)) if j != i)
        keep.append(not dom)
    return keep


def vs_full(pw):
    pw_mean = pw.groupby(["dataset", "method", "seed"])["macro_f1"].mean().reset_index()
    rows = []
    for ds, g in pw_mean.groupby("dataset"):
        piv = g.pivot(index="seed", columns="method", values="macro_f1")
        fr = piv["Full Retraining"]
        for m in CHEAP + ["RAPT", "RAPT-Enhanced"]:
            if m not in piv:
                continue
            a, b = piv[m].values, fr.values
            d = a - b
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = np.nan
            rows.append({"dataset": ds, "method": m,
                         "delta_vs_full_retraining": float(d.mean()),
                         "wilcoxon_p": float(p), "n_seeds": len(d)})
    return pd.DataFrame(rows)


def main():
    pw, sm, pooled = load()
    t = table(pw, sm, pooled)
    t.to_csv(os.path.join(FINAL, "T3_cost_matched.csv"), index=False)
    rows = []
    for ds, g in t.groupby("dataset"):
        g = g.copy()
        g["pareto"] = pareto(g)
        rows.append(g)
    par = pd.concat(rows, ignore_index=True)
    par.to_csv(os.path.join(FINAL, "T3_pareto.csv"), index=False)
    vf = vs_full(pw)
    vf.to_csv(os.path.join(FINAL, "T3_vs_full.csv"), index=False)
    print("Step 3")
    for ds, g in par.groupby("dataset"):
        print(f"-- {SLUG_NAME[ds]}")
        for _, r in g.sort_values("macro_f1_mean", ascending=False).iterrows():
            print(f"   {r['method']:20s} {r['macro_f1_mean']:.4f} "
                  f"cpu={r['adaptation_cpu_sec_mean']:.3f} "
                  f"{'*' if r['pareto'] else ''}")
    print("\nvs Full Retraining:")
    print(vf.to_string(index=False))


if __name__ == "__main__":
    main()
