"""Task B statistics for the final2 run.

Reads results/final2/raw/per_window_all_<tag>.csv, computes per-seed and
per-window summaries, then:
  * paired Wilcoxon (window-level) + Holm correction
  * Cohen's dz (seed-level)
  * TOST equivalence at +/-0.02 macro-F1
for RAPT-Guard and RAPT-Anchor vs Frozen and vs Event-Driven, plus a few extra
pairs. Writes results/final2/summary.json and results/final2/tables/*.csv.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS2 = os.path.join(ROOT, "results", "final2")
RAW2 = os.path.join(RESULTS2, "raw")
TABLES2 = os.path.join(RESULTS2, "tables")
os.makedirs(TABLES2, exist_ok=True)

TOST_MARGIN = 0.02
SLUGS = ["5g_campus", "ugr16", "5g_nr", "kpiops"]
METHOD_ORDER = ["Frozen", "Frozen+Periodic", "Event-Driven", "Event-Driven+Reuse",
                "ADWIN", "Page-Hinkley", "Full Retraining", "RAPT", "RAPT-Guard",
                "RAPT-ED-Guard"]
PAIRS = [("RAPT-Guard", "Frozen"), ("RAPT-Guard", "Event-Driven"),
         ("RAPT", "Frozen"), ("RAPT", "Event-Driven"),
         ("RAPT-ED-Guard", "Frozen"), ("RAPT-ED-Guard", "Event-Driven"),
         ("Event-Driven+Reuse", "Event-Driven"), ("Frozen+Periodic", "Frozen"),
         ("Full Retraining", "Frozen"), ("RAPT", "Full Retraining"),
         ("RAPT-Guard", "Full Retraining"), ("ADWIN", "Frozen"),
         ("Page-Hinkley", "Frozen")]


def _holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_window(df, m1, m2):
    a = df[df.method == m1][["seed", "window_id", "macro_f1"]]
    b = df[df.method == m2][["seed", "window_id", "macro_f1"]]
    m = a.merge(b, on=["seed", "window_id"], suffixes=("_1", "_2"))
    d = (m.macro_f1_1 - m.macro_f1_2).values
    if len(d) == 0 or np.all(d == 0):
        return dict(n=int(len(d)), mean_diff=0.0, p_value=1.0, cohen_d=0.0)
    _, p = stats.wilcoxon(d, zero_method="pratt")
    return dict(n=int(len(d)), mean_diff=float(np.mean(d)), p_value=float(p),
                cohen_d=float(np.mean(d) / (np.std(d) + 1e-12)))


def seed_stat(df, m1, m2):
    a = df[df.method == m1].groupby("seed").macro_f1.mean()
    b = df[df.method == m2].groupby("seed").macro_f1.mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (j.a - j.b).values; n = len(d)
    if n < 2:
        return dict(n=n, mean=np.nan, dz=np.nan, t_p=np.nan, tost_p=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan)
    dz = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
    _, tp = stats.ttest_rel(j.a, j.b)
    se = stats.sem(d); tc = stats.t.ppf(0.975, n - 1)
    if se < 1e-12:
        # zero variance: test is undefined; equivalent iff the constant
        # difference lies inside the margin
        tost_p = 0.0 if abs(np.mean(d)) < TOST_MARGIN else 1.0
    else:
        p_lo = 1 - stats.t.cdf((np.mean(d) + TOST_MARGIN) / se, n - 1)
        p_hi = stats.t.cdf((np.mean(d) - TOST_MARGIN) / se, n - 1)
        tost_p = float(max(p_lo, p_hi))
    return dict(n=n, mean=float(np.mean(d)), dz=dz, t_p=float(tp),
                tost_p=tost_p,
                ci_lo=float(np.mean(d) - tc * se),
                ci_hi=float(np.mean(d) + tc * se))


def summarize(df):
    rows = []
    for s in SLUGS:
        sub = df[df.dataset_slug == s]
        for m, g in sub.groupby("method"):
            ps = g.groupby("seed").agg(
                f1=("macro_f1", "mean"), acc=("accuracy", "mean"),
                bal=("balanced_accuracy", "mean"), prec=("precision", "mean"),
                rec=("recall", "mean"), adapt=("adaptation_cpu_sec", "sum"),
                pred=("prediction_cpu_sec", "sum"), total=("total_cpu_sec", "sum"),
                retr=("is_retrain", "sum"), reuse=("is_reuse", "sum"))
            rows.append(dict(
                stream=s, method=m,
                macro_f1=ps.f1.mean(), macro_f1_std=ps.f1.std(ddof=1),
                accuracy=ps.acc.mean(), balanced_accuracy=ps.bal.mean(),
                precision=ps.prec.mean(), recall=ps.rec.mean(),
                adaptation_cpu_sec=ps.adapt.mean(),
                prediction_cpu_sec=ps.pred.mean(),
                total_cpu_sec=ps.total.mean(),
                retrains=ps.retr.mean(), reuses=ps.reuse.mean(),
                n_seeds=int(ps.shape[0])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()
    allw = pd.read_csv(os.path.join(RAW2, f"per_window_all_{a.tag}.csv"))
    if "dataset_slug" not in allw.columns:
        slugmap = {"5G Campus QoS": "5g_campus", "UGR'16": "ugr16",
                   "5G NR Latency": "5g_nr", "Network operator KPIs": "kpiops"}
        allw["dataset_slug"] = allw["dataset"].map(slugmap)

    summ = summarize(allw)
    summ.to_csv(os.path.join(TABLES2, "summary_by_method.csv"), index=False)

    stat_rows = []
    for s in SLUGS:
        df = allw[allw.dataset_slug == s]
        present = set(df.method)
        tmp, pv = [], []
        for m1, m2 in PAIRS:
            if m1 not in present or m2 not in present:
                continue
            w = wilcoxon_window(df, m1, m2); sd = seed_stat(df, m1, m2)
            tmp.append(dict(stream=s, comparison=f"{m1} vs {m2}",
                            window_n=w["n"], window_mean_diff=w["mean_diff"],
                            window_cohen_d=w["cohen_d"], window_p=w["p_value"],
                            seed_n=sd["n"], seed_mean_diff=sd["mean"],
                            seed_dz=sd["dz"], seed_t_p=sd["t_p"], tost_p=sd["tost_p"],
                            ci95_low=sd["ci_lo"], ci95_high=sd["ci_hi"],
                            tost_equiv=bool(sd["tost_p"] < 0.05)))
            pv.append(w["p_value"])
        if tmp:
            for r, adj in zip(tmp, _holm(pv)):
                r["window_p_holm"] = float(adj)
            stat_rows.extend(tmp)
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(TABLES2, "statistics.csv"), index=False)

    out = dict(tost_margin=TOST_MARGIN, n_seeds=int(summ.n_seeds.max()),
               summary=summ.to_dict(orient="records"),
               statistics=stats_df.to_dict(orient="records"))
    with open(os.path.join(RESULTS2, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote summary.json + tables/")
    print(summ.pivot_table(index="method", columns="stream",
                           values="macro_f1").round(4).to_string())


if __name__ == "__main__":
    main()
