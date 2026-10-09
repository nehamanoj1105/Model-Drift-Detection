"""
Aggregation and statistics for the final results.

Reads results/final/raw/per_window_<slug>_final.csv and produces:
  * results/final/summary.json
  * results/final/tables/<...>.csv
  * statistics: paired Wilcoxon (window-level) with Holm correction, Cohen's dz
    (seed-level and window-level), and TOST equivalence at +/-0.02 macro-F1.

Statistical protocol notes (stated in the paper):
  * The primary effect size is Cohen's dz on the per-seed macro-F1 difference
    (n = 30 seeds, matched design). Window-level Wilcoxon pools windows and is
    reported as the window-level test.
  * Holm correction is applied across the family of window-level comparisons.
  * TOST uses a two one-sided test on the per-seed mean difference with a
    +/-0.02 macro-F1 margin.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from final_config import RAW_FINAL, RESULTS_FINAL, TABLES_FINAL, ensure_dirs

SLUGS = ["5g_campus", "ugr16", "nordicdat", "5g_nr"]
METHOD_ORDER = ["Frozen", "Frozen+Periodic", "Event-Driven", "Event-Driven+Reuse",
                "Full Retraining", "RAPT", "RAPT-Enhanced", "RAPT-Guard",
                "RAPT-Anchor", "ADWIN", "Page-Hinkley", "EDD", "EDMA"]
TOST_MARGIN = 0.02


def load(tag="final"):
    frames = {}
    for s in SLUGS:
        p = os.path.join(RAW_FINAL, f"per_window_{s}_{tag}.csv")
        if os.path.exists(p):
            frames[s] = pd.read_csv(p)
    return frames


def per_seed_macro_f1(df):
    return df.groupby(["method", "seed"])["macro_f1"].mean().reset_index()


def summarize(frames):
    rows = []
    for s, df in frames.items():
        for m, g in df.groupby("method"):
            ps = g.groupby("seed").agg(
                f1=("macro_f1", "mean"), acc=("accuracy", "mean"),
                bal=("balanced_accuracy", "mean"), prec=("precision", "mean"),
                rec=("recall", "mean"), adapt=("adaptation_cpu_sec", "sum"),
                pred=("prediction_cpu_sec", "sum"), total=("total_cpu_sec", "sum"),
                retr=("is_retrain", "sum"), reuse=("is_reuse", "sum"))
            rows.append(dict(
                stream=s, method=m,
                macro_f1=ps["f1"].mean(), macro_f1_std=ps["f1"].std(ddof=1),
                accuracy=ps["acc"].mean(), balanced_accuracy=ps["bal"].mean(),
                precision=ps["prec"].mean(), recall=ps["rec"].mean(),
                adaptation_cpu_sec=ps["adapt"].mean(),
                prediction_cpu_sec=ps["pred"].mean(),
                total_cpu_sec=ps["total"].mean(),
                retrains=ps["retr"].mean(), reuses=ps["reuse"].mean(),
                n_seeds=ps.shape[0],
                gate_allow=(g["gate_allow"].sum() if "gate_allow" in g else np.nan),
                gate_block=(g["gate_block"].sum() if "gate_block" in g else np.nan)))
    return pd.DataFrame(rows)


def _holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_window(df, m1, m2):
    """Window-level paired Wilcoxon on per-window correctness is not available
    (we store macro-F1 per window); we instead pair per-window macro-F1 by
    (seed, window_id), which is the window-level comparison the paper reports."""
    a = df[df["method"] == m1][["seed", "window_id", "macro_f1"]]
    b = df[df["method"] == m2][["seed", "window_id", "macro_f1"]]
    m = a.merge(b, on=["seed", "window_id"], suffixes=("_1", "_2"))
    d = (m["macro_f1_1"] - m["macro_f1_2"]).values
    if len(d) == 0 or np.all(d == 0):
        return dict(n=len(d), mean_diff=0.0, p_value=1.0, cohen_d=0.0)
    stat, p = stats.wilcoxon(d, zero_method="pratt")
    return dict(n=int(len(d)), mean_diff=float(np.mean(d)),
                p_value=float(p), cohen_d=float(np.mean(d) / (np.std(d) + 1e-12)))


def seed_stat(df, m1, m2):
    """Seed-level paired statistics: Cohen's dz and TOST."""
    a = df[df["method"] == m1].groupby("seed")["macro_f1"].mean()
    b = df[df["method"] == m2].groupby("seed")["macro_f1"].mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (j["a"] - j["b"]).values
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float(np.mean(d)) if n else np.nan, dz=np.nan,
                    t_p=np.nan, tost_p=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    dz = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
    tstat, tp = stats.ttest_rel(j["a"], j["b"])
    se = stats.sem(d)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (float(np.mean(d) - tcrit * se), float(np.mean(d) + tcrit * se))
    # TOST: H0_lower: d <= -margin, H0_upper: d >= margin. Equivalence is
    # declared when the larger one-sided p is below alpha.
    p_lower = 1 - stats.t.cdf((np.mean(d) + TOST_MARGIN) / se, n - 1)
    p_upper = stats.t.cdf((np.mean(d) - TOST_MARGIN) / se, n - 1)
    tost_p = float(max(p_lower, p_upper))
    return dict(n=n, mean=float(np.mean(d)), dz=dz, t_p=float(tp),
                tost_p=tost_p, ci_lo=ci[0], ci_hi=ci[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()
    ensure_dirs()
    frames = load(a.tag)
    summ = summarize(frames)
    summ.to_csv(os.path.join(TABLES_FINAL, "summary_by_method.csv"), index=False)

    pairs = [("RAPT-Guard", "Frozen"), ("RAPT-Guard", "Event-Driven"),
             ("RAPT-Anchor", "Frozen"), ("RAPT-Anchor", "Event-Driven"),
             ("RAPT", "Frozen"), ("RAPT", "Event-Driven"),
             ("RAPT-Enhanced", "Frozen"), ("RAPT-Enhanced", "Event-Driven"),
             ("Frozen+Periodic", "Frozen"), ("Event-Driven+Reuse", "Event-Driven")]
    stat_rows = []
    for s, df in frames.items():
        pvals = []
        tmp = []
        for m1, m2 in pairs:
            if m1 not in set(df["method"]) or m2 not in set(df["method"]):
                continue
            w = wilcoxon_window(df, m1, m2)
            sd = seed_stat(df, m1, m2)
            tmp.append(dict(stream=s, comparison=f"{m1} vs {m2}",
                            window_n=w["n"], window_mean_diff=w["mean_diff"],
                            window_cohen_d=w["cohen_d"], window_p=w["p_value"],
                            seed_n=sd["n"], seed_mean_diff=sd["mean"],
                            seed_dz=sd["dz"], seed_t_p=sd["t_p"],
                            tost_p=sd["tost_p"], ci95_low=sd["ci_lo"],
                            ci95_high=sd["ci_hi"]))
            pvals.append(w["p_value"])
        if tmp:
            adj = _holm(pvals)
            for r, ap_ in zip(tmp, adj):
                r["window_p_holm"] = float(ap_)
            stat_rows.extend(tmp)
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(os.path.join(TABLES_FINAL, "statistics.csv"), index=False)

    # hochberg/holm family also for seed-level t-tests across all comparisons
    out = dict(
        tost_margin=TOST_MARGIN,
        n_seeds=int(summ["n_seeds"].max()),
        summary=summ.to_dict(orient="records"),
    )
    with open(os.path.join(RESULTS_FINAL, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote summary.json and tables/ from", a.tag)
    print(summ.pivot_table(index="method", columns="stream", values="macro_f1").round(4).to_string())


if __name__ == "__main__":
    main()
