"""Step 5: statistics, in the form the paper already uses plus seed-level.

Three families are reported separately and never mixed:
  window  : per-window macro-F1 averaged over seeds, paired across windows
            (this reproduces the paper's n_windows=143/144 tests)
  seed    : per-seed pooled macro-F1, paired exact Wilcoxon over 5 seeds
            (min attainable two-sided p = 0.0625)
  blocks  : block bootstrap over day blocks (UGR'16) / regime visits

Outputs
  T5_statistics.csv          every comparison, all three families
  T5_reproduce_9a_stats.csv  window-level reproduction of statistics_9a.csv
  T5_holm_campus_ladder.csv  Holm-corrected family for the Campus ladder
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

SLUG_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
             "nordicdat": "NordicDat", "5g_nr": "5G NR"}
PAIRS = [("RAPT", "Frozen"), ("RAPT", "Event-Driven"), ("RAPT", "Full Retraining"),
         ("RAPT-Enhanced", "RAPT"), ("RAPT-Enhanced", "Full Retraining"),
         ("Event-Driven", "Full Retraining")]
LADDER = ["RAPT-Cheap", "RAPT-Floor", "RAPT-Full", "RAPT-Incremental"]
BOOT_N = 2000
BOOT_SEED = 0


def load():
    pw = pd.read_csv(os.path.join(REVAL, "raw", "A1_per_window_all_streams.csv"))
    pooled = pd.read_csv(os.path.join(REVAL, "A1_pooled_all_streams.csv"))
    a3pw = pd.read_csv(os.path.join(REVAL, "raw", "A3_per_window.csv"))
    a3pool = pd.read_csv(os.path.join(REVAL, "A3_cost_matched.csv"))
    boot = pd.read_csv(os.path.join(REVAL, "A7_block_bootstrap.csv"))
    return pw, pooled, a3pw, a3pool, boot


def window_test_df(df, ds, m1, m2, idcol="window_id"):
    """Per-window F1 averaged over seeds, paired across windows."""
    g = df[df.dataset == ds]
    w1 = g[g.method == m1].groupby(idcol)["macro_f1"].mean()
    w2 = g[g.method == m2].groupby(idcol)["macro_f1"].mean()
    idx = w1.index.intersection(w2.index)
    a, b = w1.reindex(idx).values, w2.reindex(idx).values
    d = a - b
    sd = d.std(ddof=1)
    cohen_d = float(d.mean() / sd) if sd > 0 else np.nan
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    rng = np.random.RandomState(BOOT_SEED)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(BOOT_N)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_windows": len(idx), "window_mean_diff": float(d.mean()),
            "window_cohen_d": cohen_d, "window_wilcoxon_p": float(p),
            "window_ci_low": float(lo), "window_ci_high": float(hi)}


def window_test(pw, ds, m1, m2):
    return window_test_df(pw, ds, m1, m2)


def seed_test_df(df, ds, m1, m2):
    def per_seed(m):
        return df[(df.dataset == ds) & (df.method == m)] \
            .groupby("seed")["macro_f1"].mean().sort_index()
    a, b = per_seed(m1), per_seed(m2)
    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx).values, b.reindex(idx).values
    if len(a) == 0:
        return {}
    d = a - b
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else np.nan
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    return {"n_seeds": len(d), "seed_mean_diff": float(d.mean()),
            "seed_cohen_dz": dz, "seed_wilcoxon_p": float(p)}


def seed_test(pooled, ds, m1, m2):
    a = pooled[(pooled.dataset == ds) & (pooled.method == m1)].sort_values("seed")["pooled_macro_f1"].values
    b = pooled[(pooled.dataset == ds) & (pooled.method == m2)].sort_values("seed")["pooled_macro_f1"].values
    if len(a) == 0 or len(b) == 0:
        return {}
    d = a - b
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else np.nan
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    return {"n_seeds": len(d), "seed_mean_diff": float(d.mean()),
            "seed_cohen_dz": dz, "seed_wilcoxon_p": float(p)}


def main():
    pw, pooled, a3pw, a3pool, boot = load()
    rows = []
    for ds in SLUG_NAME:
        for m1, m2 in PAIRS:
            r = {"dataset": ds, "method_1": m1, "method_2": m2}
            r.update(window_test(pw, ds, m1, m2))
            r.update(seed_test(pooled, ds, m1, m2))
            bb = boot[(boot.dataset == ds) & (boot.method_1 == m1) & (boot.method_2 == m2)]
            if len(bb):
                r["boot_ci_low"] = float(bb["ci95_low"].iloc[0])
                r["boot_ci_high"] = float(bb["ci95_high"].iloc[0])
            rows.append(r)
    # campus ladder vs Full Retraining: the ablation lives in Final_Experiments
    # (Campus only); its per-window F1 is already a window mean, so we compare
    # those windows directly and pair over seeds for the seed-level test.
    fe = pd.read_csv(os.path.join(ROOT, "Final_Experiments", "results", "raw",
                                  "per_window_full.csv"))
    fe["dataset"] = "5g_campus"
    fe_ren = {"RAPT_CHEAP": "RAPT-Cheap", "RAPT_FLOOR": "RAPT-Floor",
              "RAPT_FULL": "RAPT-Full", "RAPT_INCR": "RAPT-Incremental",
              "Full Retraining": "Full Retraining"}
    fe["method"] = fe["method"].replace(fe_ren)
    for m in LADDER:
        if m not in set(fe["method"]):
            continue
        r = {"dataset": "5g_campus", "method_1": m, "method_2": "Full Retraining"}
        r.update(window_test_df(fe, "5g_campus", m, "Full Retraining"))
        r.update(seed_test_df(fe, "5g_campus", m, "Full Retraining"))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(FINAL, "T5_statistics.csv"), index=False)

    # reproduce the paper's 9A statistics (window-level)
    s9 = pd.read_csv(os.path.join(ROOT, "experiments", "exp9a", "tables", "statistics_9a.csv"))
    rep = []
    for _, row in s9.iterrows():
        ds = {v: k for k, v in SLUG_NAME.items()}[row["dataset"]]
        mine = window_test(pw, ds, row["method_a"], row["method_b"])
        rep.append({"dataset": row["dataset"], "method_a": row["method_a"],
                    "method_b": row["method_b"],
                    "paper_mean_diff": row["mean_diff_f1"],
                    "mine_mean_diff": mine["window_mean_diff"],
                    "paper_p": row["p_value"], "mine_p": mine["window_wilcoxon_p"],
                    "paper_cohen_d": row["cohen_d"], "mine_cohen_d": mine["window_cohen_d"],
                    "paper_n": row["n_windows"], "mine_n": mine["n_windows"]})
    rep = pd.DataFrame(rep)
    rep["mean_diff_match"] = (rep["paper_mean_diff"] - rep["mine_mean_diff"]).abs() < 1e-3
    rep.to_csv(os.path.join(FINAL, "T5_reproduce_9a_stats.csv"), index=False)

    # Holm family: campus ladder vs Full Retraining
    lad = t[(t.dataset == "5g_campus") & (t.method_2 == "Full Retraining")
            & (t.method_1.isin(LADDER))].copy()
    lad = lad.sort_values("window_wilcoxon_p").reset_index(drop=True)
    m = len(lad)
    adj = []
    for i, p in enumerate(lad["window_wilcoxon_p"]):
        adj.append(min(1.0, p * (m - i)))
    lad["holm_adjusted_p"] = adj
    lad.to_csv(os.path.join(FINAL, "T5_holm_campus_ladder.csv"), index=False)

    print("Step 5")
    print(f"  T5_statistics.csv rows={len(t)}")
    print(f"  9A statistics reproduction: {int(rep['mean_diff_match'].sum())}/{len(rep)} mean diffs match")
    print(rep[["dataset", "method_a", "method_b", "paper_mean_diff", "mine_mean_diff",
               "paper_p", "mine_p", "paper_cohen_d", "mine_cohen_d"]].to_string(index=False))
    print("\nHolm campus ladder:")
    print(lad[["method_1", "window_mean_diff", "window_wilcoxon_p",
               "holm_adjusted_p", "seed_wilcoxon_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
