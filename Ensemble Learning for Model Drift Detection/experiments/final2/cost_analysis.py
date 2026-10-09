"""Cost-job analysis: summary, three tables, verdict block, suspicious checks.

Reads results/final2/raw/per_window_all_cost_final.csv (produced by cost_runner)
and writes:
  results/final2/cost_summary_by_method.csv
  results/final2/cost_table_a_cpu_change.csv
  results/final2/cost_table_b_pareto.csv
  results/final2/cost_table_c_stats.csv
  results/final2/cost_verdict.json
  results/final2/cost_suspicious.csv

Usage: python experiments/final2/cost_analysis.py --tag cost_final
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "results", "final2")
T = os.path.join(R, "tables")
os.makedirs(T, exist_ok=True)

TOST_MARGIN = 0.02
STREAM_ORDER = ["campus", "ugr16", "5g_nr", "kpiops"]
METHOD_ORDER = ["Frozen", "Full Retraining", "Full Retraining-cheap",
                "Event-Driven", "Event-Driven-cheap", "Event-Driven+Reuse",
                "Page-Hinkley", "RAPT", "RAPT-Guard"]
FOCUS = ["Event-Driven+Reuse", "RAPT-Guard"]
REF = ["Frozen", "Full Retraining", "Event-Driven", "Page-Hinkley"]


def load(tag):
    p = os.path.join(R, "raw", f"per_window_all_{tag}.csv")
    d = pd.read_csv(p)
    return d


def per_seed(d):
    g = d.groupby(["dataset_slug", "method", "seed"]).agg(
        macro_f1=("macro_f1", "mean"),
        adapt=("adaptation_cpu_sec", "sum"),
        pred=("prediction_cpu_sec", "sum"),
        total=("total_cpu_sec", "sum"),
        retrains=("is_retrain", "sum"),
        reuses=("is_reuse", "sum"),
        gate_skipped=("gate_skipped", "sum")).reset_index()
    return g


def summary_by_method(ps):
    g = ps.groupby(["dataset_slug", "method"]).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        adapt=("adapt", "mean"), pred=("pred", "mean"), total=("total", "mean"),
        total_sd=("total", "std"),
        retrains=("retrains", "mean"), reuses=("reuses", "mean"),
        gate_skipped=("gate_skipped", "mean"), n_seeds=("seed", "nunique")).reset_index()
    g["method"] = pd.Categorical(g["method"], METHOD_ORDER, ordered=True)
    g["dataset_slug"] = pd.Categorical(g["dataset_slug"], STREAM_ORDER, ordered=True)
    return g.sort_values(["method", "dataset_slug"]).reset_index(drop=True)


def table_a(sm):
    rows = []
    for st in STREAM_ORDER:
        s = sm[sm.dataset_slug == st].set_index("method")
        for meth in ["Event-Driven+Reuse", "RAPT-Guard", "Full Retraining-cheap",
                     "Event-Driven-cheap", "RAPT"]:
            if meth not in s.index:
                continue
            tot = s.loc[meth, "total"]; f1 = s.loc[meth, "macro_f1"]
            fr = s.loc["Full Retraining", "total"]
            frc = s.loc["Full Retraining-cheap", "total"]
            rows.append(dict(stream=st, method=meth,
                             total_cpu=tot, macro_f1=f1,
                             pct_vs_full_retraining=100 * (tot - fr) / fr,
                             pct_vs_full_retraining_cheap=100 * (tot - frc) / frc))
    return pd.DataFrame(rows)


def pareto(sm):
    rows = []
    for st in STREAM_ORDER:
        s = sm[sm.dataset_slug == st][["method", "macro_f1", "total"]].copy()
        # Pareto: no other method has >= f1 and <= cpu (strictly better in one)
        front = []
        for _, a in s.iterrows():
            dominated = any((b.macro_f1 >= a.macro_f1 and b.total <= a.total) and
                            (b.macro_f1 > a.macro_f1 or b.total < a.total)
                            for _, b in s.iterrows())
            front.append(not dominated)
        s["on_frontier"] = front
        s["cpu_rank"] = s["total"].rank()
        s["f1_rank"] = s["macro_f1"].rank(ascending=False)
        for _, r in s.iterrows():
            rows.append(dict(stream=st, method=r["method"], macro_f1=r.macro_f1,
                             total_cpu=r["total"], on_frontier=bool(r["on_frontier"]),
                             cpu_rank=int(r["cpu_rank"]), f1_rank=int(r["f1_rank"])))
    # mean over streams with ranks
    m = sm.groupby("method").agg(macro_f1=("macro_f1", "mean"),
                                 total=("total", "mean")).reset_index()
    m["cpu_rank"] = m["total"].rank()
    m["f1_rank"] = m["macro_f1"].rank(ascending=False)
    front = []
    for _, a in m.iterrows():
        dominated = any((b.macro_f1 >= a.macro_f1 and b.total <= a.total) and
                        (b.macro_f1 > a.macro_f1 or b.total < a.total)
                        for _, b in m.iterrows())
        front.append(not dominated)
    m["on_frontier"] = front
    for _, r in m.iterrows():
        rows.append(dict(stream="MEAN", method=r["method"], macro_f1=r.macro_f1,
                         total_cpu=r["total"], on_frontier=bool(r["on_frontier"]),
                         cpu_rank=int(r["cpu_rank"]), f1_rank=int(r["f1_rank"])))
    return pd.DataFrame(rows)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order returned)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def paired_stats(ps):
    rows = []
    for st in STREAM_ORDER:
        s = ps[ps.dataset_slug == st]
        for focus in FOCUS:
            a = s[s.method == focus].set_index("seed")["macro_f1"]
            for ref in REF:
                b = s[s.method == ref].set_index("seed")["macro_f1"]
                j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
                if len(j) < 2:
                    continue
                d = (j.a - j.b).values
                try:
                    _, p = stats.wilcoxon(j.a, j.b)
                except ValueError:
                    p = 1.0
                sd = np.std(d, ddof=1)
                dz = np.mean(d) / (sd + 1e-12)
                se = stats.sem(d); tc = stats.t.ppf(0.975, len(d) - 1)
                if se < 1e-12:
                    tost_p = 0.0 if abs(np.mean(d)) < TOST_MARGIN else 1.0
                else:
                    lo = 1 - stats.t.cdf((np.mean(d) + TOST_MARGIN) / se, len(d) - 1)
                    hi = stats.t.cdf((np.mean(d) - TOST_MARGIN) / se, len(d) - 1)
                    tost_p = float(max(lo, hi))
                rows.append(dict(stream=st, focus=focus, reference=ref, n=len(d),
                                 mean_diff=float(np.mean(d)), dz=float(dz),
                                 p_wilcoxon=float(p),
                                 ci_lo=float(np.mean(d) - tc * se),
                                 ci_hi=float(np.mean(d) + tc * se),
                                 tost_p=tost_p,
                                 tost_equiv=bool(tost_p < 0.05)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].values)     # one family
    out["sig_holm"] = out["p_holm"] < 0.05
    return out


def verdict(sm, tab_a, tab_b, stats_c):
    def get(st, meth, col):
        r = sm[(sm.dataset_slug == st) & (sm.method == meth)]
        return float(r.iloc[0][col]) if len(r) else np.nan

    # Q1: on frontier on >=2 streams?
    q1 = {}
    for st in STREAM_ORDER:
        r = tab_b[(tab_b.stream == st) & (tab_b.method == "Event-Driven+Reuse")]
        q1[st] = bool(r.iloc[0].on_frontier) if len(r) else False
    q1_yes = sum(q1.values()) >= 2

    # Q2: total CPU within 30% of Frozen on >=2 streams?
    q2 = {}
    for st in STREAM_ORDER:
        edr = get(st, "Event-Driven+Reuse", "total")
        fr = get(st, "Frozen", "total")
        q2[st] = bool(edr <= 1.30 * fr)
    q2_yes = sum(q2.values()) >= 2

    # Q3: macro-F1 within 0.02 of Full Retraining (TOST) on >=2 streams?
    q3 = {}
    for st in STREAM_ORDER:
        r = stats_c[(stats_c.stream == st) & (stats_c.focus == "Event-Driven+Reuse")
                    & (stats_c.reference == "Full Retraining")]
        q3[st] = bool(r.iloc[0].tost_equiv) if len(r) else False
    q3_yes = sum(q3.values()) >= 2

    # Q4: cheap refit lose >0.02 F1 vs standard refit on any stream?
    q4 = {}
    for st in STREAM_ORDER:
        edc = get(st, "Event-Driven-cheap", "macro_f1")
        ed = get(st, "Event-Driven", "macro_f1")
        frc = get(st, "Full Retraining-cheap", "macro_f1")
        fr = get(st, "Full Retraining", "macro_f1")
        loss_ed = ed - edc
        loss_fr = fr - frc
        q4[st] = dict(event_driven_loss=float(loss_ed), full_retraining_loss=float(loss_fr),
                      loses_more_than_002=bool(max(loss_ed, loss_fr) > 0.02))
    q4_any = any(v["loses_more_than_002"] for v in q4.values())

    return dict(
        q1_on_frontier_2plus=q1_yes, q1_per_stream=q1,
        q2_cpu_within_30pct_2plus=q2_yes, q2_per_stream=q2,
        q3_f1_within_002_fullretrain_2plus=q3_yes, q3_per_stream=q3,
        q4_cheap_loses_over_002_any=q4_any, q4_per_stream=q4)


def suspicious(d, ps):
    rows = []
    # zero / near-zero variance macro-F1 across seeds
    for (st, meth), g in ps.groupby(["dataset_slug", "method"]):
        sd = float(g.macro_f1.std())
        if sd < 1e-9:
            rows.append(dict(stream=st, method=meth, kind="zero_variance_macro_f1",
                             detail=f"sd={sd:.2e}, mean={g.macro_f1.mean():.4f}"))
    # identical per-seed F1 vectors across methods (within a stream)
    for st, g in ps.groupby("dataset_slug"):
        vecs = {}
        for meth, gg in g.groupby("method"):
            vecs[meth] = tuple(np.round(np.sort(gg.macro_f1.values), 6))
        seen = {}
        for meth, v in vecs.items():
            seen.setdefault(v, []).append(meth)
        for v, meths in seen.items():
            if len(meths) > 1:
                rows.append(dict(stream=st, method="+".join(sorted(meths)),
                                 kind="identical_macro_f1_vectors",
                                 detail="sorted per-seed F1 identical"))
    # methods that never reused on a stream -> by construction equal to a
    # trigger-only counterpart (report, since a reader may misread it as reuse)
    for (st, meth), g in d.groupby(["dataset_slug", "method"]):
        if "Reuse" in meth or meth in ("RAPT", "RAPT-Guard"):
            if g.is_reuse.sum() == 0 and g.is_retrain.sum() > 0:
                rows.append(dict(stream=st, method=meth,
                                 kind="never_reused",
                                 detail="0 reuse events in 30 seeds; equals a trigger-only run"))
    # zero cpu with retrains (timing resolution)
    for (st, meth), g in ps.groupby(["dataset_slug", "method"]):
        if g.total.mean() < 1e-9 and g.retrains.sum() > 0:
            rows.append(dict(stream=st, method=meth, kind="zero_cpu_with_retrains",
                             detail=f"total~0, retrains={g.retrains.sum()}"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cost_final")
    a = ap.parse_args()
    d = load(a.tag)
    ps = per_seed(d)
    sm = summary_by_method(ps)
    sm.to_csv(os.path.join(R, "cost_summary_by_method.csv"), index=False)
    sm.to_csv(os.path.join(T, "cost_summary_by_method.csv"), index=False)
    ta = table_a(sm); ta.to_csv(os.path.join(R, "cost_table_a_cpu_change.csv"), index=False)
    tb = pareto(sm); tb.to_csv(os.path.join(R, "cost_table_b_pareto.csv"), index=False)
    tc = paired_stats(ps); tc.to_csv(os.path.join(R, "cost_table_c_stats.csv"), index=False)
    v = verdict(sm, ta, tb, tc)
    json.dump(v, open(os.path.join(R, "cost_verdict.json"), "w"), indent=2)
    sp = suspicious(d, ps)
    sp.to_csv(os.path.join(R, "cost_suspicious.csv"), index=False)

    pd.set_option("display.width", 220)
    print("=== summary ==="); print(sm.round(4).to_string(index=False))
    print("\n=== table A (CPU % change) ==="); print(ta.round(3).to_string(index=False))
    print("\n=== table B (Pareto) ==="); print(tb.round(4).to_string(index=False))
    print("\n=== table C (stats) ==="); print(tc.round(4).to_string(index=False))
    print("\n=== verdict ==="); print(json.dumps(v, indent=2))
    print("\n=== suspicious ==="); print(sp.to_string(index=False))


if __name__ == "__main__":
    main()
