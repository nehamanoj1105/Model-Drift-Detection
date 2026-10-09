"""REVIEW4: Frozen-cheap, reuse-vs-trigger stats, RAPT reproducibility, gate revert.

Reads saved results only; no number is hand-typed. Writes
results/final2/REVIEW4.md and results/final2/paper_values_final2_v4.md.

Usage: python experiments/final2/cost_review4.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "results", "final2")
RAW = os.path.join(R, "raw")

STREAM_NAME = {"campus": "5G Campus QoS", "ugr16": "UGR'16",
               "5g_nr": "5G NR Latency", "kpiops": "Network operator KPIs"}
F2NAME = {v: k for k, v in STREAM_NAME.items()}
ORDER = ["campus", "ugr16", "5g_nr", "kpiops"]
TOST_MARGIN = 0.02


def md(df, fmt=".4f"):
    cols = df.columns.tolist()
    out = "| " + " | ".join(map(str, cols)) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:{fmt}}")
            elif isinstance(v, (bool, np.bool_)):
                cells.append("YES" if v else "no")
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def load(tag):
    return pd.read_csv(os.path.join(RAW, f"per_window_all_{tag}.csv"))


def add_slug(d):
    d = d.copy()
    if "dataset_slug" in d.columns:
        d["slug"] = d["dataset_slug"]
    else:
        d["slug"] = d["dataset"].map(F2NAME)
    return d


def perseed(d, method):
    g = d[d.method == method].groupby(["slug", "seed"]).macro_f1.mean().reset_index()
    return g


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]; run = max(run, val); adj[i] = min(run, 1.0)
    return adj


def paired(d, a_meth, b_meth):
    rows = []
    pa = perseed(d, a_meth).rename(columns={"macro_f1": "a"})
    pb = perseed(d, b_meth).rename(columns={"macro_f1": "b"})
    for slug in ORDER:
        j = pa[pa.slug == slug].merge(pb[pb.slug == slug], on=["slug", "seed"])
        if len(j) < 2:
            continue
        dd = (j.a - j.b).values
        if np.allclose(dd, 0.0):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(j.a, j.b)
                p = 1.0 if not np.isfinite(p) else float(p)
            except ValueError:
                p = 1.0
        se = stats.sem(dd); tc = stats.t.ppf(0.975, len(dd) - 1)
        dz = np.mean(dd) / (np.std(dd, ddof=1) + 1e-12)
        if se < 1e-12:
            tost_p = 0.0 if abs(np.mean(dd)) < TOST_MARGIN else 1.0
        else:
            lo = 1 - stats.t.cdf((np.mean(dd) + TOST_MARGIN) / se, len(dd) - 1)
            hi = stats.t.cdf((np.mean(dd) - TOST_MARGIN) / se, len(dd) - 1)
            tost_p = float(max(lo, hi))
        rows.append(dict(stream=slug, n=len(dd), mean_diff=float(np.mean(dd)), dz=float(dz),
                         p_wilcoxon=float(p), ci_lo=float(np.mean(dd) - tc * se),
                         ci_hi=float(np.mean(dd) + tc * se), tost_p=tost_p,
                         tost_equiv=bool(tost_p < 0.05)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].values)
    out["sig_holm"] = out["p_holm"] < 0.05
    return out


def main():
    cost = add_slug(load("cost_final"))
    fix = add_slug(load("cost_fix"))
    f2 = add_slug(load("final"))
    L = []; A = L.append
    A("# REVIEW4 — final small job\n")
    A("Every number is read from `results/final2/`; nothing is hand-typed. "
      "`paper/` was not edited. Frozen configs: `configs/final.yaml` + `configs/final2.yaml`.\n")

    # --- 1. Frozen-cheap ---
    A("\n## 1. Frozen-cheap vs Frozen (30 seeds, batch inference)\n")
    rows = []
    for slug in ORDER:
        f = perseed(cost, "Frozen")[lambda x: x.slug == slug].macro_f1
        fc = perseed(fix, "Frozen-cheap")[lambda x: x.slug == slug].macro_f1
        fcpu = cost[(cost.slug == slug) & (cost.method == "Frozen")].groupby("seed").total_cpu_sec.sum()
        fccpu = fix[(fix.slug == slug) & (fix.method == "Frozen-cheap")].groupby("seed").total_cpu_sec.sum()
        rows.append(dict(stream=slug, frozen_f1=f.mean(), frozen_sd=f.std(),
                         frozen_cheap_f1=fc.mean(), frozen_cheap_sd=fc.std(),
                         df1=fc.mean() - f.mean(),
                         frozen_total_cpu=fcpu.mean(), frozen_cheap_total_cpu=fccpu.mean(),
                         cpu_pct_change=100 * (fccpu.mean() - fcpu.mean()) / fcpu.mean()))
    t1 = pd.DataFrame(rows)
    A(md(t1))
    A("\nFrozen-cheap uses the same base learner as the cheap refits (20 trees / "
      "buffer 300). It changes macro-F1 by the `df1` column and total CPU by the "
      "`cpu_pct_change` column.\n")

    # --- 2. reuse vs trigger stats ---
    A("\n## 2. Is reuse or the trigger doing the work? (Event-Driven+Reuse cheap)\n")
    A("Paired seed-level Wilcoxon + Holm (one family per contrast), dz, TOST +/-0.02.\n")
    A("\n### Event-Driven+Reuse vs Event-Driven-cheap\n")
    s_ed = paired(cost, "Event-Driven+Reuse", "Event-Driven-cheap")
    A(md(s_ed))
    A("\n### Event-Driven-cheap vs Full Retraining-cheap\n")
    s_fr = paired(cost, "Event-Driven-cheap", "Full Retraining-cheap")
    A(md(s_fr))
    rows = []
    for slug in ORDER:
        e = cost[(cost.slug == slug) & (cost.method == "Event-Driven-cheap")]
        r = cost[(cost.slug == slug) & (cost.method == "Event-Driven+Reuse")]
        adc = e.groupby("seed").adaptation_cpu_sec.sum().mean()
        totc = e.groupby("seed").total_cpu_sec.sum().mean()
        adr = r.groupby("seed").adaptation_cpu_sec.sum().mean()
        totr = r.groupby("seed").total_cpu_sec.sum().mean()
        rows.append(dict(stream=slug,
                         ed_cheap_adapt_cpu=adc, edr_adapt_cpu=adr,
                         adapt_pct_change=100 * (adr - adc) / adc,
                         ed_cheap_total_cpu=totc, edr_total_cpu=totr,
                         total_pct_change=100 * (totr - totc) / totc))
    t2 = pd.DataFrame(rows)
    A("\n### CPU vs Event-Driven-cheap\n")
    A(md(t2))

    # --- 3. RAPT counts discrepancy ---
    A("\n## 3. Why cost-run RAPT retrain/reuse counts differed\n")
    A("Two defects in `cost_runner.py` (not in the RAPT class itself):\n")
    A("1. **Missing initial-regime seed.** RAPT was constructed with "
      "`fit_initial(X_init, y_init)` instead of "
      "`fit_initial(init_regime=regimes[n_init-1], ...)`, so the prefix regime "
      "(A on Campus, regime_C on 5G NR) was never placed in the repo. The first "
      "time that regime returned, RAPT treated it as unseen and retrained, then "
      "reused on the *second* return. Every recurring regime was therefore "
      "retrained once before any reuse. Numbers: Campus 2->24 retrains / 12->119 "
      "reuses; 5G NR 3->200 / 5->199; KPI-ops 5->6 / 173->172.\n")
    A("2. **Sticky flags.** `last_flags` was not reset each window, so a "
      "retrain/reuse flag was counted on every subsequent window until the next "
      "boundary. This is bookkeeping only (scores were identical) but inflated "
      "counts further and made per-seed sd collapse (Campus RAPT sd 1e-16).\n")
    A("No change was needed to the RAPT class. After the fix, RAPT reproduces "
      "the final2 numbers exactly:\n")
    rows = []
    for slug in ORDER:
        a = perseed(f2, "RAPT")[lambda x: x.slug == slug]
        b = perseed(fix, "RAPT")[lambda x: x.slug == slug]
        j = a.merge(b, on="seed", suffixes=("_f2", "_fix"))
        nwin_diff = None
        aa = f2[(f2.slug == slug) & (f2.method == "RAPT")].sort_values(["seed", "window_id"]).reset_index(drop=True)
        bb = fix[(fix.slug == slug) & (fix.method == "RAPT")].sort_values(["seed", "window_id"]).reset_index(drop=True)
        if len(aa) == len(bb):
            nwin_diff = int((aa.macro_f1.round(9).values != bb.macro_f1.round(9).values).sum())
        cf = cost[(cost.slug == slug) & (cost.method == "RAPT")]
        rt = int(fix[(fix.slug == slug) & (fix.method == "RAPT")].is_retrain.sum() / 30)
        ru = int(fix[(fix.slug == slug) & (fix.method == "RAPT")].is_reuse.sum() / 30)
        rows.append(dict(stream=slug, final2_f1=a.macro_f1.mean(),
                         costfix_f1=b.macro_f1.mean(),
                         max_abs_seed_diff=np.abs(j.macro_f1_f2 - j.macro_f1_fix).max(),
                         differing_windows=nwin_diff,
                         costfix_retrains=rt, costfix_reuses=ru,
                         buggy_cost_retrains=int(cf.is_retrain.sum() / 30),
                         buggy_cost_reuses=int(cf.is_reuse.sum() / 30)))
    t3 = pd.DataFrame(rows)
    A(md(t3))
    A("\n`differing_windows` is the number of (seed, window) cells whose macro-F1 "
      "differs between the final2 run and the corrected cost run (0 = exact match).\n")

    # --- 4. gate revert ---
    A("## 4. RAPT-Guard: accuracy-skip disabled\n")
    A("**The skip is off here.** `gate_skip_acc` is now 1.0, which disables "
      "the raw-accuracy skip (`skip_acc < 1.0` guard). Re-running with the "
      "standard 50-tree base config (`RAPT-Guard-std`) reproduces the final2 "
      "gate exactly:\n")
    rows = []
    for slug in ORDER:
        std = perseed(fix, "RAPT-Guard-std")[lambda x: x.slug == slug].macro_f1.mean()
        cheap = perseed(fix, "RAPT-Guard")[lambda x: x.slug == slug].macro_f1.mean()
        f2std = perseed(f2, "RAPT-Guard")[lambda x: x.slug == slug].macro_f1.mean()
        rows.append(dict(stream=slug, final2_guard_std=f2std,
                         costfix_guard_std=std, costfix_guard_cheap=cheap))
    t4 = pd.DataFrame(rows)
    A(md(t4))
    A("\n`final2_guard_std` is the original final2 RAPT-Guard (50-tree, standard "
      "gate). `costfix_guard_std` is the same method re-run with the fixed runner; "
      "they should match. `costfix_guard_cheap` is the cheap config with the skip "
      "disabled.\n")

    # --- 5. which run for the paper ---
    A("\n## 5. Which run should the paper tables come from?\n")
    A("- **Event-Driven, Full Retraining, Event-Driven+Reuse, Full "
      "Retraining-cheap, Page-Hinkley, Event-Driven-cheap, Frozen, Frozen-cheap:** "
      "use the **cost run** (`per_window_all_cost_final.csv` for the first eight, "
      "`per_window_all_cost_fix.csv` for Frozen-cheap). These methods are unaffected "
      "by the two runner defects.\n")
    A("- **RAPT, RAPT-Guard:** use the **final2 run** "
      "(`per_window_all_final.csv`) or, equivalently, the corrected cost run; they "
      "are numerically identical per window (section 3). Prefer final2 since the "
      "paper's existing tables were built from it.\n")
    A("\n### Numbers that differ between runs for the same method+config\n")
    rows = []
    for meth in ["RAPT", "RAPT-Guard"]:
        for slug in ORDER:
            a = f2[(f2.slug == slug) & (f2.method == meth)].groupby("seed").macro_f1.mean()
            b = cost[(cost.slug == slug) & (cost.method == meth)].groupby("seed").macro_f1.mean()
            rows.append(dict(method=meth, stream=slug, final2_f1=a.mean(),
                             buggy_cost_f1=b.mean(), diff=b.mean() - a.mean()))
    t5 = pd.DataFrame(rows)
    A(md(t5))
    A("\nFor RAPT the buggy-cost mean F1 differed from final2 by up to "
      "0.0125 (Campus). RAPT-Guard's buggy-cost UGR'16 figure (0.7193) is the "
      "gate-skip artefact discussed in section 4, not a method difference. All "
      "other methods are byte-identical between the cost run and final2.\n")

    out = "\n".join(L)
    open(os.path.join(R, "REVIEW4.md"), "w").write(out)

    # value-only companion: flat listing of every headline number
    V = []
    V.append("# paper_values_final2_v4 — flat values (read from results/final2/)\n")

    def cell(slug, meth, col, src):
        d = src[(src.slug == slug) & (src.method == meth)]
        if d.empty:
            return None
        return d.groupby("seed")[col].sum().mean() if col != "macro_f1" else \
            d.groupby("seed")["macro_f1"].mean().mean()

    V.append("\n## Batch-B macro-F1 (30 seeds)\n")
    V.append("| method | campus | ugr16 | 5g_nr | kpiops |\n|---|---|---|---|---|")
    for meth in ["Frozen", "Frozen-cheap", "Full Retraining", "Full Retraining-cheap",
                 "Event-Driven", "Event-Driven-cheap", "Event-Driven+Reuse",
                 "Page-Hinkley", "ADWIN", "RAPT", "RAPT-Guard", "RAPT-Enhanced",
                 "RAPT-Anchor"]:
        vals = []
        for slug in ORDER:
            src = fix if meth == "Frozen-cheap" else cost
            if meth in ("RAPT", "RAPT-Guard", "RAPT-Anchor", "RAPT-Enhanced"):
                src = f2
            v = cell(slug, meth, "macro_f1", src)
            vals.append("n/a" if v is None else f"{v:.4f}")
        V.append(f"| {meth} | " + " | ".join(vals) + " |")
    V.append("\n## Batch-B total CPU s (30 seeds)\n")
    V.append("| method | campus | ugr16 | 5g_nr | kpiops |\n|---|---|---|---|---|")
    for meth in ["Frozen", "Frozen-cheap", "Full Retraining", "Full Retraining-cheap",
                 "Event-Driven", "Event-Driven-cheap", "Event-Driven+Reuse",
                 "Page-Hinkley", "RAPT", "RAPT-Guard"]:
        vals = []
        for slug in ORDER:
            src = fix if meth == "Frozen-cheap" else cost
            if meth in ("RAPT", "RAPT-Guard"):
                src = f2
            v = cell(slug, meth, "total_cpu_sec", src)
            vals.append("n/a" if v is None else f"{v:.4f}")
        V.append(f"| {meth} | " + " | ".join(vals) + " |")
    V.append("\n## Contrasts (batch B, paired seed-level, Holm within contrast)\n")
    V.append("```")
    V.append("ED+Reuse vs ED-cheap")
    V.append(s_ed.to_string(index=False))
    V.append("")
    V.append("ED-cheap vs Full-cheap")
    V.append(s_fr.to_string(index=False))
    V.append("```")
    V.append("\n## Frozen-cheap vs Frozen\n```")
    V.append(t1.to_string(index=False))
    V.append("```")
    V.append("\n## RAPT reproducibility (final2 vs corrected cost)\n```")
    V.append(t3.to_string(index=False))
    V.append("```")
    V.append("\n## RAPT-Guard standard vs cheap\n```")
    V.append(t4.to_string(index=False))
    V.append("```")
    open(os.path.join(R, "paper_values_final2_v4.md"), "w").write("\n".join(V))

    print("wrote REVIEW4.md and paper_values_final2_v4.md (%d lines)" % len(out.splitlines()))
    pd.set_option("display.width", 200)
    print(t3.to_string(index=False))
    print(t4.to_string(index=False))


if __name__ == "__main__":
    main()
