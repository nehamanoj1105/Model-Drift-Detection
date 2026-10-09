"""Assemble REVIEW3.md and paper_values_final2_v3.md from saved cost-job files.

Purely a formatter: every number is read from results/final2/ files. No number
is hand-typed.

Usage: python experiments/final2/cost_review.py
"""

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "results", "final2")

STREAM_NAME = {"campus": "5G Campus QoS", "ugr16": "UGR'16",
               "5g_nr": "5G NR Latency", "kpiops": "Network operator KPIs"}
ORDER = ["Frozen", "Full Retraining", "Full Retraining-cheap", "Event-Driven",
         "Event-Driven-cheap", "Event-Driven+Reuse", "Page-Hinkley", "RAPT",
         "RAPT-Guard"]


def md(df, floatfmt=".4f"):
    cols = df.columns.tolist()
    out = "| " + " | ".join(str(c) for c in cols) + " |\n"
    out += "|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:{floatfmt}}")
            elif isinstance(v, (bool, np.bool_)):
                cells.append("YES" if v else "no")
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def main():
    sm = pd.read_csv(os.path.join(R, "cost_summary_by_method.csv"))
    sm["method"] = pd.Categorical(sm["method"], ORDER, ordered=True)
    sm["stream_name"] = sm["dataset_slug"].map(STREAM_NAME)
    ta = pd.read_csv(os.path.join(R, "cost_table_a_cpu_change.csv"))
    tb = pd.read_csv(os.path.join(R, "cost_table_b_pareto.csv"))
    tc = pd.read_csv(os.path.join(R, "cost_table_c_stats.csv"))
    v = json.load(open(os.path.join(R, "cost_verdict.json")))
    sp = pd.read_csv(os.path.join(R, "cost_suspicious.csv"))
    audit = pd.read_csv(os.path.join(R, "timing_audit.csv"))
    choice = json.load(open(os.path.join(R, "cost_choice.json")))
    sweep = pd.read_csv(os.path.join(R, "dev_refit_sweep.csv"))

    L = []
    A = L.append
    A("# REVIEW3 — cost-reduction job (RAPT)\n")
    A("Every number below is read from `results/final2/` files; nothing is "
      "hand-typed. `paper/` was not edited. Frozen configs come from "
      "`configs/final.yaml` + `configs/final2.yaml`.\n")

    A("\n## 1. Timing audit\n")
    A("Prediction CPU is measured with `time.process_time()` wrapped around a "
      "single vectorised `m.predict(X_window)` call per window. No `predict` is "
      "called row-by-row in any method (only `detector models` consume the "
      "per-sample error stream, which is not inference). Every estimator is "
      "constructed with `n_jobs=1`, confirmed below.\n")
    A(md(audit[["stream", "n_jobs", "pred_cpu_batched_sec", "pred_cpu_rowwise_sec",
                "inflation_x", "inference_calls_batched", "inference_calls_rowwise"]]))
    infl = ", ".join(f"{r.inflation_x:.1f}x on {STREAM_NAME[r.stream]}"
                     for _, r in audit.iterrows())
    A(f"\nRow-by-row inference would inflate prediction CPU by: {infl}. "
      "The final run uses the batched path; **all methods were rerun** so "
      "numbers are comparable.\n")

    A("\n## 2. Cheap refit mode (dev sweep, 5 seeds, first 60%)\n")
    A("Criterion: macro-F1 per total CPU, mean over Event-Driven+Reuse and "
      "RAPT-Guard. Chosen mode: **%s** (%d trees, buffer %d).\n"
      % (choice["cheap_mode"], choice["cheap_n_trees"], choice["cheap_buffer"]))
    A(md(sweep[["mode", "n_trees", "buffer", "method", "macro_f1", "adapt_cpu",
                "total_cpu", "f1_per_total_cpu"]]))

    A("\n## 3. Summary (30 seeds, mean per seed; CPU seconds)\n")
    piv = sm.pivot_table(index="method", columns="stream_name", values="macro_f1").reindex(ORDER)
    A("\n### macro-F1 (+/- sd)\n")
    t = sm.pivot_table(index="method", columns="stream_name", values="macro_f1_sd").reindex(ORDER)
    comb = piv.copy()
    for c in comb.columns:
        comb[c] = [f"{piv.loc[i,c]:.4f} +/- {t.loc[i,c]:.4f}" for i in comb.index]
    A(md(comb.reset_index()))
    for col, label in [("adapt", "adaptation CPU"), ("pred", "prediction CPU"),
                       ("total", "total CPU")]:
        p = sm.pivot_table(index="method", columns="stream_name", values=col).reindex(ORDER)
        A(f"\n### {label}\n")
        A(md(p.reset_index()))
    for col, label in [("retrains", "retrains"), ("reuses", "reuses")]:
        p = sm.pivot_table(index="method", columns="stream_name", values=col).reindex(ORDER)
        A(f"\n### {label}\n")
        A(md(p.reset_index()))

    A("\n## 4. Table A — total CPU % change vs Full Retraining and vs Full Retraining-cheap\n")
    A(md(ta))

    A("\n## 5. Table B — Pareto frontier (macro-F1 vs total CPU)\n")
    A(md(tb))

    A("\n## 6. Table C — paired stats (Wilcoxon + Holm, one family; Cohen's dz; TOST +/-0.02)\n")
    A(md(tc))

    A("\n## 7. Verdict\n")
    def yn(x): return "**YES**" if x else "**NO**"
    A(f"1. Is Event-Driven+Reuse (cheap) on the Pareto frontier on at least 2 streams? "
      f"{yn(v['q1_on_frontier_2plus'])} ({sum(v['q1_per_stream'].values())}/4 streams: "
      f"{', '.join(k for k, x in v['q1_per_stream'].items() if x) or 'none'})\n")
    A(f"2. Is its total CPU within 30% of Frozen on at least 2 streams? "
      f"{yn(v['q2_cpu_within_30pct_2plus'])} ({sum(v['q2_per_stream'].values())}/4: "
      f"{', '.join(k for k, x in v['q2_per_stream'].items() if x) or 'none'})\n")
    A(f"3. Is its macro-F1 within 0.02 of Full Retraining (TOST) on at least 2 streams? "
      f"{yn(v['q3_f1_within_002_fullretrain_2plus'])} "
      f"({sum(v['q3_per_stream'].values())}/4: "
      f"{', '.join(k for k, x in v['q3_per_stream'].items() if x) or 'none'})\n")
    A(f"4. Does the cheap refit lose more than 0.02 macro-F1 vs the standard refit on any "
      f"stream? {yn(v['q4_cheap_loses_over_002_any'])}\n")
    for st, dd in v["q4_per_stream"].items():
        A(f"   - {STREAM_NAME[st]}: Event-Driven loss {dd['event_driven_loss']:.4f}, "
          f"Full Retraining loss {dd['full_retraining_loss']:.4f}\n")

    A("\n## 8. Suspicious / identical numbers\n")
    if len(sp):
        A(md(sp))
    else:
        A("None detected (no zero-variance macro-F1, no identical per-seed vectors, "
          "no zero-CPU-with-retrains).\n")
    A("\nFull list: `results/final2/cost_suspicious.csv`.\n")

    out = "\n".join(L)
    for fname in ("REVIEW3.md", "paper_values_final2_v3.md"):
        with open(os.path.join(R, fname), "w") as f:
            f.write(out)
    print("wrote REVIEW3.md and paper_values_final2_v3.md (%d lines)" % len(out.splitlines()))


if __name__ == "__main__":
    main()
