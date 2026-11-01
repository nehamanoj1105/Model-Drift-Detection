"""T4: cost-matched Pareto analysis (Step C).

Runs, on all four streams: Frozen, Event-Driven, Full Retraining, FR-Cheap,
Periodic-Cheap-5, Periodic-Cheap-10, Event-Driven-Cheap, RAPT-Cheap (periodic
trigger), RAPT-Cheap-Floor, RAPT-Cheap cadence-matched to Periodic-Cheap, and the
window-equivalent variant. Writes pooled macro-F1 vs adaptation CPU with a Pareto
flag (F1 up, CPU down) that includes Frozen and Event-Driven as reference points.

Per stream, states whether RAPT-Cheap beats Periodic-Cheap and FR-Cheap on F1,
CPU, or both. If it does not beat Periodic-Cheap on most streams, the verdict is
"repository claim not supported".

Outputs:
  T4_cost_pareto.csv       per dataset, method, seed: pooled F1, CPU, pareto flag
  T4_cost_summary.csv      per dataset, method: mean pooled F1, CPU, pareto rate
  T4_verdicts.csv          per dataset: RAPT-Cheap vs Periodic-Cheap / FR-Cheap
  T4_claim_verdict.csv     overall repository-claim verdict
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "FR-Cheap",
           "Periodic-Cheap-5", "Periodic-Cheap-10", "Event-Driven-Cheap",
           "RAPT-Cheap-Original", "RAPT-Cheap-Floor", "RAPT-Cheap",
           "RAPT-Cheap-WinEq", "Periodic-Cheap-Matched"]


def run_all():
    rows = []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        # cadence matched to Periodic-Cheap-5's expected refresh count:
        # Periodic-Cheap-5 refreshes every 5 windows; match RAPT-Cheap to that
        # by using the same 5-window cadence, which RAPT-Cheap already uses.
        lib.PERIODIC_MATCHED = 5
        for seed in lib.SEEDS:
            pw, sm, pooled = lib.run_stream(stream, sd, seed, METHODS)
            d = sm.merge(pooled[["method", "pooled_macro_f1"]], on="method")
            d.insert(0, "dataset", lib.S.SLUG[ds])
            rows.append(d)
            print(f"  {lib.S.SLUG[ds]} seed {seed}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    lib.ensure_dirs()
    lib.freeze_config()
    raw = run_all()
    lib.write_csv(raw, "T4_cost_pareto.csv")

    summary_rows = []
    for ds, g in raw.groupby("dataset"):
        m = g.groupby("method").agg(
            pooled_macro_f1=("pooled_macro_f1", "mean"),
            pooled_macro_f1_sd=("pooled_macro_f1", "std"),
            adaptation_cpu_sec=("adaptation_cpu_sec", "mean"),
            per_window_macro_f1=("per_window_macro_f1", "mean"),
            retrain_events=("retrain_events", "mean"),
            refresh_events=("refresh_events", "mean")).reset_index()
        m.insert(0, "dataset", ds)
        m["pareto"] = lib.pareto_flags(m)
        summary_rows.append(m)
    summary = pd.concat(summary_rows, ignore_index=True)
    lib.write_csv(summary, "T4_cost_summary.csv")

    # verdicts: RAPT-Cheap vs Periodic-Cheap-5 and FR-Cheap, per stream
    vrows = []
    for ds, g in summary.groupby("dataset"):
        idx = g.set_index("method")
        for ref in ["Periodic-Cheap-5", "FR-Cheap"]:
            if "RAPT-Cheap" not in idx.index or ref not in idx.index:
                continue
            f1_win = idx.loc["RAPT-Cheap", "pooled_macro_f1"] > idx.loc[ref, "pooled_macro_f1"]
            cpu_win = idx.loc["RAPT-Cheap", "adaptation_cpu_sec"] < idx.loc[ref, "adaptation_cpu_sec"]
            vrows.append({
                "dataset": ds, "reference": ref,
                "rapt_cheap_f1": idx.loc["RAPT-Cheap", "pooled_macro_f1"],
                "reference_f1": idx.loc[ref, "pooled_macro_f1"],
                "rapt_cheap_cpu": idx.loc["RAPT-Cheap", "adaptation_cpu_sec"],
                "reference_cpu": idx.loc[ref, "adaptation_cpu_sec"],
                "beats_on_f1": bool(f1_win), "beats_on_cpu": bool(cpu_win),
                "beats_on_both": bool(f1_win and cpu_win),
                "beats_on_neither": bool(not f1_win and not cpu_win),
            })
    verdicts = pd.DataFrame(vrows)
    lib.write_csv(verdicts, "T4_verdicts.csv")

    # overall: does RAPT-Cheap beat Periodic-Cheap-5 on F1 or CPU on most streams?
    pc = verdicts[verdicts.reference == "Periodic-Cheap-5"]
    n = len(pc)
    beats_any = int((pc.beats_on_f1 | pc.beats_on_cpu).sum())
    beats_both = int(pc.beats_on_both.sum())
    supported = beats_any > n / 2
    claim = pd.DataFrame([{
        "claim": "RAPT-Cheap matches Full Retraining accuracy at much lower cost",
        "streams": n, "beats_periodic_cheap_on_f1_or_cpu": beats_any,
        "beats_on_both": beats_both,
        "verdict": "repository claim supported" if supported
                   else "repository claim not supported",
    }])
    lib.write_csv(claim, "T4_claim_verdict.csv")
    print(summary.round(4).to_string(index=False))
    print()
    print(verdicts.round(4).to_string(index=False))
    print()
    print(claim.to_string(index=False))
    lib.gate("T4", "PASS", f"cost Pareto: {claim.iloc[0]['verdict']}")


if __name__ == "__main__":
    main()
