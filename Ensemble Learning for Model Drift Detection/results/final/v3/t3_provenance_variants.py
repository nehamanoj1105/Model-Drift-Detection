"""T3: provenance-fixed RAPT variants as the primary table.

The published RAPT stores a checkpoint under the *new* regime key although the
buffer it trains on still ends in the *previous* regime (see T2/A5 provenance).
RAPT_v2 / RAPT-Enhanced_v2 fix this by storing only after the new regime's first
window labels are known (deferred). RAPT_v2_pure additionally trains the stored
policy on the new regime's own window rows plus the class anchor.

This step reruns the main four-stream table, the Campus ladder and the headline
configs on the provenance-fixed variants, next to the original implementation
labelled "as published".

Outputs:
  T3_primary_four_stream.csv    pooled + per-window macro-F1, provenance variants
  T3_campus_ladder.csv          the Campus variant ladder
  T3_headline_configs.csv       headline cheap/periodic configs
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

BASE = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
PROV = ["RAPT_v2", "RAPT_v2_pure", "RAPT-Enhanced_v2", "RAPT-Enhanced_v2_pure"]
LADDER = ["RAPT", "RAPT_v2", "RAPT_v2_pure", "RAPT-Enhanced", "RAPT-Enhanced_v2",
          "RAPT-Enhanced_v2_pure"]
HEADLINE = ["RAPT-Cheap-Original", "RAPT-Cheap", "RAPT-Cheap-WinEq", "RAPT-Cheap-Floor"]


def run(methods, datasets=None):
    datasets = datasets or lib.DATASETS
    rows = []
    for ds in datasets:
        stream, sd = lib.S.get_stream(ds)
        for seed in lib.SEEDS:
            pw, sm, pooled = lib.run_stream(stream, sd, seed, methods)
            sm = sm.copy()
            sm.insert(0, "dataset", lib.S.SLUG[ds])
            rows.append(sm.merge(pooled[["method", "pooled_macro_f1"]], on="method"))
            print(f"  {lib.S.SLUG[ds]} seed {seed} done", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    lib.ensure_dirs()
    lib.freeze_config()

    four = run(BASE + PROV)
    lib.write_csv(four, "T3_primary_four_stream.csv")

    ladder = run(LADDER, datasets=["5G Campus QoS"])
    lib.write_csv(ladder, "T3_campus_ladder.csv")

    head = run(HEADLINE + ["Full Retraining"], datasets=["5G Campus QoS", "5G NR"])
    lib.write_csv(head, "T3_headline_configs.csv")

    agg = four.groupby(["dataset", "method"]).agg(
        pooled_macro_f1=("pooled_macro_f1", "mean"),
        per_window_macro_f1=("per_window_macro_f1", "mean"),
        adapt_cpu=("adaptation_cpu_sec", "mean"),
        retrains=("retrain_events", "mean"),
        reuses=("reuse_events", "mean")).round(4)
    print(agg.to_string())
    lib.gate("T3", "PASS", "provenance-fixed variants as primary")


if __name__ == "__main__":
    main()
