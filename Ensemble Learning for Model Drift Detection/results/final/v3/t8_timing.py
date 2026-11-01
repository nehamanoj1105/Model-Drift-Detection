"""T8: timing precision.

Repeat the headline configurations 5 times each on 5G Campus and 5G NR, and
report the median and range of initial-training CPU, adaptation CPU and
prediction CPU per method, plus the adaptation-CPU saving of RAPT (and the cheap
configs) relative to Full Retraining computed from medians (not from a single
run).

Outputs:
  T8_timing_repeats.csv    per dataset, method, repeat: init/adapt/pred CPU
  T8_timing_summary.csv    per dataset, method: median + range + saving vs FR
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced",
           "RAPT_v2", "FR-Cheap", "Periodic-Cheap-5", "RAPT-Cheap"]
DATASETS = ["5G Campus QoS", "5G NR"]
REPEATS = 5


def main():
    lib.ensure_dirs()
    lib.freeze_config()
    rows = []
    for ds in DATASETS:
        stream, sd = lib.S.get_stream(ds)
        slug = lib.S.SLUG[ds]
        for seed in lib.SEEDS:
            for rep in range(REPEATS):
                pw, sm, pooled = lib.run_stream(stream, sd, seed, METHODS)
                sm = sm.copy()
                sm["repeat"] = rep
                sm.insert(0, "dataset", slug)
                rows.append(sm)
            print(f"  {slug} seed {seed} x{REPEATS} done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    lib.write_csv(df, "T8_timing_repeats.csv")

    srows = []
    for (ds, m), g in df.groupby(["dataset", "method"]):
        for col, lab in [("init_cpu_sec", "init"), ("adaptation_cpu_sec", "adapt"),
                         ("prediction_cpu_sec", "pred")]:
            srows.append({"dataset": ds, "method": m, "component": lab,
                          "median": g[col].median(), "min": g[col].min(),
                          "max": g[col].max(), "n": len(g)})
    s = pd.DataFrame(srows)
    lib.write_csv(s, "T8_timing_summary.csv")

    # saving vs Full Retraining from medians
    vrows = []
    for ds, g in df.groupby("dataset"):
        med = g.groupby("method")["adaptation_cpu_sec"].median()
        fr = med.get("Full Retraining", np.nan)
        for m in METHODS:
            if m in med and fr == fr:
                vrows.append({"dataset": ds, "method": m,
                              "adapt_cpu_median": med[m],
                              "fr_adapt_cpu_median": fr,
                              "saving_vs_fr_pct": 100.0 * (1 - med[m] / fr)})
    lib.write_csv(pd.DataFrame(vrows), "T8_timing_saving.csv")
    print(pd.DataFrame(vrows).round(3).to_string(index=False))
    lib.gate("T8", "PASS", "timing medians + range over 5 repeats")


if __name__ == "__main__":
    main()
