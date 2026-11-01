"""T6: label-delay sensitivity.

Some streams carry a genuine one-window label delay: the QoS target for a
telemetry window is only known once the following window has been observed
(e.g. the next-window p90 latency). The existing harness supports a `label_delay`
mode that buffers the previous window's rows and appends them only after the
current window has been scored, so the update never sees a label that would not
have been available at that time.

This step runs delay 0 and delay 1 on 5G Campus, NordicDat and 5G NR for all
models. UGR'16 is not applicable: its label is the current-flow attack label and
the harness already buffers per flow, so there is no next-window target.

Outputs:
  T6_label_delay.csv       per dataset, delay, method, seed: pooled F1 + metrics
  T6_label_delay_delta.csv per dataset, method: pooled F1 at delay 1 minus delay 0
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced",
           "RAPT_v2"]
DATASETS = ["5G Campus QoS", "NordicDat", "5G NR"]


def main():
    lib.ensure_dirs()
    lib.freeze_config()
    rows = []
    for ds in DATASETS:
        stream, sd = lib.S.get_stream(ds)
        slug = lib.S.SLUG[ds]
        for delay in (False, True):
            for seed in lib.SEEDS:
                pw, sm, pooled = lib.run_stream(stream, sd, seed, METHODS,
                                                label_delay=delay)
                d = sm.merge(pooled[["method", "pooled_macro_f1"]], on="method")
                d["delay"] = int(delay)
                d.insert(0, "dataset", slug)
                rows.append(d)
            print(f"  {slug} delay={int(delay)} done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    lib.write_csv(df, "T6_label_delay.csv")

    drows = []
    for (ds, m), g in df.groupby(["dataset", "method"]):
        d0 = g[g.delay == 0].set_index("seed")["pooled_macro_f1"]
        d1 = g[g.delay == 1].set_index("seed")["pooled_macro_f1"]
        p, md = lib.paired_wilcoxon(d1.to_numpy(), d0.to_numpy())
        drows.append({"dataset": ds, "method": m,
                      "pooled_f1_delay0": d0.mean(), "pooled_f1_delay1": d1.mean(),
                      "delta_f1": md, "wilcoxon_p": p})
    dl = pd.DataFrame(drows)
    lib.write_csv(dl, "T6_label_delay_delta.csv")
    print(dl.round(4).to_string(index=False))
    lib.gate("T6", "PASS", "label-delay sensitivity (UGR'16 not applicable)")


if __name__ == "__main__":
    main()
