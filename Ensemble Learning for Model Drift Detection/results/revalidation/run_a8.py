"""A8: label-availability delay.

Campus, NordicDat and 5G NR derive the label of window k from window k+1, but
the runners append (X_k, y_k) to the training buffer immediately after
predicting window k. This script re-runs all methods with delayed labelling
(window k's label is only used after window k+1 is observed) and reports the
change in pooled macro-F1 and the RAPT-vs-baseline ordering.

5G NR is window-level so delay == no change (one row per window).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SEEDS, write_csv  # noqa: E402
from streams import get_stream, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
DATASETS = ["5G Campus QoS", "NordicDat", "5G NR"]


def main():
    rows = []
    for ds in DATASETS:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        print(f"### {ds}", flush=True)
        for seed in SEEDS:
            for delay in (False, True):
                pw, sm, pooled, _ = R.run_seed_v2(stream, sd, seed, METHODS,
                                                  label_delay=delay)
                pooled.insert(0, "label_delay", int(delay))
                pooled.insert(0, "dataset", slug)
                rows.append(pooled)
            print(f"   seed {seed} done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    write_csv(df, "A8_label_delay.csv")
    agg = df.groupby(["dataset", "label_delay", "method"]).agg(
        pooled_macro_f1=("pooled_macro_f1", "mean")).reset_index()
    print("\n=== pooled macro-F1: immediate vs delayed ===")
    for ds in df.dataset.unique():
        s = agg[agg.dataset == ds].pivot(index="method", columns="label_delay",
                                         values="pooled_macro_f1")
        s.columns = ["immediate", "delayed"]
        s["delta"] = s["delayed"] - s["immediate"]
        print(f"-- {ds}")
        print(s.round(4).to_string())


if __name__ == "__main__":
    main()
