"""A10: detector wiring re-check (derived from the Step B runs).

The manuscript states "ADWIN, Page-Hinkley and EDMA recorded zero adaptation
events on all three streams and so reproduce the frozen baseline by
construction." A8 showed the ADWIN / Page-Hinkley zeros are a river 0.26.1
wiring artefact (`update()` returns None, so `bool(...)` is always False) and the
EDMA zero is a comparison-order bug (the new sample is absorbed into its own EWMA
before the test). The fixed wiring (DetectorAdaptiveModelV2 + FixedDetector) is
what Step B used, so the corrected detector results are already in
B_perwindow_all_streams.csv / B_pooled_all_streams.csv.

Outputs:
  A10_detector_corrected.csv   events/retrains/F1/CPU under fixed vs original wiring
  A10_frozen_reference.csv     Frozen per-window and pooled F1 per stream
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

DETS = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]


def original_events():
    p = os.path.join(lib.ROOT, "results", "revalidation", "A6_detector_events.csv")
    d = pd.read_csv(p)
    d = d[d.signal == "window"]
    return {(ds, det): int(g["events"].max())
            for (ds, det), g in d.groupby(["dataset", "detector"])}


def main():
    P = pd.read_csv(os.path.join(HERE, "B_perwindow_all_streams.csv"))
    B = pd.read_csv(os.path.join(HERE, "B_pooled_all_streams.csv"))
    orig = original_events()

    pw = P[P.method.isin(DETS)].groupby(["dataset", "method"]).agg(
        per_window_macro_f1=("per_window_macro_f1", "mean"),
        events=("retrain_events", "mean"),
        adapt_cpu=("adaptation_cpu_sec", "mean"),
        trees=("trees_trained", "mean")).reset_index()
    po = B[B.method.isin(DETS)].groupby(["dataset", "method"])["pooled_macro_f1"].mean()

    rows = []
    for _, r in pw.iterrows():
        rows.append({
            "dataset": r.dataset, "detector": r.method,
            "events_fixed": r.events,
            "events_original_wiring": orig.get((r.dataset, r.method), np.nan),
            "macro_f1_per_window": r.per_window_macro_f1,
            "macro_f1_pooled": float(po.loc[(r.dataset, r.method)]),
            "adapt_cpu": r.adapt_cpu, "trees_trained": r.trees,
        })
    d = pd.DataFrame(rows).sort_values(["dataset", "detector"])
    lib.write_csv(d, "A10_detector_corrected.csv")
    print(d.to_string(index=False))

    fr = P[P.method == "Frozen"].groupby("dataset").agg(
        frozen_per_window_f1=("per_window_macro_f1", "mean")).reset_index()
    fr["frozen_pooled_f1"] = fr.dataset.map(
        B[B.method == "Frozen"].groupby("dataset")["pooled_macro_f1"].mean())
    lib.write_csv(fr, "A10_frozen_reference.csv")
    print()
    print(fr.to_string(index=False))


if __name__ == "__main__":
    main()
