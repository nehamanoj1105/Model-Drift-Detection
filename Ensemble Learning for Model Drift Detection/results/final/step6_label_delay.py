"""Step 6: label-availability delay.

Tests whether the conclusions survive when the true label for a window is only
available after a delay (the network-telemetry reality). Delay is in windows.
The existing revalidation ran delay in {0, 1}; this step reports those and flags
that longer delays are untested rather than fabricating them.

Source: revalidation A8_label_delay.csv (pooled macro-F1, per its writer).

Outputs
  T6_label_delay.csv            F1 by dataset/delay/method, with delta vs delay 0
  T6_label_delay_verdict.csv    does the headline ranking change with delay?
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")

MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
SLUG_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
             "nordicdat": "NordicDat", "5g_nr": "5G NR"}


def main():
    a = pd.read_csv(os.path.join(REVAL, "A8_label_delay.csv"))
    delays = sorted(a["label_delay"].unique())
    agg = a.groupby(["dataset", "label_delay", "method"])["pooled_macro_f1"].mean().reset_index()
    base = agg[agg.label_delay == 0].set_index(["dataset", "method"])["pooled_macro_f1"]
    agg["delta_vs_delay0"] = [
        r.pooled_macro_f1 - base.get((r.dataset, r.method), np.nan)
        for _, r in agg.iterrows()]
    agg.to_csv(os.path.join(FINAL, "T6_label_delay.csv"), index=False)

    rows = []
    for ds in SLUG_NAME:
        g = agg[agg.dataset == ds]
        if g.empty:
            rows.append({"dataset": ds, "best_at_delay0": None,
                         "best_at_max_delay": None, "ranking_stable": None,
                         "max_abs_f1_shift": None,
                         "delays_tested": "not run for this stream"})
            continue
        delays_here = sorted(g["label_delay"].unique())
        lo, hi = int(delays_here[0]), int(delays_here[-1])
        best = {}
        for dly in (lo, hi):
            sub = g[g.label_delay == dly]
            sub = sub[sub.method.isin(MODELS)]
            best[dly] = sub.loc[sub["pooled_macro_f1"].idxmax(), "method"] \
                if len(sub) else None
        gaps = []
        for m in MODELS:
            v0 = g[(g.method == m) & (g.label_delay == lo)]["pooled_macro_f1"]
            v1 = g[(g.method == m) & (g.label_delay == hi)]["pooled_macro_f1"]
            if len(v0) and len(v1):
                gaps.append(abs(float(v0.iloc[0]) - float(v1.iloc[0])))
        rows.append({"dataset": ds, "best_at_delay0": best[lo],
                     "best_at_max_delay": best[hi],
                     "ranking_stable": bool(best[lo] == best[hi]),
                     "max_abs_f1_shift": max(gaps) if gaps else 0.0,
                     "delays_tested": str(sorted(set(int(d) for d in a["label_delay"])))})
    verdict = pd.DataFrame(rows)
    verdict.to_csv(os.path.join(FINAL, "T6_label_delay_verdict.csv"), index=False)
    print("Step 6")
    print(agg.to_string(index=False))
    print()
    print(verdict.to_string(index=False))
    if max(delays) < 2:
        print("\nNOTE: only delays 0 and 1 were run; longer delays are untested "
              "and are reported as a limitation, not extrapolated.")


if __name__ == "__main__":
    main()
