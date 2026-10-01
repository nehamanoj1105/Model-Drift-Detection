"""Step 4: does the headline (cost-aware) configuration transfer beyond Campus?

The paper's efficiency claim is made on 5G Campus only. This step asks whether
the same conclusion (RAPT-Cheap matches Full Retraining at lower adaptation CPU)
holds on UGR'16, NordicDat and 5G NR, or whether it is Campus-specific.

Sources: revalidation A3 (4 streams, seeds 42-46). Primary metric is the
per-window mean macro-F1, matching the paper.

Outputs
  T4_headline_other_streams.csv   RAPT-Cheap vs Full Retraining per stream
  T4_verdict.csv                  per-stream SUPPORTED / NOT SUPPORTED
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")

SLUG_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
             "nordicdat": "NordicDat", "5g_nr": "5G NR"}


def main():
    pw = pd.read_csv(os.path.join(REVAL, "raw", "A3_per_window.csv"))
    sm = pd.read_csv(os.path.join(REVAL, "A3_summary.csv"))
    pm = pw.groupby(["dataset", "method", "seed"])["macro_f1"].mean().reset_index()
    rows = []
    for ds in SLUG_NAME:
        piv = pm[pm.dataset == ds].pivot(index="seed", columns="method",
                                         values="macro_f1")
        a, b = piv["RAPT-Cheap"].values, piv["Full Retraining"].values
        d = a - b
        try:
            p = stats.wilcoxon(a, b).pvalue
        except ValueError:
            p = np.nan
        cpu_a = sm[(sm.dataset == ds) & (sm.method == "RAPT-Cheap")]["adaptation_cpu_sec"].mean()
        cpu_b = sm[(sm.dataset == ds) & (sm.method == "Full Retraining")]["adaptation_cpu_sec"].mean()
        rows.append({
            "dataset": ds,
            "rapt_cheap_macro_f1": float(a.mean()),
            "full_retraining_macro_f1": float(b.mean()),
            "delta_macro_f1": float(d.mean()),
            "wilcoxon_p": float(p),
            "rapt_cheap_adapt_cpu": float(cpu_a),
            "full_retraining_adapt_cpu": float(cpu_b),
            "cpu_saving_fraction": float(1 - cpu_a / cpu_b),
            "matches_within_0.01": bool(abs(d.mean()) < 0.01),
        })
    t = pd.DataFrame(rows)
    t["verdict"] = np.where(
        t["matches_within_0.01"],
        np.where(t["cpu_saving_fraction"] > 0, "SUPPORTED", "NOT SUPPORTED"),
        "NOT SUPPORTED")
    t.to_csv(os.path.join(FINAL, "T4_headline_other_streams.csv"), index=False)
    print("Step 4: RAPT-Cheap vs Full Retraining")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
