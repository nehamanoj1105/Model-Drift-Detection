"""T5b: 20-seed statistics for the cheap configs (read from T5_cheap_20seeds.csv).

The 5-seed primary comparisons sit on the exact-Wilcoxon floor (min p = 0.0625).
The cheap configs are inexpensive enough to run at 20 seeds, which raises the
power; this step reports exact Wilcoxon over the 20 seeds for RAPT-Cheap vs
FR-Cheap and vs Periodic-Cheap-5, with a Holm correction across the two
comparisons per stream.

Outputs:
  T5b_cheap_20seed_stats.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402


def main():
    d = pd.read_csv(os.path.join(lib.V3, "T5_cheap_20seeds.csv"))
    rows = []
    for ds, g in d.groupby("dataset"):
        piv = g.pivot_table(index="seed", columns="method", values="pooled_macro_f1")
        comps, ps, diffs, dzs = [], [], [], []
        for ref in ["FR-Cheap", "Periodic-Cheap-5"]:
            if "RAPT-Cheap" not in piv or ref not in piv:
                continue
            a, b = piv["RAPT-Cheap"].to_numpy(), piv[ref].to_numpy()
            p, md = lib.paired_wilcoxon(a, b)
            comps.append(f"RAPT-Cheap vs {ref}"); ps.append(p)
            diffs.append(md); dzs.append(lib.cohen_d(a, b))
        adj = lib.holm(ps)
        for c, p, md, dz, ap in zip(comps, ps, diffs, dzs, adj):
            rows.append({"dataset": ds, "comparison": c, "n_seeds": len(piv),
                         "mean_diff": md, "cohen_dz": dz,
                         "wilcoxon_p": p, "holm_p": ap,
                         "significant_holm_05": bool(ap < 0.05)})
    out = pd.DataFrame(rows)
    lib.write_csv(out, "T5b_cheap_20seed_stats.csv")
    print(out.round(4).to_string(index=False))
    lib.gate("T5b", "PASS", "20-seed cheap-config statistics")


if __name__ == "__main__":
    main()
