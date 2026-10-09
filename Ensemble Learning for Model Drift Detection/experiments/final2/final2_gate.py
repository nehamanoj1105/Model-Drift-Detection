"""Task A gate-accuracy aggregation.

Reads results/final2/raw/gate_logs/gate_<stream>_<seed>.csv (written by the
runner) and reports, per stream:
  * returns              : regime-return events the guard was consulted on
  * allows / blocks
  * bad_allows           : guard allowed, but the stored policy was materially
                           worse than the prefix policy on the incoming window
  * bad_allow_rate       : bad_allows / allows
  * missed_reuse (bad_block): guard blocked, but reuse would have helped

Writes results/final2/gate_accuracy.csv.
"""

import os
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW2 = os.path.join(ROOT, "results", "final2", "raw")
OUT = os.path.join(ROOT, "results", "final2", "gate_accuracy.csv")


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW2, "gate_logs", "gate_*.csv"))):
        df = pd.read_csv(f)
        rows.append(df)
    if not rows:
        print("no gate logs found")
        return
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(os.path.join(ROOT, "results", "final2", "gate_decisions_raw.csv"), index=False)
    # one row per event is a regime return; per-seed mean then overall
    per_seed = d.groupby(["dataset", "seed"]).agg(
        returns=("decision", "size"),
        allows=("decision", lambda s: int((s == "allow").sum())),
        blocks=("decision", lambda s: int((s == "block").sum())),
        bad_allows=("bad_allow", "sum"),
        bad_blocks=("bad_block", "sum")).reset_index()
    g = per_seed.groupby("dataset").agg(
        returns=("returns", "mean"),
        allows=("allows", "mean"),
        blocks=("blocks", "mean"),
        bad_allows=("bad_allows", "mean"),
        bad_blocks=("bad_blocks", "mean")).reset_index()
    g["bad_allow_rate"] = (g.bad_allows / g.allows.replace(0, np.nan)).round(3)
    g["missed_reuse"] = g.bad_blocks
    g.to_csv(OUT, index=False)
    print(g.to_string(index=False))
    print("\nOverall bad_allow_rate:", round(g.bad_allows.sum() / max(g.allows.sum(), 1), 3))
    print("Overall missed-reuse:", int(g.bad_blocks.sum()), "of blocks", int(g.blocks.sum()))


if __name__ == "__main__":
    main()
