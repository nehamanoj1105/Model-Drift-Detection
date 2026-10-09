"""Dev-only sweep of the cheap refit mode (Task 2).

Candidates: {20 trees / 300 buffer, 30 / 500, 50 / full}. Choose by macro-F1 per
total CPU on dev (5 seeds 42-46, first 60% of post-prefix windows), summed over
streams. Writes results/final2/dev_refit_sweep.csv and freezes the choice into
results/final2/cost_choice.json.

Usage: python experiments/final2/cost_dev_refit.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "experiments", "final"),
          os.path.join(ROOT, "experiments", "exp9a")):
    sys.path.insert(0, p)

from final_config import SEEDS_DEV, DEV_WINDOW_FRACTION
from final2_runner import prepare, SLUGS
import cost_runner as CR

R = os.path.join(ROOT, "results", "final2")
MODES = {"t20_b300": (20, 300), "t30_b500": (30, 500), "t50_full": (50, 1000)}
SWEEP_METHODS = ["Event-Driven+Reuse", "RAPT-Guard"]


def main():
    import yaml
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    base.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})
    from joblib import Parallel, delayed
    streams = list(SLUGS)

    rows = []
    for mode, (nt, buf) in MODES.items():
        cfg = dict(base); cfg["cheap_n_trees"] = nt; cfg["cheap_buffer"] = buf
        jobs = [(s, seed) for s in streams for seed in SEEDS_DEV]

        def one(slug, seed):
            stream, sd = prepare(SLUGS[slug])
            return CR.run_seed(stream, sd, seed, cfg, DEV_WINDOW_FRACTION,
                               methods=SWEEP_METHODS)

        t0 = time.time()
        res = Parallel(n_jobs=4, verbose=0)(delayed(one)(*j) for j in jobs)
        d = pd.concat(res, ignore_index=True)
        agg = d.groupby("method").agg(
            macro_f1=("macro_f1", "mean"),
            adapt=("adaptation_cpu_sec", "mean"),
            total=("total_cpu_sec", "mean"),
            retrains=("is_retrain", "mean"),
            reuses=("is_reuse", "mean")).reset_index()
        for _, r in agg.iterrows():
            rows.append(dict(mode=mode, n_trees=nt, buffer=buf, method=r["method"],
                             macro_f1=r["macro_f1"], adapt_cpu=r["adapt"],
                             total_cpu=r["total"], retrains=r["retrains"],
                             reuses=r["reuses"],
                             f1_per_total_cpu=r["macro_f1"] / max(r["total"], 1e-9)))
        print(f"{mode}: {time.time()-t0:.0f}s\n{agg.to_string()}\n")

    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(R, "dev_refit_sweep.csv"), index=False)
    # choose by macro-F1 per total CPU, averaged over the two methods
    score = sw.groupby("mode").agg(mean_f1_per_cpu=("f1_per_total_cpu", "mean"),
                                   mean_f1=("macro_f1", "mean")).reset_index()
    best = score.sort_values("mean_f1_per_cpu", ascending=False).iloc[0]
    choice = {"cheap_mode": best["mode"],
              "cheap_n_trees": int(MODES[best["mode"]][0]),
              "cheap_buffer": int(MODES[best["mode"]][1]),
              "criterion": "macro_F1 per total_CPU, mean over Event-Driven+Reuse and RAPT-Guard, dev 5 seeds"}
    json.dump(choice, open(os.path.join(R, "cost_choice.json"), "w"), indent=2)
    print("SWEEP:\n", score.to_string(index=False))
    print("CHOSEN:", choice)


if __name__ == "__main__":
    main()
