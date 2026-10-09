"""Timing audit (Task 1): quantify per-window batched inference vs row-by-row.

The harness measures prediction CPU with time.process_time() around a single
vectorised m.predict(X_window) call. This script demonstrates what the *old*
(row-by-row) pattern would cost by predicting the identical fitted Frozen
ensemble one row at a time, and verifies n_jobs=1 for every estimator.

Writes results/final2/timing_audit.csv and prints old/new prediction CPU.

Usage: python experiments/final2/cost_timing_audit.py
"""

import os
import sys
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

from run_final import _prepare_arrays
from final2_runner import prepare, SLUGS
from models_9a import create_base_ensemble, make_class_anchor

R = os.path.join(ROOT, "results", "final2")


def main():
    rows = []
    for slug in SLUGS:
        stream, sd = prepare(SLUGS[slug])
        (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = \
            _prepare_arrays(stream, sd)
        ens = create_base_ensemble(seed=42)
        ens.fit(X_init, y_init)
        # n_jobs audit
        nj = [getattr(ens.rf, "n_jobs", None), getattr(ens.et, "n_jobs", None)]

        # batched (current): one predict per window
        t0 = time.process_time()
        for w in range(n_init, n_windows):
            ens.predict(X[win_slices[w]])
        batch = time.process_time() - t0

        # row-by-row (old pattern)
        t0 = time.process_time()
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            for i in idx:
                ens.predict(X[i:i + 1])
        rowwise = time.process_time() - t0
        rows.append(dict(stream=slug, name=SLUGS[slug], n_jobs=",".join(map(str, nj)),
                         n_windows=n_windows - n_init,
                         inference_calls_batched=n_windows - n_init,
                         inference_calls_rowwise=int(sum(len(win_slices[w]) for w in range(n_init, n_windows))),
                         pred_cpu_batched_sec=batch,
                         pred_cpu_rowwise_sec=rowwise,
                         inflation_x=rowwise / max(batch, 1e-9)))
        print(f"{slug}: batched {batch:.4f}s  rowwise {rowwise:.4f}s  "
              f"inflation {rowwise/max(batch,1e-9):.1f}x  n_jobs={nj}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "timing_audit.csv"), index=False)
    print("wrote", os.path.join(R, "timing_audit.csv"))


if __name__ == "__main__":
    main()
