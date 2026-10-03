"""Detector signal-scale diagnostic.

The four drift detectors in the shared harness are driven by the per-window error
signal (one value per window), the same signal the Event-Driven baseline uses.
This script reports, from the committed config, how often each detector fires on
that window signal versus on its native input:

  * ADWIN / Page-Hinkley / ECDD-EWMA : the per-instance error value (0/1 bits)
  * EDDM                             : the per-instance binary error stream

The frozen ensemble's out-of-sample predictions are used as the error source (a
deterministic function of the seed), so the numbers are reproducible without
re-running the full prequential loop. Output: detector_diagnostics.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_loaders import build_stream  # noqa: E402
from runner import prepare  # noqa: E402
from models import create_ensemble  # noqa: E402
from detectors import make_detector, _detector_fired  # noqa: E402


def _fires(detector, values):
    n = 0
    for v in values:
        if _detector_fired(detector, v):
            n += 1
    return n


def run(cfg, dataset, seed=42):
    stream, sd = build_stream(dataset, cfg)
    X, y, regimes, _, _, slices, n_init, n_win = prepare(stream, sd)
    pre = np.arange(len(y)) < n_init * sd["window_rows"]
    model = create_ensemble(seed, cfg).fit(X[pre], y[pre])

    window_err, sample_bits = [], []
    for w in range(n_init, n_win):
        idx = slices[w]
        yp = model.predict(X[idx])
        window_err.append(1.0 - float(np.mean(yp == y[idx])))
        sample_bits.append((yp != y[idx]).astype(int))
    window_err = np.asarray(window_err)
    sample_bits = np.concatenate(sample_bits)

    rows = []
    for name in ["ADWIN", "Page-Hinkley", "EDDM", "ECDD-EWMA"]:
        w_fires = _fires(make_detector(name, cfg), window_err)
        s_fires = _fires(make_detector(name, cfg), sample_bits)
        rows.append({"dataset": dataset, "detector": name,
                     "window_fires": w_fires, "sample_fires": s_fires,
                     "window_err_mean": float(window_err.mean()),
                     "n_windows": len(window_err),
                     "n_samples": len(sample_bits)})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final.yaml")
    args = ap.parse_args()
    with open(os.path.join(HERE, args.config)) as f:
        cfg = yaml.safe_load(f)
    project = os.path.abspath(os.path.join(HERE, ".."))
    out_root = os.path.join(project, cfg["output_dir"], "final")

    rows = []
    for ds in cfg["datasets"]:
        rows.extend(run(cfg, ds["name"]))
        print(f"  detectors: {ds['name']} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "detector_diagnostics.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
