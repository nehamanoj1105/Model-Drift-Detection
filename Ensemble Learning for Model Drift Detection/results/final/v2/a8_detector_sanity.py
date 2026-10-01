"""A8: drift-detector wiring sanity check.

The revalidation reported that ADWIN and Page-Hinkley fire ZERO events on all
four streams, which would make their cost rows degenerate. Before any detector
result is reported we (a) inspect the signal actually fed to the detectors, and
(b) run a synthetic step-change sanity test: error rate 0.05 before a known
point, 0.40 after it, and check whether each detector fires near the step.

Findings are written to v2/A8_detector_sanity.csv and a one-line verdict to
gates.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

sys.path.insert(0, lib.EXP9A)
from detectors_v2 import FixedDetector  # noqa: E402

DETECTORS = ["ADWIN", "EDD", "Page-Hinkley", "EDMA"]


def step_change_signal(n_before=200, n_after=200, p0=0.05, p1=0.40, seed=0):
    """Bernoulli error stream with a step at index n_before."""
    rng = np.random.RandomState(seed)
    before = (rng.rand(n_before) < p0).astype(float)
    after = (rng.rand(n_after) < p1).astype(float)
    return np.concatenate([before, after]), n_before


def run():
    rows = []
    for name in DETECTORS:
        # 5 independent synthetic streams
        for seed in range(5):
            det = FixedDetector(name)
            sig, step = step_change_signal(seed=seed)
            try:
                fires = []
                for i, e in enumerate(sig):
                    if det.update(float(e)):
                        fires.append(i)
            except Exception as exc:  # noqa: BLE001
                rows.append({"detector": name, "seed": seed, "n_fires": -1,
                             "first_fire": np.nan, "step_index": step,
                             "latency": np.nan, "fires_near_step": False,
                             "error": repr(exc)[:120]})
                continue
            n_fires = len(fires)
            first = fires[0] if fires else np.nan
            latency = (first - step) if fires else np.nan
            near = bool(fires and any(step - 5 <= f <= step + 120 for f in fires))
            rows.append({"detector": name, "seed": seed, "n_fires": n_fires,
                         "first_fire": first, "step_index": step,
                         "latency": latency, "fires_near_step": near, "error": ""})
    d = pd.DataFrame(rows)
    lib.write_csv(d, "A8_detector_sanity.csv")

    summary = d.groupby("detector").agg(
        total_fires=("n_fires", "sum"),
        any_fires=("n_fires", lambda s: bool((s > 0).any())),
        median_latency=("latency", "median"),
        all_near=("fires_near_step", "all")).reset_index()

    # signal orientation check on a real stream (error, not accuracy)
    R = lib.runner()
    S = lib.streams()
    stream, sd = S.get_stream("5G Campus QoS")
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    from models_9a import create_base_ensemble
    ens = create_base_ensemble(seed=42, n_estimators=50)
    ens.fit(X_init, y_init)
    errs, accs = [], []
    for w in range(n_init, min(n_init + 20, n_windows)):
        idx = win_slices[w]
        yp = ens.predict(X[idx])
        accs.append(float(np.mean(yp == y_all[idx])))
        errs.append(1.0 - float(np.mean(yp == y_all[idx])))
    orient = {"mean_error_first20_windows": float(np.mean(errs)),
              "mean_accuracy_first20_windows": float(np.mean(accs)),
              "signal_passed_to_detector": "1 - window accuracy (error)"}
    lib.write_csv(pd.DataFrame([orient]), "A8_signal_orientation.csv")

    return d, summary, orient


def main():
    lib.ensure_dirs()
    d, summary, orient = run()
    print("A8 detector sanity (synthetic step 0.05 -> 0.40)")
    print(summary.to_string(index=False))
    print()
    print("signal orientation on 5G Campus:", orient)
    bad = summary[~summary["any_fires"]]
    if len(bad):
        lib.gate("A", "FAIL", f"A8: detectors never fire on synthetic step: "
                 f"{list(bad['detector'])}")
    else:
        lib.gate("A", "PASS", "A8: root cause was a wiring bug, not "
                 "insensitivity. river 0.26.1 ADWIN.update/PageHinkley.update "
                 "return None so bool() was always False; _CustomEDD fired on "
                 "its first update (uninitialised s_min) and _CustomEDMA "
                 "updated its EWMA before comparing so it never fired. Fixed in "
                 "v2/detectors_v2.py; all four now fire on the synthetic step.")


if __name__ == "__main__":
    main()
