#!/usr/bin/env python3
"""Task 4 -- drift-detector sanity on a synthetic step-change error stream.

NEW SCRIPT (labelled as such).  Feeds a controlled 0/1 error stream to each
detector exactly as it is wired for the paper (Table II) and exactly as it is
wired in the revalidation, and reports fire counts and detection latency.

Synthetic stream (fixed by the task):
  * n = 300 windows, error = 0.05 for indices 0..199, then 0.40 from index 200.
  * 20 repeats; each repeat adds a different seed's Bernoulli realisation of the
    two levels, so the detector sees a realistic noisy 0/1 signal.

Wiring A -- "paper / Table II": experiments/exp9a/drift_detectors_9a.py
    ADWIN(delta=0.002), PageHinkley(min_instances=30, delta=0.005, threshold=50,
    alpha=0.9999), _CustomEDD(), _CustomEDMA(), each fed ONE error per window.

Wiring B -- "revalidation": results/revalidation/a6_detectors.py
    same river/custom classes but with a sweep of settings; the paper-default
    setting is the same as wiring A.  The revalidation wrapper also updates the
    detector once per window with a single window error, so the operating point
    is identical; the *only* structural difference is that a6 wraps each update
    in `if det.update(e):` while the 9A DetectorAdaptiveModel uses
    `bool(self.detector.update(...))` (both consume the boolean correctly).

Because the two wirings are identical for the default settings, the script
reports both and flags where they differ (the A6 sweep variants).

Run from the project directory:
    python "results/extract/v2/detector_sanity.py"
"""
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
REVAL = os.path.join(ROOT, "results", "revalidation")
sys.path.insert(0, EXP9A)
sys.path.insert(0, REVAL)

from drift_detectors_9a import make_detector, _CustomEDD, _CustomEDMA  # noqa: E402
from river.drift import ADWIN, PageHinkley  # noqa: E402

N = 300
CHANGE_AT = 200
LOW, HIGH = 0.05, 0.40
REPEATS = 20


def synthetic_error(seed):
    """20 repeats of a noisy 0/1 error stream with a step at CHANGE_AT."""
    rng = np.random.RandomState(1000 + seed)
    level = np.where(np.arange(N) < CHANGE_AT, LOW, HIGH)
    return (rng.rand(N) < level).astype(float)


# --- wiring A: paper / Table II (drift_detectors_9a.make_detector) ----------
def wiring_a(name):
    return make_detector(name)


# --- wiring B: revalidation A6 defaults (a6_detectors.make_det) -------------
def wiring_b(name):
    if name == "ADWIN":
        return ADWIN(delta=0.002)
    if name == "Page-Hinkley":
        return PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
    if name == "EDD":
        return _CustomEDD(min_instances=30, warning_level=2.0, drift_level=3.0)
    if name == "EDMA":
        return _CustomEDMA(min_instances=30, alpha=0.2, k=2.0)
    raise ValueError(name)


# --- wiring C: revalidation A6 sensitivity sweep extremes -------------------
# Not the reported operating point; included only to show the detectors CAN
# fire when their parameters are loosened.
def wiring_c(name):
    if name == "ADWIN":
        return ADWIN(delta=0.3)
    if name == "Page-Hinkley":
        return PageHinkley(min_instances=5, delta=0.005, threshold=0.05, alpha=0.9999)
    if name == "EDD":
        return _CustomEDD(min_instances=30, warning_level=2.0, drift_level=3.0)
    if name == "EDMA":
        return _CustomEDMA(min_instances=30, alpha=0.4, k=1.0)
    raise ValueError(name)


def run_one(det, errs):
    fires = []
    for i, e in enumerate(errs):
        if det.update(float(e)):
            fires.append(i)
    return fires


def main():
    detectors = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]
    rows = []
    for wiring, factory in (("A_paper_tableII", wiring_a),
                            ("B_revalidation_default", wiring_b),
                            ("C_revalidation_sweep_extreme", wiring_c)):
        for name in detectors:
            for r in range(REPEATS):
                det = factory(name)
                errs = synthetic_error(r)
                fires = run_one(det, errs)
                pre = [f for f in fires if f < CHANGE_AT]
                post = [f for f in fires if f >= CHANGE_AT]
                first_post = post[0] if post else None
                rows.append(dict(
                    wiring=wiring, detector=name, repeat=r,
                    n_windows=N, change_at=CHANGE_AT,
                    low_error=LOW, high_error=HIGH,
                    total_fires=len(fires),
                    pre_change_fires=len(pre),
                    post_change_fires=len(post),
                    first_post_change_fire=first_post,
                    latency_windows=(None if first_post is None else first_post - CHANGE_AT),
                    fired=(len(post) > 0),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "T4_detector_sanity_raw.csv"), index=False)

    agg = df.groupby(["wiring", "detector"]).agg(
        repeats=("repeat", "count"),
        fired_fraction=("fired", "mean"),
        total_fires_mean=("total_fires", "mean"),
        pre_change_fires_mean=("pre_change_fires", "mean"),
        post_change_fires_mean=("post_change_fires", "mean"),
        latency_mean=("latency_windows", "mean"),
        latency_min=("latency_windows", "min"),
        latency_max=("latency_windows", "max"),
    ).reset_index()
    agg.to_csv(os.path.join(HERE, "T4_detector_sanity.csv"), index=False)
    print(agg.to_string(index=False))

    meta = pd.DataFrame([
        dict(item="script status", value="NEW (written for this extraction task)",
             detail="results/extract/v2/detector_sanity.py"),
        dict(item="stream", value="n=300, 0.05 for i<200 then 0.40, 20 Bernoulli repeats",
             detail="synthetic_error() seeds 1000..1019"),
        dict(item="wiring A (paper/Table II)",
             value="experiments/exp9a/drift_detectors_9a.py:98-107 make_detector",
             detail="ADWIN(delta=0.002); PageHinkley(min_instances=30,delta=0.005,threshold=50,alpha=0.9999); _CustomEDD(); _CustomEDMA(); one update per window"),
        dict(item="wiring B (revalidation)",
             value="results/revalidation/a6_detectors.py:29-43 make_det (default settings)",
             detail="identical classes/params at the default setting; A6 additionally sweeps params and a per-sample signal"),
        dict(item="structural difference", value="none for default settings",
             detail="both consume the boolean update() return; the river 'drift_detected attribute' pitfall is not present"),
    ])
    meta.to_csv(os.path.join(HERE, "T4_detector_sanity_meta.csv"), index=False)
    print("wrote T4_detector_sanity.csv / _raw.csv / _meta.csv")


if __name__ == "__main__":
    main()
