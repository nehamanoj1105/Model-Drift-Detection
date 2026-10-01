"""A6: repair / diagnose the drift-detector baselines.

The paper's ADWIN, Page-Hinkley and EDMA fire zero times and equal Frozen.
This script (a) confirms the exact trigger rules, (b) sweeps the detector
parameters as *sensitivity* (not selection), (c) runs detectors on a
per-sample error signal where the stream allows it, and (d) re-does the
detector comparison on all four streams.

Outputs A6_detector_events.csv (events per setting) and A6_detector_pooled.csv.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from river.drift import ADWIN, PageHinkley

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SEEDS, write_csv, write_raw_csv  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402
from drift_detectors_9a import _CustomEDD, _CustomEDMA  # noqa: E402
from models_9a import create_base_ensemble, make_class_anchor, make_train_buffer  # noqa: E402
from metrics_pooled import pooled_metrics, window_confusion  # noqa: E402


def make_det(name, **kw):
    if name == "ADWIN":
        return ADWIN(delta=kw.get("delta", 0.002))
    if name == "Page-Hinkley":
        return PageHinkley(min_instances=kw.get("min_instances", 30),
                           delta=0.005, threshold=kw.get("threshold", 50),
                           alpha=0.9999)
    if name == "EDD":
        return _CustomEDD(min_instances=kw.get("min_instances", 30),
                          warning_level=kw.get("warning_level", 2.0),
                          drift_level=kw.get("drift_level", 3.0))
    if name == "EDMA":
        return _CustomEDMA(min_instances=kw.get("min_instances", 30),
                           alpha=kw.get("alpha", 0.2), k=kw.get("k", 2.0))
    raise ValueError(name)


def run_detector(stream_df, sd, seed, det_name, params, signal="window"):
    """Run one detector-triggered model; return (pooled_row, events, retrains)."""
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream_df, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    ens = create_base_ensemble(seed=seed, n_estimators=R.FULL_TREES)
    ens.fit(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    det = make_det(det_name, **params)
    events, retrains, adapt_cpu = 0, 0, 0.0
    last_retrain = -10**9
    yt_all, yp_all = [], []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        yp = ens.predict(X[idx])
        yt_all.append(y_all[idx]); yp_all.append(yp)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        if signal == "window":
            errs = [1.0 - float(np.mean(yp == y_all[idx]))]
        else:
            errs = (yp != y_all[idx]).astype(float)
        trig = False
        for e in errs:
            if det.update(e):
                trig = True
        if trig:
            events += 1
            if (w - last_retrain) >= 3:
                retrains += 1
                last_retrain = w
                import time
                t0 = time.process_time()
                trX, trY = make_train_buffer(np.asarray(buf_X), np.asarray(buf_y),
                                             anchor_X, anchor_y, R.BUFFER_CAPACITY)
                ens = create_base_ensemble(seed=seed + retrains * 17)
                ens.fit(trX, trY)
                adapt_cpu += time.process_time() - t0
    yt = np.concatenate(yt_all); ypp = np.concatenate(yp_all)
    pm = pooled_metrics(yt, ypp, n_classes)
    pm.update({"seed": seed, "detector": det_name, "signal": signal,
               "events": events, "retrains": retrains,
               "adaptation_cpu_sec": adapt_cpu, "params": json.dumps(params)})
    return pm


def main():
    grids = {
        "ADWIN": [{"delta": d} for d in [0.002, 0.01, 0.05, 0.1, 0.3]],
        "Page-Hinkley": [{"threshold": t, "min_instances": m}
                         for t in [0.05, 0.1, 0.5, 1, 5, 50] for m in [5, 10, 30]],
        "EDMA": [{"alpha": a, "k": k} for a in [0.05, 0.2, 0.4] for k in [1, 2, 3]],
        "EDD": [{}],
    }
    rows = []
    for ds in ALL_DATASETS:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        print(f"### {ds}", flush=True)
        for det, plist in grids.items():
            for params in plist:
                for signal in ("window", "sample"):
                    try:
                        r = run_detector(stream, sd, 42, det, params, signal)
                        r["dataset"] = slug
                        rows.append(r)
                    except Exception as e:
                        print("   fail", det, params, signal, e)
            print(f"   {det} done", flush=True)
    df = pd.DataFrame(rows)
    write_csv(df, "A6_detector_events.csv")
    for ds in df.dataset.unique():
        s = df[(df.dataset == ds) & (df.signal == "window")]
        print(f"\n-- {ds}: max events over settings")
        for det, g in s.groupby("detector"):
            print(f"   {det:14s} max_events={g.events.max()} "
                  f"(default_events={g.iloc[0].events})")


if __name__ == "__main__":
    main()
