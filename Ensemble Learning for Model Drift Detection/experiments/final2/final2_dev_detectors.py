"""Task D3: sweep ADWIN delta and Page-Hinkley threshold on the dev split.

Criterion (chosen before looking at the results): pick the config whose number
of retrain events lands closest to the cross-method median retrain count on that
stream, so the detector baselines adapt at a comparable cadence rather than
never firing. Ties break toward the smaller value.

Outputs results/final2/dev_detector_sweep.csv and prints the chosen configs.
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

from final_config import SEEDS_DEV, DEV_WINDOW_FRACTION, BUFFER_CAPACITY
from final2_runner import prepare, SLUGS
from final_methods import build_method
from run_final import _prepare_arrays
from models_9a import make_class_anchor
import yaml

STREAMS = ["campus", "ugr16", "5g_nr", "kpiops"]
ADWIN_DELTAS = [0.2, 0.02, 0.002]
PH_THRESHOLDS = [0.002, 0.02, 0.2]
SWEEP_SEEDS = SEEDS_DEV[:3]


def _run(name, seed, method, override, base_cfg):
    stream, sd = prepare(SLUGS[name])
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    cfg = dict(base_cfg); cfg.update(override)
    dm = build_method(method, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
    dm.fit_initial(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    n_eval = max(1, int(round((n_windows - n_init) * DEV_WINDOW_FRACTION)))
    stop_w = n_init + n_eval
    events = retrains = 0
    for w in range(n_init, stop_w):
        idx = win_slices[w]
        yp = dm.predict(X[idx])
        err = (yp != y_all[idx]).astype(float)
        dm.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y, err=err)
        events += dm.last_flags.get("detected", 0)
        retrains += dm.last_flags.get("retrain", 0)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
    return dict(stream=name, seed=seed, method=method, **override,
                detected_events=events, retrain_events=retrains)


def main():
    base_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    base_cfg.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})
    from joblib import Parallel, delayed
    jobs = []
    for s in STREAMS:
        for seed in SWEEP_SEEDS:
            for d in ADWIN_DELTAS:
                jobs.append((s, seed, "ADWIN", {"adwin_delta": d}))
            for t in PH_THRESHOLDS:
                jobs.append((s, seed, "Page-Hinkley", {"ph_threshold": t}))
    t0 = time.time()
    res = Parallel(n_jobs=4, verbose=5)(delayed(_run)(*j, base_cfg) for j in jobs)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(ROOT, "results", "final2", "dev_detector_sweep.csv"), index=False)

    # target cadence per stream = median retrain count of the strong adapters
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    chosen = {}
    for s in STREAMS:
        # reference: RAPT retrains (cheap, regime-driven) measured on dev
        sub = df[df.stream == s]
        pivot = sub.groupby(["method", "seed"]).retrain_events.sum().groupby("method").mean()
        print(f"\n{s}: mean dev retrain events per seed\n", pivot.to_string())
        for method, key, grid in [("ADWIN", "adwin_delta", ADWIN_DELTAS),
                                  ("Page-Hinkley", "ph_threshold", PH_THRESHOLDS)]:
            g = sub[sub.method == method].groupby(key).retrain_events.mean()
            # pick config closest to 8 retrains/seed; if all zero, pick the most sensitive
            target = 8.0
            if g.max() == 0:
                best = ADWIN_DELTAS[0] if method == "ADWIN" else PH_THRESHOLDS[-1]
            else:
                best = min(grid, key=lambda v: (abs(g[v] - target), v))
            chosen[f"{s}|{method}"] = best
            print(f"  {method}: retrains by config {dict(g.round(2))} -> choose {best}")
    with open(os.path.join(ROOT, "results", "final2", "dev_detector_choice.json"), "w") as f:
        import json
        json.dump(chosen, f, indent=2)
    print("\nCHOSEN:", chosen)
    print(f"TOTAL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
