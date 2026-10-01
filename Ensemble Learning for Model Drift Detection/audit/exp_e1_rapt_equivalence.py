"""E1 -- do the three RAPT copies agree?

The repository carries three RAPT implementations:
  experiments/exp9a/rapt_9a.py          (RAPTSystem / RAPTEnhancedSystem)
  experiments/exp9b/rapt_9b.py          (5G NR stream)
  Final_Experiments/rapt_ladder.py      (RAPT_T2 == Tier-2 repository only)

The paper reports exp9a numbers in Table II and Final_Experiments numbers in the
ablation ladder. This script checks that the exp9a RAPTSystem and the
Final_Experiments RAPT_T2 variant, given the same stream, buffer and seeds,
produce identical per-window predictions. If they do, the ablation baseline is
the same algorithm as Table II RAPT and the ladder is interpretable.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp9a"))
sys.path.insert(0, os.path.join(ROOT, "Final_Experiments"))

from three_dataset_load import LOADERS, build_windows  # noqa: E402
from models_9a import make_class_anchor  # noqa: E402
from rapt_9a import RAPTSystem  # noqa: E402
import exp9a_config as base  # noqa: E402
import rapt_ladder  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
BUFFER = base.BUFFER_CAPACITY


def run(seed):
    data = LOADERS["5G Campus QoS"]()
    stream, sd = build_windows(data)
    n_init, n_win = sd["initial_train_windows"], sd["total_windows"]
    feat = sd["feature_columns"]
    X_raw = stream[feat].values.astype(np.float64)
    y = stream["y"].values.astype(int)
    reg = stream["regime_id"].values
    wid = stream["window_id"].values
    init = wid < n_init
    mu = X_raw[init].mean(axis=0)
    s = np.where(X_raw[init].std(axis=0) < 1e-9, 1.0, X_raw[init].std(axis=0))
    X = (X_raw - mu) / s
    X_init, y_init = X[init], y[init]
    slices = {w: np.where(wid == w)[0] for w in range(n_init, n_win)}
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)

    # exp9a controller
    a = RAPTSystem(seed=seed, mode="full", enable_calibration=True,
                   anchor_X=anchor_X, anchor_y=anchor_y, buffer_capacity=BUFFER)
    a.fit_initial(reg[np.where(wid == n_init - 1)[0][0]], X_init, y_init)
    # Final_Experiments controller, Tier-2 repository only
    b = rapt_ladder.build_rapt("RAPT_T2", seed=seed, anchor_X=anchor_X,
                               anchor_y=anchor_y, buffer_capacity=BUFFER)
    b.fit_initial(reg[np.where(wid == n_init - 1)[0][0]], X_init, y_init)

    pa, pb = [], []
    buf_X, buf_y = list(X_init), list(y_init)
    prev = reg[np.where(wid == n_init - 1)[0][0]]
    for w in range(n_init, n_win):
        idx = slices[w]
        r = reg[idx[0]]
        if r != prev:
            a.handle_regime_transition(r, w, X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
            b.handle_regime_transition(r, X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
            prev = r
        pa.append(a.predict(X[idx]))
        pb.append(b.predict(X[idx]))
        buf_X.extend(X[idx]); buf_y.extend(y[idx])
        buf_X, buf_y = buf_X[-BUFFER:], buf_y[-BUFFER:]
        a.update(X[idx], y[idx], X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
        b.update(X[idx], y[idx], X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
    return np.concatenate(pa), np.concatenate(pb)


print("E1 -- exp9a RAPTSystem vs Final_Experiments RAPT_T2 on 5G Campus")
identical = True
for s in SEEDS:
    pa, pb = run(s)
    same = int((pa == pb).sum())
    ok = same == len(pa)
    identical &= ok
    print(f"  seed {s}: {same}/{len(pa)} predictions identical -> "
          f"{'IDENTICAL' if ok else 'DIFFER'}")
print(f"\nVerdict: implementations are {'EQUIVALENT' if identical else 'NOT equivalent'}")
