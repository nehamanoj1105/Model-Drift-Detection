"""Task A: choose the RAPT-ED-Guard margin m on the dev split only.

Seeds 42-46, first 60% of post-prefix windows. Compare m in {0.0, 0.02, 0.05}
by dev macro-F1 of RAPT-ED-Guard (averaged over windows and seeds), and record
the gate allow rate. The winner is frozen into configs/final2.yaml.
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
from final2_runner import prepare, OUT, SLUGS
from final2_methods import RAPTEDGuard
from run_final import _prepare_arrays
from models_9a import make_class_anchor
from sklearn.metrics import f1_score
import yaml

STREAMS = ["campus", "ugr16", "5g_nr", "kpiops"]
MARGINS = [0.0, 0.02, 0.05]


def dev_run(name, seed, margin, base_cfg):
    stream, sd = prepare(SLUGS[name])
    (X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows) = _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    cfg = dict(base_cfg); cfg["edguard_margin"] = margin
    g = RAPTEDGuard(seed, anchor_X, anchor_y, cfg)
    g.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = regimes[n_init - 1]
    n_eval = max(1, int(round((n_windows - n_init) * DEV_WINDOW_FRACTION)))
    stop_w = n_init + n_eval
    f1s = []
    for w in range(n_init, stop_w):
        idx = win_slices[w]; reg = regimes[w]
        yp, pc = g.predict_window(X[idx], buf_X, buf_y)
        err = 1.0 - np.mean(yp == y_all[idx])
        g.post_predict(w, idx, X[idx], y_all[idx], buf_X, buf_y,
                       err=err, regime=reg, prev_regime=prev)
        prev = reg
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        f1s.append(f1_score(y_all[idx], yp, average="macro", zero_division=0))
    gl = g.gate_log
    allow = sum(1 for r in gl if r["decision"] == "allow")
    return dict(stream=name, seed=seed, margin=margin,
                macro_f1=float(np.mean(f1s)),
                gate_events=len(gl), gate_allows=allow)


def main():
    base_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    base_cfg.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})
    from joblib import Parallel, delayed
    jobs = [(s, seed, m) for m in MARGINS for s in STREAMS for seed in SEEDS_DEV]
    t0 = time.time()
    res = Parallel(n_jobs=4, verbose=5)(
        delayed(dev_run)(s, seed, m, base_cfg) for (s, seed, m) in jobs)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(ROOT, "results", "final2", "dev_margin_sweep.csv"), index=False)
    agg = df.groupby(["stream", "margin"]).agg(
        macro_f1=("macro_f1", "mean"), gate_events=("gate_events", "sum"),
        gate_allows=("gate_allows", "sum")).reset_index()
    print(agg.to_string(index=False))
    overall = df.groupby("margin").macro_f1.agg(["mean", "std"]).reset_index()
    print("\nMARIN OVERALL:\n", overall.to_string(index=False))
    # pooled mean macro-F1 per margin (all streams equally weighted)
    print("\nper-margin mean over streams:",
          {round(m, 2): round(df[df.margin == m].groupby("stream").macro_f1.mean().mean(), 4)
           for m in MARGINS})
    print(f"TOTAL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
