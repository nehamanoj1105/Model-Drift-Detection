"""
STEP 1a gate-accuracy evaluation.

For every return of a previously-seen regime, the guard scores the stored
ensemble and the prefix-trained ensemble on the last `guard_lookback` labelled
rows and decides whether to reuse. That decision is evaluated against the
outcome on the *incoming* window (labels arrive after prediction):

  * bad allow  : guard reused, but the stored model was NOT better than the
                 prefix model on the incoming window.
  * bad block  : guard blocked, but the stored model WAS better.
  * good allow / good block: decision matched the outcome.

BAD-ALLOW RATE is the key "how often did it allow a bad reuse" number.

Writes results/final/tables/gate_accuracy.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp9a"))

from final_config import SEEDS_DEV, DEFAULT_CFG, TABLES_FINAL, CONFIG_DIR
import yaml


def _frozen_cfg():
    p = os.path.join(CONFIG_DIR, "final.yaml")
    if os.path.exists(p):
        with open(p) as f:
            return {**DEFAULT_CFG, **yaml.safe_load(f)}
    return dict(DEFAULT_CFG)


GATE_TOL = 0.02   # a reuse that ties within 2 pts of the prefix model is not "bad"
from final_methods import build_method
from models_9a import make_class_anchor
from run_final import _prepare_stream, _prepare_arrays

STREAMS = {"campus": "5G Campus QoS", "ugr16": "UGR'16",
           "nordicdat": "NordicDat", "5g_nr": "5G NR Latency"}


def run_one(name, sd, seed, cfg):
    (X, y, regimes, X_init, y_init, win_slices, n_init, n_windows) = _prepare_arrays(*_prepare_stream(name))
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    g = build_method("RAPT-Guard", seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
    g.fit_initial(init_regime=regimes[n_init - 1], X_init=X_init, y_init=y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = regimes[n_init - 1]
    rows = []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[w]
        g.last_flags = {}
        g.predict_window(X[idx], buf_X, buf_y)
        decided = False
        if reg != prev:
            g.on_regime_change(reg, w, X[idx], y[idx], buf_X, buf_y, prev_regime=prev)
            prev = reg
            decided = True
        g.post_predict(w, idx, X[idx], y[idx], buf_X, buf_y, prev_regime=prev)
        # evaluate the guard decision against the incoming window
        if decided and reg in g.repo and "gate" in g.last_flags:
            stored = g.repo[reg]
            sw = float(np.mean(stored.predict(X[idx]) == y[idx]))
            pw = float(np.mean(g.prefix_ens.predict(X[idx]) == y[idx]))
            good_reuse = (sw - pw) > GATE_TOL
            decision = "allow" if g.last_flags.get("gate_allow") else "block"
            rows.append(dict(window_id=w, regime=str(reg),
                             decision=decision, stored_win=sw, prefix_win=pw,
                             good_reuse=int(good_reuse),
                             bad_allow=int(decision == "allow" and not good_reuse),
                             bad_block=int(decision == "block" and good_reuse)))
        buf_X.extend(X[idx]); buf_y.extend(y[idx])
        if len(buf_X) > 1000:
            buf_X, buf_y = buf_X[-1000:], buf_y[-1000:]
    return rows


def main():
    all_rows = []
    for slug, name in STREAMS.items():
        cfg = _frozen_cfg()
        for seed in SEEDS_DEV[:3]:   # 3 seeds is enough for a diagnostic table
            for r in run_one(name, None, seed, cfg):
                r.update(stream=slug, seed=seed)
                all_rows.append(r)
        print("  gate", slug, flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(TABLES_FINAL, "gate_decisions_raw.csv"), index=False)
    g = df.groupby("stream").agg(
        reuse_returns=("decision", "size"),
        allows=("decision", lambda s: int((s == "allow").sum())),
        blocks=("decision", lambda s: int((s == "block").sum())),
        bad_allows=("bad_allow", "sum"), bad_blocks=("bad_block", "sum"))
    g["bad_allow_rate"] = (g["bad_allows"] / g["allows"].replace(0, np.nan)).round(3)
    g["missed_reuse"] = g["bad_blocks"]
    g.reset_index().to_csv(os.path.join(TABLES_FINAL, "gate_accuracy.csv"), index=False)
    print(g.to_string())


if __name__ == "__main__":
    main()
