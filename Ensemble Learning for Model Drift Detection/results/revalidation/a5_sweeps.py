"""A5: code-vs-description mismatches.

Produces:
  A5_component_matrix.csv   component x variant (Table II RAPT, Table II
                            RAPT-Enhanced, each Table III rung)
  A5_parity_sweep.csv       parity threshold sweep
  A5_reldrop_sweep.csv      rel_drop sweep
  A5_gate.csv               original gate vs gate-fix
  A5_novelty_buffer.csv     actual training rows at each novelty refit
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SEEDS, write_csv  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def component_matrix():
    sys.path.insert(0, os.path.join(ROOT, "Final_Experiments"))
    import rapt_ladder as RL
    variants = ["RAPT_T2", "RAPT_T1", "RAPT_T1_REFIT", "RAPT_FULL",
                "RAPT_REL_REFIT", "RAPT_REFRESH_W5", "RAPT_REFRESH_W10",
                "RAPT_EVIDENCE", "RAPT_CHEAP", "RAPT_COMBO", "RAPT_FLOOR",
                "RAPT_INCR"]
    rows = []
    # Table II variants (from rapt_9a.py, read statically)
    rows.append({"variant": "TableII_RAPT", "Tier1_learner": 0, "policy_repository": 1,
                 "novelty_refit": 1, "parity_refit": 0, "fingerprint_gate": 0,
                 "periodic_refresh": 0, "cheap_refit": 0})
    rows.append({"variant": "TableII_RAPT-Enhanced", "Tier1_learner": 0,
                 "policy_repository": 1, "novelty_refit": 1, "parity_refit": 1,
                 "fingerprint_gate": 0, "periodic_refresh": 0, "cheap_refit": 0})
    for v in variants:
        d = RL.build_rapt(v, seed=0, anchor_X=None, anchor_y=None)
        rows.append({
            "variant": v,
            "Tier1_learner": int(d.use_tier1),
            "policy_repository": 1,
            "novelty_refit": 1,
            "parity_refit": int(d.use_refit),
            "fingerprint_gate": int(d.use_gate),
            "periodic_refresh": int(d.refresh_every or 0),
            "cheap_refit": int(d.refresh_trees == 10),
            "refresh_mode": d.refresh_mode,
        })
    df = pd.DataFrame(rows)
    write_csv(df, "A5_component_matrix.csv")
    print(df.to_string(index=False))
    return df


def run_spec(stream, sd, seed, method, kwargs):
    """Run one RAPTV2 configuration directly."""
    from models_9a import make_class_anchor
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream_df_ids(stream)
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    s = R.RAPTV2(seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, **kwargs)
    s.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    yt_all, yp_all = [], []
    wreg = {int(w): str(regimes[win_slices[w][0]]) for w in range(n_init, n_windows)}
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        if reg != prev:
            if getattr(s, "gate_fix", False):
                s._incoming_fp = s._fingerprint(X[idx])
            s.on_transition(reg, w, buf_X[-R.BUFFER_CAPACITY:],
                            buf_y[-R.BUFFER_CAPACITY:], [w - 1], [prev])
            prev = reg
        yp = s.predict(X[idx])
        yt_all.append(y_all[idx]); yp_all.append(yp)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        s.update(w, X[idx], y_all[idx], buf_X, buf_y, [w], [reg])
    from metrics_pooled import pooled_metrics
    pm = pooled_metrics(np.concatenate(yt_all), np.concatenate(yp_all), n_classes)
    pm.update({"seed": seed, "method": method, "refresh_events": s.refresh_events,
               "parity_refits": s.parity_refits, "reuse_events": s.reused_policy_count,
               "gate_accept": s.gate_accept, "gate_reject": s.gate_reject,
               "adaptation_cpu_sec": s.adaptation_cpu_time,
               "max_refit_rows": int(max(s.refit_rows)) if s.refit_rows else 0})
    return pm


def stream_df_ids(stream):
    return stream["window_id"].values


def main():
    component_matrix()
    parity_rows, rel_rows, gate_rows, buf_rows = [], [], [], []
    for ds in ALL_DATASETS:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        print(f"### {ds}", flush=True)
        for seed in SEEDS:
            for th in [0.5, 0.7, 0.8, 0.9, 0.95]:
                r = run_spec(stream, sd, seed, f"RAPT-Parity-{th}",
                             dict(parity_threshold=th))
                r.update({"dataset": slug}); parity_rows.append(r)
            for rd in [0.02, 0.03, 0.05, 0.10]:
                r = run_spec(stream, sd, seed, f"RAPT-Evidence-{rd}",
                             dict(refresh_every=1, refresh_mode="evidence",
                                  rel_drop=rd, refresh_trees=R.CHEAP_TREES,
                                  refresh_buffer=R.CHEAP_BUFFER))
                r.update({"dataset": slug}); rel_rows.append(r)
            # gate: original vs fix
            r = run_spec(stream, sd, seed, "RAPT-Gate-Orig",
                         dict(use_gate=True, gate_fix=False))
            r.update({"dataset": slug}); gate_rows.append(r)
            r = run_spec(stream, sd, seed, "RAPT-Gate-Fix",
                         dict(use_gate=True, gate_fix=True))
            r.update({"dataset": slug}); gate_rows.append(r)
        # novelty buffer: RAPT-Enhanced effective training rows
        for seed in SEEDS:
            r = run_spec(stream, sd, seed, "RAPT-Novelty1500",
                         dict(refit_n=1500, buffer=1000))
            r.update({"dataset": slug}); buf_rows.append(r)
        print(f"   {ds} done", flush=True)
    write_csv(pd.DataFrame(parity_rows), "A5_parity_sweep.csv")
    write_csv(pd.DataFrame(rel_rows), "A5_reldrop_sweep.csv")
    write_csv(pd.DataFrame(gate_rows), "A5_gate.csv")
    write_csv(pd.DataFrame(buf_rows), "A5_novelty_buffer.csv")
    print("\nnovelty buffer max training rows (capacity 1000):")
    b = pd.DataFrame(buf_rows)
    print(b.groupby("dataset")["max_refit_rows"].max())


if __name__ == "__main__":
    main()
