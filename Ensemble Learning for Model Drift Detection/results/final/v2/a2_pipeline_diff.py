"""A2: reconcile the two Campus RAPT-Cheap numbers.

  * cost-matched table (T3): RAPT-Cheap 0.9817 (per-window) vs Full Retraining 0.9836
  * Table III ladder:        RAPT-Cheap 0.9851 vs Full Retraining 0.9829

Both are per-window means, so the metric is not the difference. This script
runs the ladder variant (Final_Experiments LadderRAPT) and the cost-matched
variant (RAPTV2) through the SAME v2 pipeline on 5G Campus, seed 42, and prints
a component-by-component diff, then produces a single Full Retraining row from
that one pipeline.

Output: v2/A2_pipeline_diff.csv, v2/A2_single_pipeline_campus.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

FINAL_EXP = os.path.join(lib.ROOT, "Final_Experiments")
sys.path.insert(0, FINAL_EXP)

R = lib.runner()
S = lib.streams()


def ladder_rapt_campus(seed=42):
    """Run the Final_Experiments LadderRAPT RAPT_CHEAP through its own runner
    semantics on 5G Campus, one seed, and return per-window + pooled metrics."""
    from rapt_ladder import build_rapt
    from models_9a import make_class_anchor
    from three_dataset_load import load_campus, build_windows
    stream, sd = build_windows(load_campus())
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    sys_ = build_rapt("RAPT_CHEAP", seed=seed, anchor_X=anchor_X,
                      anchor_y=anchor_y, buffer_capacity=1000)
    sys_.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    yt, yp = [], []
    f1s = []
    from sklearn.metrics import f1_score
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        if reg != prev:
            sys_.handle_regime_transition(reg, X_buffer=buf_X[-1000:], y_buffer=buf_y[-1000:])
            prev = reg
        p = sys_.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > 1000:
            buf_X, buf_y = buf_X[-1000:], buf_y[-1000:]
        sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-1000:], y_buffer=buf_y[-1000:])
        yt.append(y_all[idx]); yp.append(p)
        f1s.append(f1_score(y_all[idx], p, average="macro", zero_division=0))
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return {"pipeline": "Final_Experiments LadderRAPT",
            "per_window_macro_f1": float(np.mean(f1s)),
            "pooled_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "refresh_trees": 10, "refresh_buffer": 300,
            "refresh_counter": "per-regime (reset on transition)",
            "metric": "per-window mean"}


def v2_cheap_campus(seed=42):
    stream, sd = S.get_stream("5G Campus QoS")
    pw, sm, pooled, _ = R.run_seed_v2(stream, sd, seed,
                                      ["RAPT-Cheap", "RAPT-Cheap-WinEq", "Full Retraining",
                                       "Frozen", "Event-Driven"])
    sm = sm.set_index("method")
    rows = []
    for m in ["RAPT-Cheap", "RAPT-Cheap-WinEq", "Full Retraining"]:
        rows.append({"pipeline": "v2 RAPTV2",
                     "method": m,
                     "per_window_macro_f1": float(sm.loc[m, "per_window_macro_f1"]),
                     "pooled_macro_f1": float(pooled.set_index("method").loc[m, "pooled_macro_f1"]),
                     "refresh_trees": 20, "refresh_buffer": 300,
                     "refresh_counter": "per-regime (reset on transition)" if m != "RAPT-Cheap-WinEq" else "global",
                     "metric": "per-window mean"})
    return pd.DataFrame(rows), pw, sm, pooled


def run():
    rows = [ladder_rapt_campus()]
    v2rows, pw, sm, pooled = v2_cheap_campus()
    rows.extend(v2rows.to_dict("records"))
    d = pd.DataFrame(rows)
    lib.write_csv(d, "A2_pipeline_diff.csv")
    print(d.to_string(index=False))

    # single-pipeline Campus table (one Full Retraining row)
    single = sm.reset_index()[["method", "per_window_macro_f1", "adaptation_cpu_sec",
                               "total_runtime_sec", "retrain_events", "reuse_events",
                               "refresh_events", "trees_trained"]]
    single["pooled_macro_f1"] = single["method"].map(
        pooled.set_index("method")["pooled_macro_f1"])
    lib.write_csv(single, "A2_single_pipeline_campus.csv")
    print()
    print("single-pipeline Campus (one Full Retraining row):")
    print(single.to_string(index=False))
    return d


def main():
    lib.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
