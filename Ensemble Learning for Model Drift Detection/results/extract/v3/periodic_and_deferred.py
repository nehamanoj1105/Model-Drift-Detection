#!/usr/bin/env python3
"""v3 task 1 and task 2.

Task 1 -- periodic refresh counter in RAPT-Cheap (RAPTV2) and Periodic-Cheap
          (FRDriver): print the code paths, test empirically whether the counter
          resets at regime transitions, and re-run RAPT-Cheap and
          Periodic-Cheap-5/10 on UGR'16 seeds 42-46 with a no-reset variant.

Task 2 -- RAPT-Deferred on 5G Campus: count evaluation windows where its
          predictions differ from RAPT, and check whether the deferred
          checkpoint is ever reused.

Run from the project directory:
    python "results/extract/v3/periodic_and_deferred.py"
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
sys.path.insert(0, REVAL)

import run_stream_v2 as R                      # noqa: E402
from streams import get_stream                 # noqa: E402
from models_9a import make_class_anchor        # noqa: E402
from metrics_pooled import pooled_metrics      # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
CAMPUS = "5G Campus QoS"
UGR16 = "UGR'16"


# ---------------------------------------------------------------------------
# Task 1: no-reset RAPT-Cheap
# ---------------------------------------------------------------------------
class RAPTV2NoReset(R.RAPTV2):
    """RAPTV2 with the periodic refresh counter NOT reset on regime transitions."""

    def on_transition(self, new_regime_id, window_id, X_buffer, y_buffer,
                      train_windows, train_regimes):
        kept = self._windows_since_refresh
        out = super().on_transition(new_regime_id, window_id, X_buffer, y_buffer,
                                    train_windows, train_regimes)
        self._windows_since_refresh = kept
        return out


def _rapt_builder(kind):
    if kind == "RAPT":
        return lambda s, ax, ay: R.RAPTSystem(
            seed=s, mode="full", enable_calibration=True, anchor_X=ax, anchor_y=ay,
            buffer_capacity=R.BUFFER_CAPACITY)
    if kind == "RAPT-Cheap":
        return lambda s, ax, ay: R.RAPTV2(
            seed=s, anchor_X=ax, anchor_y=ay, refit_n=R.BUFFER_CAPACITY,
            refresh_every=5, refresh_trees=R.CHEAP_TREES, refresh_buffer=R.CHEAP_BUFFER)
    if kind == "RAPT-Cheap-NoReset":
        return lambda s, ax, ay: RAPTV2NoReset(
            seed=s, anchor_X=ax, anchor_y=ay, refit_n=R.BUFFER_CAPACITY,
            refresh_every=5, refresh_trees=R.CHEAP_TREES, refresh_buffer=R.CHEAP_BUFFER)
    raise ValueError(kind)


def _run_rapt(stream, sd, seed, kind):
    """Self-contained prequential loop for a RAPT controller; returns pw, pooled, sys."""
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    sys_ = _rapt_builder(kind)(seed, anchor_X, anchor_y)
    sys_.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    recs = []
    yt_all, yp_all = [], []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        reused = 0
        if reg != prev:
            if isinstance(sys_, R.RAPTV2):
                reused_i, _, _ = sys_.on_transition(reg, w, buf_X[-R.BUFFER_CAPACITY:],
                                                    buf_y[-R.BUFFER_CAPACITY:], [w], [reg])
            else:
                reused_i, _, _ = sys_.handle_regime_transition(
                    reg, w, X_buffer=buf_X[-R.BUFFER_CAPACITY:],
                    y_buffer=buf_y[-R.BUFFER_CAPACITY:])
            reused = int(reused_i)
            prev = reg
        yp = sys_.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        if isinstance(sys_, R.RAPTV2):
            wa, flags = sys_.update(w, X[idx], y_all[idx], buf_X, buf_y, [w], [reg])
            refreshed = int(flags.get("is_refresh", 0))
        else:
            wa = sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-R.BUFFER_CAPACITY:],
                             y_buffer=buf_y[-R.BUFFER_CAPACITY:])
            refreshed = 0
        recs.append(R._rec(seed, kind, w, reg, y_all[idx], yp, wa, 0.0,
                           0, reused, refreshed, n_classes))
        yt_all.append(y_all[idx]); yp_all.append(yp)
    pw = pd.DataFrame(recs)
    pm = pooled_metrics(np.concatenate(yt_all), np.concatenate(yp_all), n_classes)
    pm.update(dict(dataset=sd["slug"], seed=seed, method=kind))
    return pw, pd.DataFrame([pm]), sys_


def _run_periodic(stream, sd, seed, kind, every):
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    driver = R.FRDriver(seed, n_trees=R.CHEAP_TREES, buffer=R.CHEAP_BUFFER,
                        trigger="periodic", every=every,
                        anchor_X=anchor_X, anchor_y=anchor_y)
    driver.fit_initial(X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = regimes[np.where(stream["window_id"].values == n_init - 1)[0][0]]
    recs = []
    yt_all, yp_all = [], []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        is_trans = reg != prev
        prev = reg
        adapt_cpu, did = driver.maybe_adapt(w, is_trans, buf_X, buf_y)
        yp = driver.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        recs.append(R._rec(seed, kind, w, reg, y_all[idx], yp, adapt_cpu, 0.0,
                           int(did), 0, 0, n_classes))
        yt_all.append(y_all[idx]); yp_all.append(yp)
    pw = pd.DataFrame(recs)
    pm = pooled_metrics(np.concatenate(yt_all), np.concatenate(yp_all), n_classes)
    pm.update(dict(dataset=sd["slug"], seed=seed, method=kind))
    return pw, pd.DataFrame([pm]), driver


def task1():
    stream, sd = get_stream(UGR16)
    pw_rows, po_rows = [], []
    for seed in SEEDS:
        for kind in ["RAPT", "RAPT-Cheap", "RAPT-Cheap-NoReset"]:
            pw, po, sys_ = _run_rapt(stream, sd, seed, kind)
            pw_rows.append(pw); po_rows.append(po)
        for kind, every in [("Periodic-Cheap-5", 5), ("Periodic-Cheap-10", 10)]:
            pw, po, driver = _run_periodic(stream, sd, seed, kind, every)
            pw_rows.append(pw); po_rows.append(po)
    pw = pd.concat(pw_rows, ignore_index=True)
    pooled = pd.concat(po_rows, ignore_index=True)
    pw.to_csv(os.path.join(HERE, "T1_ugr16_per_window.csv"), index=False)
    pooled.to_csv(os.path.join(HERE, "T1_ugr16_pooled.csv"), index=False)

    agg = pw.groupby("method").agg(
        per_window_macro_f1=("macro_f1", "mean"),
        adaptation_cpu_sec=("adaptation_cpu_sec", "sum"),
        retrains=("is_retrain", "sum"),
        reuses=("is_reuse", "sum"),
        refreshes=("is_refresh", "sum"),
    ).reset_index()
    agg["adaptation_cpu_sec"] = agg["adaptation_cpu_sec"] / len(SEEDS)
    agg.to_csv(os.path.join(HERE, "T1_ugr16_summary_mean.csv"), index=False)
    print(agg.to_string(index=False))

    codepaths = pd.DataFrame([
        dict(item="RAPT-Cheap counter variable", value="_windows_since_refresh (RAPTV2)",
             code_path="results/revalidation/run_stream_v2.py:93 init 0; :301 +=1; :314 compare >= refresh_every; :326 reset 0"),
        dict(item="RAPT-Cheap reset at regime transition", value="YES",
             code_path="results/revalidation/run_stream_v2.py:249 on_transition sets self._windows_since_refresh = 0"),
        dict(item="Periodic-Cheap counter variable", value="_since (FRDriver)",
             code_path="results/revalidation/run_stream_v2.py:357 init 0; :371 +=1; :372 compare >= every; :386 reset 0"),
        dict(item="Periodic-Cheap reset at regime transition", value="NO explicit reset",
             code_path="results/revalidation/run_stream_v2.py:340-388 FRDriver.maybe_adapt has no transition reset; _since counts every window (:371-372) and only resets after a refresh (:386)"),
        dict(item="UGR16 regime runs", value="all single-window",
             code_path="results/revalidation/raw/A1_per_window_all_streams.csv (every regime_id run length = 1)"),
        dict(item="consequence for RAPT-Cheap on UGR16", value="counter is reset to 0 every window, so refresh_every=5 is never reached; refresh_events=0",
             code_path="results/revalidation/A3_summary.csv [refresh_events=0 for ugr16 RAPT-Cheap, seeds 42-46]"),
        dict(item="consequence for Periodic-Cheap on UGR16", value="FRDriver already does not reset on transition, so its periodic cadence is unaffected by single-window regimes",
             code_path="results/revalidation/A3_summary.csv [Periodic-Cheap-5 retrain_events=28, Periodic-Cheap-10 retrain_events=14]"),
    ])
    codepaths.to_csv(os.path.join(HERE, "T1_refresh_counter_codepaths.csv"), index=False)

    reset = []
    a3 = pd.read_csv(os.path.join(REVAL, "A3_summary.csv"))
    for seed in SEEDS:
        r = a3[(a3.dataset == "ugr16") & (a3.method == "RAPT-Cheap") & (a3.seed == seed)]
        reset.append(dict(seed=seed, source="A3_summary.csv (committed)",
                          refresh_events=int(r.refresh_events.iloc[0]),
                          retrain_events=int(r.retrain_events.iloc[0]),
                          reuse_events=int(r.reuse_events.iloc[0])))
    pd.DataFrame(reset).to_csv(os.path.join(HERE, "T1_reset_test.csv"), index=False)


# ---------------------------------------------------------------------------
# Task 2: RAPT-Deferred vs RAPT on Campus
# ---------------------------------------------------------------------------
def task2():
    stream, sd = get_stream(CAMPUS)
    all_pw, all_prov, all_pooled = [], [], []
    for seed in SEEDS:
        prov_path = os.path.join(HERE, f"_prov_seed{seed}.csv")
        pw, sm, pooled, proba = R.run_seed_v2(
            stream, sd, seed, ["RAPT", "RAPT-Deferred"], provenance_out=prov_path)
        pw["seed"] = seed
        all_pw.append(pw)
        all_pooled.append(pooled)
        prov = pd.read_csv(prov_path)
        prov["seed"] = seed
        all_prov.append(prov)
        os.remove(prov_path)
    pw = pd.concat(all_pw, ignore_index=True)
    pooled = pd.concat(all_pooled, ignore_index=True)
    prov = pd.concat(all_prov, ignore_index=True)
    pw.to_csv(os.path.join(HERE, "T2_campus_per_window.csv"), index=False)
    prov.to_csv(os.path.join(HERE, "T2_campus_provenance.csv"), index=False)

    diff_rows = []
    for seed in SEEDS:
        s = pw[pw.seed == seed]
        a = s[s.method == "RAPT"].sort_values("window_id")
        b = s[s.method == "RAPT-Deferred"].sort_values("window_id")
        m = a.merge(b, on="window_id", suffixes=("_rapt", "_def"))
        differ = (m.cm_json_rapt != m.cm_json_def)
        diff_rows.append(dict(seed=seed, n_eval_windows=len(m),
                              windows_with_different_predictions=int(differ.sum()),
                              fraction=float(differ.mean())))
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(os.path.join(HERE, "T2_deferred_vs_rapt_windows.csv"), index=False)
    print(diff.to_string(index=False))

    # The deferred path stores a checkpoint only when a NEW regime is first seen.
    # Provenance logs reuse rows carrying the reused checkpoint's created_window;
    # created_window == 0 is the initial prefix policy, > 0 is a deferred-created
    # checkpoint (RAPTV2.update's deferred branch calls _store with the window id).
    reuse_rows = []
    for seed in SEEDS:
        s = prov[(prov.seed == seed) & (prov.event == "reuse")]
        for cw in sorted(int(x) for x in s.created_window.dropna().unique()):
            sub = s[s.created_window == cw]
            reuse_rows.append(dict(seed=seed, checkpoint_created_window=cw,
                                   deferred_created=int(cw > 0),
                                   n_reuse=len(sub),
                                   regime_keys=";".join(sorted(sub.regime_key.unique())),
                                   first_reuse_window=int(sub.reused_at_window.min()),
                                   max_age_windows=int(sub.age_windows.max())))
    reuse = pd.DataFrame(reuse_rows)
    reuse.to_csv(os.path.join(HERE, "T2_deferred_checkpoint_reuse.csv"), index=False)
    print(reuse.to_string(index=False))

    deferred_cw = sorted(int(x) for x in
                         prov[(prov.event == "reuse") & (prov.created_window > 0)]
                         .created_window.dropna().unique())
    summ = pd.DataFrame([
        dict(metric="total differing eval windows (5 seeds)",
             value=int(diff.windows_with_different_predictions.sum()),
             source="T2_deferred_vs_rapt_windows.csv"),
        dict(metric="total eval windows (5 seeds)",
             value=int(diff.n_eval_windows.sum()),
             source="T2_deferred_vs_rapt_windows.csv"),
        dict(metric="reuse events (5 seeds)",
             value=int((prov.event == "reuse").sum()),
             source="T2_campus_provenance.csv [event=reuse]"),
        dict(metric="distinct deferred-created checkpoints reused (per seed)",
             value=len(deferred_cw),
             source="T2_campus_provenance.csv [created_window>0 among reuse rows]"),
        dict(metric="deferred-created windows reused",
             value=";".join(str(c) for c in deferred_cw),
             source="T2_deferred_checkpoint_reuse.csv"),
        dict(metric="deferred checkpoint reused at all?",
             value="YES" if deferred_cw else "NO",
             source="T2_campus_provenance.csv [created_window>0]"),
    ])
    summ.to_csv(os.path.join(HERE, "T2_deferred_summary.csv"), index=False)


if __name__ == "__main__":
    task1()
    task2()
