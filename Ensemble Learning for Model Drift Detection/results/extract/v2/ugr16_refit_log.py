#!/usr/bin/env python3
"""Task 2 -- UGR'16 novelty-refit training-row log for base RAPT and RAPT-Enhanced.

Replicates the *Table II* RAPT block of experiments/exp9a/three_dataset_run.py
run_seed (lines 197-233) verbatim and records, at every novelty refit, how many
rows are handed to the ensemble:

    buffer_rows  = len(buf_X[-refit_n:])  (what run_seed passes as X_buffer)
    recent_after_cap = rows kept by make_train_buffer before the anchor
    anchor_rows  = len(anchor_X)
    train_rows   = recent_after_cap + anchor_rows

Run from the project directory:
    python "results/extract/v2/ugr16_refit_log.py"

UGR'16 raw data is not committed; the script expects it at
experiments/exp9a/data/ugr16/UGR16v1.mat (see experiments/exp9a/data/DATASETS.md).
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
sys.path.insert(0, EXP9A)
sys.path.insert(0, os.path.join(ROOT, "results", "revalidation"))

import three_dataset_config as cfg            # noqa: E402
import three_dataset_load as L                # noqa: E402
import models_9a                              # noqa: E402
import exp9a_config as base                   # noqa: E402
from rapt_9a import RAPTSystem, RAPTEnhancedSystem  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
REFIT_N = {"RAPT": 500, "RAPT-Enhanced": base.ENHANCED_NOVELTY_REFIT_N}
BUFFER_CAPACITY = base.BUFFER_CAPACITY
OUT = HERE

_orig_make_train_buffer = models_9a.make_train_buffer


def instrumented_make_train_buffer(recent_X, recent_y, anchor_X, anchor_y, cap):
    """Record the row counts, then defer to the real function."""
    recent_X = np.asarray(recent_X)
    n_recent_before_cap = len(recent_X)
    n_anchor = 0 if anchor_X is None else len(anchor_X)
    out = _orig_make_train_buffer(recent_X, recent_y, anchor_X, anchor_y, cap)
    train_X = out[0]
    return out


def run_seed_rapt(stream, sd, seed, method, refit_n, log):
    feat = sd["feature_columns"]
    X_all = stream[feat].values.astype(np.float64)
    y_all = stream["y"].values.astype(int)
    regimes = stream["regime_id"].values
    n_init = sd["initial_train_windows"]
    n_windows = sd["total_windows"]
    init_mask = stream["window_id"].values < n_init
    X_init_raw = X_all[init_mask]
    y_init = y_all[init_mask]
    mu = X_init_raw.mean(axis=0)
    sd_ = np.where(X_init_raw.std(axis=0) < 1e-9, 1.0, X_init_raw.std(axis=0))
    X = (X_all - mu) / sd_
    X_init = X[init_mask]
    win_slices = {w: np.where(stream["window_id"].values == w)[0]
                  for w in range(n_init, n_windows)}
    anchor_X, anchor_y = models_9a.make_class_anchor(X_init, y_init, per_class=15, seed=seed)

    cls = RAPTSystem if method == "RAPT" else RAPTEnhancedSystem
    kwargs = dict(seed=seed, mode="full", enable_calibration=True,
                  anchor_X=anchor_X, anchor_y=anchor_y,
                  buffer_capacity=BUFFER_CAPACITY)
    if method == "RAPT-Enhanced":
        kwargs["novelty_refit_n"] = refit_n
    sys_ = cls(**kwargs)

    wid = stream["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    sys_.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev_regime = init_regime
    refit_no = 0

    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        if reg != prev_regime:
            refit_no += 1
            xb = buf_X[-refit_n:]
            yb = buf_y[-refit_n:]
            n_buffer_rows = len(xb)
            n_anchor = len(anchor_X)
            # replicate make_train_buffer to get the post-cap recent count
            recent = np.asarray(xb)
            if len(recent) > BUFFER_CAPACITY:
                recent = recent[-BUFFER_CAPACITY:]
            n_recent_after_cap = len(recent)
            train_rows = n_recent_after_cap + (n_anchor if n_anchor else 0)
            reused, wa, _ = sys_.handle_regime_transition(reg, w, X_buffer=xb, y_buffer=yb)
            log.append(dict(seed=seed, method=method, refit_no=refit_no,
                            window_id=w, regime_id=str(reg), reused=int(reused),
                            refit_n=refit_n, buffer_rows=n_buffer_rows,
                            recent_after_cap=n_recent_after_cap,
                            anchor_rows=n_anchor, train_rows=train_rows,
                            buf_len_before=len(buf_X)))
            prev_regime = reg
        yp = sys_.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > BUFFER_CAPACITY:
            buf_X, buf_y = buf_X[-BUFFER_CAPACITY:], buf_y[-BUFFER_CAPACITY:]
        sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
    return log


def main():
    mat = os.path.join(EXP9A, "data", "ugr16", "UGR16v1.mat")
    if not os.path.exists(mat):
        raise SystemExit(f"UGR'16 raw data missing at {mat} (see experiments/exp9a/data/DATASETS.md)")
    data = L.LOADERS["UGR'16"]()
    stream, sd = L.build_windows(data)
    log = []
    for seed in SEEDS:
        for method in ("RAPT", "RAPT-Enhanced"):
            run_seed_rapt(stream, sd, seed, method, REFIT_N[method], log)
    df = pd.DataFrame(log)
    df["is_novelty_refit"] = (df["reused"] == 0).astype(int)
    df.to_csv(os.path.join(OUT, "T2_ugr16_refit_rows.csv"), index=False)

    nov = df[df["is_novelty_refit"] == 1]
    summary = nov.groupby(["method", "refit_n"]).agg(
        n_novelty_refits=("refit_no", "count"),
        buffer_rows_min=("buffer_rows", "min"),
        buffer_rows_max=("buffer_rows", "max"),
        recent_after_cap_min=("recent_after_cap", "min"),
        recent_after_cap_max=("recent_after_cap", "max"),
        anchor_rows=("anchor_rows", "max"),
        train_rows_min=("train_rows", "min"),
        train_rows_max=("train_rows", "max"),
    ).reset_index()
    summary.to_csv(os.path.join(OUT, "T2_ugr16_refit_summary.csv"), index=False)
    print(summary.to_string(index=False))

    # exact code paths
    paths = pd.DataFrame([
        dict(item="base RAPT novelty refit size", value="refit_n = 500",
             code_path="experiments/exp9a/three_dataset_run.py:198 -> for method, cls, refit_n in ((\"RAPT\", RAPTSystem, 500), ...)"),
        dict(item="RAPT-Enhanced novelty refit size", value="refit_n = ENHANCED_NOVELTY_REFIT_N = 1500",
             code_path="experiments/exp9a/exp9a_config.py:86 ; passed at three_dataset_run.py:199/204"),
        dict(item="buffer slice passed to handle_regime_transition",
             value="buf_X[-refit_n:]  -> base RAPT 500 rows, Enhanced 1500 rows",
             code_path="experiments/exp9a/three_dataset_run.py:217"),
        dict(item="base RAPT _train_policy",
             value="make_train_buffer(X_buffer, y_buffer, anchor, anchor, buffer_capacity=1000) -> recent=500, then +30 anchor = 530",
             code_path="experiments/exp9a/rapt_9a.py:138-145"),
        dict(item="RAPT-Enhanced _train_policy",
             value="recent = X_buffer[-novelty_refit_n:] then make_train_buffer(..., buffer_capacity=1000) -> recent=1000, then +30 anchor = 1030",
             code_path="experiments/exp9a/rapt_9a.py:192-200"),
        dict(item="streaming buffer cap", value="BUFFER_CAPACITY = 1000",
             code_path="experiments/exp9a/exp9a_config.py:70 ; enforced at three_dataset_run.py:222-223"),
        dict(item="anchor size", value="30 rows (15 per class x 2 classes)",
             code_path="experiments/exp9a/models_9a.py:70-86 make_class_anchor(per_class=15)"),
        dict(item="WHY THEY DIFFER (the cap-1000 interaction)",
             value="base RAPT sees 500 rows; Enhanced sees 1000 rows",
             code_path=("Both use buffer_capacity=1000 inside make_train_buffer, so both look capped at 1000. "
                        "But the two slices differ: run_seed passes buf_X[-500:] for RAPT (line 217, refit_n=500) and "
                        "buf_X[-1500:] for Enhanced (refit_n=1500). The streaming buffer is itself capped at 1000 "
                        "(line 222-223), so Enhanced's -1500 slice yields min(1500, buffer_len)=1000 rows, whereas "
                        "RAPT's -500 slice yields 500 rows. make_train_buffer's own cap (1000) never binds for either: "
                        "500<1000 and 1000<=1000. Hence 500+30=530 vs 1000+30=1030 training rows.")),
        dict(item="when the buffer reaches 1000", value="only after window 36+500/240... i.e. after the first 4-5 windows",
             code_path="At window 36 (first transition) buf_len_before=8640 (initial prefix 36 windows x 240); after the cap it is 1000 for the rest of the stream."),
    ])
    paths.to_csv(os.path.join(OUT, "T2_ugr16_refit_codepaths.csv"), index=False)
    print("wrote T2_ugr16_refit_rows.csv, T2_ugr16_refit_summary.csv, T2_ugr16_refit_codepaths.csv")


if __name__ == "__main__":
    main()
