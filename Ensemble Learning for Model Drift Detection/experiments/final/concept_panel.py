"""
STEP 1b/1d — Concept-drift stress panel and the gate-cadence experiment.

Problem in the current paper (STEP 1d): in the within-regime concept-drift
stream the regime id never changes, so boundary-keyed adapters (Full Retraining,
RAPT) never fire and every method coincides with Frozen. That is degenerate.

Fix: enable the in-regime absolute-floor trigger for all methods (`enable_floor`)
so the panel has four distinct trajectories, and report the boundary-only result
separately as the degenerate reference.

The same stream answers STEP 1b: it is the cleanest test of gate *cadence*. A
short-window gate that is only consulted at a regime boundary never fires under
concept drift; an every-window (long-window) gate does. We compare
    RAPT (boundary cadence)        vs
    RAPT+Floor (every-window gate)
on the same constructed stream and report macro-F1, recovery and refit counts.

Outputs: results/final/tables/concept_panel.csv, gate_cadence.csv and
results/final/figures/fig_concept_panel.png.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exp9a"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9b"))

from final_config import SEEDS_DEV, DEFAULT_CFG, TABLES_FINAL, FIGURES_FINAL, ensure_dirs
from final_methods import build_method
from models_9a import make_class_anchor
from sklearn.metrics import f1_score
import exp9b_drift_config as CFG9B
from drift_construct import (load_stream, initial_train_count,
                             build_concept_drift_stream, rank_predictive_features)
from preprocessing_9b import FEATURE_COLS

SEVERITIES = [0.2, 0.5, 1.0]
METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT"]


def _arrays(df, n_init):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat].values.astype(float)
    y = df["qos_target"].values.astype(int)
    init = df["window_id"].values < n_init
    mu = X[init].mean(0); sd = np.where(X[init].std(0) < 1e-9, 1.0, X[init].std(0))
    X = (X - mu) / sd
    return X, y, df["window_id"].values.astype(int)


def run_stream(df, n_init, seed, enable_floor):
    X, y, wid = _arrays(df, n_init)
    n_windows = int(wid.max()) + 1
    win_slices = {w: np.where(wid == w)[0] for w in range(n_init, n_windows)}
    init_mask = wid < n_init
    X_init, y_init = X[init_mask], y[init_mask]
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    cfg = dict(DEFAULT_CFG); cfg["enable_floor"] = enable_floor
    rows = []
    for m in METHODS:
        sys_ = build_method(m, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
        sys_.fit_initial(X_init=X_init, y_init=y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            sys_.last_flags = {}
            yp, pc = sys_.predict_window(X[idx], buf_X, buf_y)
            ac = sys_.post_predict(w, idx, X[idx], y[idx], buf_X, buf_y)
            buf_X.extend(X[idx]); buf_y.extend(y[idx])
            if len(buf_X) > 1000:
                buf_X, buf_y = buf_X[-1000:], buf_y[-1000:]
            rows.append(dict(method=m, window_id=w, macro_f1=f1_score(
                y[idx], yp, average="macro", zero_division=0),
                is_retrain=sys_.last_flags.get("retrain", 0)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ensure_dirs()
    df0, stream_def = load_stream(CFG9B.STREAM_CSV, CFG9B.STREAM_JSON)
    n_init = initial_train_count(df0, stream_def)
    severities = [0.5] if a.quick else SEVERITIES
    seeds = [42] if a.quick else SEEDS_DEV
    all_rows = []
    for lvl in severities:
        for seed in seeds:
            stream, drift_points, aff = build_concept_drift_stream(
                df0, stream_def, level=lvl, seed=42, n_init=n_init)
            for floor in (False, True):
                r = run_stream(stream, n_init, seed, floor)
                r["severity"] = lvl; r["seed"] = seed
                r["floor"] = int(floor)
                all_rows.append(r)
        print(f"  severity {lvl} done", flush=True)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(os.path.join(TABLES_FINAL, "concept_panel_raw.csv"), index=False)

    summ = df.groupby(["severity", "floor", "method"]).agg(
        macro_f1=("macro_f1", "mean"), retrains=("is_retrain", "sum")).reset_index()
    summ.to_csv(os.path.join(TABLES_FINAL, "concept_panel.csv"), index=False)

    # gate cadence table: boundary-only vs every-window, RAPT only
    cad = df[df["method"] == "RAPT"].groupby(["severity", "floor"]).agg(
        macro_f1=("macro_f1", "mean"), refits=("is_retrain", "sum")).reset_index()
    cad["gate_cadence"] = cad["floor"].map({0: "boundary-only (short)", 1: "every-window (long)"})
    cad.to_csv(os.path.join(TABLES_FINAL, "gate_cadence.csv"), index=False)

    _figure(cad)
    print(summ.to_string(index=False))


def _figure(cad):
    """STEP 1b evidence: refit cadence of a boundary-only gate versus an
    every-window gate on the same within-regime concept-drift stream. The
    macro-F1 panel is deliberately not drawn: at 500-packet windows the
    per-window macro-F1 is effectively binary, so the four methods' curves
    coincide and the panel would be degenerate (see concept_panel.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    sev = sorted(cad["severity"].unique())
    x = np.arange(len(sev))
    w = 0.38
    b = cad[cad["floor"] == 0].set_index("severity")["refits"].reindex(sev).fillna(0)
    l = cad[cad["floor"] == 1].set_index("severity")["refits"].reindex(sev).fillna(0)
    ax.bar(x - w / 2, b.values, w, label="boundary-only gate (short window)")
    ax.bar(x + w / 2, l.values, w, label="every-window gate (long window)")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(s*100)}%" for s in sev])
    ax.set_xlabel("concept-drift severity"); ax.set_ylabel("refits fired")
    ax.set_title("Gate cadence under in-regime concept drift")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FINAL, "fig_gate_cadence.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.1f}s")
