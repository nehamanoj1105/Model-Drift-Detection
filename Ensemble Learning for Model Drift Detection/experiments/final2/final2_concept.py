"""Task D4: non-degenerate within-regime concept-drift stress panel.

Activates the in-regime absolute-floor trigger for every method so the four
trajectories separate, then measures how distinct they are. Reports the
per-window trajectory spread; if the spread stays below a small threshold the
honest answer is "the panel is degenerate" and we drop it.

Outputs results/final2/concept_panel_raw.csv and concept_panel.csv.
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
          os.path.join(ROOT, "experiments", "exp9a"),
          os.path.join(ROOT, "experiments", "exp9b")):
    sys.path.insert(0, p)

from final_config import SEEDS_DEV, DEFAULT_CFG
from final_methods import build_method
from final2_methods import RAPTEDGuard
from models_9a import make_class_anchor
from sklearn.metrics import f1_score
import exp9b_drift_config as CFG9B
from drift_construct import load_stream, initial_train_count, build_concept_drift_stream
from preprocessing_9b import FEATURE_COLS
import yaml

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "ADWIN",
           "Page-Hinkley", "RAPT", "RAPT-Guard", "RAPT-ED-Guard"]
SEVERITIES = [0.2, 0.5, 1.0]


def _arrays(df, n_init):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat].values.astype(float)
    y = df["qos_target"].values.astype(int)
    init = df["window_id"].values < n_init
    mu = X[init].mean(0); sd = np.where(X[init].std(0) < 1e-9, 1.0, X[init].std(0))
    return (X - mu) / sd, y, df["window_id"].values.astype(int)


def run_stream(df, n_init, seed, cfg):
    X, y, wid = _arrays(df, n_init)
    n_windows = int(wid.max()) + 1
    win_slices = {w: np.where(wid == w)[0] for w in range(n_init, n_windows)}
    im = wid < n_init
    X_init, y_init = X[im], y[im]
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    rows = []
    for m in METHODS:
        if m == "RAPT-ED-Guard":
            s = RAPTEDGuard(seed, anchor_X, anchor_y, cfg)
        else:
            s = build_method(m, seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, cfg=cfg)
        s.fit_initial(X_init=X_init, y_init=y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = None
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            s.last_flags = {}
            yp, pc = s.predict_window(X[idx], buf_X, buf_y)
            if m == "RAPT-ED-Guard":
                err = 1.0 - np.mean(yp == y[idx])
                ac = s.post_predict(w, idx, X[idx], y[idx], buf_X, buf_y, err=err)
            else:
                ac = s.post_predict(w, idx, X[idx], y[idx], buf_X, buf_y)
            buf_X.extend(X[idx]); buf_y.extend(y[idx])
            if len(buf_X) > 1000:
                buf_X, buf_y = buf_X[-1000:], buf_y[-1000:]
            rows.append(dict(method=m, window_id=w, seed=seed,
                             macro_f1=f1_score(y[idx], yp, average="macro", zero_division=0),
                             is_retrain=s.last_flags.get("retrain", 0)))
    return pd.DataFrame(rows)


def main():
    defc = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    defc.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})
    cfg = dict(defc)
    cfg["enable_floor"] = True     # the fix for the degenerate panel

    df0, sd = load_stream(CFG9B.STREAM_CSV, CFG9B.STREAM_JSON)
    n_init = initial_train_count(df0, sd)
    all_rows = []
    for lvl in SEVERITIES:
        for seed in SEEDS_DEV:
            stream, dp, aff = build_concept_drift_stream(df0, sd, level=lvl, seed=42, n_init=n_init)
            r = run_stream(stream, n_init, seed, cfg)
            r["severity"] = lvl
            all_rows.append(r)
        print("severity", lvl, "done", flush=True)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(os.path.join(ROOT, "results", "final2", "concept_panel_raw.csv"), index=False)

    summ = df.groupby(["severity", "method"]).agg(
        macro_f1=("macro_f1", "mean"), retrains=("is_retrain", "sum")).reset_index()
    summ.to_csv(os.path.join(ROOT, "results", "final2", "concept_panel.csv"), index=False)

    # trajectory spread: mean absolute deviation between method curves
    spreads = {}
    for lvl in SEVERITIES:
        p = df[df.severity == lvl].pivot_table(index="window_id", columns="method",
                                               values="macro_f1", aggfunc="mean")
        # mean over windows of (max - min) across methods
        spreads[lvl] = float((p.max(axis=1) - p.min(axis=1)).mean())
    print("trajectory spread (mean max-min macro-F1 across methods):")
    for k, v in spreads.items():
        print(f"  severity {k}: {v:.3f}")
    pd.DataFrame({"severity": list(spreads), "spread": list(spreads.values())}).to_csv(
        os.path.join(ROOT, "results", "final2", "concept_spread.csv"), index=False)
    print(summ.to_string(index=False))
    print("\nSPREAD (mean max-min across methods):")
    for k, v in spreads.items():
        print(f"  severity {k}: {v:.4f}")
    for lvl in SEVERITIES:
        print(f"\nseverity {lvl}:")
        print(summ[summ.severity == lvl].to_string(index=False))


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"done {time.time()-t0:.0f}s")
