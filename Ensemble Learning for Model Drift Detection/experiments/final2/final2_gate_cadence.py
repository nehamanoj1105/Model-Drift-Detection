"""Task D2: long-window gate versus short-window gate, with numbers.

Two independent experiments, both on the dev split (seeds 42-46, first 60%):

(A) Cadence under within-regime concept drift (the stream where the regime id
    never changes). A short-window gate keyed on regime boundaries never fires;
    a long-window gate that is consulted every window does. We compare
    RAPT (boundary key) vs RAPT-ED-Guard (every-window trigger) on the
    constructed concept-drift stream and count refits.

(B) Feasibility on the natural streams: the same two policies, reporting
    macro-F1 and refit counts. This turns the paper's previously unsupported
    positive/negative claim into a measured table + figure.

Outputs results/final2/gate_cadence.csv and figures/fig_gate_cadence.png.
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

from final_config import SEEDS_DEV, DEV_WINDOW_FRACTION, BUFFER_CAPACITY
from final2_runner import prepare, SLUGS
from final2_methods import RAPTEDGuard
from final_methods import build_method
from run_final import _prepare_arrays
from models_9a import make_class_anchor
from sklearn.metrics import f1_score
import yaml

NATURAL = ["campus", "ugr16", "5g_nr", "kpiops"]


def _policies():
    defc = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    defc.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})
    return defc


def run_natural(slug, seed, cfg):
    stream, sd = prepare(SLUGS[slug])
    (X, y_all, regimes, X_init, y_init, ws, n_init, nw) = _prepare_arrays(stream, sd)
    ax, ay = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    stop = n_init + max(1, int(round((nw - n_init) * DEV_WINDOW_FRACTION)))
    out = {}
    for label, ctor in [("short/boundary (RAPT)", lambda: build_method(
                            "RAPT", seed=seed, anchor_X=ax, anchor_y=ay, cfg=cfg)),
                        ("long/every-window (RAPT-ED-Guard)", lambda: RAPTEDGuard(
                            seed, ax, ay, cfg))]:
        s = ctor(); s.fit_initial(init_regime=regimes[n_init - 1],
                                  X_init=X_init, y_init=y_init)
        bx, by = list(X_init), list(y_init); prev = regimes[n_init - 1]
        f1s, refits = [], 0
        for w in range(n_init, stop):
            idx = ws[w]; reg = regimes[w]; s.last_flags = {}
            yp, _ = s.predict_window(X[idx], bx, by)
            if isinstance(s, RAPTEDGuard):
                err = 1.0 - np.mean(yp == y_all[idx])
                s.post_predict(w, idx, X[idx], y_all[idx], bx, by, err=err, regime=reg)
            else:
                if reg != prev:
                    s.on_regime_change(reg, w, X[idx], y_all[idx], bx, by)
                s.post_predict(w, idx, X[idx], y_all[idx], bx, by)
            prev = reg
            refits += s.last_flags.get("retrain", 0)
            f1s.append(f1_score(y_all[idx], yp, average="macro", zero_division=0))
            bx.extend(X[idx]); by.extend(y_all[idx])
            if len(bx) > BUFFER_CAPACITY:
                bx, by = bx[-BUFFER_CAPACITY:], by[-BUFFER_CAPACITY:]
        out[label] = dict(stream=slug, policy=label, seed=seed,
                          macro_f1=float(np.mean(f1s)), refits=refits)
    return list(out.values())


def _concept_arrays():
    import exp9b_drift_config as C
    from drift_construct import load_stream, initial_train_count, build_concept_drift_stream
    from preprocessing_9b import FEATURE_COLS
    df0, sd = load_stream(C.STREAM_CSV, C.STREAM_JSON)
    n_init = initial_train_count(df0, sd)
    stream, dp, aff = build_concept_drift_stream(df0, sd, level=0.5, seed=42, n_init=n_init)
    feat = [c for c in FEATURE_COLS if c in stream.columns]
    X = stream[feat].values.astype(float)
    y = stream["qos_target"].values.astype(int)
    wid = stream["window_id"].values.astype(int)
    init = wid < n_init
    mu = X[init].mean(0); sdd = np.where(X[init].std(0) < 1e-9, 1.0, X[init].std(0))
    return (X - mu) / sdd, y, wid, n_init


def concept(slug, seed, cfg):
    X, y, wid, n_init = _concept_arrays()
    nw = int(wid.max()) + 1
    ws = {w: np.where(wid == w)[0] for w in range(n_init, nw)}
    im = wid < n_init
    ax, ay = make_class_anchor(X[im], y[im], per_class=15, seed=seed)
    out = []
    for label, ctor in [("short/boundary (RAPT)", lambda: build_method(
                            "RAPT", seed=seed, anchor_X=ax, anchor_y=ay, cfg=cfg)),
                        ("long/every-window (RAPT-ED-Guard)", lambda: RAPTEDGuard(
                            seed, ax, ay, cfg))]:
        s = ctor(); s.fit_initial(init_regime="r0", X_init=X[im], y_init=y[im])
        bx, by = list(X[im]), list(y[im]); prev = "r0"
        f1s, refits = [], 0
        for w in range(n_init, nw):
            idx = ws[w]; s.last_flags = {}
            yp, _ = s.predict_window(X[idx], bx, by)
            if isinstance(s, RAPTEDGuard):
                err = 1.0 - np.mean(yp == y[idx])
                s.post_predict(w, idx, X[idx], y[idx], bx, by, err=err, regime="r0")
            else:
                s.post_predict(w, idx, X[idx], y[idx], bx, by)
            refits += s.last_flags.get("retrain", 0)
            f1s.append(f1_score(y[idx], yp, average="macro", zero_division=0))
            bx.extend(X[idx]); by.extend(y[idx])
            if len(bx) > 1000:
                bx, by = bx[-1000:], by[-1000:]
        out.append(dict(stream="concept_drift", policy=label, seed=seed,
                        macro_f1=float(np.mean(f1s)), refits=refits))
    return out


def main():
    cfg = _policies()
    from joblib import Parallel, delayed
    jobs = [(s, seed) for s in NATURAL for seed in SEEDS_DEV] + \
           [("concept_drift", seed) for seed in SEEDS_DEV]
    def one(s, seed):
        return concept(s, seed, cfg) if s == "concept_drift" else run_natural(s, seed, cfg)
    t0 = time.time()
    res = Parallel(n_jobs=4, verbose=5)(delayed(one)(s, seed) for (s, seed) in jobs)
    rows = [r for sub in res for r in sub]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "final2", "gate_cadence_raw.csv"), index=False)
    agg = df.groupby(["stream", "policy"]).agg(
        macro_f1=("macro_f1", "mean"), refits=("refits", "sum")).reset_index()
    agg.to_csv(os.path.join(ROOT, "results", "final2", "gate_cadence.csv"), index=False)
    print(agg.to_string(index=False))
    _fig(agg)
    print(f"TOTAL {time.time()-t0:.0f}s")


def _fig(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    streams = ["concept_drift"] + NATURAL
    labels = {"concept_drift": "concept drift", "campus": "Campus",
              "ugr16": "UGR'16", "5g_nr": "5G NR", "kpiops": "KPI-ops"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    short = agg[agg.policy.str.startswith("short")]
    long_ = agg[agg.policy.str.startswith("long")]
    x = np.arange(len(streams)); w = 0.38
    def vals(d, col, s):
        return [d[d.stream == ss][col].mean() if len(d[d.stream == ss]) else 0 for ss in streams]
    axes[0].bar(x - w/2, vals(short, "refits", None), w, label="short/boundary")
    axes[0].bar(x + w/2, vals(long_, "refits", None), w, label="long/every-window")
    axes[0].set_xticks(x); axes[0].set_xticklabels([labels[s] for s in streams], rotation=20, fontsize=7)
    axes[0].set_ylabel("refits fired"); axes[0].set_title("Gate cadence", fontsize=9)
    axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x - w/2, vals(short, "macro_f1", None), w, label="short/boundary")
    axes[1].bar(x + w/2, vals(long_, "macro_f1", None), w, label="long/every-window")
    axes[1].set_xticks(x); axes[1].set_xticklabels([labels[s] for s in streams], rotation=20, fontsize=7)
    axes[1].set_ylabel("macro-F1"); axes[1].set_title("Accuracy", fontsize=9)
    axes[1].legend(fontsize=7); axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "results", "final2", "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "results", "final2", "figures", "fig_gate_cadence.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
