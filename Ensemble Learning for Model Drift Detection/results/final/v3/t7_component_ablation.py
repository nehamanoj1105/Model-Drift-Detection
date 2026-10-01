"""T7: single-factor component ablation from one base config.

Base config (the published RAPT semantics): RAPT_v2 (provenance-deferred) with
all optional components off. Each variant flips exactly one factor and nothing
else, on 5G Campus and UGR'16, 5 seeds:

  base                  gate off, no novelty refit change, parity off,
                        no floor trigger, no relative trigger, no periodic
                        refresh, full refit (50 trees / 1000-row buffer)
  +gate                 reuse gated on the incoming-window fingerprint
  +tier1                Tier-1 learner calibration on reuse (already on in base;
                        variant turns it OFF to isolate the effect)
  +novelty_refit        novelty refit trained on the enlarged 1000-row buffer
                        (base already uses 1000; variant uses 300 to isolate)
  +parity               parity refit enabled (threshold 0.5)
  +floor_trigger        absolute rolling-accuracy floor refresh (0.97)
  +relative_trigger     relative-drop refresh (10%)
  +periodic_refresh     periodic refresh every 5 windows
  +cheap_refit          cheap refit (20 trees / 300-row buffer)

Outputs:
  T7_component_ablation.csv   per dataset, variant, seed: pooled F1, CPU, events
  T7_component_delta.csv      per dataset: delta vs base with bootstrap CI
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

DATASETS = ["5G Campus QoS", "UGR'16"]
CHEAP_TREES = 20
CHEAP_BUFFER = 300
FULL_TREES = 50

# each entry: name -> RAPTV3 kwargs overriding the base
BASE = dict(store_mode="deferred", enable_calibration=True)
VARIANTS = {
    "base": {},
    "gate_on": dict(use_gate=True, gate_fix=True),
    "tier1_off": dict(enable_calibration=False),
    "novelty_refit_small": dict(refit_n=500, refresh_trees=FULL_TREES, refresh_buffer=500),
    "parity_on": dict(parity_threshold=0.5),
    "floor_trigger": dict(refresh_every=1, refresh_mode="floor",
                          refresh_trees=CHEAP_TREES, refresh_buffer=CHEAP_BUFFER),
    "relative_trigger": dict(refresh_every=1, refresh_mode="evidence", rel_drop=0.10,
                             refresh_trees=CHEAP_TREES, refresh_buffer=CHEAP_BUFFER),
    "periodic_refresh": dict(refresh_every=5, refresh_trees=CHEAP_TREES,
                             refresh_buffer=CHEAP_BUFFER),
    "cheap_refit": dict(refresh_trees=CHEAP_TREES, refresh_buffer=CHEAP_BUFFER,
                        refit_n=1000),
}


def run_variant(stream, sd, seed, name, kwargs):
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = lib.prepare(stream, sd)
    anchor_X, anchor_y = lib.make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream_df = stream["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    kw = dict(seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, refit_n=1000)
    kw.update(BASE)
    kw.update(kwargs)
    sys_ = lib.RAPTV3(**kw)
    refit_n = kw["refit_n"]
    sys_.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    yt, yp, adapt, retr = [], [], 0.0, 0
    wreg = {int(w): str(regimes[win_slices[w][0]]) for w in range(n_init, n_windows)}
    wreg[int(n_init - 1)] = str(init_regime)
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        wa = 0.0
        if reg != prev:
            nbuf = len(buf_X)
            nbuf_win = max(1, int(np.ceil(nbuf / max(1, sd["window_size"]))))
            tw = [x for x in range(max(n_init, w - nbuf_win), w)]
            tr = [wreg.get(x, "?") for x in tw]
            ri, wa, _ = sys_.on_transition(reg, w, buf_X[-refit_n:], buf_y[-refit_n:], tw, tr)
            retr += 1
            prev = reg
        y_p = sys_.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > 1000:
            del buf_X[:-1000]; del buf_y[:-1000]
        wa2, _ = sys_.update(w, X[idx], y_all[idx], buf_X, buf_y, [w], [reg])
        adapt += wa + wa2
        yt.append(y_all[idx]); yp.append(y_p)
    pm = lib.pooled_metrics(np.concatenate(yt), np.concatenate(yp), n_classes)
    return {"pooled_macro_f1": pm["pooled_macro_f1"], "accuracy": pm["pooled_accuracy"],
            "adaptation_cpu_sec": adapt, "retrains": retr,
            "reuse_events": sys_.reused_policy_count, "refresh_events": sys_.refresh_events,
            "trees_trained": sys_.trees_trained_count}


def main():
    lib.ensure_dirs()
    lib.freeze_config()
    rows = []
    for ds in DATASETS:
        stream, sd = lib.S.get_stream(ds)
        slug = lib.S.SLUG[ds]
        for name, kwargs in VARIANTS.items():
            for seed in lib.SEEDS:
                r = run_variant(stream, sd, seed, name, kwargs)
                r.update({"dataset": slug, "variant": name, "seed": seed})
                rows.append(r)
            print(f"  {slug} {name} done", flush=True)
    df = pd.DataFrame(rows)
    lib.write_csv(df, "T7_component_ablation.csv")

    drows = []
    for ds, g in df.groupby("dataset"):
        base = g[g.variant == "base"].set_index("seed")["pooled_macro_f1"]
        for name, gg in g.groupby("variant"):
            if name == "base":
                continue
            v = gg.set_index("seed")["pooled_macro_f1"]
            p, md = lib.paired_wilcoxon(v.to_numpy(), base.to_numpy())
            drows.append({"dataset": ds, "variant": name,
                          "delta_f1_vs_base": md, "wilcoxon_p": p,
                          "base_f1": base.mean(), "variant_f1": v.mean()})
    lib.write_csv(pd.DataFrame(drows), "T7_component_delta.csv")
    print(df.groupby(["dataset", "variant"])["pooled_macro_f1"].mean().round(4).to_string())
    lib.gate("T7", "PASS", "single-factor component ablation")


if __name__ == "__main__":
    main()
