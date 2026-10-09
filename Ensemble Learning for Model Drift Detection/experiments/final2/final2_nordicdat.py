"""Task F: class-weighted forest diagnosis on the telecom streams.

For each stream, compare the default ensemble against a class-weighted
('balanced') ensemble under the Frozen protocol, on the dev seeds, and record
class balance, prefix balance and the strongest feature-target correlation so
leakage / bad-target concerns can be ruled in or out.

Writes results/final2/nordicdat_diagnosis.csv (covers all four streams).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "experiments", "final"),
          os.path.join(ROOT, "experiments", "exp9a")):
    sys.path.insert(0, p)

from final_config import SEEDS_DEV, BUFFER_CAPACITY
from final2_runner import prepare, SLUGS
from final_methods import build_method, make_ensemble
from run_final import _prepare_arrays
from models_9a import make_class_anchor
from sklearn.metrics import f1_score
import yaml

STREAMS = ["campus", "ugr16", "5g_nr", "kpiops"]


def run_frozen(seed, class_weight, cfg, name):
    stream, sd = prepare(SLUGS[name])
    (X, y, regimes, X_init, y_init, ws, n_init, nw) = _prepare_arrays(stream, sd)
    ax, ay = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    m = build_method("Frozen", seed=seed, anchor_X=ax, anchor_y=ay, cfg=cfg)
    if class_weight:
        m.ens = make_ensemble(seed, class_weight=class_weight)
        m.ens.fit(X_init, y_init)
    else:
        m.fit_initial(X_init, y_init)
    f1s = []
    for w in range(n_init, nw):
        idx = ws[w]
        f1s.append(f1_score(y[idx], m.ens.predict(X[idx]), average="macro", zero_division=0))
    return float(np.mean(f1s))


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"))) or {}
    cfg.update(yaml.safe_load(open(os.path.join(ROOT, "configs", "final2.yaml"))) or {})
    rows = []
    for name in STREAMS:
        stream, sd = prepare(SLUGS[name])
        feat = sd["feature_columns"]
        y = stream["y"].values
        init = stream["window_id"].values < sd["initial_train_windows"]
        dist = pd.Series(y).value_counts(normalize=True).sort_index()
        rows.append(dict(stream=name, check="class_balance_full",
                         value=float(dist.min()),
                         detail=", ".join(f"{k}={v:.3f}" for k, v in dist.items())))
        di = pd.Series(y[init]).value_counts(normalize=True).sort_index()
        rows.append(dict(stream=name, check="class_balance_prefix",
                         value=float(di.min()),
                         detail=", ".join(f"{k}={v:.3f}" for k, v in di.items())))
        corr = {}
        for f in feat:
            v = stream[f].values.astype(float)
            if np.std(v) > 1e-12:
                corr[f] = abs(np.corrcoef(v, y)[0, 1])
        top = sorted(corr.items(), key=lambda kv: -kv[1])[:3]
        rows.append(dict(stream=name, check="max_abs_feature_target_corr",
                         value=float(top[0][1]),
                         detail=", ".join(f"{k}={v:.3f}" for k, v in top)))
        for cw in [None, "balanced"]:
            vals = [run_frozen(s, cw, cfg, name) for s in SEEDS_DEV[:3]]
            rows.append(dict(stream=name, check=f"frozen_macro_f1_class_weight={cw}",
                             value=float(np.mean(vals)),
                             detail=f"std={np.std(vals):.4f}, 3 dev seeds"))
        print(name, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "final2", "nordicdat_diagnosis.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
