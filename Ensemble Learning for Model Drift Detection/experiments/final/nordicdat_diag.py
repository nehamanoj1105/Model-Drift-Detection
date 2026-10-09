"""
STEP 1f — NordicDat diagnosis.

Checks for a bad target definition / label leakage and class balance, and
compares the default ensemble against class-weighted forests.

Writes results/final/tables/nordicdat_diagnosis.csv.
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

from final_config import SEEDS_DEV, DEFAULT_CFG, TABLES_FINAL
from final_methods import build_method, make_ensemble
from models_9a import make_class_anchor
from run_final import _prepare_stream, _prepare_arrays

NAME = "NordicDat"


def build_windows_for_seed(sd, seed, X, y, windows):
    pass


def run(seed, class_weight):
    stream, sd = _prepare_stream(NAME)
    (X, y, regimes, X_init, y_init, win_slices, n_init, n_windows) = _prepare_arrays(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    m = build_method("Frozen", seed=seed, anchor_X=anchor_X, anchor_y=anchor_y,
                     cfg=dict(DEFAULT_CFG))
    if class_weight:
        m.ens = make_ensemble(seed, class_weight=class_weight)
        m.ens.fit(X_init, y_init)
    else:
        m.fit_initial(X_init, y_init)
    from sklearn.metrics import f1_score
    f1s = []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        yp = m.ens.predict(X[idx])
        f1s.append(f1_score(y[idx], yp, average="macro", zero_division=0))
    return float(np.mean(f1s))


def main():
    stream, sd = _prepare_stream(NAME)
    feat = sd["feature_columns"]
    init = stream["window_id"].values < sd["initial_train_windows"]
    dist = stream["y"].value_counts(normalize=True).sort_index()
    rows = [dict(check="class_balance_full", value=float(dist.min()),
                 detail="minority class share: " +
                 ", ".join(f"{k}={v:.3f}" for k, v in dist.items()))]
    dinit = stream.loc[init, "y"].value_counts(normalize=True).sort_index()
    rows.append(dict(check="class_balance_prefix", value=float(dinit.min()),
                     detail=", ".join(f"{k}={v:.3f}" for k, v in dinit.items())))
    # leakage-ish: correlation of any feature with the target
    corr = {}
    for f in feat:
        v = stream[f].values.astype(float)
        if np.std(v) > 0:
            corr[f] = abs(np.corrcoef(v, stream["y"].values)[0, 1])
    top = sorted(corr.items(), key=lambda kv: -kv[1])[:4]
    rows.append(dict(check="max_abs_feature_target_corr", value=float(top[0][1]),
                     detail=", ".join(f"{k}={v:.3f}" for k, v in top)))
    for cw in [None, "balanced"]:
        f1 = np.mean([run(s, cw) for s in SEEDS_DEV[:3]])
        rows.append(dict(check=f"frozen_macro_f1_class_weight={cw}", value=f1,
                         detail="3 seeds, Frozen on NordicDat"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_FINAL, "nordicdat_diagnosis.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
