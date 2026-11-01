#!/usr/bin/env python3
"""
make_stationarity.py — UGR'16 covariate-vs-concept evidence.

For each held-out window w (seeded at the first threshold-window in the stream):
  * covariate stability: two-sample KS and PSI of each discriminative feature
    between the pooled pre-window data (initial train + all earlier windows) and
    window w.
  * label map drift: train a shallow decision tree on the pre-window data and
    measure its macro-F1 on window w ("target-semantics proxy"). A collapse here
    with stable features is the signature of a changed P(Y|X).

Writes results/final/ugr_stationarity.csv and NO future samples are used for any
threshold (all thresholds are derived from the pre-window pool only).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))

from data_loaders import build_stream  # noqa: E402
from runner import prepare  # noqa: E402


def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    e = np.histogram(expected, bins=qs)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=qs)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final.yaml")
    ap.add_argument("--dataset", default="UGR'16")
    args = ap.parse_args()
    with open(os.path.join(HERE, args.config)) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")
    os.makedirs(out, exist_ok=True)

    stream, sd = build_stream(args.dataset, cfg)
    X, y, regimes, _, _, slices, n_init, n_windows = prepare(stream, sd)
    rows_per_win = sd["window_rows"]
    n_init_rows = n_init * rows_per_win
    pre_X = [X[:n_init_rows]]; pre_y = [y[:n_init_rows]]

    # pick the two windows where the stored-policy proxy drops most
    proxies = []
    for w in range(n_init, n_windows):
        idx = slices[w]
        pool_X = np.vstack(pre_X); pool_y = np.concatenate(pre_y)
        clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(pool_X, pool_y)
        proxies.append((w, float(f1_score(y[idx], clf.predict(X[idx]),
                                          average="macro", zero_division=0))))
        pre_X.append(X[idx]); pre_y.append(y[idx])
    proxies = pd.DataFrame(proxies, columns=["window_idx", "proxy_f1"])

    rows = []
    for w, proxy in proxies.itertuples(index=False):
        idx = slices[w]
        pool_X = np.vstack([X[:n_init_rows]] + [X[slices[k]] for k in range(n_init, w)])
        pool_y = np.concatenate([y[:n_init_rows]] + [y[slices[k]] for k in range(n_init, w)])
        nonconst = [j for j in range(X.shape[1])
                    if np.std(pool_X[:, j]) > 1e-9 and np.std(X[idx, j]) > 1e-9]
        ks = [ks_2samp(pool_X[:, j], X[idx, j]).statistic for j in nonconst]
        psis = [psi(pool_X[:, j], X[idx, j]) for j in nonconst]
        rows.append({"window_idx": w, "proxy_f1": proxy,
                     "mean_ks": float(np.mean(ks)) if ks else np.nan,
                     "max_ks": float(np.max(ks)) if ks else np.nan,
                     "mean_psi": float(np.mean(psis)) if psis else np.nan,
                     "max_psi": float(np.max(psis)) if psis else np.nan,
                     "n_features_compared": len(nonconst)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "ugr_stationarity.csv"), index=False)
    print("ugr_stationarity.csv written:", df.shape)


if __name__ == "__main__":
    main()
