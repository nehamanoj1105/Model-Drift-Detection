"""E4 -- equivalence testing for the headline 'matches Full Retraining' claim.

The paper reports RAPT-Cheap vs Full Retraining as Delta=+0.0022 with p=0.3125 and
calls it 'within measurement noise'. A non-significant Wilcoxon test is not evidence
of equivalence, so we run a two-one-sided-tests (TOST) procedure and report the
smallest margin that would be supported.

Inputs are the committed Final_Experiments per-seed and per-window results.
"""
import csv
import collections
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.join(HERE, "..", "Final_Experiments", "results", "raw")

A, B = "RAPT_CHEAP", "Full Retraining"


def seed_f1(path, method):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["method"] == method:
            d[r["seed"]].append(float(r["macro_f1"]))
    return np.array([np.mean(v) for _, v in sorted(d.items())])


def window_f1(path, method):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["method"] == method:
            d[int(r["window_id"])].append(float(r["macro_f1"]))
    ws = sorted(d)
    return np.array([np.mean(d[w]) for w in ws])


def tost(a, b, margin):
    """Return (p_tost, 90% CI). Rejects non-equivalence iff CI within +/- margin."""
    diff = a - b
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    df = n - 1
    t_lo = (diff.mean() - (-margin)) / se      # H0: mu <= -margin
    t_hi = (margin - diff.mean()) / se         # H0: mu >= +margin
    p_lo = stats.t.sf(t_lo, df)
    p_hi = stats.t.sf(t_hi, df)
    p_tost = max(p_lo, p_hi)
    ci = stats.t.interval(0.90, df, loc=diff.mean(), scale=se)
    return p_tost, ci


for label, fn, n_note in [("seed-paired", "per_window_full.csv", "n=5"),
                          ("window-paired", "per_window_full.csv", "n=143")]:
    path = os.path.join(FE, fn)
    a = seed_f1(path, A) if label == "seed-paired" else window_f1(path, A)
    b = seed_f1(path, B) if label == "seed-paired" else window_f1(path, B)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diff = a - b
    print(f"\n{label} ({n_note}): Delta={diff.mean():+.5f} sd={diff.std(ddof=1):.5f}")
    print(f"  Wilcoxon p={stats.wilcoxon(a, b).pvalue:.4f}")
    for margin in (0.005, 0.01, 0.02, 0.05):
        p, ci = tost(a, b, margin)
        verdict = "EQUIVALENT" if p < 0.05 else "not shown"
        print(f"  TOST margin +/-{margin:.3f}: p={p:.4f} 90% CI "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] -> {verdict}")
