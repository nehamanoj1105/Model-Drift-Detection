"""Recompute the p-values and Cohen's d quoted in Sections V-F/V-H."""
import csv
import collections
import os

import numpy as np
from scipy import stats

R9A3 = os.path.join(os.path.dirname(__file__), "..", "results",
                    "experiment_9a_three", "raw")
STREAMS = {
    "Campus": "per_window_5g_campus_full.csv",
    "UGR16": "per_window_ugr16_full.csv",
    "Nordic": "per_window_nordicdat_full.csv",
}


def win_series(fname, method):
    """mean over seeds, per window (window-paired vector)."""
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(R9A3, fname))):
        if r["method"] == method:
            d[int(r["window_id"])].append(float(r["macro_f1"]))
    ws = sorted(d)
    return np.array([np.mean(d[w]) for w in ws])


def seed_series(fname, method):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(R9A3, fname))):
        if r["method"] == method:
            d[r["seed"]].append(float(r["macro_f1"]))
    return np.array([np.mean(v) for _, v in sorted(d.items())])


def report(name, a, b, label):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    p = stats.wilcoxon(a, b).pvalue
    # Cohen's d, several conventions
    d_paired = diff.mean() / diff.std(ddof=1)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d_pooled = diff.mean() / pooled
    d_pooled0 = diff.mean() / np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
    print(f"  {name:7s} {label:28s} n={n:3d} meanDelta={diff.mean():+.4f} "
          f"p={p:.3g} d_paired={d_paired:+.3f} d_pooled={d_pooled:+.3f} "
          f"d_pooled0={d_pooled0:+.3f}")


print("WINDOW-PAIRED (mean over seeds per window)")
for name, fn in STREAMS.items():
    report(name, win_series(fn, "RAPT"), win_series(fn, "Full Retraining"),
           "RAPT vs FullRetrain")
report("UGR16", win_series(STREAMS["UGR16"], "RAPT-Enhanced"),
       win_series(STREAMS["UGR16"], "RAPT"), "Enhanced vs RAPT")

print("\nSEED-PAIRED (n=5)")
for name, fn in STREAMS.items():
    report(name, seed_series(fn, "RAPT"), seed_series(fn, "Full Retraining"),
           "RAPT vs FullRetrain")
report("UGR16", seed_series(STREAMS["UGR16"], "RAPT-Enhanced"),
       seed_series(STREAMS["UGR16"], "RAPT"), "Enhanced vs RAPT")

print("\nPAPER QUOTES: Campus p=2.6e-4 d=-0.32 | UGR16 p=2.6e-11 d=-0.67")
print("              UGR16 Enh-vs-RAPT p=5.4e-9 d=0.56 | Nordic p=0.255 d=-0.12")
