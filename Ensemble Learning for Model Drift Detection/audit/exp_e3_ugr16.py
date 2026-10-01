"""E3 -- UGR'16 diagnosis.

The paper's central claim is that on UGR'16 'regime identifiers recur but the
feature-to-label mapping does not', which makes identifier-keyed reuse unsafe.
This script tests that claim directly on the committed window-level records,
without rerunning any model.

For every recurrence of a regime (a window whose regime has been seen before) we
measure, using only data available at that window:

  P(recur)   macro-F1 of the *frozen* model on this regime occurrence
  P(store)   macro-F1 of the *frozen* model on the regime's first occurrence

If the feature-to-label mapping is stable, P(recur) ~ P(store) for a frozen
model; if the mapping has drifted, P(recur) is much lower. We report the mean
gap and the fraction of recurrences where the gap exceeds 0.10, on UGR'16 and on
the other two cross-dataset streams as controls.
"""
import collections
import csv
import os

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results", "experiment_9a_three", "raw")
STREAMS = {"Campus": "5g_campus", "UGR16": "ugr16", "Nordic": "nordicdat"}


def load(slug):
    rows = list(csv.DictReader(open(os.path.join(RAW, f"per_window_{slug}_full.csv"))))
    out = collections.defaultdict(dict)
    for r in rows:
        if r["method"] != "Frozen":
            continue
        out[int(r["seed"])].setdefault(int(r["window_id"]), {})["regime"] = r["regime_id"]
        out[int(r["seed"])][int(r["window_id"])]["f1"] = float(r["macro_f1"])
    return out


print("E3 -- does the feature-to-label mapping survive regime recurrence?\n")
print(f"{'Stream':8s} {'recur':>6s} {'P(store)':>9s} {'P(recur)':>9s} {'gap':>7s} "
      f"{'p (Wilcoxon)':>13s} {'gap>0.10':>9s}")
summary = {}
for name, slug in STREAMS.items():
    per_seed = load(slug)
    stores, recurs = [], []
    for seed, wd in per_seed.items():
        seen_first = {}
        for w in sorted(wd):
            reg = wd[w]["regime"]
            if reg not in seen_first:
                seen_first[reg] = wd[w]["f1"]
            else:
                stores.append(seen_first[reg])
                recurs.append(wd[w]["f1"])
    stores, recurs = np.array(stores), np.array(recurs)
    gap = stores - recurs
    p = stats.wilcoxon(stores, recurs).pvalue if len(stores) else float("nan")
    frac = float(np.mean(gap > 0.10))
    summary[name] = (len(recurs), stores.mean(), recurs.mean(), gap.mean(), p, frac)
    print(f"{name:8s} {len(recurs):6d} {stores.mean():9.4f} {recurs.mean():9.4f} "
          f"{gap.mean():+7.4f} {p:13.3g} {frac:8.1%}")

n, ps, pr, g, p, f = summary["UGR16"]
print(f"\nUGR'16: {n} recurrences, mean frozen-F1 drop on recurrence {g:+.4f} "
      f"(p={p:.2e}); {f:.0%} of recurrences drop more than 0.10.")
print("Campus and Nordic gaps for comparison:",
      f"Campus {summary['Campus'][3]:+.4f}, Nordic {summary['Nordic'][3]:+.4f}")
