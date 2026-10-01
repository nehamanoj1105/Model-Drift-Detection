"""Phase 1: recompute every numeric claim in paper/main.tex from raw artifacts.

Prints a PASS/FAIL line per claim. Nothing is written to disk except stdout.
"""
import csv
import json
import os
import collections

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R9A = os.path.join(ROOT, "results", "experiment_9a_three", "raw")
R9B = os.path.join(ROOT, "results", "experiment_9b")
FE = os.path.join(ROOT, "Final_Experiments", "results", "raw")

FAILS = []


def check(label, got, want, tol=5e-5):
    ok = abs(got - want) <= tol
    if not ok:
        FAILS.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label:52s} got={got:<14.6g} paper={want}")


def load_summary(path):
    return {r["method"]: r for r in csv.DictReader(open(path))}


def stat(rows, method, col):
    v = [float(r[col]) for r in rows if r["method"] == method]
    return np.mean(v), np.std(v, ddof=1)


def per_window(path, method):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["method"] == method:
            d[int(r["window_id"])].append(float(r["macro_f1"]))
    ws = sorted(d)
    return np.array([np.mean(d[w]) for w in ws])


def per_seed(path, method, col="macro_f1"):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["method"] == method:
            d[r["seed"]].append(float(r[col]))
    return np.array([np.mean(v) for _, v in sorted(d.items())])


# =========================================================================
print("=" * 78)
print("TABLE II  (tab:primary) -- cross-dataset + 5G NR")
print("=" * 78)
for lab, fname in [("Campus", "summary_5g_campus_full.csv"),
                   ("UGR16", "summary_ugr16_full.csv"),
                   ("Nordic", "summary_nordicdat_full.csv")]:
    rows = list(csv.DictReader(open(os.path.join(R9A, fname))))
    for m in ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]:
        if not any(r["method"] == m for r in rows):
            continue
        for col, tag in [("macro_f1", "F1"), ("accuracy", "Acc"),
                         ("precision", "Prec"), ("recall", "Rec")]:
            mu, sd = stat(rows, m, col)
            print(f"   {lab:7s} {m:15s} {tag:5s} {mu:.4f} +/- {sd:.4f}")

# Paper's exact Table II values (mean only) for the three cross-dataset streams.
PAPER_T2 = {
    ("Campus", "Frozen"): (0.9364, 0.0023), ("Campus", "Event-Driven"): (0.9636, 0.0006),
    ("Campus", "Full Retraining"): (0.9836, 0.0013), ("Campus", "RAPT"): (0.9381, 0.0016),
    ("Campus", "RAPT-Enhanced"): (0.9381, 0.0016),
    ("UGR16", "Frozen"): (0.9690, 0.0029), ("UGR16", "Event-Driven"): (0.8972, 0.0277),
    ("UGR16", "Full Retraining"): (0.9595, 0.0012), ("UGR16", "RAPT"): (0.8360, 0.0262),
    ("UGR16", "RAPT-Enhanced"): (0.9276, 0.0123),
    ("Nordic", "Frozen"): (0.2779, 0.0224), ("Nordic", "Event-Driven"): (0.2547, 0.0177),
    ("Nordic", "Full Retraining"): (0.4227, 0.0299), ("Nordic", "RAPT"): (0.3818, 0.0184),
    ("Nordic", "RAPT-Enhanced"): (0.3783, 0.0321),
}
for (lab, m), (mu, sd) in PAPER_T2.items():
    fname = {"Campus": "summary_5g_campus_full.csv", "UGR16": "summary_ugr16_full.csv",
             "Nordic": "summary_nordicdat_full.csv"}[lab]
    rows = list(csv.DictReader(open(os.path.join(R9A, fname))))
    gmu, gsd = stat(rows, m, "macro_f1")
    check(f"T2 {lab} {m} F1", round(gmu, 4), mu)
    check(f"T2 {lab} {m} F1 sd", round(gsd, 4), sd)

# =========================================================================
print("\n" + "=" * 78)
print("TABLE II cost columns (CPU, Run, Retr, Reuse)")
print("=" * 78)
CPU = {
    ("Campus", "Frozen"): (0.0, 1.2482, 0, None), ("Campus", "Event-Driven"): (0.1080, 1.3926, 1, None),
    ("Campus", "Full Retraining"): (1.4008, 2.6677, 14, None), ("Campus", "RAPT"): (0.2292, 1.4931, 2, 12),
    ("Campus", "RAPT-Enhanced"): (0.2337, 2.1046, 2, 12),
    ("UGR16", "Frozen"): (0.0, 2.4914, 0, None), ("UGR16", "Event-Driven"): (0.5754, 3.1303, 3, None),
    ("UGR16", "Full Retraining"): (23.6318, 26.2627, 144, None), ("UGR16", "RAPT"): (1.9711, 4.4990, 11, 133),
    ("UGR16", "RAPT-Enhanced"): (2.7182, 5.9329, 11, 133),
    ("Nordic", "Frozen"): (0.0, 2.5488, 0, None), ("Nordic", "Event-Driven"): (0.2533, 2.8435, 2.2, None),
    ("Nordic", "Full Retraining"): (2.2034, 4.7593, 18, None), ("Nordic", "RAPT"): (0.2594, 2.8257, 2, 16),
    ("Nordic", "RAPT-Enhanced"): (2.6492, 5.8812, 2, 16),
}
for (lab, m), (cpu, run, retr, reuse) in CPU.items():
    fname = {"Campus": "summary_5g_campus_full.csv", "UGR16": "summary_ugr16_full.csv",
             "Nordic": "summary_nordicdat_full.csv"}[lab]
    rows = list(csv.DictReader(open(os.path.join(R9A, fname))))
    gcpu, _ = stat(rows, m, "adaptation_cpu_sec")
    grun, _ = stat(rows, m, "total_runtime_sec")
    gret, _ = stat(rows, m, "retrain_events")
    greu, _ = stat(rows, m, "reuse_events")
    check(f"T2 {lab} {m} CPU", round(gcpu, 4), cpu)
    check(f"T2 {lab} {m} Run", round(grun, 4), run)
    check(f"T2 {lab} {m} Retr", round(gret, 4), float(retr))
    if reuse is not None:
        check(f"T2 {lab} {m} Reuse", round(greu, 4), float(reuse))

# =========================================================================
print("\n" + "=" * 78)
print("TABLE II -- 5G NR row (from results/experiment_9b/natural_drift)")
print("=" * 78)
NR = load_summary(os.path.join(R9B, "natural_drift", "summary.csv"))
PAPER_NR = {
    "Frozen": (0.8961, 0.0099, 0.9075, 0.0061, 0.9314, 0.0083, 0.8798, 0.0085, 0.0, 1.7862, 0, None),
    "Event-Driven": (0.8903, 0.0107, 0.9055, 0.0060, 0.9475, 0.0085, 0.8688, 0.0083, 1.0406, 2.7561, 11, None),
    "Full Retraining": (0.9027, 0.0059, 0.9145, 0.0045, 0.9478, 0.0053, 0.8829, 0.0055, 0.7636, 2.4689, 9, None),
    "RAPT": (0.8894, 0.0110, 0.9025, 0.0068, 0.9258, 0.0103, 0.8737, 0.0092, 0.3955, 2.1112, 3, 6),
    "RAPT-Enhanced": (0.8915, 0.0117, 0.9055, 0.0076, 0.9348, 0.0106, 0.8737, 0.0098, 0.8609, 2.5710, 3, 6),
}
for m, vals in PAPER_NR.items():
    r = NR[m]
    keys = ["macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std",
            "precision_mean", "precision_std", "recall_mean", "recall_std"]
    for k, want in zip(keys, vals[:8]):
        check(f"NR {m} {k}", round(float(r[k]), 4), want)
    check(f"NR {m} CPU", round(float(r["adaptation_cpu_sec_mean"]), 4), vals[8])
    check(f"NR {m} Run", round(float(r["total_cpu_sec_mean"]), 4), vals[9])
    check(f"NR {m} Retr", round(float(r["retrain_events_mean"]), 4), float(vals[10]))
    if vals[11] is not None:
        check(f"NR {m} Reuse", round(float(r["reused_checkpoints_mean"]), 4), float(vals[11]))

# =========================================================================
print("\n" + "=" * 78)
print("SECTION V-F statistics (window-paired Wilcoxon + paired Cohen's d)")
print("=" * 78)
STREAMS = {"Campus": "per_window_5g_campus_full.csv",
           "UGR16": "per_window_ugr16_full.csv",
           "Nordic": "per_window_nordicdat_full.csv"}
for lab, fn in STREAMS.items():
    p = os.path.join(R9A, fn)
    a, b = per_window(p, "RAPT"), per_window(p, "Full Retraining")
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diff = a - b
    check(f"{lab} deltaF1 RAPT-FR", round(diff.mean(), 4),
          {"Campus": -0.0455, "UGR16": -0.1236, "Nordic": -0.0409}[lab])
    check(f"{lab} p RAPT-FR", stats.wilcoxon(a, b).pvalue,
          {"Campus": 2.6e-4, "UGR16": 2.6e-11, "Nordic": 0.255}[lab], tol=0.0015)
    check(f"{lab} d RAPT-FR", round(diff.mean() / diff.std(ddof=1), 2),
          {"Campus": -0.32, "UGR16": -0.67, "Nordic": -0.12}[lab], tol=0.005)
a = per_window(os.path.join(R9A, STREAMS["UGR16"]), "RAPT-Enhanced")
b = per_window(os.path.join(R9A, STREAMS["UGR16"]), "RAPT")
n = min(len(a), len(b)); a, b = a[:n], b[:n]; diff = a - b
check("UGR16 deltaF1 Enh-RAPT", round(diff.mean(), 4), 0.0917)
check("UGR16 p Enh-RAPT", stats.wilcoxon(a, b).pvalue, 5.4e-9, tol=1e-9)
check("UGR16 d Enh-RAPT", round(diff.mean() / diff.std(ddof=1), 2), 0.56, tol=0.005)

# =========================================================================
print("\n" + "=" * 78)
print("SECTION V-G  cost reductions and counts")
print("=" * 78)
for lab, fr, ra in [("Campus", 1.4008, 0.2292), ("UGR16", 23.6318, 1.9711),
                    ("Nordic", 2.2034, 0.2594), ("5G NR", 0.7636, 0.3955)]:
    check(f"cost reduction {lab}", round(-100 * (fr - ra) / fr, 1),
          {"Campus": -83.6, "UGR16": -91.7, "Nordic": -88.2, "5G NR": -48.2}[lab])

# =========================================================================
print("\n" + "=" * 78)
print("TABLE III / ablation ladder (Final_Experiments)")
print("=" * 78)
fe = list(csv.DictReader(open(os.path.join(FE, "summary_full.csv"))))
ABL = {"Full Retraining": (0.9829, 0.0023, 1.4063, None),
       "RAPT_FULL": (0.9587, 0.0000, 0.4283, 0.0),
       "RAPT_EVIDENCE": (0.9587, 0.0000, 0.4260, 0.0),
       "RAPT_FLOOR": (0.9797, 0.0050, 0.5865, 7.6),
       "RAPT_CHEAP": (0.9851, 0.0029, 0.8477, 22.0),
       "RAPT_INCR": (0.9549, 0.0020, 0.5813, 0.0)}
for m, (f1, sd, cpu, ref) in ABL.items():
    g1, gs = stat(fe, m, "macro_f1")
    gc, _ = stat(fe, m, "adaptation_cpu_sec")
    check(f"abl {m} F1", round(g1, 4), f1)
    check(f"abl {m} F1 sd", round(gs, 4), sd)
    check(f"abl {m} CPU", round(gc, 4), cpu)
    if ref is not None:
        gr, _ = stat(fe, m, "refreshes")
        check(f"abl {m} refreshes", round(gr, 4), ref)

# RAPT-Cheap vs Full Retraining equivalence
a = per_seed(os.path.join(FE, "per_window_full.csv"), "RAPT_CHEAP")
b = per_seed(os.path.join(FE, "per_window_full.csv"), "Full Retraining")
diff = a - b
check("cheap deltaF1", round(diff.mean(), 4), 0.0022, tol=1e-4)
check("cheap p (wilcoxon n=5)", stats.wilcoxon(a, b).pvalue, 0.3125, tol=1e-4)
check("cheap d", round(diff.mean() / diff.std(ddof=1), 2), 0.47, tol=0.005)
ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=diff.std(ddof=1) / np.sqrt(len(diff)))
check("cheap CI low", round(ci[0], 4), -0.0036, tol=5e-4)
check("cheap CI high", round(ci[1], 4), 0.0080, tol=5e-4)
check("cheap vs periodic CPU cut %",
      round(100 * (2.7168 - 0.8477) / 2.7168, 1), 68.8, tol=0.15)
check("cheap vs fullretrain CPU cut %",
      round(100 * (1.4063 - 0.8477) / 1.4063, 1), 39.7, tol=0.15)

# =========================================================================
print("\n" + "=" * 78)
print("SECTION V-J  historical detectors")
print("=" * 78)
DET = {"Campus": ("summary_5g_campus_full.csv", 0.9847, 0.0019, 3.9824, 38.0),
       "UGR16": ("summary_ugr16_full.csv", None, None, 6.1012, 39.0),
       "Nordic": ("summary_nordicdat_full.csv", 0.4184, 0.0165, 4.6875, 39.0)}
for lab, (fn, f1, sd, cpu, ev) in DET.items():
    rows = list(csv.DictReader(open(os.path.join(R9A, fn))))
    g1, gs = stat(rows, "EDD", "macro_f1")
    gc, _ = stat(rows, "EDD", "adaptation_cpu_sec")
    ge, _ = stat(rows, "EDD", "detected_events")
    gr, _ = stat(rows, "EDD", "retrain_events")
    if f1 is not None:
        check(f"EDD {lab} F1", round(g1, 4), f1)
        check(f"EDD {lab} F1 sd", round(gs, 4), sd)
    check(f"EDD {lab} CPU", round(gc, 4), cpu)
    check(f"EDD {lab} retrains", round(gr, 4), ev, tol=1.0)
    for m in ["ADWIN", "Page-Hinkley", "EDMA"]:
        gm, _ = stat(rows, m, "detected_events")
        check(f"{m} {lab} silent", round(gm, 4), 0.0)

print("\n" + "=" * 78)
print(f"RESULT: {len(FAILS)} failing claim(s)")
for f in FAILS:
    print("   FAIL:", f)
print("=" * 78)
