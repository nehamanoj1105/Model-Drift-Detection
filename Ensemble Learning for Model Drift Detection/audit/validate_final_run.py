"""Validate every numeric claim in the manuscripts against the authoritative
final-run artifacts under results/paper_final_run/final/.

Prints a PASS/FAIL line per claim and a summary count. Read-only.
"""
import csv
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINAL = os.path.join(ROOT, "results", "paper_final_run", "final")

FAILS = []


def check(label, got, want, tol=5e-5, rel=None):
    """rel: if set, compare with relative tolerance rel*max(|want|,eps) instead of
    the absolute tol. Used for timing / CPU / cost columns, which are not
    deterministic across runs (see AGENTS.md)."""
    if rel is not None:
        ok = abs(got - want) <= rel * max(abs(want), 1e-9)
    else:
        ok = abs(got - want) <= tol
    if not ok:
        FAILS.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label:56s} got={got:<14.6g} paper={want}")


def load(fname):
    return list(csv.DictReader(open(os.path.join(FINAL, fname))))


PRIMARY = {r["dataset"]: r for r in load("table_primary.csv")}
PRIMARY_BY = {(r["dataset"], r["model"]): r for r in load("table_primary.csv")}
ABL = {(r["dataset"], r["model"]): r for r in load("table_ablation.csv")}
DET = {(r["dataset"], r["model"]): r for r in load("table_detectors.csv")}
STATS = {(r["dataset"], r["comparison"]): r for r in load("stats.csv")}
SAV = {(r["dataset"], r["model"]): r for r in load("cost_savings.csv")}
DS = {r["dataset"]: r for r in load("dataset_stats.csv")}

# -------------------------------------------------------------------------
print("=" * 84)
print("TABLE: primary comparison (mean +/- sd over seeds 42-46)")
print("=" * 84)
# dataset, model -> (f1, sd, acc, acc_sd, prec, prec_sd, rec, rec_sd, cpu, run, retr, reuse)
T1 = {
    ("5G Campus", "Frozen"): (0.9370, 0.0010, 0.9822, 0.0006, 0.9386, 0.0010, 0.9383, 0.0003, 0.0000, 1.2307, 0.0, 0.0),
    ("5G Campus", "Event-Driven"): (0.9610, 0.0030, 0.9898, 0.0018, 0.9609, 0.0033, 0.9627, 0.0019, 0.1190, 1.3700, 1.2, 0.0),
    ("5G Campus", "Full Retraining"): (0.9739, 0.0049, 0.9929, 0.0013, 0.9741, 0.0044, 0.9751, 0.0049, 1.2715, 2.5129, 13.0, 0.0),
    ("5G Campus", "RAPT"): (0.9402, 0.0028, 0.9831, 0.0017, 0.9421, 0.0012, 0.9414, 0.0032, 0.2594, 1.5088, 2.0, 11.0),
    ("5G Campus", "RAPT-Enhanced"): (0.9402, 0.0028, 0.9831, 0.0017, 0.9421, 0.0012, 0.9414, 0.0032, 0.2587, 1.5063, 2.0, 11.0),
    ("UGR'16", "Frozen"): (0.9688, 0.0038, 0.9905, 0.0006, 0.9784, 0.0034, 0.9661, 0.0037, 0.0000, 1.3471, 0.0, 0.0),
    ("UGR'16", "Event-Driven"): (0.9104, 0.0211, 0.9899, 0.0019, 0.9177, 0.0205, 0.9081, 0.0217, 0.6582, 2.0646, 3.4, 0.0),
    ("UGR'16", "Full Retraining"): (0.9591, 0.0031, 0.9877, 0.0010, 0.9722, 0.0022, 0.9576, 0.0027, 23.8637, 25.3481, 143.0, 0.0),
    ("UGR'16", "RAPT"): (0.8503, 0.0346, 0.9565, 0.0208, 0.8688, 0.0321, 0.8491, 0.0354, 2.7592, 4.1832, 11.0, 132.0),
    ("UGR'16", "RAPT-Enhanced"): (0.8988, 0.0412, 0.9781, 0.0205, 0.9131, 0.0375, 0.8972, 0.0402, 3.3564, 4.7727, 11.0, 132.0),
    ("NordicDat", "Frozen"): (0.3037, 0.0182, 0.4843, 0.0074, 0.4479, 0.0183, 0.2767, 0.0191, 0.0000, 1.4207, 0.0, 0.0),
    ("NordicDat", "Event-Driven"): (0.2735, 0.0309, 0.4620, 0.0449, 0.4059, 0.0186, 0.2476, 0.0273, 0.2280, 1.6873, 2.0, 0.0),
    ("NordicDat", "Full Retraining"): (0.4765, 0.0171, 0.5920, 0.0077, 0.5498, 0.0159, 0.4629, 0.0180, 2.1692, 3.6034, 18.0, 0.0),
    ("NordicDat", "RAPT"): (0.3978, 0.0361, 0.6037, 0.0099, 0.4863, 0.0328, 0.3767, 0.0372, 0.3126, 1.7579, 2.0, 16.0),
    ("NordicDat", "RAPT-Enhanced"): (0.4644, 0.0332, 0.5996, 0.0120, 0.5411, 0.0272, 0.4494, 0.0343, 1.4005, 2.8486, 2.0, 10.2),
    ("5G NR Lat.", "Frozen"): (0.9048, 0.0056, 0.9048, 0.0056, 0.9048, 0.0056, 0.9048, 0.0056, 0.0000, 3.3496, 0.0, 0.0),
    ("5G NR Lat.", "Event-Driven"): (0.8576, 0.0045, 0.8576, 0.0045, 0.8576, 0.0045, 0.8576, 0.0045, 1.1062, 4.4936, 12.6, 0.0),
    ("5G NR Lat.", "Full Retraining"): (0.8586, 0.0074, 0.8586, 0.0074, 0.8586, 0.0074, 0.8586, 0.0074, 0.6014, 3.9725, 7.0, 0.0),
    ("5G NR Lat.", "RAPT"): (0.7945, 0.0163, 0.7945, 0.0163, 0.7945, 0.0163, 0.7945, 0.0163, 0.2828, 3.6635, 3.0, 4.0),
    ("5G NR Lat.", "RAPT-Enhanced"): (0.8446, 0.0106, 0.8446, 0.0106, 0.8446, 0.0106, 0.8446, 0.0106, 0.5916, 3.9896, 3.0, 3.6),
}
for key, (f1, sd, acc, asd, pr, psd, rc, rsd, cpu, run, retr, reuse) in T1.items():
    r = PRIMARY_BY[key]
    for name, want, col in [("F1", f1, "macro_f1"), ("F1 sd", sd, "macro_f1_std"),
                            ("Acc", acc, "accuracy"), ("Acc sd", asd, "accuracy_std"),
                            ("Prec", pr, "precision"), ("Prec sd", psd, "precision_std"),
                            ("Rec", rc, "recall"), ("Rec sd", rsd, "recall_std"),
                            ("CPU", cpu, "adapt_cpu_s"), ("Run", run, "runtime_s"),
                            ("Retr", retr, "retrains"), ("Reuse", reuse, "reuses")]:
        rel = 0.15 if name in ("CPU", "Run") else None
        check(f"{key[0]:12s} {key[1]:15s} {name}", round(float(r[col]), 4), want, rel=rel)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: cost reductions vs Full Retraining")
print("=" * 84)
COST = {("5G Campus", "RAPT"): 79.59, ("UGR'16", "RAPT"): 88.44,
        ("NordicDat", "RAPT"): 85.59, ("5G NR Lat.", "RAPT"): 52.96,
        ("5G Campus", "RAPT-Cheap"): 22.60, ("NordicDat", "RAPT-Enhanced"): 35.43}
for key, want in COST.items():
    # Cost reductions derive from adaptation CPU and inherit its run-to-run spread.
    check(f"cost reduction {key[0]} {key[1]}", round(float(SAV[key]["reduction_pct"]), 2), want, rel=0.15)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: efficiency ablation (5G Campus)")
print("=" * 84)
AB = {
    "RAPT-Full": (0.9537, 0.0064, 0.6216, 0.0),
    "RAPT-Evidence": (0.9402, 0.0028, 0.2580, 0.0),
    "RAPT-Floor": (0.9402, 0.0028, 0.2589, 0.0),
    "RAPT-Periodic-FullRefit": (0.9745, 0.0044, 4.6550, 22.0),
    "RAPT-Cheap": (0.9717, 0.0081, 0.9840, 22.0),
    "RAPT-Incremental": (0.7164, 0.0057, 0.1909, 130.0),
}
for m, (f1, sd, cpu, ref) in AB.items():
    r = ABL[("5G Campus", m)]
    check(f"abl {m} F1", round(float(r["macro_f1"]), 4), f1)
    check(f"abl {m} F1 sd", round(float(r["macro_f1_std"]), 4), sd)
    check(f"abl {m} CPU", round(float(r["adapt_cpu_s"]), 4), cpu, rel=0.15)
    check(f"abl {m} refreshes", round(float(r["refreshes"]), 4), ref)
# Full Retraining reference row
check("abl Full Retraining F1", round(float(PRIMARY_BY[("5G Campus", "Full Retraining")]["macro_f1"]), 4), 0.9739)
check("abl Full Retraining CPU", round(float(PRIMARY_BY[("5G Campus", "Full Retraining")]["adapt_cpu_s"]), 4), 1.2715, rel=0.15)

# RAPT-Cheap statistics
s = STATS[("5G Campus", "RAPT-Cheap vs Full Retraining")]
check("cheap deltaF1", round(float(s["delta"]), 4), -0.0022, tol=5e-5)
check("cheap p (seed-level)", float(s["p_seed"]), 0.4375, tol=1e-6)
check("cheap d_z", round(float(s["d_z"]), 2), -0.34, tol=0.005)
check("cheap CI low", round(float(s["ci95_low"]), 4), -0.0102, tol=5e-5)
check("cheap CI high", round(float(s["ci95_high"]), 4), 0.0058, tol=5e-5)
# cheap vs periodic full-refit CPU cut
check("cheap vs periodic CPU cut %",
      round(100 * (4.6550 - 0.9840) / 4.6550, 1), 78.9, tol=0.15)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: paired statistics (seed-level Wilcoxon + paired d_z)")
print("=" * 84)
SP = {
    ("5G Campus", "RAPT vs Full Retraining"): (-0.0338, -13.05),
    ("UGR'16", "RAPT vs Full Retraining"): (-0.1088, -3.16),
    ("NordicDat", "RAPT vs Full Retraining"): (-0.0786, -4.04),
    ("5G NR Lat.", "RAPT vs Full Retraining"): (-0.0642, -4.58),
    ("UGR'16", "RAPT-Enhanced vs RAPT"): (0.0485, 1.05),
    ("NordicDat", "RAPT-Enhanced vs RAPT"): (0.0666, 4.00),
    ("5G NR Lat.", "RAPT-Enhanced vs RAPT"): (0.0501, 5.44),
    ("NordicDat", "RAPT-Enhanced vs Full Retraining"): (-0.0121, -0.68),
}
for key, (d, dz) in SP.items():
    r = STATS[key]
    check(f"{key[1]:34s} {key[0]:12s} delta", round(float(r["delta"]), 4), d, tol=5e-5)
    check(f"{key[1]:34s} {key[0]:12s} d_z", round(float(r["d_z"]), 2), dz, tol=0.006)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: historical detectors (shared harness)")
print("=" * 84)
# Detector names follow the corrected paper nomenclature: the paper's "EDD" is
# River's EDDM, and the paper's "EDMA" is the ECDD-EWMA control chart.
D = {
    ("5G Campus", "EDDM"): (0.9370, 0.0010, 0.00, 0.0),
    ("UGR'16", "EDDM"): (0.9375, 0.0278, 1.29, 6.8),
    ("NordicDat", "EDDM"): (0.1354, 0.0456, 0.71, 6.2),
    ("5G NR Lat.", "EDDM"): (0.9118, 0.0065, 0.09, 1.0),
}
for key, (f1, sd, cpu, retr) in D.items():
    r = DET[key]
    check(f"EDDM {key[0]} F1", round(float(r["macro_f1"]), 4), f1)
    check(f"EDDM {key[0]} F1 sd", round(float(r["macro_f1_std"]), 4), sd)
    check(f"EDDM {key[0]} CPU", round(float(r["adapt_cpu_s"]), 2), cpu, tol=0.006)
    check(f"EDDM {key[0]} retrains", round(float(r["retrains"]), 4), retr)
for ds in ["5G Campus", "UGR'16", "NordicDat", "5G NR Lat."]:
    for m in ["ADWIN", "Page-Hinkley", "ECDD-EWMA"]:
        r = DET[(ds, m)]
        check(f"{m} {ds} silent", round(float(r["adapt_cpu_s"]), 4), 0.0)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: dataset table + protocol")
print("=" * 84)
DT = {"5G Campus": (1790, 13, 3, 179, 36, 3, 16, 14),
      "UGR'16": (43200, 133, 2, 180, 36, 12, 179, 168),
      "NordicDat": (91000, 15, 3, 182, 36, 3, 19, 17),
      "5G NR Lat.": (499, 11, 3, 499, 100, 4, 9, 6)}
for ds, (n, feat, cls, win, init, reg, trans, rec) in DT.items():
    r = DS[ds]
    check(f"dataset {ds} n_samples", float(r["n_samples"]), float(n))
    check(f"dataset {ds} n_features", float(r["n_features"]), float(feat))
    check(f"dataset {ds} classes", float(len(eval(r["classes"]))), float(cls))
    check(f"dataset {ds} windows", float(r["total_windows"]), float(win))
    check(f"dataset {ds} initial windows", float(r["initial_train_windows"]), float(init))
    check(f"dataset {ds} regimes", float(r["n_regimes"]), float(reg))
    check(f"dataset {ds} transitions", float(r["n_transitions"]), float(trans))
    check(f"dataset {ds} recurrences", float(r["n_recurrences"]), float(rec))

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: leakage check + refit timing")
print("=" * 84)
lk = pd.read_csv(os.path.join(FINAL, "leakage_check.csv"))
check("leakage rows", len(lk), 62400.0)
check("leakage all ok", float(lk["ok"].astype(str).eq("True").all()), 1.0)
rt = pd.read_csv(os.path.join(FINAL, "refit_timing.csv"))
def timing(trees, buf):
    return round(float(rt[(rt.trees == trees) & (rt.buffer == buf)]["mean_ms"].iloc[0]), 1)
check("refit 20 trees/1000 buf ms", timing(20, 1000), 66.7, rel=0.15)
check("refit 100 trees/1000 buf ms", timing(100, 1000), 320.3, rel=0.15)
check("refit 20 trees/200 buf ms", timing(20, 200), 37.1, rel=0.15)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("SECTION: UGR'16 novelty-refit attribution")
print("=" * 84)
na = pd.read_csv(os.path.join(FINAL, "novelty_attribution.csv"))
for model, want in [("RAPT(novel=500)", 0.8503), ("RAPT(novel=1500)", 0.8988)]:
    got = round(float(na[na.model == model]["macro_f1"].mean()), 4)
    check(f"novelty {model}", got, want)

print("\n" + "=" * 84)
print(f"RESULT: {len(FAILS)} failing claim(s) out of "
      f"{len(T1) * 12 + len(COST) + len(AB) * 4 + 2 + 6 + len(SP) * 2 + len(D) * 4 + 12 + 32 + 5}")
for f in FAILS:
    print("   FAIL:", f)
print("=" * 84)
