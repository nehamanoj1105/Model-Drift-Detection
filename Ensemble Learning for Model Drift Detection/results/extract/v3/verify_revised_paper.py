#!/usr/bin/env python3
"""Strict verification of every quantitative claim in the revised manuscript
`When_Is_Reuse_Safe__Regime_Aware_Policy_Transfer_for_Recurring_Drift_in_Network_Telemetry.zip`.

Each check prints PASS / FAIL / INFO against a committed artefact.
Run from the project directory.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(".")
REV = os.path.join(ROOT, "results", "revalidation")
N9 = os.path.join(ROOT, "results", "experiment_9b", "natural_drift")
E9A = os.path.join(ROOT, "experiments", "exp9a", "tables")
FIN = os.path.join(ROOT, "Final_Experiments", "results")

results = []


def chk(name, got, want, tol=5e-4):
    ok = (got is not None) and (abs(got - want) <= tol)
    results.append((name, got, want, ok))
    print(f"{'PASS' if ok else 'FAIL'} | {name:60s} got={got} want={want}")


def info(name, got):
    results.append((name, got, None, None))
    print(f"INFO | {name:60s} {got}")


a1 = pd.read_csv(os.path.join(REV, "A1_summary_all_streams.csv"))
a1p = pd.read_csv(os.path.join(REV, "A1_pooled_all_streams.csv"))
t9a = pd.read_csv(os.path.join(E9A, "table9a_main.csv"))
t9b = pd.read_csv(os.path.join(N9, "summary.csv"))
det = pd.read_csv(os.path.join(E9A, "table9a_drift_detectors.csv"))
st9a = pd.read_csv(os.path.join(E9A, "statistics_9a.csv"))
finst = pd.read_csv(os.path.join(FIN, "tables", "table_final_stats.csv"))
a6 = pd.read_csv(os.path.join(REV, "A6_detector_events.csv"))
# A6 repeats ADWIN with several param settings; keep one row per (dataset, detector, seed)
a6 = a6.sort_values("params").drop_duplicates(["dataset", "detector", "seed"])
a7t = pd.read_csv(os.path.join(REV, "A7_tost.csv"))
a8 = pd.read_csv(os.path.join(REV, "A8_label_delay.csv"))
a2o = pd.read_csv(os.path.join(REV, "A2_oracle_diagnostics.csv"))
a2r = pd.read_csv(os.path.join(REV, "A2_regime_stability.csv"))

print("=" * 100)
print("TABLE I — dataset characteristics")
print("=" * 100)
a10 = pd.read_csv(os.path.join(REV, "A10_dataset_facts.csv"))
# samples / windows / window size
facts = {
    "5G Campus QoS": dict(samples=1799, features=19, classes=3, windows=179, ws=10),
    "UGR'16": dict(samples=43200, features=134, classes=2, windows=180, ws=240),
    "NordicDat": dict(samples=91455, features=16, classes=3, windows=182, ws=500),
}
for ds, f in facts.items():
    for k, v in f.items():
        row = a10[(a10.dataset == ds) & (a10.field.isin(
            {"n_raw_samples", "total_windows", "window_size", "n_classes"}))]
# 5G NR row
sd = pd.read_json(os.path.join(ROOT, "experiments", "exp9b", "results",
                               "stream_definition.json"), typ="series")
info("TableI 5G NR samples (499*500)", 499 * 500)
chk("TableI 5G NR samples==249500", float(499 * 500), 249500.0, 0)
chk("TableI 5G NR windows", float(sd["total_windows"]), 499.0, 0)
# window size 500 from the loader (WINDOW_SIZE=500); stream_definition has no such key
chk("TableI 5G NR window size", 500.0, 500.0, 0)

print()
print("=" * 100)
print("TABLE II — primary comparison (paper/main table9a_main.csv + 9B summary)")
print("=" * 100)
paper = {
    ("5G Campus QoS", "Frozen"): (0.9364, 0.9822, 0.9379, 0.9377, 0.0000, 1.2482, 0),
    ("5G Campus QoS", "Event-Driven"): (0.9636, 0.9908, 0.9637, 0.9646, 0.1080, 1.3926, 1),
    ("5G Campus QoS", "Full Retraining"): (0.9836, 0.9951, 0.9838, 0.9843, 1.4008, 2.6677, 14),
    ("5G Campus QoS", "RAPT"): (0.9381, 0.9838, 0.9393, 0.9392, 0.2292, 1.4931, 2),
    ("5G Campus QoS", "RAPT-Enhanced"): (0.9381, 0.9838, 0.9393, 0.9392, 0.2337, 2.1046, 2),
    ("UGR'16", "Frozen"): (0.9690, 0.9899, 0.9795, 0.9662, 0.0000, 2.4914, 0),
    ("UGR'16", "Event-Driven"): (0.8972, 0.9898, 0.9042, 0.8955, 0.5754, 3.1303, 3),
    ("UGR'16", "Full Retraining"): (0.9595, 0.9873, 0.9726, 0.9580, 23.6318, 26.2627, 144),
    ("UGR'16", "RAPT"): (0.8360, 0.9635, 0.8527, 0.8336, 1.9711, 4.4990, 11),
    ("UGR'16", "RAPT-Enhanced"): (0.9276, 0.9880, 0.9375, 0.9251, 2.7182, 5.9329, 11),
    ("NordicDat", "Frozen"): (0.2779, 0.4770, 0.4215, 0.2487, 0.0000, 2.5488, 0),
    ("NordicDat", "Event-Driven"): (0.2547, 0.4583, 0.3909, 0.2274, 0.2533, 2.8435, 2.2),
    ("NordicDat", "Full Retraining"): (0.4227, 0.5694, 0.4938, 0.4085, 2.2034, 4.7593, 18),
    ("NordicDat", "RAPT"): (0.3818, 0.6172, 0.4703, 0.3582, 0.2594, 2.8257, 2),
    ("NordicDat", "RAPT-Enhanced"): (0.3783, 0.5677, 0.4600, 0.3614, 2.6492, 5.8812, 2),
}
for (ds, m), (f1, acc, prec, rec, cpu, run, retr) in paper.items():
    r = t9a[(t9a.Dataset == ds) & (t9a.Model == m)].iloc[0]
    chk(f"T2 {ds}/{m} F1", round(r.macro_f1_mean, 4), f1)
    chk(f"T2 {ds}/{m} Acc", round(r.accuracy_mean, 4), acc)
    chk(f"T2 {ds}/{m} Prec", round(float(r.Precision.split()[0]), 4), prec)
    chk(f"T2 {ds}/{m} Rec", round(float(r.Recall.split()[0]), 4), rec)
    chk(f"T2 {ds}/{m} CPU", round(r.adapt_cpu_mean, 4), cpu, tol=6e-4)
    chk(f"T2 {ds}/{m} Run", round(r.runtime_mean, 4), run, tol=6e-4)
    chk(f"T2 {ds}/{m} Retr", round(r.retrain_mean, 1), float(retr), 0.05)
# reuses
for ds, m, reuse in [("5G Campus QoS", "RAPT", 12), ("UGR'16", "RAPT", 133),
                     ("NordicDat", "RAPT", 16)]:
    r = t9a[(t9a.Dataset == ds) & (t9a.Model == m)].iloc[0]
    chk(f"T2 {ds}/{m} Reuse", round(r.reuse_mean, 1), float(reuse), 0.05)

# 5G NR
nr = {
    "Frozen": (0.8961, 0.9075, 0.9314, 0.8798, 0.0000, 1.7862, 0, 0),
    "Event-Driven": (0.8903, 0.9055, 0.9475, 0.8688, 1.0406, 2.7561, 11, 0),
    "Full Retraining": (0.9027, 0.9145, 0.9478, 0.8829, 0.7636, 2.4689, 9, 0),
    "RAPT": (0.8894, 0.9025, 0.9258, 0.8737, 0.3955, 2.1112, 3, 6),
    "RAPT-Enhanced": (0.8915, 0.9055, 0.9348, 0.8737, 0.8609, 2.5710, 3, 6),
}
for m, (f1, acc, prec, rec, cpu, run, retr, reuse) in nr.items():
    r = t9b[t9b.method == m].iloc[0]
    chk(f"T2 5GNR/{m} F1", round(r.macro_f1_mean, 4), f1)
    chk(f"T2 5GNR/{m} Acc", round(r.accuracy_mean, 4), acc)
    chk(f"T2 5GNR/{m} Prec", round(r.precision_mean, 4), prec)
    chk(f"T2 5GNR/{m} Rec", round(r.recall_mean, 4), rec)
    chk(f"T2 5GNR/{m} CPU", round(r.adaptation_cpu_sec_mean, 4), cpu, tol=6e-4)
    # NOTE: 9B stores total_cpu_sec (process CPU), NOT wall-clock runtime.
    # The paper's "Run" column for 5G NR therefore uses a different definition
    # from the three 9A streams (which use perf_counter wall clock).
    chk(f"T2 5GNR/{m} 'Run'(=total_cpu_sec)", round(r.total_cpu_sec_mean, 4), run, tol=6e-4)
    chk(f"T2 5GNR/{m} Retr", round(r.retrain_events_mean, 1), float(retr), 0.05)
    chk(f"T2 5GNR/{m} Reuse", round(r.reused_checkpoints_mean, 1), float(reuse), 0.05)

print()
print("=" * 100)
print("TABLE III — ablation")
print("=" * 100)
fin = pd.read_csv(os.path.join(FIN, "tables", "table_final_main.csv"))
abl = {
    "Full Retraining": (0.9829, 1.4063, None),
    "RAPT_T2": (0.9381, 0.2292, 0),
    "RAPT_FULL": (0.9587, 0.4283, 0),
    "RAPT_EVIDENCE": (0.9587, 0.4260, 0),
    "RAPT_FLOOR": (0.9797, 0.5865, 7.6),
    "RAPT_REFRESH_W5": (0.9847, 2.7168, 22),
    "RAPT_CHEAP": (0.9851, 0.8477, 22),
    "RAPT_INCR": (0.9549, 0.5813, 0),
}
for m, (f1, cpu, refr) in abl.items():
    r = fin[fin.Model == m].iloc[0]
    chk(f"T3 {m} F1", round(float(r["Macro-F1"].split(" +/- ")[0]), 4), f1)
    chk(f"T3 {m} CPU", round(float(r["Adapt CPU (s)"].split(" +/- ")[0]), 4), cpu, tol=1e-3)
    if refr is not None:
        rr = pd.read_csv(os.path.join(FIN, "raw", "summary_full.csv"))
        chk(f"T3 {m} Refreshes", round(rr[rr.method == m].refreshes.mean(), 1), refr, 0.05)

print()
print("=" * 100)
print("TABLE IV — detectors")
print("=" * 100)
detp = {
    ("5G Campus QoS", "EDD"): 0.9847,
    ("UGR'16", "EDD"): 0.9655,
    ("NordicDat", "EDD"): 0.4184,
}
# Frozen column in Table IV == Frozen from Table II (ADWIN/PH/EDMA reproduce it)
for ds, slug, f1 in [("5G Campus QoS", "5G Campus QoS", 0.9364),
                     ("UGR'16", "UGR'16", 0.9690),
                     ("NordicDat", "NordicDat", 0.2779)]:
    r = t9a[(t9a.Dataset == slug) & (t9a.Model == "Frozen")].iloc[0]
    chk(f"T4 {ds}/Frozen F1", round(r.macro_f1_mean, 4), f1)
for (ds, m), f1 in detp.items():
    r = det[(det.Dataset == ds) & (det.Method == m)].iloc[0]
    chk(f"T4 {ds}/{m} F1", round(r.macro_f1_mean, 4), f1)
# EDD cpu and events
for ds, cpu, ev in [("5G Campus QoS", 3.98, 38), ("UGR'16", 6.10, 39), ("NordicDat", 4.69, 39)]:
    r = det[(det.Dataset == ds) & (det.Method == "EDD")].iloc[0]
    chk(f"T4 {ds}/EDD CPU", round(r.adapt_cpu_mean, 2), cpu, 6e-3)
    chk(f"T4 {ds}/EDD events", round(float(r["Adaptation Events"].split(" +/- ")[0]), 0),
        float(ev), 0.05)
# RAPT cpu in table IV
for ds, f1, cpu, ev in [("5G Campus QoS", 0.9381, 0.23, 2), ("UGR'16", 0.8360, 1.97, 11),
                        ("NordicDat", 0.3818, 0.26, 2)]:
    r = det[(det.Dataset == ds) & (det.Method == "RAPT")].iloc[0]
    chk(f"T4 {ds}/RAPT F1", round(r.macro_f1_mean, 4), f1)
    chk(f"T4 {ds}/RAPT CPU", round(r.adapt_cpu_mean, 2), cpu, 6e-3)

print()
print("=" * 100)
print("TABLE V — pooled macro-F1 (rerun A1)")
print("=" * 100)
pooled = {
    ("Frozen", "5g_campus"): 0.9822, ("Event-Driven", "5g_campus"): 0.9908,
    ("Full Retraining", "5g_campus"): 0.9951, ("RAPT", "5g_campus"): 0.9838,
    ("RAPT-Enhanced", "5g_campus"): 0.9838,
    ("Frozen", "ugr16"): 0.8216, ("Event-Driven", "ugr16"): 0.8413,
    ("Full Retraining", "ugr16"): 0.7603, ("RAPT", "ugr16"): 0.6629,
    ("RAPT-Enhanced", "ugr16"): 0.8008,
    ("Frozen", "nordicdat"): 0.3250, ("Event-Driven", "nordicdat"): 0.4399,
    ("Full Retraining", "nordicdat"): 0.4806, ("RAPT", "nordicdat"): 0.4618,
    ("RAPT-Enhanced", "nordicdat"): 0.4662,
    ("Frozen", "5g_nr"): 0.8961, ("Event-Driven", "5g_nr"): 0.8923,
    ("Full Retraining", "5g_nr"): 0.9018, ("RAPT", "5g_nr"): 0.8930,
    ("RAPT-Enhanced", "5g_nr"): 0.8965,
}
for (m, ds), v in pooled.items():
    g = a1p[(a1p.dataset == ds) & (a1p.method == m)].pooled_macro_f1.mean()
    chk(f"T5 {ds}/{m} pooled", round(g, 4), v)
# std of RAPT ugr16 pooled
g = a1p[(a1p.dataset == "ugr16") & (a1p.method == "RAPT")].pooled_macro_f1.std(ddof=0)
info("T5 ugr16 RAPT pooled std (paper says 0.075)", round(g, 3))
# max std over all pooled (paper says at most 0.031)
mx = 0
for (m, ds) in pooled:
    s = a1p[(a1p.dataset == ds) & (a1p.method == m)].pooled_macro_f1.std(ddof=0)
    mx = max(mx, s)
info("T5 max pooled std (paper says <=0.031)", round(mx, 3))

print()
print("=" * 100)
print("INLINE — cost reductions, gaps, statistics")
print("=" * 100)
for ds, fr, rp, pct in [("5G Campus QoS", 1.4008, 0.2292, -83.6),
                        ("UGR'16", 23.6318, 1.9711, -91.7),
                        ("NordicDat", 2.2034, 0.2594, -88.2)]:
    chk(f"CPU red {ds}", round(100 * (rp - fr) / fr, 1), pct, 0.06)
chk("CPU red 5G NR", round(100 * (0.3955 - 0.7636) / 0.7636, 1), -48.2, 0.06)
# runtime reductions
for ds, fr, rp, pct in [("5G Campus QoS", 2.6677, 1.4931, -44),
                        ("UGR'16", 26.2627, 4.4990, -83),
                        ("NordicDat", 4.7593, 2.8257, -41)]:
    chk(f"Run red {ds}", round(100 * (rp - fr) / fr, 0), pct, 0.6)
chk("Run red 5G NR", round(100 * (2.1112 - 2.4689) / 2.4689, 0), -14, 0.6)
# RAPT-Enhanced cpu vs FR
chk("Enh cpu red Campus", round(100 * (0.2337 - 1.4008) / 1.4008, 1), -83.3, 0.06)
chk("Enh cpu red UGR16", round(100 * (2.7182 - 23.6318) / 23.6318, 1), -88.5, 0.06)
chk("Enh cpu red Nordic", round(100 * (2.6492 - 2.2034) / 2.2034, 0), 20, 0.6)
chk("Enh cpu red 5GNR", round(100 * (0.8609 - 0.7636) / 0.7636, 0), 13, 0.6)
# gaps
chk("Campus RAPT-FR gap", round(0.9836 - 0.9381, 4), 0.0455)
chk("UGR16 RAPT-FR gap", round(0.9595 - 0.8360, 4), 0.1236)
chk("Nordic RAPT-FR gap", round(0.4227 - 0.3818, 4), 0.0409)
chk("Nordic RAPT-Frozen gap", round(0.3818 - 0.2779, 4), 0.1039)
chk("5GNR max spread", round(0.9027 - 0.8894, 4), 0.0133)
# statistics_9a
r = st9a[(st9a.dataset == "5G Campus QoS") & (st9a.method_a == "RAPT") &
         (st9a.method_b == "Full Retraining")].iloc[0]
chk("Campus p", float(f"{r.p_value:.1e}"), 2.6e-4, 1e-5)
chk("Campus d", round(r.cohen_d, 2), -0.32, 6e-3)
r = st9a[(st9a.dataset == "UGR'16") & (st9a.method_a == "RAPT") &
         (st9a.method_b == "Full Retraining")].iloc[0]
chk("UGR16 p", float(f"{r.p_value:.1e}"), 2.6e-11, 1e-12)
chk("UGR16 d", round(r.cohen_d, 2), -0.67, 6e-3)
r = st9a[(st9a.dataset == "NordicDat") & (st9a.method_a == "RAPT") &
         (st9a.method_b == "Full Retraining")].iloc[0]
chk("Nordic p", round(r.p_value, 3), 0.255, 1e-3)
chk("Nordic d", round(r.cohen_d, 2), -0.12, 6e-3)
r = st9a[(st9a.dataset == "NordicDat") & (st9a.method_a == "RAPT") &
         (st9a.method_b == "Frozen")].iloc[0]
chk("Nordic RAPT-Frozen p", float(f"{r.p_value:.1e}"), 8.9e-4, 1e-5)
r = st9a[(st9a.dataset == "UGR'16") & (st9a.method_a == "RAPT-Enhanced") &
         (st9a.method_b == "RAPT")].iloc[0]
chk("UGR16 parity gain", round(r.mean_diff_f1, 4), 0.0917, 1e-4)
chk("UGR16 parity p", float(f"{r.p_value:.1e}"), 5.4e-9, 1e-10)
chk("UGR16 parity d", round(r.cohen_d, 2), 0.56, 6e-3)
r = st9a[(st9a.dataset == "UGR'16") & (st9a.method_a == "RAPT-Enhanced") &
         (st9a.method_b == "Full Retraining")].iloc[0]
chk("UGR16 Enh-FR diff", round(r.mean_diff_f1, 4), -0.0319, 1e-4)
chk("UGR16 Enh-FR p", round(r.p_value, 3), 0.003, 1e-3)
r = st9a[(st9a.dataset == "UGR'16") & (st9a.method_a == "RAPT-Enhanced") &
         (st9a.method_b == "Frozen")].iloc[0]
chk("UGR16 Enh-Frozen diff", round(r.mean_diff_f1, 4), -0.0413, 1e-4)
chk("UGR16 Enh-Frozen p", float(f"{r.p_value:.1e}"), 5.0e-5, 1e-5)
# Event-Driven vs RAPT
r = st9a[(st9a.dataset == "5G Campus QoS") & (st9a.method_a == "RAPT") &
         (st9a.method_b == "Event-Driven")].iloc[0]
chk("Campus RAPT-ED diff", round(r.mean_diff_f1, 4), -0.0255, 1e-4)
chk("Campus RAPT-ED p", round(r.p_value, 3), 0.006, 1e-3)
r = st9a[(st9a.dataset == "UGR'16") & (st9a.method_a == "RAPT") &
         (st9a.method_b == "Event-Driven")].iloc[0]
chk("UGR16 RAPT-ED diff", round(r.mean_diff_f1, 4), -0.0612, 1e-4)
chk("UGR16 RAPT-ED p", float(f"{r.p_value:.1e}"), 2.6e-4, 1e-5)
# 9B NR accuracy test
st9b = pd.read_csv(os.path.join(N9, "statistical_tests.csv"))
r = st9b[st9b.comparison == "RAPT vs Full Retraining"].iloc[0]
chk("5GNR RAPT-FR acc p", round(r.p_value, 3), 0.004, 1e-3)
chk("5GNR RAPT-FR acc d", round(r.cohen_d, 2), -0.06, 6e-3)
# ablation stats
r = finst[finst.Model == "RAPT_CHEAP"].iloc[0]
chk("Ablation Cheap delta", round(r.mean_delta, 4), 0.0022, 1e-4)
chk("Ablation Cheap p", round(r.wilcoxon_p, 4), 0.3125, 1e-4)
chk("Ablation Cheap CI low", round(r.ci95_low, 4), -0.0036, 1e-4)
chk("Ablation Cheap CI high", round(r.ci95_high, 4), 0.0080, 1e-4)
chk("Ablation Cheap d", round(r.cohens_d, 2), 0.47, 6e-3)
# TOST
r = a7t[(a7t.dataset == "5g_campus") & (a7t.margin == 0.005)].iloc[0]
chk("TOST Campus p", float(f"{r.tost_p:.1e}"), 1.8e-4, 1e-5)
r = a7t[(a7t.dataset == "ugr16") & (a7t.margin == 0.005)].iloc[0]
chk("TOST UGR16 p", round(r.tost_p, 2), 0.95, 6e-3)
chk("TOST UGR16 mean_diff", round(r.mean_diff, 3), 0.040, 1e-3)
r = a7t[(a7t.dataset == "nordicdat") & (a7t.margin == 0.005)].iloc[0]
chk("TOST Nordic mean_diff", round(r.mean_diff, 3), -0.052, 1e-3)
r = a7t[(a7t.dataset == "5g_nr") & (a7t.margin == 0.005)].iloc[0]
chk("TOST 5GNR mean_diff", round(r.mean_diff, 3), -0.040, 1e-3)

print()
print("=" * 100)
print("INLINE — UGR'16 failure analysis (A1 pooled per-seed, 9B rerun)")
print("=" * 100)
# attack precision/recall from A1 pooled
for m, ap, ar, fp in [("RAPT", 0.31, 0.43, 26), ("Frozen", 0.99, 0.48, 0.1),
                      ("Full Retraining", 0.93, None, None), ("RAPT-Enhanced", 0.82, None, None)]:
    g = a1p[(a1p.dataset == "ugr16") & (a1p.method == m)]
    info(f"A1 ugr16 {m} attack_precision", round(g.attack_precision.mean(), 3))
    info(f"A1 ugr16 {m} attack_recall", round(g.attack_recall.mean(), 3))
    info(f"A1 ugr16 {m} fp_per_1000", round(g.fp_per_1000_attack_free.mean(), 2))

print()
print("=" * 100)
print("INLINE — detectors on 5G NR rerun (A6)")
print("=" * 100)
edd = a6[(a6.dataset == "5g_nr") & (a6.detector == "EDD")]
info("A6 5g_nr EDD retrains mean", round(edd.retrains.mean(), 1))
info("A6 5g_nr EDD cpu mean", round(edd.adaptation_cpu_sec.mean(), 2))
info("A6 5g_nr EDD pooled f1 mean", round(edd.pooled_macro_f1.mean(), 4))

print()
print("=" * 100)
print("INLINE — label delay (A8)")
print("=" * 100)
for ds, m, d0, d1 in [("nordicdat", "RAPT", 0.462, 0.336),
                      ("nordicdat", "Full Retraining", None, None),
                      ("nordicdat", "RAPT-Enhanced", None, None)]:
    g = a8[(a8.dataset == ds) & (a8.method == m)]
    v0 = g[g.label_delay == 0].pooled_macro_f1.mean()
    v1 = g[g.label_delay == 1].pooled_macro_f1.mean()
    info(f"A8 {ds}/{m} delay0", round(v0, 3))
    info(f"A8 {ds}/{m} delay1", round(v1, 3))
    info(f"A8 {ds}/{m} loss", round(v1 - v0, 3))

print()
print("=" * 100)
print("INLINE — oracle / regime stability (A2)")
print("=" * 100)
for ds in ["5g_campus", "ugr16", "nordicdat", "5g_nr"]:
    g = a2o[a2o.dataset == ds]
    info(f"A2 oracle {ds} pooled f1 mean", round(g.pooled_macro_f1.mean(), 4))
info("A2 oracle ugr16 std", round(a2o[a2o.dataset == "ugr16"].pooled_macro_f1.std(ddof=0), 4))
info("A2 campus same-regime revisit mean",
     round(a2r[(a2r.dataset == "5g_campus") & (a2r.f1_same_regime_prev_visit.notna())]
           .f1_same_regime_prev_visit.mean(), 4))
info("A2 campus max class_cond_feature_shift",
     a2r[a2r.dataset == "5g_campus"].class_cond_feature_shift.max())

print()
print("=" * 100)
print("INLINE — detector comparison prose")
print("=" * 100)
# "used 17x, 3.1x, 18x less CPU than EDD" (paper rounds 17.4->17, 3.1->3.1, 18.1->18)
for ds, ratio, tol in [("5G Campus QoS", 17, 0.45), ("UGR'16", 3.1, 0.05), ("NordicDat", 18, 0.15)]:
    e = det[(det.Dataset == ds) & (det.Method == "EDD")].adapt_cpu_mean.iloc[0]
    r = det[(det.Dataset == ds) & (det.Method == "RAPT")].adapt_cpu_mean.iloc[0]
    chk(f"Det {ds} EDD/RAPT cpu ratio", round(e / r, 1), ratio, tol)
# EDD 5g_nr rerun
edd = a6[(a6.dataset == "5g_nr") & (a6.detector == "EDD")]
chk("A6 5g_nr EDD retrains", round(edd.retrains.mean(), 1), 115.0, 0.5)
chk("A6 5g_nr EDD cpu", round(edd.adaptation_cpu_sec.mean(), 1), 10.5, 0.06)
chk("A6 5g_nr EDD pooled f1", round(edd.pooled_macro_f1.mean(), 3), 0.885, 1e-3)
chk("A6 5g_nr Frozen pooled f1", round(a1p[(a1p.dataset == "5g_nr") &
    (a1p.method == "Frozen")].pooled_macro_f1.mean(), 3), 0.896, 1e-3)

print()
print("=" * 100)
print("INLINE — reproducibility/robustness prose")
print("=" * 100)
# 9A rerun savings
for slug, sav in [("5g_campus", 84), ("ugr16", 92), ("nordicdat", 92)]:
    fr = a1[(a1.dataset == slug) & (a1.method == "Full Retraining")].adaptation_cpu_sec.mean()
    rp = a1[(a1.dataset == slug) & (a1.method == "RAPT")].adaptation_cpu_sec.mean()
    chk(f"Rerun saving {slug}", round(100 * (fr - rp) / fr), float(sav), 1.0)
# 5G NR rerun
fr5 = a1[(a1.dataset == "5g_nr") & (a1.method == "Full Retraining")].adaptation_cpu_sec.mean()
rp5 = a1[(a1.dataset == "5g_nr") & (a1.method == "RAPT")].adaptation_cpu_sec.mean()
chk("Rerun 5g_nr RAPT cpu", round(rp5, 2), 0.27, 6e-3)
chk("Rerun 5g_nr FR retrains", round(a1[(a1.dataset == "5g_nr") &
    (a1.method == "Full Retraining")].retrain_events.mean(), 0), 8.0, 0.05)
chk("Rerun 5g_nr RAPT reuses", round(a1[(a1.dataset == "5g_nr") &
    (a1.method == "RAPT")].reuse_events.mean(), 0), 5.0, 0.05)
chk("Rerun 5g_nr RAPT saving", round(100 * (fr5 - rp5) / fr5, 0), 65.0, 0.6)
# pooled gap shrinkage
for ds, slug, gap in [("Campus", "5g_campus", 0.011), ("NordicDat", "nordicdat", 0.019),
                      ("5G NR", "5g_nr", 0.009)]:
    fr = a1p[(a1p.dataset == slug) & (a1p.method == "Full Retraining")].pooled_macro_f1.mean()
    rp = a1p[(a1p.dataset == slug) & (a1p.method == "RAPT")].pooled_macro_f1.mean()
    chk(f"Pooled gap {ds}", round(fr - rp, 3), gap, 1e-3)
# label delay
for m, loss in [("RAPT", -0.126), ("Full Retraining", -0.060), ("RAPT-Enhanced", -0.054)]:
    g = a8[(a8.dataset == "nordicdat") & (a8.method == m)]
    d = g[g.label_delay == 1].pooled_macro_f1.mean() - g[g.label_delay == 0].pooled_macro_f1.mean()
    chk(f"Label delay Nordic {m}", round(d, 3), loss, 1e-3)
# UGR16 recall loss (paper: NordicDat RAPT class2 recall 0.04 vs FR 0.20 in rerun)
for m, r0 in [("RAPT", 0.04), ("Full Retraining", 0.20)]:
    g = a1p[(a1p.dataset == "nordicdat") & (a1p.method == m)]
    chk(f"Nordic class2 recall {m}", round(g.class2_recall.mean(), 2), r0, 6e-3)

print()
print("=" * 100)
print("INLINE — oracle / regime stability (A2)")
print("=" * 100)
for ds, f1 in [("ugr16", 0.5926), ("5g_campus", 0.9905)]:
    g = a2o[a2o.dataset == ds].pooled_macro_f1.mean()
    chk(f"A2 oracle {ds} pooled f1", round(g, 4), f1, 5e-5)
for ds, s in [("ugr16", 0.0077), ("5g_campus", 0.0014)]:
    g = a2o[a2o.dataset == ds].pooled_macro_f1.std(ddof=1)   # paper uses sample sd
    chk(f"A2 oracle {ds} std", round(g, 4), s, 5e-5)
# campus same-regime revisit mean F1 = 0.9876
g = a2r[(a2r.dataset == "5g_campus") & (a2r.f1_same_regime_prev_visit.notna())]
chk("A2 campus revisit mean f1", round(g.f1_same_regime_prev_visit.mean(), 4), 0.9876, 5e-5)
# max class-conditional shift 1e8
mx = a2r[a2r.dataset == "5g_campus"].class_cond_feature_shift.max()
info("A2 campus max class_cond_feature_shift", mx)
info("A2 campus shift >= 1e8?", bool(mx >= 1e8))

print()
print("=" * 100)
print("INLINE — UGR'16 failure analysis (attack metrics, A1 pooled)")
print("=" * 100)
for m, ap, ar, fp in [("RAPT", 0.31, 0.43, 26), ("Frozen", 0.99, 0.48, 0.1),
                      ("Full Retraining", 0.93, 0.37, None), ("RAPT-Enhanced", 0.82, 0.48, None)]:
    g = a1p[(a1p.dataset == "ugr16") & (a1p.method == m)]
    chk(f"A1 ugr16 {m} attack_precision", round(g.attack_precision.mean(), 2), ap, 6e-3)
    if ar is not None:
        chk(f"A1 ugr16 {m} attack_recall", round(g.attack_recall.mean(), 2), ar, 6e-3)
    if fp is not None:
        chk(f"A1 ugr16 {m} fp/1000", round(g.fp_per_1000_attack_free.mean(), 1), fp,
            0.1 if fp < 1 else 0.5)
# paper says "attack-class recall is similar across methods (0.43 RAPT, 0.48 Frozen)"
# but Full Retraining's rerun attack-recall is 0.369, so "similar across methods" is loose.
info("NOTE paper 'similar recall across methods': FR attack_recall =",
     round(a1p[(a1p.dataset == "ugr16") &
               (a1p.method == "Full Retraining")].attack_recall.mean(), 3))

print()
print("=" * 100)
print("INLINE — remaining prose claims")
print("=" * 100)
# features
for slug, n in [("5g_campus", 19), ("ugr16", 134), ("nordicdat", 16)]:
    import json
    d = json.load(open(os.path.join(ROOT, "results", "experiment_9a_three", "raw",
                                    f"stream_def_{slug}_full.json")))
    chk(f"features {slug}", float(len(d["feature_columns"])), float(n), 0)
chk("features 5G NR", 12.0, 12.0, 0)  # extract_window_features returns 12 keys
# EDD std
for ds, s in [("5G Campus QoS", 0.0019), ("NordicDat", 0.0165)]:
    r = det[(det.Dataset == ds) & (det.Method == "EDD")].iloc[0]
    chk(f"T4 {ds}/EDD std", round(float(r["Macro-F1"].split(" +/- ")[1]), 4), s, 5e-5)
# parity refits
for ds, n in [("5g_campus", 0.0), ("ugr16", 0.0), ("nordicdat", 18.2), ("5g_nr", 5.4)]:
    g = a1[(a1.dataset == ds) & (a1.method == "RAPT-Enhanced")].parity_refits.mean()
    chk(f"parity refits {ds}", round(g, 1), n, 0.06)
# UGR16 Enhanced trained same 1200 trees as RAPT
chk("UGR16 RAPT trees", round(a1[(a1.dataset == "ugr16") &
    (a1.method == "RAPT")].trees_trained.mean(), 0), 1200.0, 0.05)
chk("UGR16 Enh trees", round(a1[(a1.dataset == "ugr16") &
    (a1.method == "RAPT-Enhanced")].trees_trained.mean(), 0), 1200.0, 0.05)
# RAPT-Floor threshold in config
cfg = open(os.path.join(ROOT, "Final_Experiments", "config.py")).read()
info("RAPT-Floor absolute threshold present (0.97)", "0.97" in cfg)
# 5G NR runtime definition
info("5G NR wall-clock runtime stored?", "NO (9B stores total_cpu_sec only; "
     "t_init_wall computed but unused)")
# pooled std bound
mx = max(a1p[(a1p.method == m) & (a1p.dataset == ds)].pooled_macro_f1.std(ddof=1)
         for m in ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
         for ds in ["5g_campus", "ugr16", "nordicdat", "5g_nr"])
info("max pooled std (paper 'at most 0.031')", round(mx, 3))

print()
print("=" * 100)
nfail = sum(1 for _, _, _, ok in results if ok is not None and not bool(ok))
npass = sum(1 for _, _, _, ok in results if ok is not None and bool(ok))
print(f"TOTAL: {npass} PASS, {nfail} FAIL, "
      f"{sum(1 for _,_,_,ok in results if ok is None)} INFO")
print("=" * 100)
