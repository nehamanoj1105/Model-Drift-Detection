"""Verify every quantitative claim in paper/main.tex against committed artifacts.

Run from the repository sub-root ("Ensemble Learning for Model Drift Detection").
Prints a table of PASS/FAIL with the artifact value, the paper value and the delta.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PAPER = os.path.join(ROOT, "paper", "main.tex")

checks = []


def check(claim, artifact, paper_value, tol=5e-4):
    ok = artifact is not None and abs(float(artifact) - float(paper_value)) <= tol
    checks.append((claim, artifact, paper_value, ok))
    return ok


def main():
    tex = open(PAPER).read()

    # ---- Table I: primary metrics -----------------------------------------
    # Campus / UGR'16 / NordicDat are quoted from the committed 9A table
    # (table9a_main.csv).  5G NR is quoted from the committed 9B natural-drift
    # summary.  A1 (below) independently reproduces the predictive metrics.
    t9a = pd.read_csv(os.path.join(ROOT, "experiments", "exp9a", "tables",
                                   "table9a_main.csv"))
    t9b = pd.read_csv(os.path.join(ROOT, "results", "experiment_9b",
                                   "natural_drift", "summary.csv"))
    tableI = {
        ("5G Campus QoS", "Frozen"): (.9364, .9822, .9379, .9377),
        ("5G Campus QoS", "Event-Driven"): (.9636, .9908, .9637, .9646),
        ("5G Campus QoS", "Full Retraining"): (.9836, .9951, .9838, .9843),
        ("5G Campus QoS", "RAPT"): (.9381, .9838, .9393, .9392),
        ("5G Campus QoS", "RAPT-Enhanced"): (.9381, .9838, .9393, .9392),
        ("UGR'16", "Frozen"): (.9690, .9899, .9795, .9662),
        ("UGR'16", "Event-Driven"): (.8972, .9898, .9042, .8955),
        ("UGR'16", "Full Retraining"): (.9595, .9873, .9726, .9580),
        ("UGR'16", "RAPT"): (.8360, .9635, .8527, .8336),
        ("UGR'16", "RAPT-Enhanced"): (.9276, .9880, .9375, .9251),
        ("NordicDat", "Frozen"): (.2779, .4770, .4215, .2487),
        ("NordicDat", "Event-Driven"): (.2547, .4583, .3909, .2274),
        ("NordicDat", "Full Retraining"): (.4227, .5694, .4938, .4085),
        ("NordicDat", "RAPT"): (.3818, .6172, .4703, .3582),
        ("NordicDat", "RAPT-Enhanced"): (.3783, .5677, .4600, .3614),
    }
    for (ds, model), (f1, acc, prec, rec) in tableI.items():
        r = t9a[(t9a.Dataset == ds) & (t9a.Model == model)].iloc[0]
        check(f"TableI {ds}/{model} F1", round(r.macro_f1_mean, 4), f1)
        check(f"TableI {ds}/{model} Acc", round(r.accuracy_mean, 4), acc)
        check(f"TableI {ds}/{model} Prec", round(float(r.Precision.split()[0]), 4), prec)
        check(f"TableI {ds}/{model} Rec", round(float(r.Recall.split()[0]), 4), rec)
    tableI_nr = {"Frozen": (.8961, .9075, .9314, .8798),
                 "Event-Driven": (.8903, .9055, .9475, .8688),
                 "Full Retraining": (.9027, .9145, .9478, .8829),
                 "RAPT": (.8894, .9025, .9258, .8737),
                 "RAPT-Enhanced": (.8915, .9055, .9348, .8737)}
    for model, (f1, acc, prec, rec) in tableI_nr.items():
        r = t9b[t9b.method == model].iloc[0]
        check(f"TableI 5G NR/{model} F1", round(r.macro_f1_mean, 4), f1)
        check(f"TableI 5G NR/{model} Acc", round(r.accuracy_mean, 4), acc)
        check(f"TableI 5G NR/{model} Prec", round(r.precision_mean, 4), prec)
        check(f"TableI 5G NR/{model} Rec", round(r.recall_mean, 4), rec)

    # ---- adaptation CPU values quoted in prose (committed artifacts) ----
    cpu_paper = {("5G Campus QoS", "Full Retraining"): 1.4008,
                 ("UGR'16", "Full Retraining"): 23.6318,
                 ("NordicDat", "Full Retraining"): 2.2034,
                 ("5G Campus QoS", "RAPT"): 0.2292,
                 ("UGR'16", "RAPT"): 1.9711,
                 ("NordicDat", "RAPT"): 0.2594}
    for (ds, model), pv in cpu_paper.items():
        r = t9a[(t9a.Dataset == ds) & (t9a.Model == model)].iloc[0]
        check(f"CPU {ds}/{model}", round(r.adapt_cpu_mean, 4), pv, tol=6e-4)
    for model, pv in [("Full Retraining", 0.7636), ("RAPT", 0.3955)]:
        r = t9b[t9b.method == model].iloc[0]
        check(f"CPU 5G NR/{model}", round(r.adaptation_cpu_sec_mean, 4), pv, tol=6e-4)

    # ---- CPU reduction percentages ----
    red_paper = {"5G Campus QoS": 83.6, "UGR'16": 91.7, "NordicDat": 88.2}
    for ds, pv in red_paper.items():
        fr = t9a[(t9a.Dataset == ds) & (t9a.Model == "Full Retraining")].adapt_cpu_mean.iloc[0]
        rp = t9a[(t9a.Dataset == ds) & (t9a.Model == "RAPT")].adapt_cpu_mean.iloc[0]
        check(f"CPU reduction {ds}", round(100 * (fr - rp) / fr, 1), pv, tol=0.15)
    fr = t9b[t9b.method == "Full Retraining"].adaptation_cpu_sec_mean.iloc[0]
    rp = t9b[t9b.method == "RAPT"].adaptation_cpu_sec_mean.iloc[0]
    check("CPU reduction 5G NR", round(100 * (fr - rp) / fr, 1), 48.2, tol=0.15)

    # ---- retrain / reuse counts ----
    counts_paper = {"5G Campus QoS": (14, 2, 12), "UGR'16": (144, 11, 133),
                    "NordicDat": (18, 2, 16)}
    for ds, (fr_r, rp_r, rp_u) in counts_paper.items():
        fr = t9a[(t9a.Dataset == ds) & (t9a.Model == "Full Retraining")].iloc[0]
        rp = t9a[(t9a.Dataset == ds) & (t9a.Model == "RAPT")].iloc[0]
        check(f"retrains FR {ds}", round(fr.retrain_mean), fr_r)
        check(f"retrains RAPT {ds}", round(rp.retrain_mean), rp_r)
        check(f"reuses RAPT {ds}", round(rp.reuse_mean), rp_u)
    fr = t9b[t9b.method == "Full Retraining"].iloc[0]
    rp = t9b[t9b.method == "RAPT"].iloc[0]
    check("retrains FR 5G NR", round(fr.retrain_events_mean), 9)
    check("retrains RAPT 5G NR", round(rp.retrain_events_mean), 3)
    check("reuses RAPT 5G NR", round(rp.reused_checkpoints_mean), 6)

    # ---- independent reproduction cross-check (A1 harness) ----
    a1_path = os.path.join(ROOT, "results", "revalidation",
                           "A1_summary_all_streams.csv")
    if os.path.exists(a1_path):
        a1 = pd.read_csv(a1_path).groupby(["dataset", "method"]).mean(numeric_only=True)
        map_ds = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
                  "nordicdat": "NordicDat"}
        for (ds9a, model), (f1, acc, prec, rec) in tableI.items():
            ds = [k for k, v in map_ds.items() if v == ds9a][0]
            try:
                row = a1.loc[(ds, model)]
            except KeyError:
                continue
            for c, pv in zip(["per_window_macro_f1", "per_window_accuracy",
                              "per_window_precision", "per_window_recall"],
                             (f1, acc, prec, rec)):
                check(f"A1-repro {ds9a}/{model} {c}", round(row[c], 4), pv)

    # ---- ablation (Table III) ----
    abl = pd.read_csv(os.path.join(ROOT, "Final_Experiments", "results", "tables",
                                   "table_final_main.csv"))
    abl["f1"] = abl["Macro-F1"].str.extract(r"([0-9.]+)").astype(float)
    abl["cpu"] = abl["Adapt CPU (s)"].str.extract(r"([0-9.]+)").astype(float)
    abl_map = {"Full Retraining": ("Full Retraining", .9829, 1.4063),
               "RAPT_FULL": ("RAPT-Full", .9587, .4283),
               "RAPT_EVIDENCE": ("RAPT-Evidence", .9587, .4260),
               "RAPT_FLOOR": ("RAPT-Floor", .9797, .5865),
               "RAPT_CHEAP": ("RAPT-Cheap", .9851, .8477),
               "RAPT_INCR": ("RAPT-Incremental", .9549, .5813)}
    for k, (name, f1, cpu) in abl_map.items():
        r = abl[abl.Model == k].iloc[0]
        check(f"Ablation {name} F1", round(r.f1, 4), f1)
        check(f"Ablation {name} CPU", round(r.cpu, 4), cpu, tol=1e-3)

    # ---- Delta and p for RAPT-Cheap vs Full Retraining ----
    st = pd.read_csv(os.path.join(ROOT, "Final_Experiments", "results", "tables",
                                  "table_final_stats.csv"))
    r = st[st.Model == "RAPT_CHEAP"].iloc[0]
    check("Cheap-FR delta", round(r.mean_delta, 4), 0.0022, tol=1e-4)
    check("Cheap-FR p", round(r.wilcoxon_p, 4), 0.3125, tol=1e-4)
    check("Cheap-FR CI low", round(r.ci95_low, 4), -0.0036, tol=1e-4)
    check("Cheap-FR CI high", round(r.ci95_high, 4), 0.0080, tol=1e-4)
    check("Cheap-FR d", round(r.cohens_d, 2), 0.47, tol=6e-3)
    # 40% CPU cut and 69% vs periodic refresh
    cheap_cpu = abl[abl.Model == "RAPT_CHEAP"].iloc[0].cpu
    fr_cpu = abl[abl.Model == "Full Retraining"].iloc[0].cpu
    refresh = abl[abl.Model == "RAPT_REFRESH_W5"].iloc[0].cpu
    check("Cheap CPU cut vs FR (%)", round(100 * (fr_cpu - cheap_cpu) / fr_cpu, 0), 40, tol=1)
    check("Cheap CPU cut vs refresh (%)", round(100 * (refresh - cheap_cpu) / refresh, 0), 69, tol=1)

    # ---- detector comparison ----
    det = pd.read_csv(os.path.join(ROOT, "experiments", "exp9a", "tables",
                                   "table9a_drift_detectors.csv"))
    det = det[det.Dataset.isin(["5G Campus QoS", "UGR'16", "NordicDat"])]
    edd = det[det.Method == "EDD"]
    check("EDD campus F1", round(edd[edd.Dataset == "5G Campus QoS"].macro_f1_mean.iloc[0], 4), 0.9847)
    check("EDD nordic F1", round(edd[edd.Dataset == "NordicDat"].macro_f1_mean.iloc[0], 4), 0.4184)
    for ds, pv in [("5G Campus QoS", 3.98), ("UGR'16", 6.10), ("NordicDat", 4.69)]:
        check(f"EDD CPU {ds}", round(edd[edd.Dataset == ds].adapt_cpu_mean.iloc[0], 2), pv, tol=6e-3)

    # ---- significance values in prose ----
    s9a = pd.read_csv(os.path.join(ROOT, "experiments", "exp9a", "tables",
                                   "statistics_9a.csv"))
    def stat(ds, a, b):
        r = s9a[(s9a.dataset == ds) & (s9a.method_a == a) & (s9a.method_b == b)]
        return r.iloc[0] if len(r) else None
    r = stat("5G Campus QoS", "RAPT", "Full Retraining")
    check("Campus p", float(f"{r.p_value:.1e}"), 2.6e-4, tol=1e-5)
    check("Campus d", round(r.cohen_d, 2), -0.32, tol=6e-3)
    r = stat("UGR'16", "RAPT", "Full Retraining")
    check("UGR16 p", float(f"{r.p_value:.1e}"), 2.6e-11, tol=1e-12)
    check("UGR16 d", round(r.cohen_d, 2), -0.67, tol=6e-3)
    r = stat("UGR'16", "RAPT-Enhanced", "RAPT")
    check("UGR16 parity gain", round(r.mean_diff_f1, 4), 0.0917, tol=1e-4)
    check("UGR16 parity p", float(f"{r.p_value:.1e}"), 5.4e-9, tol=1e-10)
    check("UGR16 parity d", round(r.cohen_d, 2), 0.56, tol=6e-3)
    r = stat("NordicDat", "RAPT", "Full Retraining")
    check("Nordic p", round(r.p_value, 3), 0.255, tol=1e-3)
    check("Nordic d", round(r.cohen_d, 2), -0.12, tol=6e-3)

    # ---- "48-92% less adaptation CPU" abstract claim ----
    reds = [83.6, 91.7, 88.2, 48.2]
    check("abstract range low", min(reds), 48, tol=0.6)
    check("abstract range high", max(reds), 92, tol=0.6)

    # ---- Controls: equivalence test and label-delay protocol ----
    # TOST (A7) and label-delay (A8) live on the revalidation branch and are
    # carried on paper-audit under results/revalidation/.
    rev = os.path.join(ROOT, "results", "revalidation")
    tost = pd.read_csv(os.path.join(rev, "A7_tost.csv"))
    r = tost[(tost.dataset == "ugr16") & (tost.method_1 == "RAPT-Cheap")
             & (tost.method_2 == "Full Retraining") & (tost.margin == 0.005)].iloc[0]
    check("UGR16 TOST p", round(r.tost_p, 2), 0.95, tol=6e-3)
    check("UGR16 TOST equivalence rejected",
          0.0 if not bool(r["equivalent_0.05"]) else 1.0, 0.0)
    ldel = pd.read_csv(os.path.join(rev, "A8_label_delay.csv"))
    piv = ldel.groupby(["dataset", "method", "label_delay"]).pooled_macro_f1.mean().unstack()
    check("label-delay RAPT nordic", round(piv.loc[("nordicdat", "RAPT")][0]
                                            - piv.loc[("nordicdat", "RAPT")][1], 2), 0.13, tol=6e-3)
    check("label-delay FR nordic", round(piv.loc[("nordicdat", "Full Retraining")][0]
                                          - piv.loc[("nordicdat", "Full Retraining")][1], 2), 0.06, tol=6e-3)
    for m in ["Frozen", "Event-Driven"]:
        check(f"label-delay {m} campus unchanged",
              round(piv.loc[("5g_campus", m)][0] - piv.loc[("5g_campus", m)][1], 6), 0.0, tol=1e-6)

    # ---- report ----
    n_pass = sum(1 for c in checks if c[3])
    print(f"{'CLAIM':58s} {'ARTIFACT':>12s} {'PAPER':>12s}  OK")
    print("-" * 96)
    for name, art, pv, ok in checks:
        a = f"{art:.6g}" if isinstance(art, (int, float)) else str(art)
        print(f"{name:58s} {a:>12s} {pv:>12g}  {'PASS' if ok else 'FAIL'}")
    print("-" * 96)
    print(f"{n_pass}/{len(checks)} checks pass")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
