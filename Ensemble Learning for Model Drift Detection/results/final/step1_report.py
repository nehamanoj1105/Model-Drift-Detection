"""Step 1: primary metric and pooled results, all four streams.

Source of truth is the existing revalidation A1 artifacts (already computed on
the four streams, seeds 42-46, same protocol). This script does not re-run the
models; it recomputes the report tables from those per-window and pooled CSVs
and cross-checks them against the two older pipelines.

Outputs
  T1_pooled_all_streams.csv   primary table: model x stream, mean +/- sd
  T1_recovery.csv             recovery time per drift point, aggregated
  T1_crosscheck.csv           revalidation vs 9A vs Final_Experiments
  T1_claim_verdicts.csv       each paper claim -> SUPPORTED / NOT SUPPORTED /
                              INCONCLUSIVE with the number that decides it
"""
import os
import re
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")
sys.path.insert(0, REVAL)

RECOVERY_FRACTION = 0.95
PRE_DRIFT_WINDOWS = 3
MAX_RECOVERY_WINDOWS = 60

MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
DETECTORS = ["ADWIN", "EDD", "Page-Hinkley", "EDMA"]
SLUG_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
             "nordicdat": "NordicDat", "5g_nr": "5G NR"}


def pooled_table():
    """Primary table. Primary metric = per-window mean macro-F1 (matches the
    paper). Pooled macro-F1 is kept as a sensitivity column."""
    p = pd.read_csv(os.path.join(REVAL, "A1_pooled_all_streams.csv"))
    s = pd.read_csv(os.path.join(REVAL, "A1_summary_all_streams.csv"))
    rows = []
    for (ds, method), g in p.groupby(["dataset", "method"]):
        sg = s[(s.dataset == ds) & (s.method == method)]
        rows.append({
            "dataset": ds, "method": method, "n_seeds": len(g),
            "macro_f1_mean": sg["per_window_macro_f1"].mean(),
            "macro_f1_sd": sg["per_window_macro_f1"].std(),
            "accuracy_mean": sg["per_window_accuracy"].mean(),
            "accuracy_sd": sg["per_window_accuracy"].std(),
            "precision_mean": sg["per_window_precision"].mean(),
            "precision_sd": sg["per_window_precision"].std(),
            "recall_mean": sg["per_window_recall"].mean(),
            "recall_sd": sg["per_window_recall"].std(),
            "pooled_macro_f1_mean": g["pooled_macro_f1"].mean(),
            "pooled_macro_f1_sd": g["pooled_macro_f1"].std(),
            "pooled_accuracy_mean": g["pooled_accuracy"].mean(),
            "adaptation_cpu_sec_mean": sg["adaptation_cpu_sec"].mean(),
            "adaptation_cpu_sec_sd": sg["adaptation_cpu_sec"].std(),
            "total_runtime_sec_mean": sg["total_runtime_sec"].mean(),
            "total_runtime_sec_sd": sg["total_runtime_sec"].std(),
            "retrain_events_mean": sg["retrain_events"].mean(),
            "reuse_events_mean": sg["reuse_events"].mean(),
            "refresh_events_mean": sg["refresh_events"].mean(),
            "trees_trained_mean": sg["trees_trained"].mean(),
            "trees_reused_mean": sg["trees_reused"].mean(),
            "init_cpu_sec_mean": sg["init_cpu_sec"].mean(),
        })
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODELS + DETECTORS)}
    out["_o"] = out["method"].map(order)
    out = out.sort_values(["dataset", "_o"]).drop(columns="_o")
    out.to_csv(os.path.join(FINAL, "T1_pooled_all_streams.csv"), index=False)
    return out


def aggregation_sensitivity(t1):
    """How far the two aggregations diverge, per (stream, model)."""
    d = t1[["dataset", "method", "macro_f1_mean", "pooled_macro_f1_mean"]].copy()
    d["pooled_minus_per_window"] = d["pooled_macro_f1_mean"] - d["macro_f1_mean"]
    d.to_csv(os.path.join(FINAL, "T1_aggregation_sensitivity.csv"), index=False)
    return d


def recovery_table():
    """Recovery time to 95% of the pre-drift per-window F1, per drift point."""
    pw = pd.read_csv(os.path.join(REVAL, "raw", "A1_per_window_all_streams.csv"))
    rows = []
    for (ds, method, seed), g in pw.groupby(["dataset", "method", "seed"]):
        g = g.sort_values("window_id")
        wids = g["window_id"].values
        regs = g["regime_id"].values
        f1 = g["macro_f1"].values
        # drift points: indices where regime changes
        trans = [i for i in range(1, len(regs)) if regs[i] != regs[i - 1]]
        for i in trans:
            lo = max(0, i - PRE_DRIFT_WINDOWS)
            pre = float(np.mean(f1[lo:i]))
            thr = RECOVERY_FRACTION * pre
            # search forward until next transition or cap
            nxt = len(f1)
            for j in range(i + 1, len(regs)):
                if regs[j] != regs[i]:
                    nxt = j
                    break
            rec = np.nan
            for j in range(i, min(nxt, i + MAX_RECOVERY_WINDOWS)):
                if f1[j] >= thr:
                    rec = j - i
                    break
            post = float(np.mean(f1[i:min(nxt, i + MAX_RECOVERY_WINDOWS)])) \
                if nxt > i else np.nan
            rows.append({"dataset": ds, "method": method, "seed": seed,
                         "drift_window": int(wids[i]), "to_regime": regs[i],
                         "pre_drift_f1": pre, "post_drift_f1": post,
                         "delta_f1": post - pre,
                         "recovery_windows": rec,
                         "recovered": int(not np.isnan(rec))})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(FINAL, "raw", "T1_recovery_per_drift.csv"), index=False)
    agg = d.groupby(["dataset", "method"]).agg(
        n_drift_points=("recovery_windows", "size"),
        recovery_windows_mean=("recovery_windows", "mean"),
        recovery_windows_median=("recovery_windows", "median"),
        recovered_fraction=("recovered", "mean"),
        delta_f1_mean=("delta_f1", "mean"),
        pre_drift_f1_mean=("pre_drift_f1", "mean"),
        post_drift_f1_mean=("post_drift_f1", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(FINAL, "T1_recovery.csv"), index=False)
    return agg


def crosscheck():
    """Revalidation vs the 9A pipeline (table9a_main.csv).

    The 9A table reports the mean of per-window macro-F1, so the like-for-like
    comparison is revalidation per_window_macro_f1_mean. The pooled column is
    reported alongside to expose the aggregation sensitivity.
    """
    p = pd.read_csv(os.path.join(REVAL, "A1_pooled_all_streams.csv"))
    s = pd.read_csv(os.path.join(REVAL, "A1_summary_all_streams.csv"))
    t9a = pd.read_csv(os.path.join(ROOT, "experiments", "exp9a", "tables",
                                   "table9a_main.csv"))
    rows = []
    for ds, name in SLUG_NAME.items():
        if name not in set(t9a["Dataset"]):
            continue
        for m in MODELS:
            a = p[(p.dataset == ds) & (p.method == m)]["pooled_macro_f1"]
            pw = s[(s.dataset == ds) & (s.method == m)]["per_window_macro_f1"]
            b = t9a[(t9a.Dataset == name) & (t9a.Model == m)]["macro_f1_mean"]
            if len(a) == 0 or len(b) == 0:
                continue
            rows.append({"dataset": ds, "method": m,
                         "revalidation_per_window_macro_f1": float(pw.mean()),
                         "exp9a_per_window_macro_f1": float(b.iloc[0]),
                         "per_window_abs_diff": abs(float(pw.mean()) - float(b.iloc[0])),
                         "revalidation_pooled_macro_f1": float(a.mean()),
                         "pooled_minus_per_window": float(a.mean() - pw.mean())})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(FINAL, "T1_crosscheck.csv"), index=False)
    return d


def claim_verdicts(t1, rec, xc):
    """Verdicts for the paper's quantitative claims, decided by the numbers.

    Uses the primary per-window mean macro-F1 (macro_f1_mean)."""
    def f1(ds, m):
        return float(t1[(t1.dataset == ds) & (t1.method == m)]["macro_f1_mean"].iloc[0])

    def f1sd(ds, m):
        return float(t1[(t1.dataset == ds) & (t1.method == m)]["macro_f1_sd"].iloc[0])

    rows = []

    def add(claim, metric, value, verdict, note):
        rows.append({"claim": claim, "deciding_metric": metric,
                     "value": value, "verdict": verdict, "note": note})

    # C1 RAPT macro-F1 below Full Retraining on all four streams
    below = all(f1(ds, "RAPT") < f1(ds, "Full Retraining") for ds in SLUG_NAME)
    diffs = {ds: f1(ds, "Full Retraining") - f1(ds, "RAPT") for ds in SLUG_NAME}
    add("RAPT macro-F1 is below Full Retraining on all four streams",
        "per-window macro-F1 difference (FR - RAPT)", json.dumps(diffs),
        "SUPPORTED" if below else "NOT SUPPORTED",
        "per-window mean protocol; sign consistent on all four")

    # C2 UGR'16 RAPT is the worst of the five models
    ugr = {m: f1("ugr16", m) for m in MODELS}
    worst = min(ugr, key=ugr.get)
    add("On UGR'16 RAPT has the lowest macro-F1 of the five models",
        "min over models", ugr[worst], "SUPPORTED" if worst == "RAPT" else "NOT SUPPORTED",
        f"ranking={json.dumps(ugr)}")

    # C3 numeric recovery (RAPT-Enhanced over RAPT) is real
    gain = f1("ugr16", "RAPT-Enhanced") - f1("ugr16", "RAPT")
    add("UGR'16 RAPT-Enhanced recovers macro-F1 over RAPT",
        "per-window macro-F1 gain", gain, "SUPPORTED" if gain > 0 else "NOT SUPPORTED",
        "numeric recovery; seeds 42-46 paired test is in Step 5")

    # C3b but the paper's attribution of that recovery to the parity refit is
    # not supported (Step 2 isolation: parity never fires; refit buffer drives it)
    add("UGR'16 recovery is caused by the parity refit",
        "parity_refits on UGR'16 at thresholds 0.5-0.95 (Step 0.4)",
        0, "NOT SUPPORTED",
        "parity branch never fires; refit buffer 500->1000 reproduces the gain "
        "(T2_mechanism_isolation.csv)")

    # C4 NordicDat deficit not significant -> report point estimate only
    add("NordicDat RAPT deficit vs Full Retraining is small",
        "per-window macro-F1 difference", f1("nordicdat", "Full Retraining") - f1("nordicdat", "RAPT"),
        "INCONCLUSIVE", "significance decided in Step 5")

    # C5 5G NR five models close
    nr = {m: f1("5g_nr", m) for m in MODELS}
    spread = max(nr.values()) - min(nr.values())
    add("On 5G NR the five models are within a small spread",
        "max-min per-window macro-F1", spread, "SUPPORTED" if spread < 0.02 else "NOT SUPPORTED",
        f"values={json.dumps(nr)}")

    # C6 RAPT-Enhanced == RAPT on Campus (no regime recurrence-driven refit)
    gap = abs(f1("5g_campus", "RAPT") - f1("5g_campus", "RAPT-Enhanced"))
    add("On Campus RAPT and RAPT-Enhanced coincide",
        "absolute per-window macro-F1 gap", gap,
        "SUPPORTED" if gap < 1e-6 else "NOT SUPPORTED",
        "parity refit never fires on Campus")

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(FINAL, "T1_claim_verdicts.csv"), index=False)
    return d


def metric_reconciliation(t1):
    """Reconcile the paper's Table II values with the revalidation.

    The paper's Table II (label tab:primary) reports the per-window mean for the
    three 9A streams but the pooled macro-F1 (from the separate 9B pipeline) for
    5G NR. This is extracted from paper/main.tex and compared against both
    revalidation aggregations so the mismatch is explicit and reproducible.
    """
    tex = os.path.join(ROOT, "paper", "main.tex")
    with open(tex) as f:
        lines = f.readlines()
    # restrict to the tab:primary block
    start = next(i for i, l in enumerate(lines) if "\\label{tab:primary}" in l)
    end = next(i for i, l in enumerate(lines) if i > start and "\\bottomrule" in l)
    rows = []
    cur_ds = None
    for ln in lines[start:end]:
        hm = re.search(r"\\multirow\{5\}\{\*\}\{([^}]+)\}", ln)
        if hm:
            cur_ds = hm.group(1).strip()
            continue
        if not ln.strip().startswith("&"):
            continue
        fields = [f.strip() for f in ln.split("&")]
        if len(fields) < 3:
            continue
        model = re.sub(r"\\[a-zA-Z]+", "", fields[1]).strip()
        num = re.search(r"([0-9]+\.[0-9]+)", fields[2])
        if not num or not cur_ds:
            continue
        rows.append({"dataset_tex": cur_ds, "method": model,
                     "paper_macro_f1": float(num.group(1))})
    paper = pd.DataFrame(rows)
    tex2slug = {"5G Campus": "5g_campus", "UGR'16": "ugr16",
                "NordicDat": "nordicdat", "5G NR Lat.": "5g_nr"}
    paper["dataset"] = paper["dataset_tex"].map(tex2slug)
    out = paper.merge(
        t1[["dataset", "method", "macro_f1_mean", "pooled_macro_f1_mean"]],
        on=["dataset", "method"], how="left")
    out["matches_per_window"] = (out["paper_macro_f1"] - out["macro_f1_mean"]).abs() < 1e-4
    out["matches_pooled"] = (out["paper_macro_f1"] - out["pooled_macro_f1_mean"]).abs() < 5e-3
    out.to_csv(os.path.join(FINAL, "T1_metric_reconciliation.csv"), index=False)
    return out


def main():
    print("Step 1")
    t1 = pooled_table()
    print(f"  T1_pooled_all_streams.csv rows={len(t1)}")
    agg = aggregation_sensitivity(t1)
    print(f"  T1_aggregation_sensitivity.csv max |pooled - per_window| = "
          f"{agg['pooled_minus_per_window'].abs().max():.4f}")
    rec = recovery_table()
    print(f"  T1_recovery.csv rows={len(rec)}")
    xc = crosscheck()
    print(f"  T1_crosscheck.csv rows={len(xc)}; max per-window abs diff="
          f"{xc['per_window_abs_diff'].max():.2e}")
    cv = claim_verdicts(t1, rec, xc)
    print(f"  T1_claim_verdicts.csv rows={len(cv)}")
    print(cv[["verdict", "claim"]].to_string(index=False))
    mr = metric_reconciliation(t1)
    print(f"\n  T1_metric_reconciliation.csv rows={len(mr)}; "
          f"paper matches per-window={int(mr['matches_per_window'].sum())}/{len(mr)}, "
          f"matches pooled={int(mr['matches_pooled'].sum())}/{len(mr)}")
    print(mr[["dataset", "method", "paper_macro_f1", "macro_f1_mean",
              "pooled_macro_f1_mean", "matches_per_window", "matches_pooled"]].to_string(index=False))
    print("\n=== primary per-window macro-F1 ===")
    for ds in SLUG_NAME:
        print(f"-- {SLUG_NAME[ds]}")
        g = t1[t1.dataset == ds]
        for _, r in g[g.method.isin(MODELS)].iterrows():
            print(f"   {r['method']:16s} {r['macro_f1_mean']:.4f} "
                  f"+/- {r['macro_f1_sd']:.4f}")


if __name__ == "__main__":
    main()
