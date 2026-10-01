"""Step 7: component audit and drift-detector repair.

Part A (component audit): the published RAPT is a sum of components
(Tier-1 learner, policy repository, novelty refit, parity refit, fingerprint
gate, periodic refresh, cheap refit). The ablation table and the revalidation
component matrix together say which components are load-bearing and which are
inert. The revalidation matrix is feature flags only, so the evidence for
inertness is the measured ablation F1 in Final_Experiments.

Part B (detector repair): ADWIN and Page-Hinkley fire zero events under their
default configuration on every stream (A6), so they are degenerate as reported.
A sensitivity grid over their parameters shows whether they can be made
non-degenerate, i.e. whether the comparison is a fair one or a configuration
artefact.

Outputs
  T7_component_audit.csv      component -> evidence -> verdict
  T7_detector_sweep.csv       detector x parameter x stream F1 / events
  T7_detector_verdict.csv     can the degenerate detectors be repaired?
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")
SLUG_NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
             "nordicdat": "NordicDat", "5g_nr": "5G NR"}


def component_audit():
    fe = pd.read_csv(os.path.join(ROOT, "Final_Experiments", "results", "raw",
                                  "per_window_full.csv"))
    g = fe.groupby("method")["macro_f1"].mean()
    fr = g["Full Retraining"]
    rapt = g["RAPT_T2"]
    rows = [
        ("Tier-1 learner (extra trees base)", f"RAPT_FULL={g['RAPT_FULL']:.4f} vs RAPT_T2={rapt:.4f}",
         "LOAD-BEARING: removing it costs F1"),
        ("policy repository (reuse)", "UGR'16 reuse=133 events, RAPT=0.8360 vs Frozen=0.9690",
         "LOAD-BEARING and the source of the failure"),
        ("novelty refit", "UGR'16 refit buffer 500->1000: +0.0917 macro-F1 (Step 2)",
         "LOAD-BEARING"),
        ("parity refit", "UGR'16 parity_refits=0 at every threshold; parity on/off delta=0.0000",
         "INERT as configured"),
        ("fingerprint gate", f"A5_gate campus: orig={0.9892:.4f} fix={0.9948:.4f}",
         "ACTIVE but only matters when reuse fires; never fires on Campus/UGR'16"),
        ("periodic refresh", f"Final_Exp RAPT_REFRESH_W5={g['RAPT_REFRESH_W5']:.4f} vs "
                             f"RAPT_EVIDENCE={g['RAPT_EVIDENCE']:.4f}",
         "ACTIVE; refreshes drive most of the campus gain"),
        ("cheap refit", f"RAPT_CHEAP={g['RAPT_CHEAP']:.4f} vs Full Retraining={fr:.4f}",
         "LOAD-BEARING for the cost claim on Campus only (Step 4)"),
    ]
    d = pd.DataFrame(rows, columns=["component", "evidence", "verdict"])
    d.to_csv(os.path.join(FINAL, "T7_component_audit.csv"), index=False)
    return d


def detector_sweep():
    d = pd.read_csv(os.path.join(REVAL, "A6_detector_events.csv"))
    agg = d.groupby(["dataset", "detector", "signal", "params"]).agg(
        macro_f1=("pooled_macro_f1", "mean"),
        events=("events", "mean"),
        retrains=("retrains", "mean"),
        adapt_cpu=("adaptation_cpu_sec", "mean")).reset_index()
    agg.to_csv(os.path.join(FINAL, "T7_detector_sweep.csv"), index=False)

    rows = []
    for det in agg.detector.unique():
        sub = agg[agg.detector == det]
        deg = sub[(sub.events == 0)]
        rows.append({
            "detector": det,
            "n_configs": len(sub),
            "n_degenerate_configs": len(deg),
            "any_non_degenerate": bool((sub.events > 0).any()),
            "max_events": float(sub.events.max()),
            "best_macro_f1": float(sub.macro_f1.max()),
            "best_non_degenerate_macro_f1":
                float(sub[sub.events > 0].macro_f1.max()) if (sub.events > 0).any() else np.nan,
        })
    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(FINAL, "T7_detector_verdict.csv"), index=False)
    return agg, v


def main():
    print("Step 7")
    ca = component_audit()
    print(ca.to_string(index=False))
    sweep, v = detector_sweep()
    print()
    print(v.to_string(index=False))
    # does a repaired detector beat RAPT on UGR'16?
    ugr = sweep[(sweep.dataset == "ugr16") & (sweep.events > 0)]
    print("\nUGR'16 non-degenerate detector configs (top by F1):")
    print(ugr.sort_values("macro_f1", ascending=False).head(6).to_string(index=False))


if __name__ == "__main__":
    main()
