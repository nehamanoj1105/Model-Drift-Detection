"""Step B: pooled results for all four streams, all models, seeds 42-46.

Primary metric is pooled macro-F1 (concatenate y_true/y_pred over all evaluation
windows of a run, then macro-F1). Per-window macro-F1 is also written and is
always labelled secondary.

Models: Frozen, Event-Driven, Full Retraining, RAPT, RAPT-Enhanced and the four
detectors ADWIN, Page-Hinkley, EDD, EDMA (post A8 fix).

Outputs:
  v2/B_pooled_all_streams.csv        one row per dataset, method, seed
  v2/raw/B_<slug>_seed<seed>.csv     per-window raw records
  v2/B_reference_crosscheck.csv      UGR'16 vs the reference pooled values
  v2/B_statement_checks.csv          the five paper statements under both metrics
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

R = lib.runner()
S = lib.streams()

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced",
           "ADWIN", "Page-Hinkley", "EDD", "EDMA"]
DATASETS = ["5G Campus QoS", "UGR'16", "NordicDat", "5G NR"]

# reference pooled macro-F1 from the earlier revalidation (tolerance 0.002)
REFERENCE = {"Frozen": 0.8216, "Event-Driven": 0.8413, "Full Retraining": 0.7603,
             "EDD": 0.8256, "RAPT": 0.6629, "RAPT-Enhanced": 0.8008}


def main():
    lib.ensure_dirs()
    pooled_rows, pwin_rows = [], []
    for ds in DATASETS:
        stream, sd = S.get_stream(ds)
        slug = sd["slug"]
        print(f"### {ds} windows={sd['total_windows']} prefix={sd['initial_train_windows']} "
              f"classes={sd['n_classes']}", flush=True)
        for seed in lib.SEEDS:
            pw, sm, pooled, _ = R.run_seed_v2(stream, sd, seed, METHODS)
            pw.to_csv(os.path.join(lib.RAW, f"B_{slug}_seed{seed}.csv"), index=False)
            pooled = pooled.copy()
            pooled.insert(0, "dataset", slug)
            pooled.insert(1, "dataset_name", ds)
            pooled_rows.append(pooled)
            sm = sm.copy()
            sm.insert(0, "dataset", slug)
            sm["seed"] = seed
            pwin_rows.append(sm)
            print(f"  seed {seed}: " + " | ".join(
                f"{r.method}={r.pooled_macro_f1:.4f}" for r in pooled.itertuples()),
                flush=True)
    B = pd.concat(pooled_rows, ignore_index=True)
    lib.write_csv(B, "B_pooled_all_streams.csv")
    P = pd.concat(pwin_rows, ignore_index=True)
    lib.write_csv(P, "B_perwindow_all_streams.csv")

    # cross-check UGR'16 against the reference values
    ugr = B[B.dataset == "ugr16"].groupby("method")["pooled_macro_f1"].mean()
    ref_rows = []
    for m, ref in REFERENCE.items():
        if m in ugr.index:
            got = float(ugr[m])
            ref_rows.append({"dataset": "ugr16", "method": m,
                             "pooled_macro_f1_reference": ref,
                             "pooled_macro_f1_v2": got,
                             "delta": got - ref,
                             "within_tolerance_0.002": abs(got - ref) <= 0.002})
    lib.write_csv(pd.DataFrame(ref_rows), "B_reference_crosscheck.csv")
    print("\nUGR'16 cross-check:")
    print(pd.DataFrame(ref_rows).to_string(index=False))

    statements(B, P)


def _by(B, metric):
    return B.groupby(["dataset", "method"])[metric].mean()


def statements(B, P):
    """Evaluate the five paper statements under pooled and per-window metrics.

    Restricted to the paper's five models (Frozen, Event-Driven, Full Retraining,
    RAPT, RAPT-Enhanced); the detector models are excluded from the ranking so the
    statement is about the model set the paper actually compares. Ties are treated
    as satisfying "best"/"worst".
    """
    PAPER_MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
    pooled = _by(B[B.method.isin(PAPER_MODELS)], "pooled_macro_f1")
    pw = _by(P[P.method.isin(PAPER_MODELS)], "per_window_macro_f1")
    rows = []

    def add(text, metric_name, verdict, evidence):
        rows.append({"statement": text, "metric": metric_name,
                     "verdict": verdict, "evidence": evidence})

    for metric_name, m, table in (("pooled_macro_f1", pooled, B),
                                  ("per_window_macro_f1 (secondary)", pw, P)):
        # Frozen best on UGR'16 (ties count as best)
        u = m.loc["ugr16"]
        best = u.idxmax()
        add("Frozen is best on UGR'16", metric_name,
            "SUPPORTED" if np.isclose(u["Frozen"], u.max(), atol=1e-9) else "NOT SUPPORTED",
            f"best={best} ({u.max():.4f}); Frozen={u.get('Frozen', float('nan')):.4f}")
        # RAPT worst on UGR'16 (ties count as worst)
        worst = u.idxmin()
        add("RAPT is worst on UGR'16", metric_name,
            "SUPPORTED" if np.isclose(u["RAPT"], u.min(), atol=1e-9) else "NOT SUPPORTED",
            f"worst={worst} ({u.min():.4f}); RAPT={u.get('RAPT', float('nan')):.4f}")
        # Full Retraining best on three of four streams
        n_best = sum(1 for ds in ["5g_campus", "ugr16", "nordicdat", "5g_nr"]
                     if ds in m.index.get_level_values(0)
                     and m.loc[ds].idxmax() == "Full Retraining")
        add("Full Retraining is best on three of four streams", metric_name,
            "SUPPORTED" if n_best >= 3 else "NOT SUPPORTED",
            f"best on {n_best} of 4 streams")
        # RAPT below Full Retraining on all four streams
        below = 0
        for ds in ["5g_campus", "ugr16", "nordicdat", "5g_nr"]:
            if ds in m.index.get_level_values(0):
                if m.loc[ds].get("RAPT", np.nan) < m.loc[ds].get("Full Retraining", np.nan):
                    below += 1
        add("RAPT is below Full Retraining on all four streams", metric_name,
            "SUPPORTED" if below == 4 else "NOT SUPPORTED",
            f"RAPT below FR on {below} of 4 streams")
        # RAPT highest accuracy on NordicDat
        acccol = "pooled_accuracy" if metric_name.startswith("pooled") else "per_window_accuracy"
        a = table[table.dataset == "nordicdat"].groupby("method")[acccol].mean()
        add("RAPT has the highest accuracy on NordicDat", metric_name,
            "SUPPORTED" if a.idxmax() == "RAPT" else "NOT SUPPORTED",
            f"best accuracy={a.idxmax()} ({a.max():.4f}); RAPT={a.get('RAPT', float('nan')):.4f}")
    lib.write_csv(pd.DataFrame(rows), "B_statement_checks.csv")
    print("\nStatement checks:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
