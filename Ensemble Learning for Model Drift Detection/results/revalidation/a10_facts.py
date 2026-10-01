"""A10: dataset and Table I facts, regenerated from the loaders / stream defs.

No paper edits; writes facts to results/revalidation/A10_dataset_facts.csv and
prints a human-readable summary.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import OUT, write_csv  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def regime_recurrence(seq):
    """Count regime ids that appear in >= 2 disjoint runs (visits)."""
    visits = {}
    prev = None
    for r in seq:
        if r != prev:
            visits.setdefault(r, 0)
            visits[r] += 1
        prev = r
    recur = {r: v for r, v in visits.items() if v >= 2}
    return visits, recur


def main():
    rows = []
    for ds in ALL_DATASETS:
        stream, sd = get_stream(ds)
        seq = list(sd["regime_sequence"])
        if not seq:
            seq = stream.groupby("window_id")["regime_id"].first().tolist()
        visits, recur = regime_recurrence(seq)
        n_recur_ids = len(recur)
        n_recur_visits = int(sum(v - 1 for v in recur.values()))
        n_attacks = None
        if ds == "UGR'16":
            y = stream["y"].values
            n_init = sd["initial_train_windows"]
            wid = stream["window_id"].values
            n_attacks = int((y == 1).sum())
            att_pref = int(((y == 1) & (wid < n_init)).sum())
            rows.append({"field": "attack_minutes_total", "value": n_attacks})
            rows.append({"field": "attack_minutes_prefix", "value": att_pref})
            rows.append({"field": "attack_minutes_eval", "value": n_attacks - att_pref})
        rows += [
            {"dataset": ds, "field": "window_size", "value": sd["window_size"]},
            {"dataset": ds, "field": "n_raw_samples", "value": sd["n_raw_samples"]},
            {"dataset": ds, "field": "n_usable_samples", "value": sd["n_usable_samples"]},
            {"dataset": ds, "field": "total_windows", "value": sd["total_windows"]},
            {"dataset": ds, "field": "initial_train_windows", "value": sd["initial_train_windows"]},
            {"dataset": ds, "field": "eval_windows", "value": sd["total_windows"] - sd["initial_train_windows"]},
            {"dataset": ds, "field": "n_classes", "value": sd["n_classes"]},
            {"dataset": ds, "field": "n_distinct_regimes", "value": len(set(seq))},
            {"dataset": ds, "field": "n_recurring_regime_ids", "value": n_recur_ids},
            {"dataset": ds, "field": "n_recurrence_events", "value": n_recur_visits},
            {"dataset": ds, "field": "n_windows_with_duplicate_rows", "value": np.nan},
        ]
        if ds == "5G NR":
            rows.append({"dataset": ds, "field": "prefix_round_499x0.2",
                         "value": int(round(sd["total_windows"] * 0.2))})
    df = pd.DataFrame(rows)
    write_csv(df, "A10_dataset_facts.csv")
    print(df.to_string(index=False))

    # NordicDat duplicate rows
    import three_dataset_load as L  # noqa
    nd = L.load_nordicdat()
    df_nd = pd.read_csv(L.NORDICDAT_CSV)
    dups = int(df_nd.duplicated().sum())
    print(f"\nNordicDat raw rows={len(df_nd)} duplicate rows={dups} "
          f"usable={nd and 0}")
    print("NordicDat regime field =", "operator" if "operator" in df_nd.columns else "band")
    print("NordicDat columns:", [c for c in df_nd.columns][:20])

    # UGR'16 loader comment vs code
    print("\nUGR'16 loader regime comment says '10 WD + 10 WE'; code builds "
          "WD/WE x 6 four-hour periods.")

    # Campus regime source
    camp = pd.read_csv(os.path.join(ROOT, "experiments", "exp9", "data",
                                    "processed_exp9_stream.csv"))
    print("Campus rows:", len(camp), "regime col present:",
          "regime_label" in camp.columns)
    if "regime_label" in camp.columns:
        print("Campus distinct regimes:", camp["regime_label"].nunique())


if __name__ == "__main__":
    main()
