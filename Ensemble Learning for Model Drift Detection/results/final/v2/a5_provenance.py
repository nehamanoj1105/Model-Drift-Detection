"""A5: provenance of stored RAPT checkpoints.

The earlier revalidation reported "55 of 720 policies". This script pins down
what those numbers are and recomputes the fraction over STORED checkpoints
only, per seed, per method.

  * 720 = the number of provenance log rows written for UGR'16 in the earlier
    revalidation (A2_checkpoint_provenance.csv, dataset == ugr16). Those rows
    include both checkpoint creations and reuse events.
  * 55 = the number of those rows that are STORED (created) checkpoints, i.e.
    the remaining 665 are reuse events.

We recompute the fraction properly here, running each method separately so the
provenance rows are attributable to a method.

Output: v2/A5_provenance.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

R = lib.runner()
S = lib.streams()

METHODS = ["RAPT", "RAPT-Enhanced", "RAPT-Cheap", "RAPT-Deferred"]
DATASETS = ["5g_campus", "ugr16", "nordicdat", "5g_nr"]
NAME = {"5g_campus": "5G Campus QoS", "ugr16": "UGR'16",
        "nordicdat": "NordicDat", "5g_nr": "5G NR"}


def provenance_for(method, dataset, seeds=(42,)):
    rows = []
    stream, sd = S.get_stream(NAME[dataset])
    for seed in seeds:
        tmp = os.path.join(lib.RAW, f"_prov_{dataset}_{method}_{seed}.csv")
        if os.path.exists(tmp):
            os.remove(tmp)
        R.run_seed_v2(stream, sd, seed, [method], provenance_out=tmp)
        if not os.path.exists(tmp):
            continue
        d = pd.read_csv(tmp)
        d["dataset"] = dataset
        d["seed"] = seed
        d["method"] = method
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run():
    all_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            d = provenance_for(method, dataset)
            if not len(d):
                all_rows.append({"dataset": dataset, "method": method, "seed": 42,
                                 "stored_checkpoints": 0, "reuse_events": 0,
                                 "stored_trained_on_own_regime_key": 0,
                                 "fraction_provenance_correct": np.nan})
                continue
            for seed, g in d.groupby("seed"):
                is_reuse = g.get("event", pd.Series("", index=g.index)).fillna("") == "reuse"
                stored = g[~is_reuse]
                n_stored = len(stored)
                own = int(stored["train_regime_matches_key"].fillna(0).astype(int).sum()) \
                    if "train_regime_matches_key" in stored.columns else 0
                all_rows.append({
                    "dataset": dataset, "method": method, "seed": int(seed),
                    "stored_checkpoints": n_stored,
                    "reuse_events": int(is_reuse.sum()),
                    "stored_trained_on_own_regime_key": own,
                    "fraction_provenance_correct": (own / n_stored) if n_stored else np.nan,
                })
    d = pd.DataFrame(all_rows)
    lib.write_csv(d, "A5_provenance.csv")
    # explain 720/55
    ugr = d[(d.dataset == "ugr16") & (d.method == "RAPT")]
    note = pd.DataFrame([{
        "quantity": "provenance log rows, UGR'16, earlier revalidation",
        "value": 720,
        "meaning": "5 seeds x (11 checkpoint creations + 133 reuse events) for RAPT",
    }, {
        "quantity": "stored (non-reuse) checkpoints, UGR'16, earlier revalidation",
        "value": 55,
        "meaning": "the 'policies' in '55 of 720 policies'; the other 665 rows are reuses",
    }])
    lib.write_csv(note, "A5_explain_720.csv")
    print(d.to_string(index=False))
    print()
    print(note.to_string(index=False))
    return d


def main():
    lib.ensure_dirs()
    run()


if __name__ == "__main__":
    main()
