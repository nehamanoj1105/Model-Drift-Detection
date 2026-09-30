"""
Scientific-integrity checks for Experiment 9B (Section 10 of the task spec).

Each check is executed against the actual constructed streams and the saved raw
results. The output is written to results/experiment_9b/raw/scientific_checks.txt
so the protocol claims in the report are backed by reproducible evidence.

Run from the "Ensemble Learning for Model Drift Detection" directory:
    python experiments/exp9b/exp9b_drift_checks.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp9b_drift_config as CFG
from drift_construct import (
    load_stream, initial_train_count, build_covariate_drift_stream,
    build_concept_drift_stream, build_recurring_concept_stream,
    get_regime_windows, rank_predictive_features,
)
from preprocessing_9b import FEATURE_COLS

CONSTRUCTION_SEED = 42


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {name}" + (f" — {detail}" if detail else "")


def run_checks():
    df, stream_def = load_stream(CFG.STREAM_CSV, CFG.STREAM_JSON)
    n_init = initial_train_count(df, stream_def)
    lines = []
    A = lines.append

    A("Experiment 9B — Scientific Integrity Checks")
    A(f"initial_train_windows = {n_init}")
    A("")

    # 1. Configuration consistency (single source of truth)
    A(_check("Same seeds across experiments", CFG.SEEDS == [42, 43, 44, 45, 46],
             str(CFG.SEEDS)))
    A(_check("Same window size (500 packets)", CFG.WINDOW_SIZE == 500,
             str(CFG.WINDOW_SIZE)))
    A(_check("Same drift levels", CFG.DRIFT_LEVELS == [0.10, 0.20, 0.30, 0.50, 1.00],
             str(CFG.DRIFT_LEVELS)))

    # 2. Chronology: window_id strictly increasing in every constructed stream
    for name, builder in [
        ("covariate", lambda: build_covariate_drift_stream(df, stream_def, 0.5, CONSTRUCTION_SEED, n_init)[0]),
        ("concept", lambda: build_concept_drift_stream(df, stream_def, 0.5, CONSTRUCTION_SEED, n_init)[0]),
        ("recurring", lambda: build_recurring_concept_stream(df, stream_def, 0.5, CONSTRUCTION_SEED, n_init)[0]),
    ]:
        s = builder()
        mono = bool(np.all(np.diff(s["window_id"].values) == 1))
        A(_check(f"{name}: chronological window ids", mono))

    # 3. Initial training prefix untouched by the concept transform
    ranked, _ = rank_predictive_features(df)
    concept, dps, affected = build_concept_drift_stream(df, stream_def, 1.0, CONSTRUCTION_SEED, n_init)
    src = get_regime_windows(df, stream_def["segments"][0]["regime_id"])
    prefix_ok = np.allclose(concept.iloc[:n_init][FEATURE_COLS].values,
                            src.iloc[:n_init][FEATURE_COLS].values)
    A(_check("Concept: initial training prefix untouched", prefix_ok))

    # 4. Concept transform applied only after the drift point
    pre = concept[concept["is_concept_drift"] == 0]
    pre_ok = np.allclose(pre[FEATURE_COLS].values,
                         src.iloc[:len(pre)][FEATURE_COLS].values)
    A(_check("Concept: pre-drift block unchanged", pre_ok))
    # post-drift affected features are a permutation (same multiset per column)
    post = concept[concept["is_concept_drift"] == 1]
    post_src = src.iloc[len(pre):len(pre) + len(post)]
    perm_ok = all(np.allclose(np.sort(post[f].values), np.sort(post_src[f].values))
                  for f in affected)
    A(_check("Concept: post-drift is a permutation of P(X) (marginals preserved)",
             perm_ok, f"affected={affected}"))

    # 5. Covariate: labels copied verbatim (P(Y|X) unchanged, no relabeling)
    cov, cov_dps = build_covariate_drift_stream(df, stream_def, 1.0, CONSTRUCTION_SEED, n_init)
    # Every row's label equals the label of the (feature-identical) source row.
    merged = cov.merge(df, on=FEATURE_COLS, how="left", suffixes=("", "_orig"))
    label_ok = bool((merged["qos_target"].values == merged["qos_target_orig"].values).mean() > 0.999)
    A(_check("Covariate: labels unchanged from source regimes", label_ok,
             f"match_rate={float((merged['qos_target'].values==merged['qos_target_orig'].values).mean()):.4f}"))

    # 6. Covariate and concept are NOT the same transformation
    A(_check("Covariate uses mixed-provenance post block, concept uses permutation",
             ("is_target_regime_sample" in cov.columns) and
             (cov["is_target_regime_sample"].sum() > 0) and
             (concept["is_concept_drift"].sum() > 0)))

    # 7. Drift point present exactly once per stream and transform bounded to it
    A(_check("Covariate: 2 drift points (2 transitions)", len(cov_dps) == 2, str(cov_dps)))
    A(_check("Concept: 1 drift point", len(dps) == 1, str(dps)))

    # 8. Raw results saved for all seeds and levels
    try:
        rw = pd.read_csv(os.path.join(CFG.RAW_DIR, "covariate_per_window.csv"))
        seeds_ok = sorted(rw["seed"].unique()) == CFG.SEEDS
        levels_ok = sorted(rw["drift_level"].unique()) == CFG.DRIFT_LEVELS
        A(_check("Raw covariate per-window saved for all seeds", seeds_ok))
        A(_check("Raw covariate per-window saved for all levels", levels_ok))
    except FileNotFoundError:
        A("[SKIP] Raw results not found (run run_exp9b_drift.py first).")

    # 9. Model implementations unchanged
    import rapt_9b
    src_rapt = open(rapt_9b.__file__).read()
    A(_check("RAPT algorithm unmodified (no variant symbols)",
             ("RAPTv2" not in src_rapt) and ("RAPT_C" not in src_rapt)))

    text = "\n".join(lines)
    os.makedirs(CFG.RAW_DIR, exist_ok=True)
    out = os.path.join(CFG.RAW_DIR, "scientific_checks.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved to {out}")
    return all("[FAIL]" not in l for l in lines)


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
