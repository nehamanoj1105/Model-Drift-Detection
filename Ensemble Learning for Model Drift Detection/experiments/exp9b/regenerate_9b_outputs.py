"""
Regenerate Experiment 9B drift figures, tables and final report from the saved
raw results (results/experiment_9b/...). Useful when only the presentation
layer changes; it performs no model re-evaluation.

Run from the "Ensemble Learning for Model Drift Detection" directory:
    python experiments/exp9b/regenerate_9b_outputs.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp9b_drift_config as CFG


def _read(*parts):
    return pd.read_csv(os.path.join(*parts))


def main():
    CFG.ensure_dirs()
    nat = CFG.NATURAL_DIR
    raw = CFG.RAW_DIR

    with open(os.path.join(CFG.EXP9B_DIR, "results", "stream_definition.json")) as f:
        stream_def = json.load(f)

    natural = {
        "per_window": _read(nat, "per_window.csv"),
        "per_seed": _read(nat, "per_seed.csv"),
        "summary": _read(nat, "summary.csv"),
        "segments": _read(nat, "regime_segments.csv"),
        "stream_def": stream_def,
    }

    cov_pw = _read(raw, "covariate_per_window.csv")
    cov_agg = _read(CFG.COVARIATE_DIR, "aggregated.csv")
    cov_rec = _read(raw, "covariate_recovery.csv")
    cov_agg_rec = _read(CFG.COVARIATE_DIR, "recovery_aggregated.csv")

    con_pw = _read(raw, "concept_per_window.csv")
    con_agg = _read(CFG.CONCEPT_DIR, "aggregated.csv")
    con_rec = _read(raw, "concept_recovery.csv")
    con_agg_rec = _read(CFG.CONCEPT_DIR, "recovery_aggregated.csv")

    rec_pw = _read(raw, "recurring_per_window.csv")
    rec_agg = _read(CFG.RECURRING_DIR, "aggregated.csv")
    rec_rec = _read(raw, "recurring_recovery.csv")
    rec_agg_rec = _read(CFG.RECURRING_DIR, "recovery_aggregated.csv")
    phase_df = _read(CFG.RECURRING_DIR, "per_phase_f1.csv")

    stats = _read(raw, "paired_statistics.csv")

    print("Generating figures ...", flush=True)
    from exp9b_drift_figures import generate_all as gen_figs
    gen_figs(natural, cov_pw, cov_agg, cov_rec, cov_agg_rec,
             con_pw, con_agg, con_rec, con_agg_rec,
             rec_pw, rec_agg, rec_rec, rec_agg_rec, phase_df, stats)

    print("Generating tables ...", flush=True)
    from exp9b_drift_tables import generate_all as gen_tables
    gen_tables(natural, cov_agg, con_agg, rec_agg, cov_rec, con_rec, rec_rec)

    print("Writing report ...", flush=True)
    from exp9b_drift_report import write_report
    write_report(natural, cov_agg, cov_agg_rec, con_agg, con_agg_rec,
                 rec_agg, rec_agg_rec, phase_df, stats)
    print("Done.")


if __name__ == "__main__":
    main()
