#!/usr/bin/env python3
"""v2 extraction driver -- turns the task-2/3/5 scripts' outputs and the
existing committed artifacts into the compact CSVs printed in EXTRACT_V2.md.

Task 1 (printing E3 / E5_mismatch_fraction / E7_detector_meta / E4_gate_conflict_note)
is pure reproduction of results/extract/*.csv and is done by print_e3_e5_e7_e4.py.

Run from the project directory:
    python "results/extract/v2/extract_v2.py"
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXTRACT = os.path.join(ROOT, "results", "extract")
REVAL = os.path.join(ROOT, "results", "revalidation")


def w(df, name):
    df.to_csv(os.path.join(HERE, name), index=False)
    print(f"wrote {name} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Task 3 -- 5G NR loader facts and Event-Driven definitions
# ---------------------------------------------------------------------------
def task3():
    sd9b = json.load(open(os.path.join(ROOT, "experiments", "exp9b", "results",
                                       "stream_definition.json")))
    df = pd.read_csv(os.path.join(ROOT, "experiments", "exp9b", "data",
                                  "processed_exp9b_stream.csv"))
    a1s = pd.read_csv(os.path.join(REVAL, "A1_summary_all_streams.csv"))
    a1p = pd.read_csv(os.path.join(REVAL, "A1_pooled_all_streams.csv"))
    b = pd.read_csv(os.path.join(ROOT, "results", "experiment_9b",
                                 "natural_drift", "summary.csv"))

    facts = pd.DataFrame([
        dict(item="one row in the Table-II 5G NR stream",
             value="one 500-packet telemetry window (a fixed-size aggregate), with 12 QoS features",
             source="experiments/exp9b/load_and_prepare_stream_9b.py:34-59 extract_window_features (window = WINDOW_SIZE=500 packets, line 15/56-62)"),
        dict(item="number of rows / windows", value=str(sd9b["total_windows"]),
             source="experiments/exp9b/results/stream_definition.json [total_windows]"),
        dict(item="initial train prefix", value=str(sd9b["initial_train_windows"]),
             source="experiments/exp9b/results/stream_definition.json [initial_train_windows]"),
        dict(item="label granularity", value="3-class QoS class of the window's own p90 latency (GOOD/DEGRADED/BAD), thresholds frozen on the initial 20% prefix",
             source="experiments/exp9b/load_and_prepare_stream_9b.py:189-197 assign_qos_class(df_stream['target_p90_lat']); thresholds in stream_definition.json [qos_thresholds_ms]"),
        dict(item="loader that produced Table II (5G NR)",
             value="9B: experiments/exp9b/load_and_prepare_stream_9b.py -> processed_exp9b_stream.csv; consumed by experiments/exp9b/run_exp9b.py / drift_harness.py",
             source="results/experiment_9b/natural_drift/summary.csv ; experiments/exp9b/run_exp9b.py"),
        dict(item="loader that produced the revalidation numbers",
             value="results/revalidation/streams.py:33-62 load_5g_nr (reads the SAME processed_exp9b_stream.csv but re-declares window_size=1, one row per window)",
             source="results/revalidation/streams.py:33-62"),
        dict(item="window_size as seen by the revalidation",
             value="1 (one pre-built window per streaming step)",
             source="results/revalidation/streams.py:52"),
    ])
    w(facts, "T3_5gnr_loader_facts.csv")

    ev = []
    for m in ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]:
        ss = a1s[(a1s.dataset == "5g_nr") & (a1s.method == m)]
        pp = a1p[(a1p.dataset == "5g_nr") & (a1p.method == m)]
        ev.append(dict(method=m,
                       per_window_macro_f1=ss.per_window_macro_f1.mean(),
                       pooled_macro_f1=pp.pooled_macro_f1.mean(),
                       table_ii_value=(b[b.method == m].macro_f1_mean.iloc[0]
                                       if len(b[b.method == m]) else None),
                       per_window_accuracy=ss.per_window_accuracy.mean(),
                       pooled_accuracy=pp.pooled_accuracy.mean(),
                       source_pw="results/revalidation/A1_summary_all_streams.csv [per_window_macro_f1]",
                       source_pooled="results/revalidation/A1_pooled_all_streams.csv [pooled_macro_f1]"))
    w(pd.DataFrame(ev), "T3_5gnr_event_driven.csv")


# ---------------------------------------------------------------------------
# Task 5 -- assemble the day-block vs regime-run table
# ---------------------------------------------------------------------------
def task5():
    day = pd.read_csv(os.path.join(HERE, "T5_ugr16_dayblock.csv"))
    reg = pd.read_csv(os.path.join(HERE, "T5_ugr16_regimerun_for_comparison.csv"))
    rows = []
    for _, r in day.iterrows():
        rows.append(dict(comparison=f'{r.method_1} vs {r.method_2}',
                         bootstrap_type="day-block (24 calendar days)",
                         metric="pooled_macro_f1",
                         obs_diff=r.obs_diff_mean,
                         ci_low=r.ci_low_mean, ci_high=r.ci_high_mean,
                         n_blocks=r.n_blocks_mean, n_seeds=r.n_seeds,
                         source="results/extract/v2/T5_ugr16_dayblock.csv"))
    for _, r in reg.iterrows():
        rows.append(dict(comparison=f'{r.method_1} vs {r.method_2}',
                         bootstrap_type="regime-run (contiguous regime visits) [existing A7]",
                         metric=r.metric,
                         obs_diff=r.mean_diff,
                         ci_low=r.ci95_low, ci_high=r.ci95_high,
                         n_blocks=r.n_blocks_mean, n_seeds=5,
                         source="results/revalidation/A7_block_bootstrap.csv"))
    w(pd.DataFrame(rows), "T5_ugr16_bootstrap_comparison.csv")


# ---------------------------------------------------------------------------
# Task 2 -- compact refit table (reads the script's raw output if present)
# ---------------------------------------------------------------------------
def task2():
    fp = os.path.join(HERE, "T2_ugr16_refit_summary.csv")
    if not os.path.exists(fp):
        w(pd.DataFrame([dict(status="MISSING",
                             searched_in="results/extract/v2/T2_ugr16_refit_summary.csv",
                             note="run results/extract/v2/ugr16_refit_log.py first")]),
          "T2_ugr16_refit_summary.csv")
        return
    df = pd.read_csv(fp)
    df["source_file"] = ("results/extract/v2/ugr16_refit_log.py (replicates "
                         "experiments/exp9a/three_dataset_run.py:197-233)")
    w(df, "T2_ugr16_refit_summary.csv")


def main():
    task2(); task3(); task5()


if __name__ == "__main__":
    main()
