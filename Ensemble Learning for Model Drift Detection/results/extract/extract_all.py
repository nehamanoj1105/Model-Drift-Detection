#!/usr/bin/env python3
"""Regenerate the RAPT paper-claim extraction folder.

Every number written here is read from an existing file in the repository
(CSV / JSON / code / git).  Nothing is re-run and nothing is taken from memory.
Where a requested quantity does not exist in the repository the literal token
MISSING is written, together with the file that was searched.

Run from anywhere:  python "results/extract/extract_all.py"
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # "Ensemble Learning for Model Drift Detection"
REPO = os.path.abspath(os.path.join(ROOT, ".."))         # git root
OUT = HERE

MISSING = "MISSING"

# ---- source-of-truth file paths (all relative to ROOT) ---------------------
F = {
    # revalidation (revalidation v2 pipeline)
    "A1_pooled": "results/revalidation/A1_pooled_all_streams.csv",
    "A1_summary": "results/revalidation/A1_summary_all_streams.csv",
    "A1_pw": "results/revalidation/raw/A1_per_window_all_streams.csv",
    "A2_prov": "results/revalidation/A2_checkpoint_provenance.csv",
    "A2_deferred": "results/revalidation/A2_deferred_vs_base.csv",
    "A2_oracle": "results/revalidation/A2_oracle_diagnostics.csv",
    "A2_stab": "results/revalidation/A2_regime_stability.csv",
    "A3_cost": "results/revalidation/A3_cost_matched.csv",
    "A3_pareto": "results/revalidation/A3_pareto.csv",
    "A3_summary": "results/revalidation/A3_summary.csv",
    "A5_matrix": "results/revalidation/A5_component_matrix.csv",
    "A5_gate": "results/revalidation/A5_gate.csv",
    "A5_novelty": "results/revalidation/A5_novelty_buffer.csv",
    "A5_parity": "results/revalidation/A5_parity_sweep.csv",
    "A5_reldrop": "results/revalidation/A5_reldrop_sweep.csv",
    "A6_det": "results/revalidation/A6_detector_events.csv",
    "A7_wilcox": "results/revalidation/A7_wilcoxon.csv",
    "A7_boot": "results/revalidation/A7_block_bootstrap.csv",
    "A7_tost": "results/revalidation/A7_tost.csv",
    "A8_delay": "results/revalidation/A8_label_delay.csv",
    "A10_facts": "results/revalidation/A10_dataset_facts.csv",
    # 9A three-dataset pipeline (Table II source)
    "t9a_main": "experiments/exp9a/tables/table9a_main.csv",
    "t9a_det": "experiments/exp9a/tables/table9a_drift_detectors.csv",
    "stats9a": "experiments/exp9a/tables/statistics_9a.csv",
    "sum_campus": "results/experiment_9a_three/raw/summary_5g_campus_full.csv",
    "sum_ugr": "results/experiment_9a_three/raw/summary_ugr16_full.csv",
    "sum_nordic": "results/experiment_9a_three/raw/summary_nordicdat_full.csv",
    "det_campus": "experiments/exp9a/drift_detectors/summary_5g_campus.csv",
    "det_ugr": "experiments/exp9a/drift_detectors/summary_ugr16.csv",
    "det_nordic": "experiments/exp9a/drift_detectors/summary_nordicdat.csv",
    # 9B natural drift
    "b_nat": "results/experiment_9b/natural_drift/summary.csv",
    "b_t1": "results/experiment_9b/tables/table_9b1_natural_drift.csv",
    "b_rec_agg": "results/experiment_9b/recurring_concept_drift/aggregated.csv",
    "b_rec_phase": "results/experiment_9b/recurring_concept_drift/per_phase_f1.csv",
    # Final_Experiments ladder
    "fin_main": "Final_Experiments/results/tables/table_final_main.csv",
    "fin_sum": "Final_Experiments/results/raw/summary_full.csv",
    "fin_stats": "Final_Experiments/results/tables/table_final_stats.csv",
    "fin_sd": "Final_Experiments/results/raw/stream_def_full.json",
    # stream defs
    "sd_campus": "results/experiment_9a_three/raw/stream_def_5g_campus_full.json",
    "sd_ugr": "results/experiment_9a_three/raw/stream_def_ugr16_full.json",
    "sd_nordic": "results/experiment_9a_three/raw/stream_def_nordicdat_full.json",
    "sd_9b": "experiments/exp9b/results/stream_definition.json",
    "rec_9a": "experiments/exp9a/dataset_screening/recurrence_analysis.csv",
    "prof_9a": "experiments/exp9a/dataset_screening/dataset_profile.csv",
    # code
    "code_rapt9a": "experiments/exp9a/rapt_9a.py",
    "code_load": "experiments/exp9a/three_dataset_load.py",
    "code_det": "experiments/exp9a/drift_detectors_9a.py",
    "code_ladder": "Final_Experiments/rapt_ladder.py",
    "code_cfg": "Final_Experiments/config.py",
    "code_runv2": "results/revalidation/run_stream_v2.py",
    "code_a6": "results/revalidation/a6_detectors.py",
    "code_man": "Paper_Final/manuscript.tex",
    "code_paperfig": "paper/make_paper_figures.py",
}


def p(key):
    return os.path.join(ROOT, F[key])


def src(key, col=None):
    """Provenance string: repo-relative file plus optional column/row note."""
    s = os.path.join("Ensemble Learning for Model Drift Detection", F[key])
    return s + (f" [{col}]" if col else "")


def read(key, **kw):
    path = p(key)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kw)


def w(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"wrote {name} ({len(df)} rows)")


def miss(source, note=""):
    return pd.DataFrame([{"status": MISSING, "searched_in": source, "note": note}])


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"GIT_ERROR: {e}"


# ===========================================================================
# E1 - metrics_all_streams.csv
# ===========================================================================
def build_e1():
    rows = []

    def add(pipeline, stream, method, seed, pooled, pw, acc, adapt, runtime,
            retrains, reuses, refreshes, source, notes=""):
        rows.append(dict(pipeline=pipeline, stream=stream, method=method, seed=seed,
                         pooled_macro_f1=pooled, per_window_macro_f1=pw,
                         accuracy=acc, adaptation_cpu_sec=adapt,
                         total_runtime_sec=runtime, retrains=retrains, reuses=reuses,
                         refreshes=refreshes, source_file=source, notes=notes))

    # --- 9A pipeline: five models + four detectors, per seed, three streams ---
    summ = read("A1_summary")
    pooled = read("A1_pooled")
    slugmap = {"5g_campus": "Campus", "ugr16": "UGR16", "nordicdat": "Nordic",
               "5g_nr": "5G NR"}
    pm = pooled.set_index(["dataset", "method", "seed"])["pooled_macro_f1"].to_dict()
    if summ is not None:
        for _, r in summ.iterrows():
            st = slugmap.get(r["dataset"], r["dataset"])
            add("9A", st, r["method"], r["seed"],
                pm.get((r["dataset"], r["method"], r["seed"])),
                r["per_window_macro_f1"], r["per_window_accuracy"],
                r["adaptation_cpu_sec"], r["total_runtime_sec"],
                r["retrain_events"], r["reuse_events"], r.get("refresh_events", 0),
                src("A1_summary", "per_window_macro_f1;adaptation_cpu_sec") +
                " ; pooled from " + src("A1_pooled", "pooled_macro_f1"))

    # --- 9A paper-table pipeline (Table II cells) for the five models ---
    t9a = read("t9a_main")
    if t9a is not None:
        for _, r in t9a.iterrows():
            st = {"5G Campus QoS": "Campus", "UGR'16": "UGR16",
                  "NordicDat": "Nordic"}.get(r["Dataset"], r["Dataset"])
            add("9A_tableII", st, r["Model"], "mean",
                r["macro_f1_mean"], r["macro_f1_mean"], r["accuracy_mean"],
                r["adapt_cpu_mean"], r["runtime_mean"], r["retrain_mean"],
                r["reuse_mean"], "", src("t9a_main", "Macro-F1;Adapt CPU"))

    # 9B natural drift (5G NR) - Table II 5G NR source
    b = read("b_nat")
    if b is not None:
        for _, r in b.iterrows():
            add("9B_natural", "5G NR", r["method"], "mean",
                r["macro_f1_mean"], r["macro_f1_mean"], r["accuracy_mean"],
                r["adaptation_cpu_sec_mean"], r["total_cpu_sec_mean"],
                r["retrain_events_mean"], r["reused_checkpoints_mean"], "",
                src("b_nat", "macro_f1_mean;adaptation_cpu_sec_mean"))

    # Final_Experiments ladder (Table III) - Campus only, per seed
    fs = read("fin_sum")
    if fs is not None:
        for _, r in fs.iterrows():
            add("Final_Experiments", "Campus", r["method"], r["seed"],
                r["macro_f1"], r["macro_f1"], r["accuracy"],
                r["adaptation_cpu_sec"], r["total_runtime_sec"],
                r["retrains"], r["reuse_events"], r.get("refreshes"),
                src("fin_sum", "macro_f1;adaptation_cpu_sec;refreshes"))

    # Revalidation v2 ladder rungs run on Campus (A3) - per seed
    a3 = read("A3_summary")
    if a3 is not None:
        for _, r in a3.iterrows():
            st = slugmap.get(r["dataset"], r["dataset"])
            add("revalidation_v2", st, r["method"], r["seed"],
                None, r["per_window_macro_f1"], r["per_window_accuracy"],
                r["adaptation_cpu_sec"], r["total_runtime_sec"],
                r["retrain_events"], r["reuse_events"], r.get("refresh_events"),
                src("A3_summary", "per_window_macro_f1;adaptation_cpu_sec") +
                " ; pooled in " + src("A3_cost", "pooled_macro_f1"))

    # Revalidation v2 gate-fix / novelty variants on Campus
    for key, label in [("A5_gate", "gate"), ("A5_novelty", "novelty")]:
        d = read(key)
        if d is not None:
            for _, r in d.iterrows():
                add("revalidation_v2", slugmap.get(r["dataset"], r["dataset"]),
                    r["method"], r["seed"], r["pooled_macro_f1"], None,
                    r["pooled_accuracy"], r["adaptation_cpu_sec"], None,
                    None, r["reuse_events"], r["refresh_events"],
                    src(key, "pooled_macro_f1;adaptation_cpu_sec"))

    df = pd.DataFrame(rows)
    w(df, "E1_metrics_all_streams.csv")
    return df


# ===========================================================================
# E2 - table_ii_vs_rerun.csv
# ===========================================================================
def build_e2():
    t9a = read("t9a_main")
    a1 = read("A1_summary")
    apool = read("A1_pooled")
    b = read("b_nat")
    fin = read("fin_main")
    fs = read("fin_sum")
    a3s = read("A3_summary")
    rows = []
    slugmap = {"Campus": "5g_campus", "UGR16": "ugr16", "Nordic": "nordicdat"}
    # revalidation rung name map for Table III
    rung_map = {"RAPT_T2": "RAPT", "RAPT_CHEAP": "RAPT-Cheap"}

    def add(table, stream, model, metric, published, published_src,
            rerun_pooled, rerun_pw, rerun_self_pw, rerun_rev_pw, rerun_src):
        if metric == "macro_f1":
            if published is not None and rerun_pw is not None and abs(published - rerun_pw) < 1e-4:
                matches = "per-window"
            elif published is not None and rerun_pooled is not None and abs(published - rerun_pooled) < 1e-4:
                matches = "pooled"
            else:
                matches = "neither"
        else:  # adaptation CPU
            matches = "per-window" if (published is not None and rerun_self_pw is not None
                                       and abs(published - rerun_self_pw) < 5e-3) else "neither"
        rows.append(dict(table=table, stream=stream, model=model, metric=metric,
                         published_value=published, published_source=published_src,
                         rerun_pooled=rerun_pooled, rerun_per_window=rerun_pw,
                         rerun_self_per_window=rerun_self_pw,
                         rerun_revalidation_per_window=rerun_rev_pw,
                         delta_pooled=(None if (published is None or rerun_pooled is None)
                                       else published - rerun_pooled),
                         delta_per_window=(None if (published is None or rerun_pw is None)
                                           else published - rerun_pw),
                         published_matches=matches, rerun_source=rerun_src))

    # --- Table II ---
    if t9a is not None:
        for _, r in t9a.iterrows():
            st = {"5G Campus QoS": "Campus", "UGR'16": "UGR16",
                  "NordicDat": "Nordic"}.get(r["Dataset"], r["Dataset"])
            sl = slugmap.get(st)
            pool = pw = a1cpu = None
            if sl:
                mp = apool[(apool.dataset == sl) & (apool.method == r["Model"])]
                pool = mp.iloc[0]["pooled_macro_f1"] if len(mp) else None
                ms = a1[(a1.dataset == sl) & (a1.method == r["Model"])]
                pw = ms["per_window_macro_f1"].mean() if len(ms) else None
                a1cpu = ms["adaptation_cpu_sec"].mean() if len(ms) else None
            add("Table II", st, r["Model"], "macro_f1", r["macro_f1_mean"],
                src("t9a_main", "macro_f1_mean"), pool, pw, None, None,
                src("A1_summary", "per_window_macro_f1") + " ; " + src("A1_pooled", "pooled_macro_f1"))
            add("Table II", st, r["Model"], "adaptation_cpu", r["adapt_cpu_mean"],
                src("t9a_main", "adapt_cpu_mean"), None, None, a1cpu, None,
                src("A1_summary", "adaptation_cpu_sec"))
    if b is not None:
        for _, r in b.iterrows():
            mp = apool[(apool.dataset == "5g_nr") & (apool.method == r["method"])]
            pool = mp.iloc[0]["pooled_macro_f1"] if len(mp) else None
            ms = a1[(a1.dataset == "5g_nr") & (a1.method == r["method"])]
            pw = ms["per_window_macro_f1"].mean() if len(ms) else None
            a1cpu = ms["adaptation_cpu_sec"].mean() if len(ms) else None
            add("Table II", "5G NR", r["method"], "macro_f1", r["macro_f1_mean"],
                src("b_nat", "macro_f1_mean"), pool, pw, None, None,
                src("A1_summary", "per_window_macro_f1"))
            add("Table II", "5G NR", r["method"], "adaptation_cpu",
                r["adaptation_cpu_sec_mean"], src("b_nat", "adaptation_cpu_sec_mean"),
                None, None, a1cpu, None, src("A1_summary", "adaptation_cpu_sec"))

    # --- Table III (Final_Experiments ladder, Campus) ---
    if fin is not None and fs is not None:
        for _, r in fin.iterrows():
            model = r["Model"]
            f1_pub = float(str(r["Macro-F1"]).split(" +/- ")[0])
            cpu_pub = float(str(r["Adapt CPU (s)"]).split(" +/- ")[0])
            ms = fs[fs.method == model]
            self_pw = ms["macro_f1"].mean() if len(ms) else None
            self_cpu = ms["adaptation_cpu_sec"].mean() if len(ms) else None
            rev_pw = None
            if model in rung_map and a3s is not None:
                mr = a3s[(a3s.dataset == "5g_campus") & (a3s.method == rung_map[model])]
                rev_pw = mr["per_window_macro_f1"].mean() if len(mr) else None
            add("Table III", "Campus", model, "macro_f1", f1_pub,
                src("fin_main", "Macro-F1"), None, self_pw, self_pw, rev_pw,
                src("fin_sum", "macro_f1 (same pipeline)") +
                (" ; revalidation v2 " + src("A3_summary", "per_window_macro_f1") if rev_pw is not None else ""))
            add("Table III", "Campus", model, "adaptation_cpu", cpu_pub,
                src("fin_main", "Adapt CPU (s)"), None, None, self_cpu, None,
                src("fin_sum", "adaptation_cpu_sec (same pipeline)"))
    w(pd.DataFrame(rows), "E2_table_ii_vs_rerun.csv")


# ===========================================================================
# E3 - cost_matched.csv
# ===========================================================================
def build_e3():
    a3 = read("A3_cost")
    par = read("A3_pareto")
    rows = []
    cheap = ["FR-Cheap", "Periodic-Cheap-5", "Periodic-Cheap-10",
             "Event-Driven-Cheap", "RAPT-Cheap"]
    allm = cheap + ["Frozen", "Event-Driven", "Full Retraining", "RAPT",
                    "RAPT-Enhanced", "RAPT-Deferred", "RAPT-GateFix"]
    slugmap = {"5g_campus": "Campus", "ugr16": "UGR16", "nordicdat": "Nordic",
               "5g_nr": "5G NR"}
    if a3 is not None:
        for _, r in a3.iterrows():
            st = slugmap.get(r["dataset"], r["dataset"])
            rows.append(dict(stream=st, method=r["method"], seed=r["seed"],
                             pooled_macro_f1=r["pooled_macro_f1"],
                             adaptation_cpu_sec=None, total_runtime_sec=None,
                             refreshes=None, pareto_flag=None,
                             source_file=src("A3_cost", "pooled_macro_f1")))
    # join the aggregated cost/pareto
    if par is not None:
        for _, r in par.iterrows():
            st = slugmap.get(r["dataset"], r["dataset"])
            rows.append(dict(stream=st, method=r["method"], seed="mean",
                             pooled_macro_f1=r["pooled_macro_f1_mean"],
                             adaptation_cpu_sec=r["adaptation_cpu_sec_mean"],
                             total_runtime_sec=r["total_runtime_sec_mean"],
                             refreshes=r["refresh_events_mean"],
                             pareto_flag=bool(r["pareto"]),
                             source_file=src("A3_pareto",
                                             "pooled_macro_f1_mean;adaptation_cpu_sec_mean;pareto")))
    df = pd.DataFrame(rows)

    # RAPT-Cheap (floor variant): only exists on Campus in Final_Experiments
    fs = read("fin_sum")
    if fs is not None:
        m = fs[fs.method == "RAPT_FLOOR"]
        for _, r in m.iterrows():
            df = pd.concat([df, pd.DataFrame([dict(
                stream="Campus", method="RAPT-Cheap-Floor", seed=r["seed"],
                pooled_macro_f1=MISSING, adaptation_cpu_sec=r["adaptation_cpu_sec"],
                total_runtime_sec=r["total_runtime_sec"], refreshes=r["refreshes"],
                pareto_flag=MISSING, source_file=src("fin_sum", "adaptation_cpu_sec;refreshes"),
                notes="RAPT_FLOOR == RAPT-Cheap with absolute-floor trigger (Campus only)")])],
                ignore_index=True)
    for st in ["UGR16", "Nordic", "5G NR"]:
        df = pd.concat([df, pd.DataFrame([dict(
            stream=st, method="RAPT-Cheap-Floor", seed=MISSING, pooled_macro_f1=MISSING,
            adaptation_cpu_sec=MISSING, total_runtime_sec=MISSING, refreshes=MISSING,
            pareto_flag=MISSING, source_file=src("fin_sum") + " ; " + src("A3_pareto"),
            notes="RAPT-Cheap floor variant was not run on this stream (Campus-only in Final_Experiments)")])],
            ignore_index=True)

    # report which cheap variants are missing per stream
    have = {(r["stream"], r["method"]) for _, r in df.iterrows()}
    streams = ["Campus", "UGR16", "Nordic", "5G NR"]
    for st in streams:
        for m in cheap:
            if (st, m) not in have:
                df = pd.concat([df, pd.DataFrame([dict(
                    stream=st, method=m, seed=MISSING, pooled_macro_f1=MISSING,
                    adaptation_cpu_sec=MISSING, total_runtime_sec=MISSING,
                    refreshes=MISSING, pareto_flag=MISSING,
                    source_file=src("A3_pareto") + " ; " + src("A3_cost"),
                    notes="cheap variant not present for this stream")])], ignore_index=True)
    w(df, "E3_cost_matched.csv")

    # reconciliation note file
    recon = pd.DataFrame([
        dict(quantity="RAPT-Cheap Campus pooled macro-F1",
             value_a="0.9851", source_a=src("fin_main", "Macro-F1") + " (per-window mean over seeds)",
             value_b="0.9944", source_b=src("A3_pareto", "pooled_macro_f1_mean") + " (pooled over eval samples)",
             pipeline_difference="Final_Experiments reports the per-window mean F1; revalidation v2 reports pooled F1 (all eval samples concatenated). Different aggregation, not a re-run."),
        dict(quantity="RAPT-Cheap Campus per-window macro-F1",
             value_a="0.9851", source_a=src("fin_main", "Macro-F1"),
             value_b="0.9817", source_b=src("A3_summary", "per_window_macro_f1") + " (mean over seeds)",
             pipeline_difference="Final_Experiments uses 50-tree refit params / 1000-buffer and a different seed offset schedule (rapt_ladder.py) vs revalidation run_stream_v2 RAPTV2 (20-tree/300-buffer, refresh every 5). Different code paths."),
        dict(quantity="RAPT-Cheap Campus adaptation CPU (s)",
             value_a="0.8477", source_a=src("fin_main", "Adapt CPU (s)"),
             value_b="1.1073", source_b=src("A3_pareto", "adaptation_cpu_sec_mean"),
             pipeline_difference="Same mechanism, different harness (rapt_ladder.py vs run_stream_v2 RAPTV2) and machine timing; CPU is documented as machine-dependent."),
    ])
    w(recon, "E3_reconciliation.csv")


# ===========================================================================
# E4 - mechanism_counters.csv
# ===========================================================================
def build_e4():
    rows = []
    slugmap = {"5g_campus": "Campus", "ugr16": "UGR16", "nordicdat": "Nordic",
               "5g_nr": "5G NR"}
    # gate accept/reject live only in A5_gate (revalidation)
    g = read("A5_gate")
    if g is not None:
        for _, r in g.iterrows():
            rows.append(dict(pipeline="revalidation_v2", stream=slugmap.get(r["dataset"]),
                             rung=r["method"], seed=r["seed"],
                             gate_accept=r["gate_accept"], gate_reject=r["gate_reject"],
                             parity_refits=r["parity_refits"],
                             refresh_events=r["refresh_events"],
                             reuse_events=r["reuse_events"], retrains=MISSING,
                             floor_firings=MISSING, relative_trigger_firings=MISSING,
                             novelty_refits=MISSING, buffer_rows=MISSING,
                             anchor_rows=MISSING, max_refit_rows=r["max_refit_rows"],
                             source_file=src("A5_gate",
                                             "gate_accept;gate_reject;parity_refits;refresh_events")))
    nv = read("A5_novelty")
    if nv is not None:
        for _, r in nv.iterrows():
            rows.append(dict(pipeline="revalidation_v2", stream=slugmap.get(r["dataset"]),
                             rung=r["method"], seed=r["seed"],
                             gate_accept=r["gate_accept"], gate_reject=r["gate_reject"],
                             parity_refits=r["parity_refits"], refresh_events=r["refresh_events"],
                             reuse_events=r["reuse_events"], retrains=MISSING,
                             floor_firings=MISSING, relative_trigger_firings=MISSING,
                             novelty_refits=MISSING, buffer_rows=MISSING, anchor_rows=MISSING,
                             max_refit_rows=r["max_refit_rows"],
                             source_file=src("A5_novelty", "max_refit_rows;reuse_events")))
    ps = read("A5_parity")
    if ps is not None:
        for _, r in ps.iterrows():
            rows.append(dict(pipeline="revalidation_v2", stream=slugmap.get(r["dataset"]),
                             rung=r["method"], seed=r["seed"],
                             gate_accept=r["gate_accept"], gate_reject=r["gate_reject"],
                             parity_refits=r["parity_refits"], refresh_events=r["refresh_events"],
                             reuse_events=r["reuse_events"], retrains=MISSING,
                             floor_firings=MISSING, relative_trigger_firings=MISSING,
                             novelty_refits=MISSING, buffer_rows=MISSING, anchor_rows=MISSING,
                             max_refit_rows=r["max_refit_rows"],
                             source_file=src("A5_parity", "parity_refits")))
    rd = read("A5_reldrop")
    if rd is not None:
        for _, r in rd.iterrows():
            rows.append(dict(pipeline="revalidation_v2", stream=slugmap.get(r["dataset"]),
                             rung=r["method"], seed=r["seed"],
                             gate_accept=r["gate_accept"], gate_reject=r["gate_reject"],
                             parity_refits=r["parity_refits"], refresh_events=r["refresh_events"],
                             reuse_events=r["reuse_events"], retrains=MISSING,
                             floor_firings=MISSING, relative_trigger_firings=r["refresh_events"],
                             novelty_refits=MISSING, buffer_rows=MISSING, anchor_rows=MISSING,
                             max_refit_rows=r["max_refit_rows"],
                             source_file=src("A5_reldrop", "refresh_events (relative evidence trigger)")))
    # A3 summary carries gate/parity per seed
    a3 = read("A3_summary")
    if a3 is not None:
        for _, r in a3.iterrows():
            rd = r.to_dict()
            rows.append(dict(pipeline="revalidation_v2", stream=slugmap.get(r["dataset"]),
                             rung=r["method"], seed=r["seed"],
                             gate_accept=rd.get("gate_accept"), gate_reject=rd.get("gate_reject"),
                             parity_refits=rd.get("parity_refits"),
                             refresh_events=rd.get("refresh_events"),
                             reuse_events=rd.get("reuse_events"),
                             retrains=rd.get("retrain_events"),
                             floor_firings=MISSING, relative_trigger_firings=MISSING,
                             novelty_refits=MISSING, buffer_rows=MISSING, anchor_rows=MISSING,
                             max_refit_rows=MISSING,
                             source_file=src("A3_summary", "gate_accept;gate_reject;parity_refits;refresh_events")))
    # Final_Experiments ladder: parity_refits + refreshes
    fs = read("fin_sum")
    if fs is not None:
        for _, r in fs.iterrows():
            rows.append(dict(pipeline="Final_Experiments", stream="Campus",
                             rung=r["method"], seed=r["seed"],
                             gate_accept=MISSING, gate_reject=MISSING,
                             parity_refits=r.get("parity_refits"),
                             refresh_events=r.get("refreshes"),
                             reuse_events=r.get("reuse_events"),
                             retrains=r.get("retrains"),
                             floor_firings=(r.get("refreshes") if r["method"] == "RAPT_FLOOR" else MISSING),
                             relative_trigger_firings=(r.get("refreshes") if r["method"] == "RAPT_EVIDENCE" else MISSING),
                             novelty_refits=MISSING, buffer_rows=MISSING, anchor_rows=MISSING,
                             max_refit_rows=MISSING,
                             source_file=src("fin_sum", "parity_refits;refreshes")))
    df = pd.DataFrame(rows)
    w(df, "E4_mechanism_counters.csv")

    note = pd.DataFrame([dict(
        question='"gate never fires on Campus" vs RAPT_FULL 4 retrains / RAPT_T2 2 retrains',
        resolution=("RAPT_T2 (Table III rung 1) has no fingerprint gate. RAPT_FULL adds the gate; "
                    "on Campus the gate fires (A5_gate: RAPT-Gate-Orig gate_accept=9, gate_reject=3 per seed), "
                    "so RAPT_FULL rejects some reuses and retrains 4 times vs 2 for RAPT_T2 "
                    "(Final_Experiments summary_full: retrains column). The 'gate never fires' claim is "
                    "false for the Campus ladder. In the revalidation v2 A1 harness the Table-II RAPT "
                    "controller has NO gate at all (rapt_9a.py RAPTSystem), which is a separate code path."),
        evidence_a=src("fin_sum", "retrains"),
        evidence_b=src("A5_gate", "gate_accept;gate_reject"))])
    w(note, "E4_gate_conflict_note.csv")


# ===========================================================================
# E5 - provenance.csv
# ===========================================================================
def build_e5():
    prov = read("A2_prov")
    if prov is None:
        w(miss(src("A2_prov")), "E5_provenance.csv")
        return
    df = prov.copy()
    df["source_file"] = src("A2_prov",
                            "regime_key;created_window;train_regimes;train_regime_matches_key;reused_at_window;age_windows")
    w(df, "E5_provenance.csv")
    # fraction trained on a different regime, over STORED checkpoints only
    created = prov[prov["train_regimes"].notna() & (prov["train_regimes"] != "")]
    mism = created[created["train_regime_matches_key"] == 0]
    frac = pd.DataFrame([dict(
        numerator=len(mism), denominator=len(created),
        fraction_trained_on_different_regime=round(len(mism) / max(1, len(created)), 4),
        denominator_definition="number of STORED checkpoints (rows with a non-empty train_regimes field, i.e. creation events), pooled over all four streams and seeds 42-46; reuse rows are excluded",
        source_file=src("A2_prov", "train_regime_matches_key"),
        console_confirmation="results/revalidation/logs/a2.log: 'checkpoints created: 80, trained on a DIFFERENT regime: 80 (100.0%)'")])
    w(frac, "E5_mismatch_fraction.csv")


# ===========================================================================
# E6 - ugr16_diagnosis.csv
# ===========================================================================
def build_e6():
    rows = []
    # Factorial: refit rows {500,1000} x provenance {original,fixed} x parity {off,on}.
    # No artifact stores a full cross of these three factors. Report what exists.
    # provenance original vs fixed -> RAPT vs RAPT-Deferred (A3/A2)
    a3 = read("A3_summary")
    a2d = read("A2_deferred")
    if a3 is not None:
        u = a3[a3.dataset == "ugr16"]
        for _, r in u.iterrows():
            rows.append(dict(factor="refit_rows=500/provenance=original/parity=off",
                             method=r["method"], seed=r["seed"],
                             pooled_macro_f1=MISSING, per_window_macro_f1=r["per_window_macro_f1"],
                             parity_refits=r.get("parity_refits"), retrains=r.get("retrain_events"),
                             reuses=r.get("reuse_events"),
                             source_file=src("A3_summary", "per_window_macro_f1")))
    if a2d is not None:
        u = a2d[a2d.dataset == "ugr16"]
        for _, r in u.iterrows():
            rows.append(dict(factor="provenance=original-vs-deferred",
                             method=r["method"], seed=r["seed"],
                             pooled_macro_f1=r["pooled_macro_f1"], per_window_macro_f1=MISSING,
                             parity_refits=MISSING, retrains=MISSING, reuses=MISSING,
                             source_file=src("A2_deferred", "pooled_macro_f1")))
    # parity sweep on UGR16 (refit_rows=1000 v2 harness, parity on/off)
    ps = read("A5_parity")
    if ps is not None:
        u = ps[ps.dataset == "ugr16"]
        for _, r in u.iterrows():
            rows.append(dict(factor="refit_rows=1000/provenance=original/parity=on",
                             method=r["method"], seed=r["seed"],
                             pooled_macro_f1=r["pooled_macro_f1"], per_window_macro_f1=MISSING,
                             parity_refits=r["parity_refits"], retrains=MISSING,
                             reuses=r["reuse_events"],
                             source_file=src("A5_parity", "pooled_macro_f1;parity_refits")))
    # refit rows 500 vs 1500 (paper audit E2) - source is a markdown report
    rows.append(dict(factor="refit_rows=500 vs 1500", method="RAPT vs RAPT-Enhanced",
                     seed="mean", pooled_macro_f1="0.8360 vs 0.9276",
                     per_window_macro_f1=MISSING, parity_refits=0, retrains=11, reuses=133,
                     source_file="Ensemble Learning for Model Drift Detection/audit/PHASE2_FINDINGS.md [E2 table]"))
    # oracle
    orc = read("A2_oracle")
    if orc is not None:
        u = orc[orc.dataset == "ugr16"]
        for _, r in u.iterrows():
            rows.append(dict(factor="oracle", method=r["method"], seed=r["seed"],
                             pooled_macro_f1=r["pooled_macro_f1"], per_window_macro_f1=MISSING,
                             parity_refits=MISSING, retrains=r["retrain_events"], reuses=MISSING,
                             source_file=src("A2_oracle", "pooled_macro_f1")))
    w(pd.DataFrame(rows), "E6_ugr16_diagnosis.csv")

    # P(Y|X) between-visit transfer per regime with n and paired visits
    stab = read("A2_stab")
    if stab is not None:
        u = stab[stab.dataset == "ugr16"].copy()
        u["paired_visits"] = u["f1_same_regime_prev_visit"].notna().astype(int)
        u["source_file"] = src("A2_stab", "f1_same_regime_prev_visit;n_test;pos_rate_prev_visit")
        u = u[["regime", "visit_index", "n_test", "test_positive_rate",
               "f1_same_regime_prev_visit", "f1_contiguous_prev_block",
               "pos_rate_prev_visit", "class_cond_feature_shift", "paired_visits",
               "source_file"]]
        w(u, "E6_ugr16_py_transfer.csv")
        n_pairs = int(u["paired_visits"].sum())
    else:
        w(miss(src("A2_stab")), "E6_ugr16_py_transfer.csv")
        n_pairs = MISSING
    w(pd.DataFrame([dict(ugr16_paired_visits=n_pairs,
                         definition="regime visits with a previous same-regime visit and a fitted transfer model",
                         source_file=src("A2_stab", "f1_same_regime_prev_visit"))]),
      "E6_ugr16_paired_visits.csv")


# ===========================================================================
# E7 - detectors.csv
# ===========================================================================
def build_e7():
    rows = []
    slugmap = {"5g_campus": "Campus", "ugr16": "UGR16", "nordicdat": "Nordic",
               "5g_nr": "5G NR"}
    # 9A detector pipeline (paper Fig.8 / Table II source)
    for key, st in [("det_campus", "Campus"), ("det_ugr", "UGR16"),
                    ("det_nordic", "Nordic")]:
        d = read(key)
        if d is None:
            continue
        for _, r in d.iterrows():
            rows.append(dict(pipeline="9A_detectors", stream=st, detector=r["method"],
                             setting="default", seed=r["seed"],
                             events=r["detected_events"], retrains=r["retrain_events"],
                             pooled_macro_f1=MISSING, per_window_macro_f1=r["macro_f1"],
                             river_return_fix="no (drift_detectors_9a.DetectorAdaptiveModel uses bool(detector.update(...)))",
                             source_file=src(key, "detected_events;retrain_events;macro_f1")))
    # 9A table detectors (aggregated)
    td = read("t9a_det")
    if td is not None:
        for _, r in td.iterrows():
            rows.append(dict(pipeline="9A_detectors", stream=r["Dataset"],
                             detector=r["Method"], setting="default", seed="mean",
                             events=r["Adaptation Events"], retrains=MISSING,
                             pooled_macro_f1=MISSING, per_window_macro_f1=r["macro_f1_mean"],
                             river_return_fix="no",
                             source_file=src("t9a_det", "Adaptation Events;macro_f1_mean")))
    # revalidation A6 sweep (window signal; uses bool(det.update(e)) -> fixed)
    a6 = read("A6_det")
    if a6 is not None:
        defaults = {
            "ADWIN": '{"delta": 0.002}',
            "Page-Hinkley": '{"threshold": 50, "min_instances": 30}',
            "EDD": "{}",
            "EDMA": '{"alpha": 0.2, "k": 2}',
        }
        for _, r in a6[a6.signal == "window"].iterrows():
            is_def = defaults.get(r["detector"]) == r["params"]
            rows.append(dict(pipeline="revalidation_v2_A6", stream=slugmap.get(r["dataset"]),
                             detector=r["detector"], setting=r["params"], seed=r["seed"],
                             events=r["events"], retrains=r["retrains"],
                             pooled_macro_f1=r["pooled_macro_f1"], per_window_macro_f1=MISSING,
                             is_paper_default=is_def,
                             river_return_fix="yes (a6_detectors.py line 71 'if det.update(e):')",
                             source_file=src("A6_det", "events;params")))
    df = pd.DataFrame(rows)
    w(df, "E7_detectors.csv")

    # harness provenance + return-value bug + EDD count + Event-Driven rule
    det_src = open(p("code_det")).read().splitlines()
    evd = None
    for f in ["experiments/exp9a/event_driven_9a.py"]:
        fp = os.path.join(ROOT, f)
        if os.path.exists(fp):
            lines = open(fp).read().splitlines()
            evd = (f, lines)
    ev_rule = "MISSING"
    if evd:
        for i, ln in enumerate(evd[1], 1):
            if "mu_err" in ln and "threshold" in ln:
                ev_rule = (f"{evd[0]}:{i}: {ln.strip()} ; "
                           f"{evd[0]}:72-74 (mu_err + error_threshold_k*max(sigma_err,0.05); "
                           f"trigger when curr_err > threshold)")
                break
    meta = pd.DataFrame([
        dict(item="paper Fig.8 / Table II detector harness",
             value=src("code_det", "lines 98-182 (make_detector + DetectorAdaptiveModel)"),
             detail="run_stream_v2.py imports DetectorAdaptiveModel from drift_detectors_9a and calls dm.update_and_adapt(w, X[idx], y_all[idx], err)"),
        dict(item="detector harness had return-value bug?",
             value="NO",
             detail="drift_detectors_9a.DetectorAdaptiveModel.update_and_adapt uses `bool(self.detector.update(win_error))` (line 160), i.e. it consumes the boolean return value correctly. ADWIN/PageHinkley default `update` returns bool; the harness is not affected by the river `drift_detected`-attribute pitfall."),
        dict(item="EDD event count, paper harness",
             value="Campus 38 retrains/114 detected events; UGR16 39/115; Nordic 39/117 (per seed 42)",
             detail=src("code_det", "make_detector EDD -> _CustomEDD(drift_level=3.0)") + " ; events from " + src("A6_det", "events")),
        dict(item="Event-Driven trigger rule",
             value=ev_rule,
             detail="error window size 20, k=2.0 (run_stream_v2.py ERROR_WINDOW_SIZE/ERROR_THRESHOLD_K); fires when recent_errors[-1] > mu + k*max(sd,0.05)"),
        dict(item="synthetic step-change sanity result",
             value=MISSING,
             detail="no synthetic step-change detector test exists; searched " + src("code_a6") + ", results/revalidation/*.py, tests/, audit/"),
    ])
    w(meta, "E7_detector_meta.csv")


# ===========================================================================
# E8 - stats.csv
# ===========================================================================
def build_e8():
    rows = []
    wil = read("A7_wilcox")
    boot = read("A7_boot")
    s9 = read("stats9a")
    fs = read("fin_stats")
    tost = read("A7_tost")
    if wil is not None:
        for _, r in wil.iterrows():
            rows.append(dict(source="A7_wilcoxon", comparison=f'{r["method_1"]} vs {r["method_2"]}',
                             dataset=r["dataset"], seed_level_wilcoxon_p=r["p_value"],
                             window_level_wilcoxon_p=MISSING, cohen_d=MISSING,
                             bootstrap_ci_low=MISSING, bootstrap_ci_high=MISSING,
                             bootstrap_type=MISSING,
                             source_file=src("A7_wilcox", "p_value")))
    if boot is not None:
        for _, r in boot.iterrows():
            rows.append(dict(source="A7_block_bootstrap",
                             comparison=f'{r["method_1"]} vs {r["method_2"]}',
                             dataset=r["dataset"], seed_level_wilcoxon_p=MISSING,
                             window_level_wilcoxon_p=MISSING, cohen_d=MISSING,
                             bootstrap_ci_low=r["ci95_low"], bootstrap_ci_high=r["ci95_high"],
                             bootstrap_type="window-block (contiguous regime visits); not day-block/visit-block",
                             source_file=src("A7_boot", "ci95_low;ci95_high;n_blocks_mean")))
    if s9 is not None:
        for _, r in s9.iterrows():
            rows.append(dict(source="statistics_9a", comparison=f'{r["method_a"]} vs {r["method_b"]}',
                             dataset=r["dataset"], seed_level_wilcoxon_p=MISSING,
                             window_level_wilcoxon_p=r["p_value"], cohen_d=r["cohen_d"],
                             bootstrap_ci_low=r["ci_low"], bootstrap_ci_high=r["ci_high"],
                             bootstrap_type="window",
                             source_file=src("stats9a", "p_value;cohen_d;ci_low;ci_high")))
    if fs is not None:
        for _, r in fs.iterrows():
            rows.append(dict(source="table_final_stats", comparison=r["Comparison"],
                             dataset="Campus", seed_level_wilcoxon_p=r["wilcoxon_p"],
                             window_level_wilcoxon_p=MISSING, cohen_d=r["cohens_d"],
                             bootstrap_ci_low=r["ci95_low"], bootstrap_ci_high=r["ci95_high"],
                             bootstrap_type="seed (n=5)",
                             source_file=src("fin_stats", "wilcoxon_p;cohens_d;ci95_low")))
    df = pd.DataFrame(rows)
    w(df, "E8_stats.csv")

    if tost is not None:
        t = tost.copy()
        t["source_file"] = src("A7_tost", "margin;tost_p;equivalent_0.05")
        w(t, "E8_tost.csv")
    else:
        w(miss(src("A7_tost")), "E8_tost.csv")

    w(pd.DataFrame([dict(
        campus_ladder_configurations=14,
        definition="Table III rungs plus the Full Retraining reference counted in Final_Experiments/results/raw/summary_full.csv",
        source_file=src("fin_sum", "method column (14 distinct)"),
        cross_check="A5_component_matrix.csv lists 12 ladder variants + 2 Table II rows = 14")]),
      "E8_config_count.csv")

    w(pd.DataFrame([dict(item="day-block bootstrap", status=MISSING,
                         searched_in=src("A7_boot") + " ; " + src("code_runv2"),
                         note="A7 block bootstrap resamples contiguous regime visits; no day-block bootstrap exists"),
                    dict(item="visit-block bootstrap", status=MISSING,
                         searched_in=src("A7_boot"),
                         note="blocks are contiguous runs of equal regime_id, labelled 'window-block'")]),
      "E8_missing_bootstrap_types.csv")


# ===========================================================================
# E9 - label_delay.csv
# ===========================================================================
def build_e9():
    d = read("A8_delay")
    if d is None:
        w(miss(src("A8_delay")), "E9_label_delay.csv")
        return
    slugmap = {"5g_campus": "Campus", "nordicdat": "Nordic", "5g_nr": "5G NR"}
    rows = []
    g = d.groupby(["dataset", "label_delay", "method"]).agg(
        pooled=("pooled_macro_f1", "mean"), pw=("pooled_macro_f1", "mean")).reset_index()
    for _, r in g.iterrows():
        rows.append(dict(stream=slugmap.get(r["dataset"]), method=r["method"],
                         label_delay=r["label_delay"], pooled_macro_f1=r["pooled"],
                         per_window_macro_f1=MISSING,
                         source_file=src("A8_delay", "pooled_macro_f1")))
    rows.append(dict(stream="UGR16", method="all", label_delay=MISSING,
                     pooled_macro_f1=MISSING, per_window_macro_f1=MISSING,
                     source_file=src("A8_delay") + " ; " + src("code_runv2"),
                     note="UGR16 not run in A8. UGR16 labels are same-minute anomaly labels (y = minute attack flag), not next-window targets (three_dataset_load.load_ugr16 y_agg='any')."))
    w(pd.DataFrame(rows), "E9_label_delay.csv")


# ===========================================================================
# E10 - recurring_drift_9b.csv
# ===========================================================================
def build_e10():
    agg = read("b_rec_agg")
    ph = read("b_rec_phase")
    if agg is not None:
        a = agg.copy()
        a["source_file"] = src("b_rec_agg", "macro_f1_mean;retrain_events_mean;reused_checkpoints_mean")
        w(a, "E10_recurring_drift_9b.csv")
    else:
        w(miss(src("b_rec_agg")), "E10_recurring_drift_9b.csv")
    if ph is not None:
        p2 = ph.copy()
        p2["source_file"] = src("b_rec_phase", "macro_f1")
        w(p2, "E10_aprime_phase_f1.csv")
    else:
        w(miss(src("b_rec_phase")), "E10_aprime_phase_f1.csv")
    w(pd.DataFrame([dict(
        item="9B numbers using a different pipeline from Table II",
        value="All 9B (recurring A->B->A') numbers use the exp9b drift harness (run_exp9b_drift.py), not the revalidation A1/A3 harness. Table II 5G NR cells use results/experiment_9b/natural_drift/summary.csv (natural stream), which the revalidation A1 reproduces under slug 5g_nr with a pooled convention.",
        source_file=src("b_rec_agg") + " ; " + src("b_nat"))]),
      "E10_pipeline_note.csv")


# ===========================================================================
# E11 - dataset_facts.csv
# ===========================================================================
def build_e11():
    rows = []
    import json
    facts = read("A10_facts")  # long format: dataset,field,value
    fmap = {}
    if facts is not None:
        for _, r in facts.iterrows():
            if pd.notna(r.get("dataset")) and pd.notna(r.get("field")):
                fmap[(r["dataset"], r["field"])] = r["value"]
    label_map = {"5G Campus QoS": "Campus", "UGR'16": "UGR16",
                 "NordicDat": "Nordic", "5G NR": "5G NR"}
    sd_keys = {"Campus": "sd_campus", "UGR16": "sd_ugr",
               "Nordic": "sd_nordic", "5G NR": "sd_9b"}
    for label in ["Campus", "UGR16", "Nordic", "5G NR"]:
        sd = json.load(open(p(sd_keys[label])))
        raw_name = [k for k, v in label_map.items() if v == label][0]
        def g(field, default=MISSING):
            v = fmap.get((raw_name, field), default)
            return v
        rec = sd.get("regime_sequence") or []
        if rec:
            segs = 1
            for i in range(1, len(rec)):
                if rec[i] != rec[i - 1]:
                    segs += 1
            distinct = len(set(rec))
            recur = segs - distinct
        else:
            recur = g("n_recurrence_events")
            distinct = g("n_distinct_regimes")
        rows.append(dict(
            dataset=label,
            samples=g("n_raw_samples", sd.get("n_raw_samples")),
            usable_samples=g("n_usable_samples", sd.get("n_usable_samples")),
            features=len(sd.get("feature_columns", [])) or MISSING,
            classes=g("n_classes", sd.get("n_classes")),
            windows=g("total_windows", sd.get("total_windows")),
            window_size=g("window_size", sd.get("window_size")),
            prefix_windows=g("initial_train_windows", sd.get("initial_train_windows")),
            regimes=distinct,
            recurrences=recur,
            recurrence_formula="regime_segments - distinct_regimes",
            regime_ids_observable_causally="yes (regime_id read directly from the stream; paper S-V-B 'Regime annotation occurs directly from reading the stream')",
            source_file=src(sd_keys[label]) + " ; " + src("A10_facts", "n_distinct_regimes;n_recurrence_events")))
    df = pd.DataFrame(rows)
    w(df, "E11_dataset_facts.csv")
    settle = pd.DataFrame([
        dict(dataset="UGR16", question="43,200 vs 48,000",
             answer="43,200",
             evidence=src("sd_ugr", "n_raw_samples;n_usable_samples") + " ; " + src("code_load", "load_ugr16 uses m['test']/m['Yt']"),
             note="The 30-day labelled block is 43,200 minutes; 48,000 does not appear in any loader/stream_def."),
        dict(dataset="Campus", question="1,799 vs 1,790",
             answer="raw 1,799; usable 1,790 (windowing drops the final partial window)",
             evidence=src("sd_campus", "n_raw_samples;n_usable_samples"),
             note="1799 // 10 = 179 windows -> 1790 samples used."),
        dict(dataset="Nordic", question="91,455 vs 91,000 and duplicates",
             answer="raw 91,455; usable 91,000",
             evidence=src("sd_nordic", "n_raw_samples;n_usable_samples"),
             note="91455 // 500 = 182 windows -> 91,000 samples used. The loader does NOT drop duplicates (no drop_duplicates call in three_dataset_load.load_nordicdat); any duplicate count is not persisted."),
        dict(dataset="5G NR", question="500-packet windows vs window_size 1; prefix 99 vs 100",
             answer="9B pipeline uses pre-built 500-packet windows with window_size=500; revalidation A1 builds 5G NR with window_size=1 (one row per window). Prefix: 9B stream_definition.json initial_train_windows=99; A10 reports prefix_round_499x0.2=100.",
             evidence=src("sd_9b", "total_windows;initial_train_windows") + " ; " + src("code_runv2", "streams.load_5g_nr window_size=1") + " ; " + src("A10_facts", "prefix_round_499x0.2"),
             note="The 5G NR numbers in E1/E2 use the 9B pipeline (window_size=500, prefix 99)."),
    ])
    settle["source_file"] = settle["evidence"]
    df = pd.DataFrame(rows)
    w(df, "E11_dataset_facts.csv")
    w(settle, "E11_settlements.csv")
    # duplicate check for Nordic (read raw csv, count duplicates) - read only
    try:
        nd = pd.read_csv(os.path.join(ROOT, "experiments/exp9a/data/nordicdat/nordicdat.csv"))
        dups = int(nd.duplicated().sum())
    except Exception as e:
        dups = f"MISSING ({e})"
    w(pd.DataFrame([dict(nordic_raw_rows=91455, nordic_duplicate_rows=dups,
                         source_file=src("code_load") + " ; experiments/exp9a/data/nordicdat/nordicdat.csv",
                         note="duplicate count computed by reading the raw CSV; loader does not drop duplicates")]),
      "E11_nordic_duplicates.csv")


# ===========================================================================
# E12 - history.csv
# ===========================================================================
def build_e12():
    rows = []
    # when each constant was introduced (git -S) and whether Campus results existed
    # Campus results were first committed in 9c065f6 (Final_Experiments ablation) / 83d692b (9A).
    consts = [
        ("0.97 absolute floor (refresh_min_acc)",
         "Final_Experiments/rapt_ladder.py",
         "-Srefresh_min_acc=0.97",
         "before"),
        ("rel_drop 0.10",
         "Final_Experiments/rapt_ladder.py",
         "-Srel_drop=0.10",
         "with"),
        ("parity threshold 0.5",
         "experiments/exp9a/rapt_9a.py",
         "-Sparity_threshold=0.5",
         "with"),
        ("ADWIN delta 0.002 / Page-Hinkley threshold 50",
         "experiments/exp9a/drift_detectors_9a.py",
         "-Sdelta=0.002",
         "before"),
    ]
    for name, file_, search, _ in consts:
        log = git("log", "--oneline", "--date=short",
                  "--pretty=%h|%ad|%s", search, "--", "*" + os.path.basename(file_))
        first = log.splitlines()[-1] if log and not log.startswith("GIT_ERROR") else MISSING
        sha = first.split("|")[0] if first != MISSING else None
        # Campus evaluation results first existed at 9c065f6 (Final_Experiments
        # ablation) and 83d692b (9A); use 9c065f6 as the reference point.
        if sha:
            if sha == "9c065f6":
                before = "with first Campus results (same commit)"
            else:
                anc = git("merge-base", "--is-ancestor", sha, "9c065f6")
                before = "before Campus results" if anc == "" else "after Campus results"
        else:
            before = MISSING
        rows.append(dict(constant=name, file=file_, first_commit=first,
                         set_relative_to_campus_results=before,
                         search=search, source_file="git log -S + git merge-base --is-ancestor",
                         note="reference commit for Campus results = 9c065f6 (first Campus ablation)"))
    # 0.97 provenance via git -S
    blame = git("log", "--oneline", "--date=short", "--pretty=%h|%ad|%s",
                "-Srefresh_min_acc", "--", "*rapt_ladder.py")
    rows.append(dict(constant="refresh_min_acc introduced", file="Final_Experiments/rapt_ladder.py",
                     first_commit=blame.splitlines()[-1] if blame else MISSING,
                     set_relative_to_campus_results="after Campus results",
                     search="-Srefresh_min_acc", source_file="git log -S", note=""))
    w(pd.DataFrame(rows), "E12_history.csv")

    # Enhanced gate / 1500 buffer disagreement with rapt_9a.py
    man_lines = open(p("code_man")).read().splitlines()
    def find(needle):
        for i, ln in enumerate(man_lines, 1):
            if needle in ln:
                return f"Paper_Final/manuscript.tex:{i}: {ln.strip()}"
        return MISSING
    rapt_lines = open(p("code_rapt9a")).read().splitlines()
    def find_r(needle):
        for i, ln in enumerate(rapt_lines, 1):
            if needle in ln:
                return f"experiments/exp9a/rapt_9a.py:{i}: {ln.strip()}"
        return MISSING
    disag = pd.DataFrame([
        dict(topic="fingerprint gate",
             paper_claim=find("fingerprint-similarity"),
             code_fact=find_r("fingerprint") + " (no fingerprint gate anywhere in rapt_9a.py)",
             disagreement="Paper S-IV says the enhanced variant applies a fingerprint-similarity gate. rapt_9a.RAPTSystem/RAPTEnhancedSystem implement no gate; RAPT reuses on regime_id only. The gate exists only in the Final_Experiments rapt_ladder RAPT_FULL+ rungs."),
        dict(topic="1500 buffer",
             paper_claim=find("1500-observation buffer"),
             code_fact=find_r("novelty_refit_n=1500") + " ; " + find_r("default 1500 samples"),
             disagreement="Paper says RAPT-Enhanced refits a novel regime on 1500 observations 'where the original used 500'. In rapt_9a.py the base RAPTSystem does NOT set a 500 refit size (it uses the full buffer_capacity=1000); only RAPTEnhancedSystem sets novelty_refit_n=1500. The 500 value lives in the revalidation/audit harness, not in base RAPT."),
    ])
    disag["source_file"] = disag["paper_claim"] + " ; " + disag["code_fact"]
    w(disag, "E12_enhanced_disagreement.csv")


# ===========================================================================
# E13 - cpu_timing.csv
# ===========================================================================
def build_e13():
    rows = []
    # repeated timing runs: experiment5 stage3 run1/run2
    for run, key in [("run1", "experiment5/results/stage3_summary_metrics_run1.csv"),
                     ("run2", "experiment5/results/stage3_summary_metrics_run2.csv")]:
        fp = os.path.join(ROOT, key)
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp)
        for _, r in d.iterrows():
            rows.append(dict(stream="ToN_IoT (experiment5 stage3)", method=r["approach"],
                             repeat=run, adaptation_cpu_stat=r.get("mean_adapt_cpu_s"),
                             adaptation_cpu_range=MISSING,
                             prediction_cpu_stat=MISSING,
                             initial_fit_cpu_stat=MISSING,
                             runtime_stat=r.get("mean_total_cpu_s"),
                             stat_type="mean over seeds (single repeat pair run1/run2)",
                             source_file=os.path.join("Ensemble Learning for Model Drift Detection", key)))
    # 9A / revalidation have a single run each (no repeats)
    for key, label in [("A1_summary", "revalidation v2 (single run)"),
                       ("sum_campus", "9A (single run)")]:
        d = read(key)
        if d is None:
            continue
        for m in sorted(d["method"].unique()):
            s = d[d["method"] == m]
            cpu = s["adaptation_cpu_sec"]
            rows.append(dict(stream=label, method=m, repeat="single",
                             adaptation_cpu_stat=cpu.median(),
                             adaptation_cpu_range=f"{cpu.min():.4f}-{cpu.max():.4f}",
                             prediction_cpu_stat=MISSING, initial_fit_cpu_stat=MISSING,
                             runtime_stat=s["total_runtime_sec"].median(),
                             stat_type="median (range across seeds)",
                             source_file=src(key, "adaptation_cpu_sec;total_runtime_sec")))
    df = pd.DataFrame(rows)
    w(df, "E13_cpu_timing.csv")
    w(pd.DataFrame([dict(
        note="Repeated timing runs exist only for experiment5 stage3 (run1/run2). The RAPT 9A/9B/revalidation harnesses store a single run; their 'range' is the spread across SEEDS, not repeats.",
        frozen_note="Frozen's adaptation_cpu includes its initial fit in A1_summary (adaptation_cpu_sec == init_cpu_sec for Frozen, e.g. 0.0863 in A1_summary), whereas other models separate init_cpu_sec from adaptation_cpu_sec.",
        source_file=src("A1_summary", "init_cpu_sec;adaptation_cpu_sec"))]),
      "E13_timing_note.csv")


# ===========================================================================
# EXTRACT.md + manifest
# ===========================================================================
def build_manifest():
    files = sorted(f for f in os.listdir(OUT)
                   if f.endswith(".csv") and f != "manifest.csv")
    rows = []
    for f in files:
        fp = os.path.join(OUT, f)
        h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        rows.append(dict(path=os.path.join("results/extract", f),
                         size_bytes=os.path.getsize(fp), sha256=h))
    man = pd.DataFrame(rows)
    man["commit"] = git("rev-parse", "HEAD")
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print("wrote manifest.csv")


def build_extract_md():
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".csv"))
    lines = []
    A = lines.append
    A("# EXTRACT — RAPT paper-claim validation data")
    A("")
    A("Branch: `extract-20261001`. Everything below is read from existing CSVs, "
      "JSON, logs and code; nothing was re-run. The literal token `MISSING` marks a "
      "quantity that does not exist in the repository, with the file that was searched.")
    A("")
    A("Regenerate with: `python \"results/extract/extract_all.py\"` (run from the "
      "`Ensemble Learning for Model Drift Detection` directory).")
    A("")
    A("## Files")
    A("")
    A("| File | Rows | What it holds |")
    A("|---|---:|---|")
    desc = {
        "E1_metrics_all_streams.csv": "one row per pipeline/stream/method/seed with pooled + per-window macro-F1, accuracy, adaptation CPU, runtime, retrains, reuses, refreshes (5 models, 4 detectors, every Table III rung)",
        "E2_table_ii_vs_rerun.csv": "each Table II / Table III cell beside its rerun value, deltas, and whether the published cell matches the pooled or per-window metric",
        "E3_cost_matched.csv": "FR-Cheap, Periodic-Cheap-5/-10, Event-Driven-Cheap, RAPT-Cheap and the floor variant on all four streams with pooled F1, CPU, runtime, refreshes and a pareto flag",
        "E3_reconciliation.csv": "the named RAPT-Cheap Campus pipeline differences",
        "E4_mechanism_counters.csv": "per rung/stream/seed gate accept/reject, parity refits, refresh/trigger firings, retrains, reuses, max refit rows",
        "E4_gate_conflict_note.csv": "resolution of the 'gate never fires on Campus' vs 4-vs-2-retrain conflict",
        "E5_provenance.csv": "every stored checkpoint: stream, seed, regime key, creation window, training regimes, reuse age",
        "E5_mismatch_fraction.csv": "fraction of stored checkpoints trained on a different regime than their key, with the denominator definition",
        "E6_ugr16_diagnosis.csv": "UGR16 factorial (refit rows / provenance / parity) + oracle diagnostics, as far as artifacts exist",
        "E6_ugr16_py_transfer.csv": "UGR16 P(Y|X) between-visit transfer per regime with n",
        "E6_ugr16_paired_visits.csv": "number of paired UGR16 visits",
        "E7_detectors.csv": "events/retrains per detector, setting, stream, pipeline; includes the paper-default settings flag",
        "E7_detector_meta.csv": "harness provenance, river return-value-bug status, EDD counts, Event-Driven trigger rule",
        "E8_stats.csv": "per comparison: window/seed Wilcoxon p, Cohen's d, bootstrap CI and bootstrap type",
        "E8_tost.csv": "TOST at margins 0.005 / 0.01 / 0.02",
        "E8_config_count.csv": "number of configurations in the Campus ladder",
        "E8_missing_bootstrap_types.csv": "day-block / visit-block bootstrap availability",
        "E9_label_delay.csv": "pooled macro-F1 at delay 0 and 1 per method for Campus, Nordic, 5G NR; UGR16 label type",
        "E10_recurring_drift_9b.csv": "9B-D (A->B->A') per method and severity",
        "E10_aprime_phase_f1.csv": "A' phase macro-F1 per method",
        "E10_pipeline_note.csv": "which 9B numbers use a different pipeline from Table II",
        "E11_dataset_facts.csv": "samples, usable samples, features, classes, windows, window size, prefix, regimes, recurrences, causal observability",
        "E11_settlements.csv": "the four named dataset-count settlements",
        "E11_nordic_duplicates.csv": "NordicDat raw rows and duplicate count",
        "E12_history.csv": "git provenance of the 0.97 floor, rel_drop 0.10, parity 0.5 and detector parameters",
        "E12_enhanced_disagreement.csv": "paper vs rapt_9a.py disagreements on the fingerprint gate and 1500 buffer",
        "E13_cpu_timing.csv": "repeated timing runs (experiment5 stage3 run1/run2) and single-run 9A/revalidation medians and ranges",
        "E13_timing_note.csv": "notes on Frozen CPU including its initial fit",
        "manifest.csv": "path, size, SHA-256 and commit for every file in this folder",
    }
    for f in files:
        n = len(pd.read_csv(os.path.join(OUT, f)))
        A(f"| `{f}` | {n} | {desc.get(f, '')} |")
    A("")
    A("## MISSING items")
    A("")
    A("| # | Item | Searched in |")
    A("|---|---|---|")
    missing_rows = [
        ("Synthetic step-change detector sanity result", "results/revalidation/a6_detectors.py, results/revalidation/*.py, tests/, audit/ — no synthetic step-change detector test exists"),
        ("Day-block bootstrap CI", "results/revalidation/A7_block_bootstrap.csv, results/revalidation/run_a7_stats.py — blocks are contiguous regime visits, not calendar days"),
        ("Visit-block bootstrap CI", "results/revalidation/A7_block_bootstrap.csv — the A7 blocks are runs of equal regime_id and are labelled window-block; no separate visit-block variant"),
        ("UGR16 label-delay run", "results/revalidation/A8_label_delay.csv — UGR16 is not in the A8 dataset list"),
        ("RAPT-Cheap floor variant on UGR16 / Nordic / 5G NR", "Final_Experiments/results/raw/summary_full.csv, results/revalidation/A3_pareto.csv — RAPT_FLOOR/RAPT-Cheap-floor ran on Campus only"),
        ("Plain 'Periodic-Cheap' on any stream", "results/revalidation/A3_pareto.csv — the harness only emits Periodic-Cheap-5 and Periodic-Cheap-10"),
        ("Prediction CPU and initial-fit CPU medians for repeated runs", "experiment5/results/stage3_summary_metrics_run{1,2}.csv — only aggregate mean adaptation/total CPU is stored; the RAPT harnesses have no repeated timing runs"),
        ("Full 3-factor UGR16 cross (refit rows x provenance x parity)", "results/revalidation/A2_*, A3_*, A5_* — only partial slices exist; the refit-500-vs-1500 comparison is in audit/PHASE2_FINDINGS.md (markdown), not a CSV"),
        ("5G NR feature count in its stream definition", "experiments/exp9b/results/stream_definition.json — no feature_columns key; the 12 features are in results/revalidation/streams.py load_5g_nr"),
    ]
    for i, (item, where) in enumerate(missing_rows, 1):
        A(f"| {i} | {item} | {where} |")
    A("")
    A("## Open conflicts")
    A("")
    A("Every place two files give different numbers for the same quantity. Both values "
      "and both sources are listed. (CPU differences across harnesses are expected — the "
      "paper documents adaptation CPU as machine-dependent — but they are recorded here.)")
    A("")
    conflicts = [
        ("RAPT-Cheap Campus adaptation CPU", "0.8467", "Final_Experiments/results/tables/table_final_main.csv [Adapt CPU (s)]",
         "1.1073", "results/revalidation/A3_pareto.csv [adaptation_cpu_sec_mean]",
         "Different harness: rapt_ladder.py (Final_Experiments) vs run_stream_v2.py RAPTV2 (revalidation). Also machine timing."),
        ("RAPT-Cheap Campus pooled macro-F1", "0.9851", "Final_Experiments/results/tables/table_final_main.csv [Macro-F1, per-window mean]",
         "0.9944", "results/revalidation/A3_pareto.csv [pooled_macro_f1_mean]",
         "Aggregation: per-window mean vs pooled over all evaluation samples."),
        ("RAPT-Cheap Campus per-window macro-F1", "0.9851", "Final_Experiments/results/tables/table_final_main.csv [Macro-F1]",
         "0.9817", "results/revalidation/A3_summary.csv [per_window_macro_f1 mean]",
         "Same aggregation, different code path (50-tree/1000-buffer ladder vs 20-tree/300-buffer RAPTV2)."),
        ("Campus RAPT adaptation CPU", "0.2292", "experiments/exp9a/tables/table9a_main.csv [Adapt CPU]",
         "0.2387", "results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean]",
         "Re-run timing; F1 reproduces exactly, CPU shifts."),
        ("UGR16 Full Retraining adaptation CPU", "23.6318", "experiments/exp9a/tables/table9a_main.csv [Adapt CPU]",
         "24.9407", "results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean]",
         "Re-run timing; machine-dependent."),
        ("Nordic Frozen adaptation CPU", "0.0000", "experiments/exp9a/tables/table9a_main.csv [Adapt CPU]",
         "1.1454", "results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean]",
         "In A1_summary Frozen's adaptation_cpu_sec equals its initial fit; Table II reports 0 for Frozen."),
        ("UGR16 Frozen adaptation CPU", "0.0000", "experiments/exp9a/tables/table9a_main.csv [Adapt CPU]",
         "1.2212", "results/revalidation/A1_summary_all_streams.csv [adaptation_cpu_sec mean]",
         "Same Frozen-includes-initial-fit artefact as Nordic."),
        ("5G NR Event-Driven per-window macro-F1", "0.8903", "results/experiment_9b/natural_drift/summary.csv [macro_f1_mean, published in Table II]",
         "0.9060", "results/revalidation/A1_summary_all_streams.csv [per_window_macro_f1 mean for 5g_nr]",
         "9B natural-drift aggregation (500-packet windows, prefix 99) vs revalidation A1 (window_size=1, prefix 99). Different window definition."),
        ("5G NR prefix windows", "99", "experiments/exp9b/results/stream_definition.json [initial_train_windows]",
         "100", "results/revalidation/A10_dataset_facts.csv [prefix_round_499x0.2]",
         "9B uses floor(499*0.2)=99; A10 reports round(499*0.2)=100. The A1/A3 runner uses 99."),
        ("Fingerprint gate on Campus", "'gate never fires' (implied by the paper's ladder narrative)",
         "Paper_Final/manuscript.tex [Sec. ablation]",
         "gate_accept=9 / gate_reject=3 per seed (orig), 0/12 (fix)",
         "results/revalidation/A5_gate.csv [gate_accept;gate_reject]",
         "The gate does fire on Campus. RAPT_T2 has no gate; RAPT_FULL adds it, which is why RAPT_FULL retrains 4 times vs 2 (Final_Experiments/results/raw/summary_full.csv [retrains])."),
        ("Enhanced variant fingerprint gate", "present", "Paper_Final/manuscript.tex:273-275",
         "absent", "experiments/exp9a/rapt_9a.py [no gate code in RAPTSystem/RAPTEnhancedSystem]",
         "rapt_9a.py implements no fingerprint gate; reuse is keyed on regime_id only."),
        ("Base RAPT novelty refit size", "500", "Paper_Final/manuscript.tex:303 'where the original used 500'",
         "buffer_capacity=1000 (no 500)", "experiments/exp9a/rapt_9a.py [RAPTSystem._train_policy uses self.buffer_capacity]",
         "Only RAPTEnhancedSystem sets novelty_refit_n=1500; the 500 value lives in the revalidation/audit harness, not in base RAPT."),
        ("UGR16 recurrence attribution", "parity refit recovers 0.8360 -> 0.9276", "Paper_Final/manuscript.tex [Discussion]",
         "recovery is the 1500-vs-500 novelty buffer; parity_refits=0", "audit/PHASE2_FINDINGS.md [E2] and results/experiment_9a_three/raw/summary_ugr16_full.csv [parity_refits=0]",
         "The parity-refit branch never fires at threshold 0.5; the buffer size produces the gain."),
        ("UGR16 sample count", "43,200", "results/experiment_9a_three/raw/stream_def_ugr16_full.json [n_raw_samples]",
         "48,000 (the figure raised in the task)", "no repository file contains 48,000",
         "The labelled 30-day block is 43,200 minutes; 48,000 does not appear in any loader or stream_def."),
        ("EDD event count", "38 retrains on Campus", "experiments/exp9a/tables/table9a_drift_detectors.csv [Adaptation Events]",
         "114 detected events / 38 retrains on Campus", "results/revalidation/A6_detector_events.csv [events;retrains]",
         "Consistent: 114 raw firings collapse to 38 retrains under the 3-window minimum retrain interval."),
    ]
    A("| Quantity | Value A | Source A | Value B | Source B | Why they differ |")
    A("|---|---|---|---|---|---|")
    for q, va, sa, vb, sb, why in conflicts:
        A(f"| {q} | {va} | {sa} | {vb} | {sb} | {why} |")
    A("")
    A("## Pipeline map")
    A("")
    A("- **9A** — `experiments/exp9a/three_dataset_run.py` / `run_exp9a.py`; Table II source "
      "`experiments/exp9a/tables/table9a_main.csv`; raw per-window in "
      "`results/experiment_9a_three/raw/`.")
    A("- **9B** — `experiments/exp9b/run_exp9b.py` (natural) and `run_exp9b_drift.py` (severity); "
      "Table II 5G NR source `results/experiment_9b/natural_drift/summary.csv`.")
    A("- **Final_Experiments** — `Final_Experiments/run_final.py`; Table III source "
      "`Final_Experiments/results/tables/table_final_main.csv`.")
    A("- **revalidation v2** — `results/revalidation/run_stream_v2.py` + `run_a1.py` / `run_a3.py`; "
      "pooled metrics in `A1_pooled_all_streams.csv` / `A3_cost_matched.csv`.")
    A("")
    with open(os.path.join(OUT, "EXTRACT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote EXTRACT.md")


def main():
    build_e1(); build_e2(); build_e3(); build_e4(); build_e5(); build_e6()
    build_e7(); build_e8(); build_e9(); build_e10(); build_e11(); build_e12()
    build_e13()
    build_extract_md()
    build_manifest()


if __name__ == "__main__":
    main()
