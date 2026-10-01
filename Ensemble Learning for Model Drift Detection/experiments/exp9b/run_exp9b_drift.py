"""
Experiment 9B — Drift Severity and Concept-Drift Evaluation (main runner).

Extends the existing Experiment 9B into a systematic evaluation of:
  * 9B-A  natural recurring drift        (existing experiment, reused unchanged)
  * 9B-B  controlled covariate drift     (P(X) changes, P(Y|X) fixed)
  * 9B-C  controlled concept drift       (P(Y|X) changes, P(X) fixed)
  * 9B-D  recurring concept drift        (A -> B -> A', blind policy reuse)

All experiments share the existing 9B protocol: chronological streaming,
Test-Then-Train, no future leakage, the same 20% initial-training prefix, the
same window size, the same seeds [42,43,44,45,46], and the same four models.

The RAPT algorithm is NOT modified anywhere in this file.

Run from the "Ensemble Learning for Model Drift Detection" directory:
    python experiments/exp9b/run_exp9b_drift.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp9b_drift_config as CFG
from drift_construct import (
    load_stream, initial_train_count, build_covariate_drift_stream,
    build_concept_drift_stream, build_recurring_concept_stream,
    class_conditional_stats, feature_target_association,
    conditional_proba_estimate, get_regime_windows, model_based_py_change,
    transform_invariance,
)
from drift_harness import evaluate_stream
from drift_metrics import (
    recovery_analysis, aggregate_recovery, paired_wilcoxon, seed_summary,
)

# Fixed construction seed: the drift *data* must be identical across the model
# seeds so that every seed is evaluated on the same experimental configuration.
CONSTRUCTION_SEED = 42

METHOD_ORDER = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]


def _log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 9B-A  Natural drift (reuse the existing experiment outputs)
# --------------------------------------------------------------------------
def run_natural():
    _log("=" * 78)
    _log("9B-A  NATURAL RECURRING DRIFT (existing experiment, same protocol)")
    _log("=" * 78)
    # Evaluate the *unchanged* natural 9B stream with the shared harness so that
    # all five final models are measured under the identical protocol. The four
    # original models reproduce their existing 9B results; RAPT-Enhanced is
    # added. No drift construction is applied to this stream.
    df, stream_def = load_stream(CFG.STREAM_CSV, CFG.STREAM_JSON)
    n_init = initial_train_count(df, stream_def)
    drift_points = sorted(df["regime_id"].drop_duplicates().index.tolist())[1:]
    # Actual regime-change window ids (first window of each new regime segment).
    segs0 = stream_def["segments"]
    drift_points = [int(s["start_window"]) for s in segs0[1:]]

    all_window, all_seed = [], []
    for seed in CFG.SEEDS:
        _log(f"    seed {seed} ...")
        pw, ps = evaluate_stream(df, n_init=n_init, seed=seed,
                                 drift_points=drift_points,
                                 scenario="natural", level=np.nan)
        all_window.append(pw)
        all_seed.append(ps)
    pw = pd.concat(all_window, ignore_index=True)
    ps = pd.concat(all_seed, ignore_index=True)

    # Summary (mean/std across seeds) per model.
    summ_rows = []
    for m in METHOD_ORDER:
        g = ps[ps["method"] == m]
        row = {"method": m}
        for col in ["macro_f1", "accuracy", "precision", "recall",
                    "total_cpu_sec", "adaptation_cpu_sec", "retrain_events",
                    "reused_checkpoints", "trees_trained", "trees_reused"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std())
        summ_rows.append(row)
    summ = pd.DataFrame(summ_rows)

    trans = pd.DataFrame()

    pw.to_csv(os.path.join(CFG.NATURAL_DIR, "per_window.csv"), index=False)
    ps.to_csv(os.path.join(CFG.NATURAL_DIR, "per_seed.csv"), index=False)
    summ.to_csv(os.path.join(CFG.NATURAL_DIR, "summary.csv"), index=False)
    pw.to_csv(os.path.join(CFG.RAW_DIR, "natural_per_window.csv"), index=False)
    ps.to_csv(os.path.join(CFG.RAW_DIR, "natural_per_seed.csv"), index=False)

    segs = stream_def["segments"]
    recs = []
    for i, s in enumerate(segs):
        recs.append({
            "segment_index": s["segment_index"],
            "regime_id": s["regime_id"],
            "start_window": s["start_window"],
            "end_window": s["end_window"],
            "window_count": s["window_count"],
            "is_transition": int(i > 0 and s["regime_id"] != segs[i - 1]["regime_id"]),
        })
    seg_df = pd.DataFrame(recs)
    seg_df.to_csv(os.path.join(CFG.NATURAL_DIR, "regime_segments.csv"), index=False)
    _log(f"  stream: {stream_def['total_windows']} windows, "
         f"initial_train={stream_def['initial_train_windows']}, "
         f"regime sequence={''.join(stream_def['regime_sequence'])}")
    _log(f"  regime recurrences: {seg_df['regime_id'].value_counts().to_dict()}")
    _log(summ[["method", "macro_f1_mean", "accuracy_mean", "adaptation_cpu_sec_mean",
               "retrain_events_mean", "reused_checkpoints_mean"]].to_string(index=False))
    return {"per_window": pw, "per_seed": ps, "summary": summ,
            "transitions": trans, "stats": pd.DataFrame(), "segments": seg_df,
            "stream_def": stream_def}


# --------------------------------------------------------------------------
# Generic severity sweep (covariate / concept / recurring)
# --------------------------------------------------------------------------
def _sweep(build_fn, scenario, out_dir, boundary_key=None):
    df, stream_def = load_stream(CFG.STREAM_CSV, CFG.STREAM_JSON)
    n_init = initial_train_count(df, stream_def)

    all_window, all_seed = [], []
    for level in CFG.DRIFT_LEVELS:
        tag = CFG.severity_tag(level)
        # Build the drift data once; reuse for every model seed.
        built = build_fn(df, stream_def, level, CONSTRUCTION_SEED, n_init)
        if boundary_key is None:
            stream, drift_points = built[0], built[1]
            extra = built[2] if len(built) > 2 else None
        else:
            stream, bounds, extra = built
            drift_points = [bounds[boundary_key]]
        _log(f"  [level {tag}%] stream={len(stream)} windows, "
             f"drift_points={drift_points}, "
             f"{'affected=' + str(extra) if scenario != 'covariate' else ''}")

        for seed in CFG.SEEDS:
            _log(f"    seed {seed} ...")
            pw, ps = evaluate_stream(stream, n_init=n_init, seed=seed,
                                     drift_points=drift_points,
                                     scenario=scenario, level=level)
            all_window.append(pw)
            all_seed.append(ps)

    pw_all = pd.concat(all_window, ignore_index=True)
    ps_all = pd.concat(all_seed, ignore_index=True)
    pw_all.to_csv(os.path.join(CFG.RAW_DIR, f"{scenario}_per_window.csv"), index=False)
    ps_all.to_csv(os.path.join(CFG.RAW_DIR, f"{scenario}_per_seed.csv"), index=False)
    return pw_all, ps_all, n_init, df, stream_def


def _recovery_table(pw_all, scenario, pre_reference_fn=None):
    recs = []
    for level, df_l in pw_all.groupby("drift_level"):
        dps = sorted(df_l["window_id"][df_l["is_drift_point"] == 1].unique())
        pre_ref = pre_reference_fn(level, dps) if pre_reference_fn else None
        df_rec = recovery_analysis(df_l, dps, pre_reference=pre_ref)
        df_rec["scenario"] = scenario
        df_rec["drift_level"] = level
        recs.append(df_rec)
    rec_all = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    rec_all.to_csv(os.path.join(CFG.RAW_DIR, f"{scenario}_recovery.csv"), index=False)
    return rec_all


def run_covariate():
    _log("=" * 78)
    _log("9B-B  CONTROLLED COVARIATE DRIFT SEVERITY")
    _log("=" * 78)
    pw, ps, n_init, df, stream_def = _sweep(
        build_covariate_drift_stream, "covariate", CFG.COVARIATE_DIR)
    rec = _recovery_table(pw, "covariate")

    agg = seed_summary(ps)
    agg.to_csv(os.path.join(CFG.COVARIATE_DIR, "aggregated.csv"), index=False)
    agg_rec = aggregate_recovery(rec, by=("method", "drift_level"))
    agg_rec.to_csv(os.path.join(CFG.COVARIATE_DIR, "recovery_aggregated.csv"), index=False)
    _log(agg[["drift_level", "method", "macro_f1_mean", "macro_f1_std",
              "adaptation_cpu_sec_mean", "retrain_events_mean"]].to_string(index=False))
    return pw, ps, agg, rec, agg_rec


def run_concept():
    _log("=" * 78)
    _log("9B-C  CONTROLLED CONCEPT DRIFT SEVERITY")
    _log("=" * 78)
    df, stream_def = load_stream(CFG.STREAM_CSV, CFG.STREAM_JSON)
    n_init = initial_train_count(df, stream_def)

    # ---- Diagnostics: evidence that P(Y|X) changes and P(X) does not ----
    diag_rows = []
    cond_rows = []
    assoc_rows = []
    model_diag_rows = []
    for level in CFG.DRIFT_LEVELS:
        stream, dps, affected = build_concept_drift_stream(
            df, stream_def, level, CONSTRUCTION_SEED, n_init)
        ccs = class_conditional_stats(stream, affected)
        ccs["drift_level"] = level
        ccs["affected_features"] = ",".join(affected)
        diag_rows.append(ccs)
        fa = feature_target_association(stream, affected)
        fa["drift_level"] = level
        assoc_rows.append(fa)
        mb = model_based_py_change(stream, n_init)
        if mb:
            mb["drift_level"] = level
            mb["affected_features"] = ",".join(affected)
            model_diag_rows.append(mb)
        for f in affected[:1]:
            cp = conditional_proba_estimate(stream, f)
            cond_rows.append({
                "drift_level": level, "feature": f,
                "pre": json.dumps(cp["pre"]), "post": json.dumps(cp["post"]),
            })
    pd.concat(diag_rows, ignore_index=True).to_csv(
        os.path.join(CFG.CONCEPT_DIR, "class_conditional_stats.csv"), index=False)
    pd.concat(assoc_rows, ignore_index=True).to_csv(
        os.path.join(CFG.CONCEPT_DIR, "feature_target_association.csv"), index=False)
    pd.DataFrame(cond_rows).to_csv(
        os.path.join(CFG.CONCEPT_DIR, "conditional_proba_estimates.csv"), index=False)
    if model_diag_rows:
        pd.DataFrame(model_diag_rows).to_csv(
            os.path.join(CFG.CONCEPT_DIR, "model_based_py_change.csv"), index=False)

    # Explicit P(X)-invariance check: untransformed vs transformed post-drift block.
    plain, _, _ = build_concept_drift_stream(df, stream_def, 0.0, CONSTRUCTION_SEED, n_init)
    drift, _, aff_full = build_concept_drift_stream(df, stream_def, 1.0, CONSTRUCTION_SEED, n_init)
    inv = transform_invariance(plain, drift, aff_full)
    inv.to_csv(os.path.join(CFG.CONCEPT_DIR, "px_invariance.csv"), index=False)

    pw, ps, _, _, _ = _sweep(build_concept_drift_stream, "concept", CFG.CONCEPT_DIR)
    rec = _recovery_table(pw, "concept")
    agg = seed_summary(ps)
    agg.to_csv(os.path.join(CFG.CONCEPT_DIR, "aggregated.csv"), index=False)
    agg_rec = aggregate_recovery(rec, by=("method", "drift_level"))
    agg_rec.to_csv(os.path.join(CFG.CONCEPT_DIR, "recovery_aggregated.csv"), index=False)
    _log(agg[["drift_level", "method", "macro_f1_mean", "macro_f1_std",
              "accuracy_mean", "retrain_events_mean"]].to_string(index=False))
    return pw, ps, agg, rec, agg_rec


def run_recurring():
    _log("=" * 78)
    _log("9B-D  RECURRING CONCEPT DRIFT (A -> B -> A')")
    _log("=" * 78)
    pw, ps, n_init, df, stream_def = _sweep(
        build_recurring_concept_stream, "recurring", CFG.RECURRING_DIR,
        boundary_key="A_prime_start")

    def _pre_ref(level, dps):
        st, bd, _ = build_recurring_concept_stream(
            df, stream_def, level, CONSTRUCTION_SEED, n_init)
        # Pre-drift reference = the phase-A block (the last 30 windows of A).
        return {dp: (max(0, bd["A_end"] - 30), bd["A_end"]) for dp in dps}

    rec = _recovery_table(pw, "recurring", pre_reference_fn=_pre_ref)
    agg = seed_summary(ps)
    agg.to_csv(os.path.join(CFG.RECURRING_DIR, "aggregated.csv"), index=False)

    # Per-phase F1 (A, B, A')
    from sklearn.metrics import f1_score
    stream, bounds, affected = build_recurring_concept_stream(
        df, stream_def, 1.0, CONSTRUCTION_SEED, n_init)
    phase_rows = []
    for level, df_l in pw.groupby("drift_level"):
        st, bd, _ = build_recurring_concept_stream(
            df, stream_def, level, CONSTRUCTION_SEED, n_init)
        for phase, (lo, hi) in {
            "A": (0, bd["A_end"]),
            "B": (bd["A_end"], bd["B_end"]),
            "A_prime": (bd["B_end"], bd["B_end"] + 1000),
        }.items():
            wids = st.iloc[lo:hi]["window_id"].values
            sub = df_l[df_l["window_id"].isin(wids)]
            for m in METHOD_ORDER:
                s = sub[sub["method"] == m]
                if len(s) == 0:
                    continue
                phase_rows.append({
                    "drift_level": level, "phase": phase, "method": m,
                    "macro_f1": float(f1_score(s["y_true"], s["y_pred"],
                                               average="macro", zero_division=0)),
                    "accuracy": float((s["y_true"] == s["y_pred"]).mean()),
                })
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(CFG.RECURRING_DIR, "per_phase_f1.csv"), index=False)
    agg_rec = aggregate_recovery(rec, by=("method", "drift_level"))
    agg_rec.to_csv(os.path.join(CFG.RECURRING_DIR, "recovery_aggregated.csv"), index=False)
    _log(agg[["drift_level", "method", "macro_f1_mean", "accuracy_mean",
              "retrain_events_mean", "reused_checkpoints_mean",
              "trees_trained_mean", "trees_reused_mean"]].to_string(index=False))
    return pw, ps, agg, rec, agg_rec, phase_df


# --------------------------------------------------------------------------
# Statistical validation (paired Wilcoxon, per drift type & severity)
# --------------------------------------------------------------------------
def _recovery_censoring(rec):
    """Recovered-rate per scenario, used to flag censored recovery metrics."""
    out = {}
    for scenario, df in rec.items():
        if df is None or len(df) == 0:
            out[scenario] = None
            continue
        out[scenario] = {
            "rate": float(df["recovered"].mean()),
            "n": int(len(df)),
        }
    return out


def run_statistics(cov_pw, con_pw, rec_pw):
    rows = []
    for scenario, pw in [("covariate", cov_pw), ("concept", con_pw), ("recurring", rec_pw)]:
        for level, df_l in pw.groupby("drift_level"):
            for m1, m2 in [("RAPT", "Frozen"), ("RAPT", "Event-Driven"),
                           ("RAPT", "Full Retraining"),
                           ("RAPT-Enhanced", "Frozen"), ("RAPT-Enhanced", "Event-Driven"),
                           ("RAPT-Enhanced", "Full Retraining"), ("RAPT-Enhanced", "RAPT")]:
                r = paired_wilcoxon(df_l, m1, m2)
                if r is None:
                    continue
                r.update({"scenario": scenario, "drift_level": level})
                rows.append(r)
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(CFG.RAW_DIR, "paired_statistics.csv"), index=False)
    return stats


def main():
    CFG.ensure_dirs()
    _log("EXPERIMENT 9B — DRIFT SEVERITY & CONCEPT-DRIFT EVALUATION")
    _log(f"seeds={CFG.SEEDS}  levels={CFG.DRIFT_LEVELS}  window_size={CFG.WINDOW_SIZE}")

    natural = run_natural()
    cov_pw, cov_ps, cov_agg, cov_rec, cov_agg_rec = run_covariate()
    con_pw, con_ps, con_agg, con_rec, con_agg_rec = run_concept()
    rec_pw, rec_ps, rec_agg, rec_rec, rec_agg_rec, phase_df = run_recurring()

    stats = run_statistics(cov_pw, con_pw, rec_pw)

    _log("Running scientific-integrity checks ...")
    from exp9b_drift_checks import run_checks
    run_checks()

    _log("=" * 78)
    _log("Generating figures and tables ...")
    from exp9b_drift_figures import generate_all
    generate_all(natural, cov_pw, cov_agg, cov_rec, cov_agg_rec,
                 con_pw, con_agg, con_rec, con_agg_rec,
                 rec_pw, rec_agg, rec_rec, rec_agg_rec, phase_df, stats)

    from exp9b_drift_tables import generate_all as generate_tables
    generate_tables(natural, cov_agg, con_agg, rec_agg, cov_rec, con_rec, rec_rec)

    _log("Writing final report ...")
    from exp9b_drift_report import write_report
    censoring = _recovery_censoring({"covariate": cov_rec, "concept": con_rec,
                                     "recurring": rec_rec})
    write_report(natural, cov_agg, cov_agg_rec, con_agg, con_agg_rec,
                 rec_agg, rec_agg_rec, phase_df, stats, censoring)

    _log("=" * 78)
    _log("EXPERIMENT 9B DRIFT EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
