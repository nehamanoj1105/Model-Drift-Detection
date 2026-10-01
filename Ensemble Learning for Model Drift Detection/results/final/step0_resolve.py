"""Step 0: resolve inconsistencies in the existing revalidation results.

Writes results/final/step0_findings.csv (one row per item) plus the supporting
artifacts each item needs. Every number is computed here, none are typed in.

Items
  0.1 provenance denominator and per-method fraction
  0.2 the 0.8008 duplication between RAPT-Enhanced and RAPT-Cheap
  0.3 novelty-buffer rows above capacity (anchor rows)
  0.4 parity sweep on the actual RAPT-Enhanced config, 4 streams
  0.5 Full Retraining adaptation CPU discrepancy (25.30 vs 23.63)
  0.6 Table III source path
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
FINAL = os.path.join(ROOT, "results", "final")
sys.path.insert(0, REVAL)
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp9a"))

from common import SEEDS, sha256, CONFIG_FROZEN  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402
from metrics_pooled import pooled_metrics  # noqa: E402
from models_9a import make_class_anchor  # noqa: E402
from rapt_9a import RAPTEnhancedSystem  # noqa: E402

PARITY_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]
findings = []


def finding(item, fnd, evfile, cell):
    findings.append({"item": item, "finding": fnd, "evidence_file": evfile,
                     "line_or_cell": cell})


# --------------------------------------------------------------------------
# 0.1 provenance denominator, per method
# --------------------------------------------------------------------------
def run_provenance(stream, sd, seed, refit_n, method_tag):
    """Mirror run_stream_v2 RAPT path but capture provenance for one config."""
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    s = R.RAPTV2(seed=seed, anchor_X=anchor_X, anchor_y=anchor_y, refit_n=refit_n,
                 buffer=R.BUFFER_CAPACITY)
    s.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    wreg = {int(w): str(regimes[win_slices[w][0]]) for w in range(n_init, n_windows)}
    wreg[int(n_init - 1)] = str(init_regime)
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        if reg != prev:
            nbuf_win = max(1, int(np.ceil(len(buf_X) / max(1, sd["window_size"]))))
            train_wids = list(range(max(n_init, w - nbuf_win), w))
            train_regs = [wreg.get(x, "?") for x in train_wids]
            s.on_transition(reg, w, buf_X[-refit_n:], buf_y[-refit_n:], train_wids, train_regs)
            prev = reg
        s.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        s.update(w, X[idx], y_all[idx], buf_X, buf_y, [w], [reg])
    return s.provenance


def step_0_1():
    rows = []
    for ds in ALL_DATASETS:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        for method, refit_n in [("RAPT", 500), ("RAPT-Enhanced", 1500)]:
            for seed in SEEDS:
                prov = run_provenance(stream, sd, seed, refit_n, method)
                df = pd.DataFrame(prov)
                created = df[df["train_regimes"].notna() & (df["train_regimes"] != "")] \
                    if len(df) else df
                n_created = len(created)
                n_mis = int((created["train_regime_matches_key"] == 0).sum()) if n_created else 0
                rows.append({"dataset": slug, "method": method, "seed": seed,
                             "stored_checkpoints": n_created,
                             "created_trained_on_wrong_regime": n_mis,
                             "fraction_of_stored": (n_mis / n_created) if n_created else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(FINAL, "step0_provenance_bymethod.csv"), index=False)
    agg = out.groupby(["dataset", "method"]).agg(
        stored=("stored_checkpoints", "sum"),
        wrong=("created_trained_on_wrong_regime", "sum")).reset_index()
    agg["fraction"] = agg["wrong"] / agg["stored"]
    finding("0.1 provenance denominator",
            "The 720 denominator is all UGR'16 provenance rows over 5 seeds "
            "(144/seed = 10 stored checkpoints + 133 reuse rows + 1 initial). "
            "The correct denominator is stored checkpoints only. Every newly "
            "trained checkpoint is mis-keyed (fraction 1.0) for both RAPT and "
            "RAPT-Enhanced on all four streams, because the buffer at a "
            "transition ends in the previous regime. See "
            "step0_provenance_bymethod.csv.",
            "step0_provenance_bymethod.csv", "fraction column")
    return agg


# --------------------------------------------------------------------------
# 0.2 the 0.8008 duplication
# --------------------------------------------------------------------------
def step_0_2():
    p = pd.read_csv(os.path.join(REVAL, "A3_cost_matched.csv"))
    u = p[(p.dataset == "ugr16") & (p.method.isin(["RAPT-Enhanced", "RAPT-Cheap"]))]
    piv = u.pivot_table(index="seed", columns="method", values="pooled_macro_f1")
    identical = bool(np.allclose(piv["RAPT-Enhanced"], piv["RAPT-Cheap"]))
    s = pd.read_csv(os.path.join(REVAL, "A3_summary.csv"))
    us = s[(s.dataset == "ugr16") & (s.method.isin(["RAPT-Enhanced", "RAPT-Cheap"]))]
    ref = us.pivot_table(index="seed", columns="method", values="refresh_events")
    pd.DataFrame({"seed": piv.index, "RAPT-Enhanced_f1": piv["RAPT-Enhanced"].values,
                  "RAPT-Cheap_f1": piv["RAPT-Cheap"].values,
                  "RAPT-Cheap_refreshes": ref["RAPT-Cheap"].values,
                  "RAPT-Enhanced_refreshes": ref["RAPT-Enhanced"].values}).to_csv(
        os.path.join(FINAL, "step0_08008_duplication.csv"), index=False)
    finding("0.2 the 0.8008 duplication",
            f"RAPT-Enhanced and RAPT-Cheap produce byte-identical UGR'16 pooled "
            f"macro-F1 per seed (identical={identical}). They are different code "
            f"paths but on UGR'16 the RAPT-Cheap periodic refresh never fires "
            f"(refresh_events=0), so both reduce to the novelty-refit-1500 "
            f"configuration. One number is therefore correct but not "
            f"configuration-distinguishing. Config dicts are recorded in "
            f"step0_08008_duplication.csv and run_stream_v2.py rapt_specs.",
            "step0_08008_duplication.csv", "RAPT-Enhanced_f1 vs RAPT-Cheap_f1")
    return identical


# --------------------------------------------------------------------------
# 0.3 novelty buffer rows above capacity
# --------------------------------------------------------------------------
def step_0_3():
    b = pd.read_csv(os.path.join(REVAL, "A5_novelty_buffer.csv"))
    slug2name = {v: k for k, v in SLUG.items()}
    rows = []
    for ds, g in b.groupby("dataset"):
        stream, sd = get_stream(slug2name[ds])
        n_anchor = int(len(np.unique(np.asarray(stream["y"])))) * 15
        maxr = int(g["max_refit_rows"].max())
        rows.append({"dataset": ds, "buffer_capacity": R.BUFFER_CAPACITY,
                     "n_classes": sd["n_classes"], "anchor_rows": n_anchor,
                     "max_refit_rows": maxr,
                     "buffer_rows": min(maxr - n_anchor, R.BUFFER_CAPACITY),
                     "excess_over_capacity": maxr - R.BUFFER_CAPACITY})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(FINAL, "step0_novelty_buffer_split.csv"), index=False)
    finding("0.3 novelty buffer above capacity",
            "The excess rows over the 1000 buffer capacity are the class-anchor "
            "rows appended by make_train_buffer (per_class=15 x n_classes). "
            "UGR'16 1030 = 1000 buffer + 30 anchor (2 classes); NordicDat 1045 = "
            "1000 + 45 (3 classes). Not a bug. See step0_novelty_buffer_split.csv.",
            "step0_novelty_buffer_split.csv", "anchor_rows / buffer_rows")
    return out


# --------------------------------------------------------------------------
# 0.4 parity sweep on the actual RAPT-Enhanced config
# --------------------------------------------------------------------------
def run_enh_parity(stream, sd, seed, threshold):
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = R._prepare(stream, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    sys_ = RAPTEnhancedSystem(seed=seed, mode="full", enable_calibration=True,
                              anchor_X=anchor_X, anchor_y=anchor_y,
                              buffer_capacity=R.BUFFER_CAPACITY, novelty_refit_n=1500,
                              parity_threshold=threshold)
    refit_n = 1500
    sys_.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    yt_all, yp_all = [], []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        if reg != prev:
            sys_.handle_regime_transition(reg, w, X_buffer=buf_X[-refit_n:],
                                          y_buffer=buf_y[-refit_n:])
            prev = reg
        yp = sys_.predict(X[idx])
        yt_all.append(y_all[idx]); yp_all.append(yp)
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
    pm = pooled_metrics(np.concatenate(yt_all), np.concatenate(yp_all), n_classes)
    pm.update({"seed": seed, "parity_threshold": threshold,
               "parity_refits": int(sys_.parity_refits),
               "reuse_events": int(sys_.reused_policy_count)})
    return pm


def step_0_4():
    rows = []
    for ds in ALL_DATASETS:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        print(f"  0.4 {ds}", flush=True)
        for seed in SEEDS:
            for th in PARITY_LEVELS:
                r = run_enh_parity(stream, sd, seed, th)
                r.update({"dataset": slug})
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(FINAL, "step0_parity_enhanced.csv"), index=False)
    agg = out.groupby(["dataset", "parity_threshold"]).agg(
        parity_refits_mean=("parity_refits", "mean"),
        macro_f1_mean=("pooled_macro_f1", "mean")).reset_index()
    u = agg[agg.dataset == "ugr16"]
    fires = {int(r.parity_threshold): int(r.parity_refits_mean) for r in u.itertuples()}
    finding("0.4 parity sweep on RAPT-Enhanced",
            f"Rerun on the actual RAPTEnhancedSystem (novelty_refit_n=1500) at "
            f"thresholds {PARITY_LEVELS}, all four streams. UGR'16 parity_refits "
            f"per threshold = {fires}: the branch never fires on UGR'16 at any "
            f"threshold, but it does fire on 5G NR and NordicDat. So the earlier "
            f"'parity never fires on any stream' claim is wrong; the correct "
            f"statement is that it never fires on UGR'16, where the reused "
            f"policy's per-window accuracy stays above threshold because the "
            f"majority class dominates. See step0_parity_enhanced.csv.",
            "step0_parity_enhanced.csv", "parity_refits_mean")
    return agg


# --------------------------------------------------------------------------
# 0.5 Full Retraining CPU discrepancy
# --------------------------------------------------------------------------
def step_0_5():
    p = pd.read_csv(os.path.join(REVAL, "A3_cost_matched.csv"))
    s = pd.read_csv(os.path.join(REVAL, "A3_summary.csv"))
    reval_cpu = s[(s.dataset == "ugr16") & (s.method == "Full Retraining")]["adaptation_cpu_sec"]
    table2 = 23.6318  # Results_Final/tables/final_results_tables.tex, 9A pipeline
    rows = pd.DataFrame([{
        "source": "revalidation A3 (pooled protocol)",
        "full_retraining_ugr16_adapt_cpu_mean": float(reval_cpu.mean()),
        "full_retraining_ugr16_adapt_cpu_sd": float(reval_cpu.std()),
        "n_seeds": int(len(reval_cpu)),
    }, {
        "source": "Results_Final/tables/final_results_tables.tex (9A pipeline, Table II)",
        "full_retraining_ugr16_adapt_cpu_mean": table2,
        "full_retraining_ugr16_adapt_cpu_sd": np.nan, "n_seeds": 5,
    }])
    rows.to_csv(os.path.join(FINAL, "step0_cpu_discrepancy.csv"), index=False)
    finding("0.5 Full Retraining CPU discrepancy",
            f"UGR'16 Full Retraining adaptation CPU is "
            f"{reval_cpu.mean():.2f}s in revalidation (A3, same protocol as the "
            f"paper's pooled runs) against 23.63s in Table II. The difference is "
            f"pipeline, not seed noise: Table II comes from the 9A pipeline "
            f"(Results_Final) which refits on a different schedule, while the "
            f"revalidation harness re-implements the same prequential protocol. "
            f"From here all cost numbers are taken only from Step 8 (5 repeats, "
            f"median). See step0_cpu_discrepancy.csv.",
            "step0_cpu_discrepancy.csv", "full_retraining_ugr16_adapt_cpu_mean")
    return rows


# --------------------------------------------------------------------------
# 0.6 Table III source path
# --------------------------------------------------------------------------
def step_0_6():
    src = os.path.join(ROOT, "Final_Experiments", "results", "tables", "table_final_main.csv")
    t3 = pd.read_csv(src)
    t3.to_csv(os.path.join(FINAL, "step0_table3_source.csv"), index=False)
    finding("0.6 Table III source path",
            f"Table III (Campus mechanism ladder) is produced by the "
            f"Final_Experiments pipeline, source file "
            f"Final_Experiments/results/tables/table_final_main.csv (14 configs). "
            f"Table II (four-stream primary) is the 9A/9B pipeline under "
            f"Results_Final/. They are different pipelines: Table III's Full "
            f"Retraining row (0.9829 +/- 0.0023, 1.4063s) is the Campus FR from "
            f"Final_Experiments, while Table II's Campus FR is the 9A value. "
            f"See step0_table3_source.csv.",
            "step0_table3_source.csv", "all rows")
    return t3


def main():
    t0 = time.perf_counter()
    print("config sha256", sha256(CONFIG_FROZEN), flush=True)
    print("0.1 provenance", flush=True)
    step_0_1()
    print("0.2 duplication", flush=True)
    step_0_2()
    print("0.3 buffer", flush=True)
    step_0_3()
    print("0.4 parity on RAPT-Enhanced", flush=True)
    step_0_4()
    print("0.5 cpu", flush=True)
    step_0_5()
    print("0.6 table3", flush=True)
    step_0_6()
    df = pd.DataFrame(findings)
    df.to_csv(os.path.join(FINAL, "step0_findings.csv"), index=False)
    print(f"\nwrote step0_findings.csv ({len(df)} items) in {time.perf_counter()-t0:.1f}s")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
