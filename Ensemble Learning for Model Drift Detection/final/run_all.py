#!/usr/bin/env python3
"""
Single entry point for the final RAPT experiment.

    python run_all.py --config final.yaml

Single entry point for the final RAPT experiment. Runs the primary models, the
ablation ladder and the drift detectors on all four datasets, writes raw
per-window parquet to results/raw/, and captures the environment, config
snapshot, dataset statistics, CPU breakdown and refit timing. It then regenerates
every table, figure, headline number, audit and report text from those raw files
(make_report.py, make_latex.py, make_figures.py, make_claims_audit.py,
make_paper_number_audit.py, make_summary.py), and finally runs the Experiment 9B
drift-severity suite (experiments/exp9b/run_exp9b_drift.py). One execution from
one config produces every artefact on disk. Set RAPT_SKIP_9B=1 to skip 9B.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))

from data_loaders import build_stream  # noqa: E402
from runner import run_dataset  # noqa: E402
import refit_timing  # noqa: E402


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def env_capture(cfg):
    import sklearn, scipy, matplotlib, psutil
    import river
    info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "river": river.__version__,
        "pyarrow": __import__("pyarrow").__version__,
        "psutil": psutil.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
        "seeds": cfg["seeds"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(HERE, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds or cfg["seeds"]
    out_root = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")
    raw_dir = os.path.join(out_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # --- environment + config snapshot ------------------------------------
    with open(os.path.join(out_root, "env.json"), "w") as f:
        json.dump(env_capture(cfg), f, indent=2)
    with open(os.path.join(out_root, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    names = args.datasets or [d["name"] for d in cfg["datasets"]]
    leakage_log, ds_stats, policy_log = [], [], []
    for name in names:
        print(f"=== {name} ===", flush=True)
        stream, sd = build_stream(name, cfg)
        ds_stats.append({k: v for k, v in sd.items() if k != "regime_sequence"})
        dfw, dfs = run_dataset(stream, sd, cfg, seeds, leakage_log, policy_log)
        slug = sd["slug"]
        dfw.to_parquet(os.path.join(raw_dir, f"per_window_{slug}.parquet"), index=False)
        dfs.to_parquet(os.path.join(raw_dir, f"per_seed_{slug}.parquet"), index=False)
        print(f"  windows={sd['total_windows']} eval={sd['evaluated_windows']} "
              f"features={sd['n_features']} regimes={sd['n_regimes']} "
              f"transitions={sd['n_transitions']} recurrences={sd['n_recurrences']}",
              flush=True)

    def _jsonify(v):
        return json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list)) \
            else v

    stats_df = pd.DataFrame(ds_stats)
    for c in stats_df.columns:
        stats_df[c] = stats_df[c].map(_jsonify)
    stats_df.to_csv(os.path.join(out_root, "dataset_stats.csv"), index=False)

    # Per-regime class balance for UGR'16 (spec: attacks are sparse).
    reg_rows = []
    for rec in ds_stats:
        if rec["dataset"] == "UGR'16":
            for reg, d in rec["class_balance_per_regime"].items():
                reg_rows.append({"dataset": "UGR'16", "regime": reg, **d})
    if reg_rows:
        pd.DataFrame(reg_rows).to_csv(
            os.path.join(out_root, "ugr_regime_class_balance.csv"), index=False)
    pd.DataFrame(leakage_log).to_csv(os.path.join(out_root, "leakage_check.csv"), index=False)
    pd.DataFrame(policy_log).to_csv(os.path.join(out_root, "rapt_policy_events.csv"), index=False)

    # --- automated leakage assertion (spec) -------------------------------
    leak_df = pd.DataFrame(leakage_log)
    n_future = int(leak_df["future_buffer"].sum()) if len(leak_df) else 0
    max_over = int((leak_df["max_train_window"] - leak_df["window_idx"]).max()) if len(leak_df) else 0
    assert n_future == 0, f"leakage: {n_future} evaluations trained on future windows"
    # Re-verify from the raw per-window files (max_window_in_buffer < window_idx).
    bad = 0
    for name in names:
        slug = next(d["slug"] for d in cfg["datasets"] if d["name"] == name)
        pw = pd.read_parquet(os.path.join(raw_dir, f"per_window_{slug}.parquet"))
        bad += int((pw["max_window_in_buffer"] >= pw["window_idx"]).sum())
    assert bad == 0, f"leakage: {bad} raw windows had a buffer window >= scored window"
    with open(os.path.join(out_root, "leakage_check.txt"), "w") as f:
        f.write(f"PASS: 0 evaluations used a window with index >= the scored window.\n"
                f"rows checked: {len(leak_df)}; max(buffer_window - scored_window) "
                f"= {max_over} (must be <= -1).\n")

    # --- CPU breakdown ----------------------------------------------------
    rows = []
    comp_rows = []
    for name in names:
        slug = next(d["slug"] for d in cfg["datasets"] if d["name"] == name)
        dfs = pd.read_parquet(os.path.join(raw_dir, f"per_seed_{slug}.parquet"))
        for _, r in dfs.iterrows():
            rows.append({"dataset": name, "model": r["model"], "seed": int(r["seed"]),
                         "adapt_cpu_s": r["adapt_cpu_s"], "pred_cpu_s": r["pred_cpu_s"],
                         "runtime_s": r["runtime_s"],
                         "retrains": r["retrains"], "reuses": r["reuses"],
                         "refreshes": r["refreshes"],
                         "trees_trained": r["trees_trained"],
                         "trees_reused": r["trees_reused"]})
            cj = r.get("cpu_json", "")
            if isinstance(cj, str) and cj:
                parts = json.loads(cj)
                comp = {"dataset": name, "model": r["model"], "seed": int(r["seed"])}
                comp.update(parts)
                comp_rows.append(comp)
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "cpu_breakdown.csv"), index=False)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(out_root, "cpu_breakdown_components.csv"), index=False)

    # --- refit timing -----------------------------------------------------
    refit_timing.run(cfg, cfg["refit_timing"]).to_csv(
        os.path.join(out_root, "refit_timing.csv"), index=False)

    print("run_all: core run complete; generating report artifacts.", flush=True)

    # --- tables first (diagnostics and the number audit consume these) -----
    for mod in ("make_report", "make_latex"):
        try:
            m = __import__(mod)
            sys.argv = [mod, "--config", "final.yaml"]
            m.main()
        except Exception as e:  # pragma: no cover
            print(f"{mod} failed: {e}", flush=True)

    # --- diagnostics (need the harness; write CSVs only) ------------------
    if os.environ.get("RAPT_SKIP_DIAGNOSTICS") != "1":
        import diagnostics
        try:
            diagnostics.main()
        except Exception as e:  # pragma: no cover
            print(f"diagnostics failed: {e}", flush=True)
        import make_stationarity
        try:
            sys.argv = ["make_stationarity", "--config", "final.yaml"]
            make_stationarity.main()
        except Exception as e:  # pragma: no cover
            print(f"stationarity failed: {e}", flush=True)
        import ugr_diagnostics
        try:
            sys.argv = ["ugr_diagnostics", "--config", "final.yaml"]
            ugr_diagnostics.main()
        except Exception as e:  # pragma: no cover
            print(f"ugr_diagnostics failed: {e}", flush=True)

        import make_detector_diagnostics
        try:
            sys.argv = ["make_detector_diagnostics", "--config", "final.yaml"]
            make_detector_diagnostics.main()
        except Exception as e:  # pragma: no cover
            print(f"detector diagnostics failed: {e}", flush=True)

    # --- figures, numbers, report text, claims audit -----------------------
    for mod in ("make_figures", "make_report_text", "make_claims_audit",
                "make_paper_number_audit", "make_summary"):
        try:
            m = __import__(mod)
            sys.argv = [mod, "--config", "final.yaml"]
            m.main()
        except Exception as e:  # pragma: no cover
            print(f"{mod} failed: {e}", flush=True)

    # --- Experiment 9B drift-severity suite -------------------------------
    # Same single entry point: the natural / covariate / concept / recurring
    # drift experiments, their figures, tables and report all regenerate from
    # experiments/exp9b/exp9b_drift_config.py. Skip with RAPT_SKIP_9B=1.
    if os.environ.get("RAPT_SKIP_9B") != "1":
        runner = os.path.join(PROJECT_DIR, "experiments", "exp9b", "run_exp9b_drift.py")
        try:
            subprocess.run([sys.executable, runner], cwd=PROJECT_DIR, check=True)
        except Exception as e:  # pragma: no cover
            print(f"exp9b drift run failed: {e}", flush=True)

    # --- COMSNETS submission package --------------------------------------
    try:
        import make_submission_zip
        make_submission_zip.main()
    except Exception as e:  # pragma: no cover
        print(f"submission zip failed: {e}", flush=True)

    print("run_all complete: raw + tables + figures + numbers + report + audit + 9B "
          "+ submission zip.",
          flush=True)


if __name__ == "__main__":
    main()
