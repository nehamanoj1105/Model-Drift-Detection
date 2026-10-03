"""
UGR'16 diagnostics that require re-running the harness with altered parameters:
the parity-refit ablation grid (threshold x window length). Everything else
(staleness, PSI/KS, buffer class counts, the Frozen explanation) is computed from
the saved raw files by make_report.py.

Writes results/final/parity_ablation.csv.
"""

from __future__ import annotations

import os
import sys
import copy

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))

from data_loaders import build_stream  # noqa: E402
from runner import prepare, run_model  # noqa: E402
from models import make_class_anchor  # noqa: E402
from models import RAPTEnhanced, RAPT  # noqa: E402


def parity_grid(cfg, seeds, thresholds=(0.50, 0.70, 0.85, 0.95), windows=(2, 3, 5)):
    stream, sd = build_stream("UGR'16", cfg)
    X, y_all, regimes, _, _, slices, n_init, n_windows = prepare(stream, sd)
    prefix_mask = np.arange(len(y_all)) < n_init * sd["window_rows"]
    rows = []
    for thr in thresholds:
        for pw in windows:
            for seed in seeds:
                c = copy.deepcopy(cfg)
                c["rapt_enhanced"]["parity_threshold"] = thr
                c["rapt_enhanced"]["parity_window"] = pw
                anchor = make_class_anchor(X[prefix_mask], y_all[prefix_mask],
                                           cfg["class_anchor_per_class"], seed)
                model = RAPTEnhanced(seed, c, anchor)
                recs, _ = run_model(model, X, y_all, regimes, X[prefix_mask],
                                    y_all[prefix_mask], slices, n_init, n_windows,
                                    "UGR'16", seed, [])
                dfm = pd.DataFrame(recs)
                rows.append({"parity_threshold": thr, "parity_window": pw,
                             "seed": int(seed),
                             "macro_f1": float(dfm["window_f1"].mean()),
                             "adapt_cpu_s": float(dfm["adapt_cpu_s"].sum()),
                             "parity_refits": int(getattr(model, "parity_refits", 0)),
                             "refreshes": int((dfm["event_type"] == "refresh").sum())})
    return pd.DataFrame(rows)


def novelty_attribution(cfg, seeds):
    """Isolate whether UGR'16 recovery comes from the parity trigger or simply
    from a larger novelty-refit buffer: base RAPT at novel_refit_n in {500,1500}."""
    stream, sd = build_stream("UGR'16", cfg)
    X, y_all, regimes, _, _, slices, n_init, n_windows = prepare(stream, sd)
    mask = np.arange(len(y_all)) < n_init * sd["window_rows"]
    rows = []
    for n in (500, 1500):
        for seed in seeds:
            c = copy.deepcopy(cfg)
            c["rapt"]["novel_refit_n"] = n
            anchor = make_class_anchor(X[mask], y_all[mask], cfg["class_anchor_per_class"], seed)
            m = RAPT(seed, c, anchor)
            recs, _ = run_model(m, X, y_all, regimes, X[mask], y_all[mask], slices,
                                n_init, n_windows, "UGR'16", seed, [])
            d = pd.DataFrame(recs)
            rows.append({"dataset": "UGR'16", "model": f"RAPT(novel={n})", "seed": int(seed),
                         "macro_f1": float(d["window_f1"].mean()),
                         "adapt_cpu_s": float(d["adapt_cpu_s"].sum())})
    return pd.DataFrame(rows)


def main():
    import yaml
    with open(os.path.join(HERE, "final.yaml")) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(PROJECT_DIR, "results", "paper_final_run", "final")
    os.makedirs(out, exist_ok=True)
    df = parity_grid(cfg, cfg["seeds"])
    df.to_csv(os.path.join(out, "parity_ablation.csv"), index=False)
    na = novelty_attribution(cfg, cfg["seeds"])
    na.to_csv(os.path.join(out, "novelty_attribution.csv"), index=False)
    print("parity_ablation.csv:", df.shape, "novelty_attribution.csv:", na.shape)


if __name__ == "__main__":
    main()
