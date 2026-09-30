"""
Experiment 9A (three telecom datasets) — CSV/LaTeX tables and statistics.

Outputs
-------
table9a_main.csv / .tex
    Dataset | Model | Accuracy | Macro-F1 | Precision | Recall | Adapt CPU |
    Runtime | Retrains | Reuses          (mean +/- std over 5 seeds)
table9a_drift_detectors.csv / .tex
    Dataset | Method | Macro-F1 | Accuracy | Adapt CPU | Runtime | Adaptation Events
statistics_9a.csv
    Paired Wilcoxon signed-rank (per window, RAPT(-Enhanced) vs baselines),
    Cohen's d effect size and bootstrap CI on the mean F1 difference.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from three_dataset_config import (
    DATASET_SLUG, MODELS, RAW_DIR, TABLES_DIR, ensure_dirs,
)
from three_dataset_config import DATASETS as DATASET_LIST
from three_dataset_figures import DETECTOR_METHODS

COMPARISONS = [("RAPT", "Frozen"), ("RAPT", "Event-Driven"),
               ("RAPT", "Full Retraining"), ("RAPT-Enhanced", "Frozen"),
               ("RAPT-Enhanced", "Full Retraining"), ("RAPT-Enhanced", "RAPT")]


def _ms(series):
    return f"{series.mean():.4f} +/- {series.std():.4f}"


def main(mode="full"):
    ensure_dirs()
    main_rows, det_rows, stat_rows = [], [], []
    for ds in DATASET_LIST:
        slug = DATASET_SLUG[ds]
        s = pd.read_csv(os.path.join(RAW_DIR, f"summary_{slug}_{mode}.csv"))
        w = pd.read_csv(os.path.join(RAW_DIR, f"per_window_{slug}_{mode}.csv"))
        for m in MODELS:
            g = s[s["method"] == m]
            if g.empty:
                continue
            main_rows.append({
                "Dataset": ds, "Model": m,
                "Accuracy": _ms(g["accuracy"]), "Macro-F1": _ms(g["macro_f1"]),
                "Precision": _ms(g["precision"]), "Recall": _ms(g["recall"]),
                "Adapt CPU": _ms(g["adaptation_cpu_sec"]),
                "Runtime": _ms(g["total_runtime_sec"]),
                "Retrains": _ms(g["retrain_events"]),
                "Reuses": _ms(g["reuse_events"]),
                # numeric means for downstream stats/plots
                "macro_f1_mean": g["macro_f1"].mean(),
                "macro_f1_std": g["macro_f1"].std(),
                "accuracy_mean": g["accuracy"].mean(),
                "adapt_cpu_mean": g["adaptation_cpu_sec"].mean(),
                "runtime_mean": g["total_runtime_sec"].mean(),
                "retrain_mean": g["retrain_events"].mean(),
                "reuse_mean": g["reuse_events"].mean(),
                "trees_reused_mean": g["trees_reused"].mean(),
                "trees_trained_mean": g["trees_trained"].mean(),
            })
        for m in DETECTOR_METHODS:
            g = s[s["method"] == m]
            if g.empty:
                continue
            det_rows.append({
                "Dataset": ds, "Method": m,
                "Macro-F1": _ms(g["macro_f1"]), "Accuracy": _ms(g["accuracy"]),
                "Adapt CPU": _ms(g["adaptation_cpu_sec"]),
                "Runtime": _ms(g["total_runtime_sec"]),
                "Adaptation Events": _ms(g["retrain_events"]),
                "macro_f1_mean": g["macro_f1"].mean(),
                "adapt_cpu_mean": g["adaptation_cpu_sec"].mean(),
            })
        # paired per-window statistics
        for a, b in COMPARISONS:
            ga = w[w["method"] == a].groupby("window_id")["macro_f1"].mean()
            gb = w[w["method"] == b].groupby("window_id")["macro_f1"].mean()
            common = ga.index.intersection(gb.index)
            if len(common) < 5:
                continue
            va, vb = ga.loc[common].values, gb.loc[common].values
            diff = va - vb
            try:
                stat, p = stats.wilcoxon(va, vb)
            except Exception:
                stat, p = np.nan, np.nan
            d = diff.mean() / (diff.std(ddof=1) + 1e-12)
            boot = [np.mean(np.random.RandomState(s_).choice(diff, len(diff), replace=True))
                    for s_ in range(500)]
            stat_rows.append({
                "dataset": ds, "method_a": a, "method_b": b, "n_windows": len(common),
                "mean_diff_f1": diff.mean(), "cohen_d": d,
                "wilcoxon_stat": stat, "p_value": p,
                "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                "significant_0.05": bool(p < 0.05) if p == p else False,
            })

    df_main = pd.DataFrame(main_rows)
    df_det = pd.DataFrame(det_rows)
    df_stat = pd.DataFrame(stat_rows)
    df_main.to_csv(os.path.join(TABLES_DIR, "table9a_main.csv"), index=False)
    df_det.to_csv(os.path.join(TABLES_DIR, "table9a_drift_detectors.csv"), index=False)
    df_stat.to_csv(os.path.join(TABLES_DIR, "statistics_9a.csv"), index=False)

    _latex(df_main[["Dataset", "Model", "Accuracy", "Macro-F1", "Precision",
                    "Recall", "Adapt CPU", "Runtime", "Retrains", "Reuses"]],
           os.path.join(TABLES_DIR, "table9a_main.tex"), "Experiment 9A main results")
    _latex(df_det[["Dataset", "Method", "Macro-F1", "Accuracy", "Adapt CPU",
                   "Runtime", "Adaptation Events"]],
           os.path.join(TABLES_DIR, "table9a_drift_detectors.tex"),
           "Experiment 9A drift-detector comparison")
    print("Tables written to", TABLES_DIR)
    return df_main, df_det, df_stat


def _latex(df, path, caption):
    cols = "ll" + "c" * (len(df.columns) - 2)
    body = df.to_latex(index=False, escape=True, column_format=cols)
    with open(path, "w") as f:
        f.write("% " + caption + "\n")
        f.write(body)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
