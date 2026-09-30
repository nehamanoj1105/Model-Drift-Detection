"""Final_Experiments tables (CSV + LaTeX) and statistical tests."""

import os
import numpy as np
import pandas as pd
from scipy import stats

from config import RAW_DIR, TABLES_DIR, MODELS, RAPT_VARIANTS, ensure_dirs

ALL_MODELS = MODELS + RAPT_VARIANTS
BASELINE = "Full Retraining"


def _load(mode="full"):
    s = pd.read_csv(os.path.join(RAW_DIR, f"summary_{mode}.csv"))
    return s


def _ms(df, col):
    g = df.groupby("method")[col].agg(["mean", "std"])
    return g


def main_table(s):
    order = [m for m in ALL_MODELS if m in set(s.method)]
    rows = []
    for m in order:
        d = s[s.method == m]
        rows.append(dict(
            Model=m,
            **{"Macro-F1": f"{d.macro_f1.mean():.4f} +/- {d.macro_f1.std():.4f}",
               "Accuracy": f"{d.accuracy.mean():.4f} +/- {d.accuracy.std():.4f}",
               "Precision": f"{d.precision.mean():.4f} +/- {d.precision.std():.4f}",
               "Recall": f"{d.recall.mean():.4f} +/- {d.recall.std():.4f}",
               "Adapt CPU (s)": f"{d.adaptation_cpu_sec.mean():.4f} +/- {d.adaptation_cpu_sec.std():.4f}",
               "Runtime (s)": f"{d.total_runtime_sec.mean():.3f} +/- {d.total_runtime_sec.std():.3f}",
               "Retrains": f"{d.retrains.mean():.2f}",
               "Reuse events": f"{d.reuse_events.mean():.2f}",
               "Trees trained": f"{d.trees_trained.mean():.1f}",
               "Trees reused": f"{d.trees_reused.mean():.1f}"},
        ))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLES_DIR, "table_final_main.csv"), index=False)
    try:
        t.to_latex(os.path.join(TABLES_DIR, "table_final_main.tex"), index=False)
    except Exception:
        pass
    return t


def stats_table(s):
    """Paired Wilcoxon (per seed) of each model vs Full Retraining, + Cohen's d."""
    rows = []
    for m in [x for x in ALL_MODELS if x in set(s.method) and x != BASELINE]:
        a = s[s.method == m].sort_values("seed")["macro_f1"].values
        b = s[s.method == BASELINE].sort_values("seed")["macro_f1"].values
        if len(a) != len(b) or len(a) < 3:
            continue
        diff = a - b
        try:
            stat, p = stats.wilcoxon(a, b)
        except Exception:
            stat, p = np.nan, np.nan
        sd = diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        d = float(diff.mean() / sd) if sd and not np.isnan(sd) else np.nan
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(),
                              scale=stats.sem(diff)) if len(diff) > 1 else (np.nan, np.nan)
        rows.append(dict(Model=m, Comparison=f"{m} vs {BASELINE}",
                         mean_delta=float(diff.mean()),
                         ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                         cohens_d=d, wilcoxon_p=float(p) if not np.isnan(p) else np.nan,
                         n_seeds=len(a)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABLES_DIR, "table_final_stats.csv"), index=False)
    return t


def main(mode="full"):
    ensure_dirs()
    s = _load(mode)
    main_table(s); stats_table(s)
    print(f"Tables written to {TABLES_DIR}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
