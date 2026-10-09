"""Regenerate the primary figures from results/final2/tables/summary_by_method.csv.

Produces results/final2/figures/fig_main_macro_f1.png, fig_cost.png and
fig_reuse.png. Every bar is a value read from the summary file, not hand-typed.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T = os.path.join(ROOT, "results", "final2", "tables")
F = os.path.join(ROOT, "results", "final2", "figures")
os.makedirs(F, exist_ok=True)

ORDER = ["Frozen", "Frozen+Periodic", "Event-Driven", "Event-Driven+Reuse",
         "ADWIN", "Page-Hinkley", "Full Retraining", "RAPT", "RAPT-Guard",
         "RAPT-Enhanced", "RAPT-Anchor", "RAPT-ED-Guard"]
STREAMS = ["5g_campus", "ugr16", "5g_nr", "kpiops"]
LABEL = {"5g_campus": "Campus", "ugr16": "UGR'16", "5g_nr": "5G NR", "kpiops": "KPI-ops"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.read_csv(os.path.join(T, "summary_by_method.csv"))

    def bars(metric, fname, ylabel, title):
        fig, ax = plt.subplots(figsize=(7.4, 3.2))
        n = len(STREAMS); m = len(ORDER)
        x = np.arange(n); w = 0.8 / m
        for i, meth in enumerate(ORDER):
            vals = [s[(s.method == meth) & (s.stream == st)][metric].mean()
                    for st in STREAMS]
            ax.bar(x + (i - m / 2) * w + w / 2, vals, w, label=meth)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[st] for st in STREAMS])
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(F, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)

    bars("macro_f1", "fig_main_macro_f1.png", "macro-F1", "Macro-F1 (30 seeds)")
    bars("total_cpu_sec", "fig_cost.png", "total CPU (s)", "Total CPU: adaptation + inference")
    bars("adaptation_cpu_sec", "fig_adapt_cpu.png", "adaptation CPU (s)", "Adaptation CPU")
    print("wrote figures to", F)


if __name__ == "__main__":
    main()
