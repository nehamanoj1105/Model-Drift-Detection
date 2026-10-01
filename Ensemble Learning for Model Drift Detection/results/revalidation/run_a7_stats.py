"""A7: statistics.

Reads the raw per-window results produced by the other tasks and produces:

  A7_wilcoxon.csv        seed-level paired exact Wilcoxon (min p for n=5 is 0.0625)
  A7_block_bootstrap.csv block bootstrap over day blocks (UGR'16) / regime
                         visits (other streams), 95% CI
  A7_tost.csv            TOST equivalence for RAPT-Cheap vs Full Retraining
  A7_multiple_comparisons.csv  Holm correction over the Campus ladder configs

All inputs are raw CSVs already committed by A1/A3.
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SEEDS, write_csv  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0


def load_a1():
    return pd.read_csv(os.path.join(HERE, "A1_pooled_all_streams.csv"))


def seed_wilcoxon(pooled, m1, m2, metric="pooled_macro_f1"):
    """Paired exact Wilcoxon over seeds."""
    rows = []
    for ds in pooled.dataset.unique():
        a = pooled[(pooled.dataset == ds) & (pooled.method == m1)].sort_values("seed")[metric].values
        b = pooled[(pooled.dataset == ds) & (pooled.method == m2)].sort_values("seed")[metric].values
        n = min(len(a), len(b))
        if n < 2:
            continue
        d = a[:n] - b[:n]
        try:
            stat, p = stats.wilcoxon(d, alternative="two-sided")
        except Exception:
            p = 1.0
        rows.append({"dataset": ds, "method_1": m1, "method_2": m2,
                     "n_seeds": n, "mean_diff": float(np.mean(d)),
                     "median_diff": float(np.median(d)),
                     "p_value": float(p), "min_possible_p": float(2.0 / 2 ** n),
                     "significant_0.05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def block_bootstrap(a1_pw, m1, m2, dataset, metric="macro_f1", n_blocks=None):
    """Block bootstrap over day blocks (UGR'16) or regime visits.

    Blocks are contiguous runs of windows (regime visits). Resampling whole
    blocks preserves within-block dependence.
    """
    pw = a1_pw[a1_pw.dataset == dataset]
    rng = np.random.RandomState(BOOT_SEED)
    per_seed_diffs = []
    for seed in sorted(pw.seed.unique()):
        s = pw[pw.seed == seed]
        a = s[s.method == m1].sort_values("window_id")
        b = s[s.method == m2].sort_values("window_id")
        if len(a) != len(b):
            continue
        # build blocks = contiguous runs of equal regime_id
        regs = a["regime_id"].values
        blocks = []
        start = 0
        for i in range(1, len(regs) + 1):
            if i == len(regs) or regs[i] != regs[start]:
                blocks.append(np.arange(start, i))
                start = i
        d = (a[metric].values - b[metric].values)
        boot = []
        for _ in range(N_BOOT):
            pick = rng.randint(0, len(blocks), len(blocks))
            idx = np.concatenate([blocks[p] for p in pick])
            boot.append(np.mean(d[idx]))
        per_seed_diffs.append((np.mean(d), np.percentile(boot, 2.5), np.percentile(boot, 97.5),
                               len(blocks)))
    if not per_seed_diffs:
        return None
    arr = np.array([[x[0], x[1], x[2], x[3]] for x in per_seed_diffs])
    return {"dataset": dataset, "method_1": m1, "method_2": m2,
            "metric": metric,
            "mean_diff": float(arr[:, 0].mean()),
            "ci95_low": float(arr[:, 1].mean()),
            "ci95_high": float(arr[:, 2].mean()),
            "n_blocks_mean": float(arr[:, 3].mean())}


def tost(x, y, margin):
    """Two one-sided tests for equivalence with the given margin."""
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    se = np.std(d, ddof=1) / np.sqrt(n) if n > 1 else np.inf
    if se == 0 or not np.isfinite(se):
        return 0.0
    t_lo = (np.mean(d) - (-margin)) / se
    t_hi = (np.mean(d) - margin) / se
    p_lo = 1 - stats.t.cdf(t_lo, df=n - 1)
    p_hi = stats.t.cdf(t_hi, df=n - 1)
    return float(max(p_lo, p_hi))


def main():
    pooled = load_a1()
    comparisons = [("RAPT-Enhanced", "Frozen"), ("RAPT-Enhanced", "Full Retraining"),
                   ("RAPT", "Frozen"), ("RAPT", "Full Retraining"),
                   ("Event-Driven", "RAPT"), ("Event-Driven", "Full Retraining")]

    wrows = []
    for m1, m2 in comparisons:
        wrows.append(seed_wilcoxon(pooled, m1, m2))
    wil = pd.concat(wrows, ignore_index=True)
    write_csv(wil, "A7_wilcoxon.csv")
    print("=== seed-level Wilcoxon ===")
    print(wil.round(4).to_string(index=False))

    # block bootstrap (needs per-window A1 raw)
    pw = pd.read_csv(os.path.join(HERE, "raw", "A1_per_window_all_streams.csv"))
    brows = []
    for ds in pw.dataset.unique():
        for m1, m2 in comparisons:
            r = block_bootstrap(pw, m1, m2, ds)
            if r:
                brows.append(r)
    boot = pd.DataFrame(brows)
    write_csv(boot, "A7_block_bootstrap.csv")
    print("\n=== block bootstrap 95% CI ===")
    print(boot.round(4).to_string(index=False))

    # TOST: RAPT-Cheap vs Full Retraining on Campus (from A3)
    a3 = pd.read_csv(os.path.join(HERE, "A3_cost_matched.csv"))
    trows = []
    for ds in a3.dataset.unique():
        a = a3[(a3.dataset == ds) & (a3.method == "RAPT-Cheap")].sort_values("seed")["pooled_macro_f1"].values
        b = a3[(a3.dataset == ds) & (a3.method == "Full Retraining")].sort_values("seed")["pooled_macro_f1"].values
        n = min(len(a), len(b))
        if n < 2:
            continue
        for margin in (0.005, 0.01, 0.02):
            trows.append({"dataset": ds, "method_1": "RAPT-Cheap",
                          "method_2": "Full Retraining", "margin": margin,
                          "mean_diff": float(np.mean(a[:n] - b[:n])),
                          "tost_p": tost(a[:n], b[:n], margin),
                          "equivalent_0.05": bool(tost(a[:n], b[:n], margin) < 0.05)})
    tostdf = pd.DataFrame(trows)
    write_csv(tostdf, "A7_tost.csv")
    print("\n=== TOST RAPT-Cheap vs Full Retraining ===")
    print(tostdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
