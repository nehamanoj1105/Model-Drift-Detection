#!/usr/bin/env python3
"""Task 5 -- UGR'16 day-block bootstrap (24 calendar-day blocks, 5 seeds).

DISTINCT from the existing A7 regime-run bootstrap
(results/revalidation/A7_block_bootstrap.csv), which resamples contiguous runs
of equal regime_id.  Here the blocks are calendar days.

UGR'16: one window = 240 minutes; one calendar day = 1440 / 240 = 6 windows.
The evaluation stream is windows 36..179, i.e. calendar days 6..29 = 24 days.
Blocks are therefore day = window_id // 6, giving 24 blocks.

Metric: pooled macro-F1, recomputed from the per-window confusion matrices in
results/revalidation/raw/A1_per_window_all_streams.csv (cm_json).  A resample
draws the 24 day-blocks with replacement, sums the confusion matrices of every
window in the drawn blocks, and scores macro-F1 on the summed matrix.  The two
methods of a comparison are resampled on the SAME drawn blocks (paired).

Comparisons: RAPT vs Frozen, RAPT-Enhanced vs Frozen,
RAPT-Enhanced vs Full Retraining.

Run from the project directory:
    python "results/extract/v2/ugr16_dayblock_bootstrap.py"
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW = os.path.join(ROOT, "results", "revalidation", "raw", "A1_per_window_all_streams.csv")
POOLED = os.path.join(ROOT, "results", "revalidation", "A1_pooled_all_streams.csv")
SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 2000
BOOT_SEED = 0
WINDOWS_PER_DAY = 6  # 1440 min / 240 min


def cm_to_counts(cm, n_classes):
    """Return per-class (tp, fp, fn) vectors from a confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn


def macro_f1_from_counts(tp, fp, fn):
    """Macro-F1 from summed per-class tp/fp/fn (vectorised over classes)."""
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(np.mean(f1))


def load_ugr16():
    d = pd.read_csv(RAW)
    d = d[d.dataset == "ugr16"].copy()
    d["cm"] = d["cm_json"].apply(json.loads)
    d["day"] = (d["window_id"] // WINDOWS_PER_DAY).astype(int)
    return d


def dayblock_bootstrap(d, m1, m2, n_classes=2):
    out = []
    for seed in SEEDS:
        s = d[d.seed == seed]
        a = s[s.method == m1].sort_values("window_id")
        b = s[s.method == m2].sort_values("window_id")
        if len(a) != len(b) or len(a) == 0:
            continue
        assert (a.window_id.values == b.window_id.values).all()
        days = a["day"].values
        blocks = [np.where(days == dd)[0] for dd in sorted(set(days))]
        n_blocks = len(blocks)

        # per-window per-class counts for each method
        def counts(frame):
            tps, fps, fns = [], [], []
            for c in frame["cm"].values:
                tp, fp, fn = cm_to_counts(c, n_classes)
                tps.append(tp); fps.append(fp); fns.append(fn)
            return np.array(tps), np.array(fps), np.array(fns)

        ta, fa_, fna = counts(a)
        tb, fb_, fnb = counts(b)
        obs_a = macro_f1_from_counts(ta.sum(0), fa_.sum(0), fna.sum(0))
        obs_b = macro_f1_from_counts(tb.sum(0), fb_.sum(0), fnb.sum(0))

        rng = np.random.RandomState(BOOT_SEED + seed)
        diffs, fa, fb = [], [], []
        for _ in range(N_BOOT):
            pick = rng.randint(0, n_blocks, n_blocks)
            idx = np.concatenate([blocks[p] for p in pick])
            pa = macro_f1_from_counts(ta[idx].sum(0), fa_[idx].sum(0), fna[idx].sum(0))
            pb = macro_f1_from_counts(tb[idx].sum(0), fb_[idx].sum(0), fnb[idx].sum(0))
            fa.append(pa); fb.append(pb); diffs.append(pa - pb)
        out.append(dict(seed=seed, n_blocks=n_blocks,
                        obs_m1=obs_a, obs_m2=obs_b, obs_diff=obs_a - obs_b,
                        ci_low=np.percentile(diffs, 2.5),
                        ci_high=np.percentile(diffs, 97.5),
                        boot_mean=np.mean(diffs),
                        m1_ci_low=np.percentile(fa, 2.5),
                        m1_ci_high=np.percentile(fa, 97.5),
                        m2_ci_low=np.percentile(fb, 2.5),
                        m2_ci_high=np.percentile(fb, 97.5)))
    return pd.DataFrame(out)


def main():
    d = load_ugr16()
    pooled = pd.read_csv(POOLED)
    pooled = pooled[pooled.dataset == "ugr16"]

    # sanity: pooled macro-F1 recomputed from summed confusion == committed pooled
    chk = []
    for _, r in pooled.iterrows():
        s = d[(d.seed == r["seed"]) & (d.method == r["method"])]
        if len(s) == 0:
            continue
        tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2)
        for c in s["cm"].values:
            t, f_, n_ = cm_to_counts(c, 2)
            tp += t; fp += f_; fn += n_
        got = macro_f1_from_counts(tp, fp, fn)
        chk.append(dict(seed=r["seed"], method=r["method"],
                        committed_pooled=r["pooled_macro_f1"], recomputed_pooled=got,
                        abs_diff=abs(got - r["pooled_macro_f1"])))
    chk = pd.DataFrame(chk)
    chk.to_csv(os.path.join(HERE, "T5_ugr16_pooled_recompute_check.csv"), index=False)

    comparisons = [("RAPT", "Frozen"), ("RAPT-Enhanced", "Frozen"),
                   ("RAPT-Enhanced", "Full Retraining")]
    rows = []
    for m1, m2 in comparisons:
        per = dayblock_bootstrap(d, m1, m2)
        per.insert(0, "method_1", m1)
        per.insert(1, "method_2", m2)
        per.insert(2, "bootstrap_type", "day-block (24 calendar days)")
        rows.append(per)
    per_seed = pd.concat(rows, ignore_index=True)
    per_seed.to_csv(os.path.join(HERE, "T5_ugr16_dayblock_per_seed.csv"), index=False)

    agg = per_seed.groupby(["method_1", "method_2"]).agg(
        n_seeds=("seed", "count"),
        n_blocks_mean=("n_blocks", "mean"),
        obs_diff_mean=("obs_diff", "mean"),
        ci_low_mean=("ci_low", "mean"),
        ci_high_mean=("ci_high", "mean"),
        m1_ci_low_mean=("m1_ci_low", "mean"),
        m1_ci_high_mean=("m1_ci_high", "mean"),
    ).reset_index()
    agg["bootstrap_type"] = "day-block (24 calendar days)"
    agg["metric"] = "pooled_macro_f1"
    agg.to_csv(os.path.join(HERE, "T5_ugr16_dayblock.csv"), index=False)
    print(agg.to_string(index=False))

    # existing regime-run bootstrap for the same comparisons, clearly labelled
    a7 = pd.read_csv(os.path.join(ROOT, "results", "revalidation",
                                  "A7_block_bootstrap.csv"))
    a7 = a7[(a7.dataset == "ugr16")]
    a7 = a7[a7.apply(lambda r: (r.method_1, r.method_2) in comparisons, axis=1)]
    a7 = a7.copy()
    a7["bootstrap_type"] = "regime-run (contiguous regime visits) [existing A7]"
    a7.to_csv(os.path.join(HERE, "T5_ugr16_regimerun_for_comparison.csv"), index=False)
    print(a7.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
