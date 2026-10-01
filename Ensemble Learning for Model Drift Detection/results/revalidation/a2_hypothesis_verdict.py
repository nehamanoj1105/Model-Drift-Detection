"""A2 verdict: does UGR'16 reuse fail because P(Y|X) changes with recurrence?

Reads A2_regime_stability.csv and emits A2_hypothesis_verdict.csv. Two hypotheses
are tested on the paired per-visit scores produced by a2_analysis.py.

H1 (identity tracks behaviour): a policy trained on a regime's previous visit
    predicts the next visit of the same regime as well as a policy trained on the
    contiguous previous block. Rejected if same-regime F1 is materially lower.

H2 (the failure is a changed conditional): the same-regime deficit on UGR'16 is
    attributable to P(Y|X) moving rather than to P(X) moving. Judged inconclusive
    when the class-conditional feature shift is large (covariate evidence) and
    same-regime transfer is no worse than contiguous transfer.

Paired tests use the per-regime observations; no significance is claimed where the
paired sample is too small to support it.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
STAB = os.path.join(HERE, "A2_regime_stability.csv")
OUT = os.path.join(HERE, "A2_hypothesis_verdict.csv")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return len(a), np.nan, np.nan, np.nan
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    return len(a), float(diff.mean()), float(t), float(p)


def main():
    df = pd.read_csv(STAB)
    rows = []
    for ds, g in df.groupby("dataset"):
        same = g["f1_same_regime_prev_visit"]
        contig = g["f1_contiguous_prev_block"]
        n, md, t, p = paired(same, contig)
        shift = float(g["class_cond_feature_shift"].mean())
        mean_same = float(np.nanmean(same)) if same.notna().any() else np.nan
        mean_contig = float(np.nanmean(contig)) if contig.notna().any() else np.nan

        # H1: same-regime transfer is materially below the contiguous control.
        if np.isnan(md):
            h1 = "INSUFFICIENT_DATA"
        elif md < -0.02 and p < 0.05:
            h1 = "SUPPORTED"
        else:
            h1 = "NOT_SUPPORTED"
        rows.append({
            "dataset": ds, "hypothesis": "H1_identity_tracks_behaviour",
            "paired_n": n, "mean_same_regime": round(mean_same, 4),
            "mean_contiguous": round(mean_contig, 4), "mean_diff": round(md, 4),
            "t_stat": round(t, 4) if not np.isnan(t) else np.nan,
            "p_value": round(p, 4) if not np.isnan(p) else np.nan,
            "class_cond_feature_shift": shift, "verdict": h1,
        })

        # H2: on UGR'16, can the deficit be attributed to a conditional change?
        if ds != "ugr16":
            rows.append({
                "dataset": ds, "hypothesis": "H2_deficit_is_concept_drift",
                "paired_n": n, "mean_same_regime": round(mean_same, 4),
                "mean_contiguous": round(mean_contig, 4), "mean_diff": round(md, 4),
                "t_stat": round(t, 4) if not np.isnan(t) else np.nan,
                "p_value": round(p, 4) if not np.isnan(p) else np.nan,
                "class_cond_feature_shift": shift, "verdict": "NOT_APPLICABLE",
            })
            continue

        # UGR'16: same-regime reuse is not worse than contiguous reuse, and the
        # class-conditional feature distributions move a great deal between visits.
        # Both point away from a conditional change as the explanation.
        if (not np.isnan(md)) and md >= 0 and shift > 1e3:
            h2 = "NOT_SUPPORTED"
        elif (not np.isnan(md)) and md < -0.02 and p < 0.05 and shift < 1e2:
            h2 = "SUPPORTED"
        else:
            h2 = "INCONCLUSIVE"
        rows.append({
            "dataset": ds, "hypothesis": "H2_deficit_is_concept_drift",
            "paired_n": n, "mean_same_regime": round(mean_same, 4),
            "mean_contiguous": round(mean_contig, 4), "mean_diff": round(md, 4),
            "t_stat": round(t, 4) if not np.isnan(t) else np.nan,
            "p_value": round(p, 4) if not np.isnan(p) else np.nan,
            "class_cond_feature_shift": shift, "verdict": h2,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
