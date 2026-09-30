"""
Dataset profiling + regime/recurrence analysis for the three-dataset 9A screen.

Produces dataset_profile.csv and a per-dataset regime/recurrence table.
Lightweight distribution comparison via Wasserstein distance on the top-variance
numeric features.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from three_dataset_config import (
    DATASETS, SCREENING_DIR, WINDOW_SIZE,
)
from three_dataset_load import LOADERS, build_windows


def _profile_one(name, data, max_samples=None):
    stream, sd = build_windows(data, max_samples=max_samples)
    X = data["X"]
    n_raw = data["extra"]["n_raw"]
    n_feat = X.shape[1]
    # missing after Inf->NaN (before imputation happens inside build_windows)
    Xf = X.astype(np.float64).copy()
    Xf[~np.isfinite(Xf)] = np.nan
    miss = float(np.isnan(Xf).mean() * 100)
    ts = np.asarray(data["timestamp"])
    try:
        tmin, tmax = float(np.min(ts.astype(np.float64))), float(np.max(ts.astype(np.float64)))
    except Exception:
        tmin = tmax = float("nan")
    n_classes = int(sd["n_classes"])
    regimes = pd.Series(sd["regime_sequence"])
    # duplicates on raw rows
    try:
        raw_dup = int(pd.DataFrame(X).duplicated().sum())
    except Exception:
        raw_dup = -1
    return {
        "Dataset": name,
        "Samples": n_raw,
        "Usable_samples": sd["n_usable_samples"],
        "Features": n_feat,
        "Target": sd["target_desc"],
        "Temporal_field": sd["temporal_field"],
        "Classes": n_classes,
        "Regimes": regimes.nunique(),
        "Total_windows": sd["total_windows"],
        "Window_size": sd["window_size"],
        "Initial_train_windows": sd["initial_train_windows"],
        "Missing_pct": round(miss, 3),
        "Duplicate_rows": raw_dup,
        "Chronological_span": f"{tmin:.0f}..{tmax:.0f}",
    }, stream, sd


def _recurrence(stream, sd, feature_cols):
    """Recurrence analysis on the regime sequence (contiguous visits)."""
    seq = np.asarray(sd["regime_sequence"])
    n = len(seq)

    # Build contiguous visits: (regime, start, end_exclusive)
    visits = []
    start = 0
    for i in range(1, n + 1):
        if i == n or seq[i] != seq[start]:
            visits.append((seq[start], start, i))
            start = i

    visit_counts = {}
    for r, a, b in visits:
        visit_counts.setdefault(r, []).append((a, b))
    uniq = sorted(visit_counts)
    n_unique = len(uniq)
    n_repeated = int(sum(1 for r in uniq if len(visit_counts[r]) > 1))

    seen = set()
    recur_events = 0
    for r, a, b in visits:
        if r in seen:
            recur_events += 1
        seen.add(r)

    gaps = []
    for r in uniq:
        v = visit_counts[r]
        starts = [a for a, b in v]
        if len(starts) > 1:
            gaps.extend(np.diff(starts).tolist())
    avg_gap = float(np.mean(gaps)) if gaps else float("nan")

    # distribution similarity: scale-normalised Wasserstein between the earlier
    # and later halves of each regime's windows (works for any block size).
    # X is sample-level; visit indices are window indices -> scale by window size.
    win = sd.get("window_size", 1)
    X = stream[feature_cols].values.astype(np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    col_std = X.std(axis=0)
    top = np.argsort(-col_std)[:min(10, X.shape[1])]
    sims = []
    for r in uniq:
        v = visit_counts[r]
        if len(v) < 2:
            continue
        idxs = []
        for a, b in v:
            idxs.extend(range(a * win, b * win))
        idxs = np.array(idxs)
        if len(idxs) < 4:
            continue
        half = len(idxs) // 2
        A = X[idxs[:half]][:, top]
        B = X[idxs[half:]][:, top]
        ds = [wasserstein_distance(A[:, k], B[:, k]) /
              (col_std[top[k]] + 1e-9) for k in range(len(top))]
        sims.append(float(np.mean(ds)))
    dist_sim = float(np.mean(sims)) if sims else float("nan")

    recur_frac = recur_events / max(1, len(seq) - n_unique)
    return {
        "unique_regimes": n_unique,
        "repeated_regimes": n_repeated,
        "recurrence_events": int(recur_events),
        "avg_recurrence_distance_windows": round(avg_gap, 2),
        "mean_feature_wasserstein_between_visits": round(dist_sim, 4),
        "recurrence_fraction": round(recur_frac, 3),
        "samples_per_recurrence": (round(len(seq) / max(1, recur_events), 2)
                                   if recur_events else float("nan")),
    }


def main():
    os.makedirs(SCREENING_DIR, exist_ok=True)
    profiles, recur_rows = [], []
    for name in DATASETS:
        data = LOADERS[name]()
        prof, stream, sd = _profile_one(name, data)
        prof["File_size_MB"] = round(os.path.getsize(data["source"]) / 1e6, 2)
        profiles.append(prof)
        rec = _recurrence(stream, sd, sd["feature_columns"])
        rec["dataset"] = name
        recur_rows.append(rec)
        print(f"[{name}] windows={sd['total_windows']} classes={sd['n_classes']} "
              f"regimes={rec['unique_regimes']} recurrence_events={rec['recurrence_events']}")

    df_prof = pd.DataFrame(profiles)[[
        "Dataset", "Samples", "Usable_samples", "Features", "Target",
        "Temporal_field", "Classes", "Regimes", "Total_windows", "Window_size",
        "Initial_train_windows", "Missing_pct", "Duplicate_rows",
        "Chronological_span", "File_size_MB"]]
    df_prof.to_csv(os.path.join(SCREENING_DIR, "dataset_profile.csv"), index=False)
    df_rec = pd.DataFrame(recur_rows)
    df_rec.to_csv(os.path.join(SCREENING_DIR, "recurrence_analysis.csv"), index=False)
    print("\n", df_prof.to_string(index=False))
    print("\n", df_rec.to_string(index=False))
    return df_prof, df_rec


if __name__ == "__main__":
    main()
