"""
Dataset loading and stream construction for the final RAPT experiment.

Four network streams, one protocol. Each loader returns a dict with:
  X          (n_samples, d) float64 features
  y          (n_samples,) int labels (window-level broadcast to samples)
  regime     (n_samples,) str regime id per sample
  window_rows  rows per window
  meta       provenance / target / regime description

Design notes
------------
* 5G Campus / NordicDat / 5G NR carry a window-level target that is broadcast to
  every sample of the window; a window is a telemetry aggregate.
* UGR'16 is a genuine per-minute labelled stream: the label is per sample and a
  window is 240 consecutive minutes (matching the 4-hour regime granularity used
  in the manuscript). The manuscript reports an 18-window UGR'16 stream; halving
  the window to 240 rows (as below) restores that granularity while keeping the
  same 4-hour regime blocks.
* Regimes are read from the stream (semantic condition), never derived from the
  target.
* Preprocessing statistics are fit on the initial training prefix only.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _prefix_impute(X, n_init_rows):
    med = np.nanmedian(X[:n_init_rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _drop_constants(X, cols, n_init_rows):
    """Drop columns constant over the initial training prefix (documented rule)."""
    keep = []
    for j in range(X.shape[1]):
        col = X[:n_init_rows, j]
        col = col[~np.isnan(col)]
        if col.size and np.ptp(col) > 1e-12 and np.nanstd(col) > 1e-12:
            keep.append(j)
    return X[:, keep], [cols[j] for j in keep]


# ---------------------------------------------------------------------------
def load_campus(window_rows):
    df = pd.read_csv(os.path.join(PROJECT_DIR, "experiments", "exp9", "data",
                                  "processed_exp9_stream.csv"))
    df = df.sort_values("_timestamp").reset_index(drop=True)
    drop = {"_timestamp", "_scenario", "_rep", "_direction", "regime_key",
            "stream_window_id", "block_step", "regime_label", "next_p90_delay",
            "target"}
    feat = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    return {
        "X": df[feat].values.astype(np.float64),
        "y_raw": df["target"].values.astype(int),
        "y_is_window_level": True,
        "regime": df["regime_label"].astype(str).values,
        "feature_cols": feat,
        "window_rows": window_rows,
        "meta": {"kind": "measured", "label_type": "QoS",
                 "target_desc": "3-class next-window p90-delay QoS (GOOD/DEGRADED/BAD)",
                 "regime_desc": "5G campus scenario label"},
    }


# ---------------------------------------------------------------------------
_UGR16_ATTACK_IDX = [0, 1, 2, 3, 5, 6, 7]  # all but blacklist


def load_ugr16(window_rows):
    import scipy.io as sio
    m = sio.loadmat(os.path.join(PROJECT_DIR, "experiments", "exp9a", "data",
                                 "ugr16", "UGR16v1.mat"),
                    squeeze_me=True, struct_as_record=False)
    X = np.asarray(m["test"]).astype(np.float64)
    Yt = np.asarray(m["Yt"])
    classM = np.asarray(m["classMt"])
    classWD = np.asarray(m["classWDt"])
    obs = np.asarray(m["obs_lt"]).astype(str)
    feat = [str(v) for v in m["var_l"]]

    attack = (Yt[:, _UGR16_ATTACK_IDX] > 0).any(axis=1)
    y = np.where(attack, 1, 0).astype(int)  # 1 = attack/anomaly, 0 = normal

    order = np.argsort(obs, kind="stable")
    X, y, obs = X[order], y[order], obs[order]
    classM, classWD = classM[order], classWD[order]

    hour = classM.astype(int) // 60
    period = hour // 4
    wd = np.where(classWD == 1, "WD", "WE").astype(str)
    regime = np.char.add(np.char.add(wd, "_"), period.astype(str))

    return {
        "X": X, "y_raw": y, "y_is_window_level": False,
        "regime": regime, "feature_cols": feat, "window_rows": window_rows,
        "meta": {"kind": "measured", "label_type": "attack",
                 "target_desc": "binary per-minute attack label (any non-blacklist attack)",
                 "regime_desc": "workday/weekend x 4-hour period"},
    }


# ---------------------------------------------------------------------------
def load_nordicdat(window_rows):
    df = pd.read_csv(os.path.join(PROJECT_DIR, "experiments", "exp9a", "data",
                                  "nordicdat", "nordicdat.csv"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    drop = {"timestamp", "latitude", "longitude", "serving_cell_id", "heading"}
    feat = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    return {
        "X": df[feat].values.astype(np.float64),
        "delay_signal": df["delay"].values.astype(np.float64),
        "y_raw": None, "y_is_window_level": True,
        "regime": df["operator"].astype(str).values,
        "feature_cols": feat, "window_rows": window_rows,
        "meta": {"kind": "measured", "label_type": "QoS",
                 "target_desc": "3-class next-window median-delay QoS (prefix thresholds)",
                 "regime_desc": "mobile network operator (cross-border roaming)"},
    }


# ---------------------------------------------------------------------------
def load_nrlat(window_rows):
    df = pd.read_csv(os.path.join(PROJECT_DIR, "experiments", "exp9b", "data",
                                  "processed_exp9b_stream.csv"))
    drop = {"regime_id", "scenario_filename", "regime_letter", "window_id",
            "segment_index", "target_p90_lat", "qos_target",
            "ue_count", "scs", "ue_cap", "sinr", "offered_bitrate"}
    feat = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    return {
        "X": df[feat].values.astype(np.float64),
        "y_raw": df["qos_target"].values.astype(int),
        "y_is_window_level": True,
        "regime": df["regime_letter"].astype(str).values,
        "feature_cols": feat, "window_rows": window_rows,
        "meta": {"kind": "simulated", "label_type": "QoS",
                 "target_desc": "3-class next-window p90-delay QoS (frozen thresholds)",
                 "regime_desc": "simulated 5G NR scenario (UE/SCS/SINR/bitrate)"},
    }


LOADERS = {
    "5G Campus": load_campus,
    "UGR'16": load_ugr16,
    "NordicDat": load_nordicdat,
    "5G NR Lat.": load_nrlat,
}


# ---------------------------------------------------------------------------
def build_stream(name, cfg):
    """Return (stream_df, stream_def) for one dataset under the final protocol."""
    ds = next(d for d in cfg["datasets"] if d["name"] == name)
    data = LOADERS[name](ds["window_rows"])
    rows = ds["window_rows"]
    X = data["X"].copy()
    regime = np.asarray(data["regime"])
    n = len(X)
    n_windows = n // rows
    n = n_windows * rows
    X, regime = X[:n], regime[:n]

    n_init_w = _round_half_up(n_windows * cfg["initial_train_fraction"])
    n_init_rows = n_init_w * rows

    if name == "NordicDat":
        sig = data["delay_signal"][:n]
        win_med = np.median(sig.reshape(n_windows, rows), axis=1)
        next_med = np.concatenate([win_med[1:], [win_med[-1]]])
        t1 = np.median(win_med[:n_init_w])
        t2 = np.percentile(win_med[:n_init_w], 80)
        y_win = np.where(next_med <= t1, 0, np.where(next_med <= t2, 1, 2)).astype(int)
        y = np.repeat(y_win, rows)
        class_map = {0: "GOOD", 1: "DEGRADED", 2: "BAD"}
    else:
        # Campus, UGR'16 and 5G NR carry a per-row label (the campus `target` is
        # the next-window p90-delay class; UGR'16 is a per-minute attack label;
        # 5G NR has one aggregate row per window). Labels are used as-is.
        y = np.asarray(data["y_raw"])[:n].astype(int)
        class_map = ({0: "normal", 1: "attack"} if name == "UGR'16"
                     else {0: "GOOD", 1: "DEGRADED", 2: "BAD"})

    X = _prefix_impute(X, n_init_rows)
    X, kept = _drop_constants(X, data["feature_cols"], n_init_rows)

    window_id = np.arange(n) // rows
    base = pd.DataFrame({"window_id": window_id,
                         "regime_id": regime.astype(str),
                         "y": y.astype(int)})
    feats = pd.DataFrame(X, columns=kept)
    stream = pd.concat([base, feats], axis=1)

    reg_win = stream.groupby("window_id")["regime_id"].first().values
    transitions = int(np.sum(reg_win[1:] != reg_win[:-1]))
    # Recurrences = regime visits beyond the first occurrence of each regime.
    seen, visits, prev = set(), 0, None
    for r in reg_win:
        if r != prev:
            visits += 1
            seen.add(r)
        prev = r
    n_distinct = len(set(reg_win))
    recurrences = int(visits - n_distinct)

    base_stats = {
        "dataset": name, "slug": ds["slug"],
        "kind": data["meta"]["kind"], "label_type": data["meta"]["label_type"],
        "is_simulated": bool(data["meta"]["kind"] == "simulated"),
        "target_desc": data["meta"]["target_desc"],
        "regime_desc": data["meta"]["regime_desc"],
        "window_rows": rows,
        "n_samples": int(n),
        "n_features": len(kept),
        "feature_columns": kept,
        "n_classes": int(len(np.unique(y))),
        "classes": sorted(int(c) for c in np.unique(y)),
        "class_map": class_map,
        "class_balance": {int(c): int((y == c).sum()) for c in np.unique(y)},
        "total_windows": int(n_windows),
        "initial_train_windows": int(n_init_w),
        "initial_train_rows": int(n_init_rows),
        "prefix_rounding_rule": cfg["prefix_rounding"],
        "evaluated_windows": int(n_windows - n_init_w),
        "n_regimes": int(len(set(reg_win))),
        "n_transitions": transitions,
        "n_recurrences": recurrences,
        "regime_sequence": reg_win.tolist(),
    }

    if name == "UGR'16":
        # Attacks are sparse; per-window attack fraction is informative.
        af = stream.groupby("window_id")["y"].mean().values
        base_stats["attack_fraction_overall"] = float(y.mean())
        base_stats["attack_fraction_window_min"] = float(np.min(af))
        base_stats["attack_fraction_window_max"] = float(np.max(af))
        base_stats["attack_fraction_window_mean"] = float(np.mean(af))
        base_stats["n_windows_with_attack"] = int(np.sum(af > 0))
        # class balance per regime
        per_reg = {}
        for r in np.unique(regime):
            yy = y[regime == r]
            per_reg[str(r)] = {"n": int(len(yy)),
                               "attack": int(yy.sum()),
                               "attack_frac": float(yy.mean()) if len(yy) else 0.0}
        base_stats["class_balance_per_regime"] = per_reg

    stream_def = base_stats
    return stream, stream_def


def _round_half_up(x):
    import math
    return int(math.floor(x + 0.5))
