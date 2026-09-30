"""
Experiment 9A (three telecom datasets) — dataset loading, profiling and
lightweight common preprocessing.

Datasets
--------
* 5G Campus Network QoS : the repo's processed per-window QoS stream
  (`processed_exp9_stream.csv`). Each row is already a 500-packet aggregate;
  we group `WINDOW_CAMPUS` consecutive rows into one streaming step.
* UGR'16               : per-minute netflow feature aggregates + 8 attack
  count labels (Camacho feature-data mirror). Target = binary anomaly
  (any attack vs normal). Regimes = workday/weekend x 4-hour period.
* NordicDat             : per-second cross-border LTE/5G QoS/mobility traces.
  Target = 3-class QoS class of the *next* window's median delay. Regimes =
  radio band (network condition).

Preprocessing is deliberately lightweight and identical in shape:
    Load -> identify target -> identify timestamp -> drop IDs/constants ->
    keep numeric features -> Inf->NaN -> drop rows with missing target ->
    training-prefix median imputation -> deterministic target encoding ->
    chronological order -> windowing.

No PCA / SHAP / RFE / hyper-parameter search / global normalisation / shuffling.
"""

import os
import hashlib
import numpy as np
import pandas as pd

from three_dataset_config import (
    CAMPUS_WINDOW_CSV, UGR16_MAT, NORDICDAT_CSV, WINDOW_SIZE,
)

# Per-dataset streaming-step sizes (samples per window). Documented in the
# report: the raw granularity differs (packet-aggregate rows / minutes /
# seconds), so the step is chosen to give a comparable number of adaptation
# decisions (~180-350 windows) without changing any other protocol component.
WINDOW_CAMPUS = 10
WINDOW_UGR16 = 240              # 4 hours: aligned to the 4-hour regime blocks
WINDOW_NORDIC = WINDOW_SIZE     # 500 seconds


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prefix_median_impute(X, n_init):
    """Median imputation using training-prefix statistics only."""
    med = np.nanmedian(X[:n_init], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def _encode_target(y):
    classes = sorted(pd.unique(y))
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=int), mapping


# ---------------------------------------------------------------------------
# 5G Campus Network QoS
# ---------------------------------------------------------------------------
def load_campus():
    df = pd.read_csv(CAMPUS_WINDOW_CSV)
    tcol = "_timestamp"
    df = df.sort_values(tcol).reset_index(drop=True)

    # Documented target: 3-class next-window p90-delay QoS label (already built
    # by the existing exp9 pipeline).
    y_raw = df["target"].values
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = {tcol, "stream_window_id", "block_step", "target"}
    feature_cols = [c for c in numeric if c not in drop]
    X = df[feature_cols].values.astype(np.float64)

    regime = df["regime_label"].astype(str).values
    ts = df[tcol].values
    return {
        "name": "5G Campus QoS",
        "slug": "5g_campus",
        "X": X, "y_raw": y_raw, "feature_cols": feature_cols,
        "regime": regime, "timestamp": ts, "window_size": WINDOW_CAMPUS,
        "source": CAMPUS_WINDOW_CSV,
        "temporal_field": tcol,
        "target_desc": "3-class next-window p90-delay QoS (GOOD/DEGRADED/BAD)",
        "regime_desc": "network scenario label (bandwidth/slots/ratio/scenario)",
        "y_agg": "majority",
        "extra": {"n_raw": len(df), "id_cols": sorted(drop), "time_col": ts},
    }


# ---------------------------------------------------------------------------
# UGR'16
# ---------------------------------------------------------------------------
_UGR16_ATTACK_IDX = [0, 1, 2, 3, 5, 6, 7]  # all except blacklist


def load_ugr16():
    import scipy.io as sio
    m = sio.loadmat(UGR16_MAT, squeeze_me=True, struct_as_record=False)
    # The UGR'16 calibration split (m['X']/m['Y']) contains only blacklist
    # traffic and no labelled attacks, so it cannot support a supervised stream.
    # We therefore use the labelled 30-day period (m['test']/m['Yt']), which is
    # the standard supervised UGR'16 block (1,364 attack minutes / 43,200).
    X = np.asarray(m["test"]).astype(np.float64)
    Yt = np.asarray(m["Yt"])
    classD = np.asarray(m["classDt"])
    classM = np.asarray(m["classMt"])
    classWD = np.asarray(m["classWDt"])
    obs = np.asarray(m["obs_lt"]).astype(str)

    var_l = [str(v) for v in m["var_l"]]
    feature_cols = list(var_l)

    attack = (Yt[:, _UGR16_ATTACK_IDX] > 0).any(axis=1)
    y_raw = np.where(attack, "anomaly", "normal")

    order = np.argsort(obs, kind="stable")
    X, y_raw, obs = X[order], y_raw[order], obs[order]
    classD, classM, classWD = classD[order], classM[order], classWD[order]

    hour = (classM.astype(int) // 60)
    period = hour // 4  # 6 four-hour periods
    wd = np.where(classWD == 1, "WD", "WE")
    regime = np.char.add(np.char.add(wd.astype(str), "_"), period.astype(str))

    return {
        "name": "UGR'16",
        "slug": "ugr16",
        "X": X, "y_raw": y_raw, "feature_cols": feature_cols,
        "regime": regime, "timestamp": obs, "window_size": WINDOW_UGR16,
        "source": UGR16_MAT,
        "temporal_field": "obs_lt (minute timestamp string)",
        "target_desc": "binary anomaly: any non-blacklist attack minute in window",
        "regime_desc": "workday/weekend x 4-hour period (10 WD + 10 WE regimes)",
        # Majority vote per window would collapse to all-normal (anomalies are
        # temporally clustered); a window is anomalous if it contains any attack
        # minute. Documented in the report.
        "y_agg": "any",
        "extra": {"n_raw": len(X), "day": classD},
    }


# ---------------------------------------------------------------------------
# NordicDat
# ---------------------------------------------------------------------------
def load_nordicdat():
    df = pd.read_csv(NORDICDAT_CSV)
    tcol = "timestamp"
    df = df.sort_values(tcol).reset_index(drop=True)

    # Documented target: 3-class QoS class of the NEXT window's median delay,
    # using thresholds fit on the initial training prefix (no look-ahead).
    signal = df["delay"].values.astype(np.float64)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # Drop raw GPS coordinates and mobility IDs that are pure identifiers; keep
    # network/mobility/PHY features.
    drop = {tcol, "latitude", "longitude", "serving_cell_id", "heading"}
    feature_cols = [c for c in numeric if c not in drop]
    X = df[feature_cols].values.astype(np.float64)

    regime = df["operator"].astype(str).values
    ts = df[tcol].values
    return {
        "name": "NordicDat",
        "slug": "nordicdat",
        "X": X, "y_raw": None, "feature_cols": feature_cols,
        "regime": regime, "timestamp": ts, "window_size": WINDOW_NORDIC,
        "source": NORDICDAT_CSV,
        "temporal_field": tcol,
        "target_desc": "3-class next-window median-delay QoS (prefix thresholds)",
        "regime_desc": "mobile network operator (Nordic cross-border roaming)",
        "extra": {"delay_signal": signal, "n_raw": len(df), "band": df["band"].astype(str).values},
    }


LOADERS = {"5G Campus QoS": load_campus, "UGR'16": load_ugr16,
           "NordicDat": load_nordicdat}


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def build_windows(data, max_samples=None, initial_frac=0.20):
    """Chronologically window a loaded dataset into the common stream schema.

    The returned stream is SAMPLE-level: one row per raw observation, with a
    `window_id` grouping `window_size` consecutive observations. Models train
    and predict on the raw samples inside each window (test-then-train), exactly
    as in the existing 9A/9B harnesses; the window is the adaptation step.

    Returns (stream_df, stream_def). stream_df columns:
      window_id, regime_id, y, <feature columns...>
    `y` is the sample-level encoded label (0..C-1); features are prefix-imputed.
    """
    X = data["X"].copy()
    name = data["name"]
    y_raw = data["y_raw"]
    regime = np.asarray(data["regime"])
    win = data["window_size"]

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        regime = regime[:max_samples]
        if y_raw is not None:
            y_raw = y_raw[:max_samples]
        if "delay_signal" in data["extra"]:
            data["extra"]["delay_signal"] = data["extra"]["delay_signal"][:max_samples]

    n = len(X)
    n_windows = n // win
    n = n_windows * win
    X = X[:n]
    regime = regime[:n]

    # ---- sample-level target -------------------------------------------------
    mapping = None
    if name == "NordicDat":
        signal = data["extra"]["delay_signal"][:n]
        win_med = np.median(signal.reshape(n_windows, win), axis=1)
        next_med = np.concatenate([win_med[1:], [win_med[-1]]])
        n_init_w = max(1, int(round(n_windows * initial_frac)))
        t1 = np.median(win_med[:n_init_w])
        t2 = np.percentile(win_med[:n_init_w], 80)
        y_win = np.where(next_med <= t1, 0, np.where(next_med <= t2, 1, 2)).astype(int)
        y = np.repeat(y_win, win)              # broadcast window label to samples
        mapping = {0: "GOOD", 1: "DEGRADED", 2: "BAD"}
    else:
        y_agg = data.get("y_agg", "majority")
        y_bin = np.asarray([1 if v == "anomaly" else 0 for v in y_raw])[:n]
        if y_agg in ("any", "rate>=0.5"):
            # UGR'16: sample-level minute anomaly label.
            y = y_bin
        else:
            y, mapping = _encode_target(y_raw[:n])

    # ---- prefix-only imputation ---------------------------------------------
    n_init_samples = max(win, int(round(n * initial_frac)))
    X = _prefix_median_impute(X, n_init_samples)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    window_id = np.arange(n) // win
    stream = pd.DataFrame({"window_id": window_id,
                           "regime_id": regime.astype(str),
                           "y": y.astype(int)})
    for j, c in enumerate(data["feature_cols"]):
        stream[c] = X[:, j]
    stream["_t0"] = np.asarray(data["timestamp"])[:n]

    n_init_windows = max(1, int(round(n_windows * initial_frac)))
    if mapping is None:
        mapping = {str(k): str(v) for k, v in
                   zip([0, 1], ["normal", "anomaly"])}
    stream_def = {
        "dataset": name,
        "slug": data["slug"],
        "source": data["source"],
        "temporal_field": data["temporal_field"],
        "target_desc": data["target_desc"],
        "regime_desc": data["regime_desc"],
        "window_size": win,
        "n_raw_samples": int(len(data["X"])),
        "n_usable_samples": int(n),
        "total_windows": int(n_windows),
        "initial_train_windows": int(n_init_windows),
        "feature_columns": data["feature_cols"],
        "class_map": {str(k): str(v) for k, v in mapping.items()},
        "regime_sequence": (pd.Series(regime[:n]).groupby(window_id)
                            .first().tolist()),
        "n_classes": int(pd.Series(y).nunique()),
    }
    return stream, stream_def
