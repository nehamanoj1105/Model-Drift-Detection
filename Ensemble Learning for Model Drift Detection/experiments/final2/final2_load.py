"""Task C loader: Network operator KPIs time series (Zenodo 8147768).

Builds one multivariate, sample-level telemetry stream from the 14 real KPI
series shipped with the dataset. All series share the same 5-minute timestamp
grid (verified), so we align them into a 14-column feature vector per time step.

Target  : binary anomaly. A time step is positive iff it lies inside a
          producer-annotated incident (data_real_incidents.txt) for at least one
          series. The open-ended incident on r8 (marked -1) is excluded because
          it would label the whole tail positive; this is documented, not hidden.
Regime  : time-of-day slot (window index mod 6, 4-hour slots). The same slot
          recurs every day for 37 days -> every regime recurs ~37 times, which
          is the recurrence signal RAPT keys on.
Window  : 4 hours = 48 samples of 5 minutes. 223 windows total.
Prefix  : first 20% of windows (same protocol as the other streams).
"""

import os
import re
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "kpiops")

WINDOW_SIZE = 48          # 4 h at 5-min cadence
N_SLOTS = 6               # 6 four-hour slots per day
INITIAL_FRACTION = 0.20


def _load_incidents():
    inc = {}
    with open(os.path.join(DATA, "data_real_incidents.txt")) as f:
        for line in f:
            a = line.split()
            if len(a) >= 3:
                inc.setdefault(int(a[0][1:]), []).append((int(a[1]), int(a[2])))
    return inc


def load_kpiops():
    files = sorted(glob.glob(os.path.join(DATA, "data_real", "r*.txt")),
                   key=lambda p: int(re.search(r"r(\d+)", os.path.basename(p)).group(1)))
    ids, ts, cols = [], None, []
    for p in files:
        rid = int(re.search(r"r(\d+)", os.path.basename(p)).group(1))
        d = np.loadtxt(p)
        if ts is None:
            ts = d[:, 0]
        ids.append(rid)
        cols.append(d[:, 1])
    X = np.stack(cols, axis=1).astype(np.float64)
    # align check
    assert len(ts) == X.shape[0]
    n = len(ts)

    inc = _load_incidents()
    y = np.zeros(n, dtype=int)
    for r, iv in inc.items():
        for s, e in iv:
            if e < 0:          # open-ended r8 incident, excluded (see module doc)
                continue
            e = min(e, n - 1)
            y[s:e + 1] = 1

    n_windows = n // WINDOW_SIZE
    n = n_windows * WINDOW_SIZE
    X, y, ts = X[:n], y[:n], ts[:n]
    window_id = np.arange(n) // WINDOW_SIZE
    slot = (window_id % N_SLOTS)
    regime_id = np.array([f"slot{s}" for s in slot])
    feat = [f"kpi_r{rid}" for rid in ids]

    stream = pd.DataFrame({"window_id": window_id,
                           "regime_id": regime_id,
                           "y": y.astype(int)})
    for j, c in enumerate(feat):
        stream[c] = X[:, j]
    stream["_t0"] = ts

    n_init_windows = max(1, int(round(n_windows * INITIAL_FRACTION)))
    stream_def = {
        "dataset": "Network operator KPIs",
        "slug": "kpiops",
        "source": os.path.join(DATA, "data_real"),
        "temporal_field": "5-minute timestamp (shifted)",
        "target_desc": "binary anomaly from producer-annotated incidents "
                       "(union over 14 real KPI series; r8 open-ended excluded)",
        "regime_desc": "4-hour time-of-day slot (recurs daily, 37 days)",
        "window_size": WINDOW_SIZE,
        "n_raw_samples": int(len(ts)),
        "n_usable_samples": int(n),
        "total_windows": int(n_windows),
        "initial_train_windows": int(n_init_windows),
        "feature_columns": feat,
        "class_map": {"0": "normal", "1": "anomaly"},
        "n_classes": 2,
    }
    return stream, stream_def


if __name__ == "__main__":
    s, sd = load_kpiops()
    print("windows", sd["total_windows"], "init", sd["initial_train_windows"],
          "features", len(sd["feature_columns"]))
    print("sample positive rate", round(s["y"].mean(), 4))
    from collections import Counter
    print("windows per regime", Counter(
        s.groupby("window_id")["regime_id"].first().values))
