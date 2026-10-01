"""E2 -- which mechanism actually recovers UGR'16?

The paper attributes RAPT-Enhanced's UGR'16 recovery (0.8360 -> 0.9276) to
'selective parity refitting'. But the committed summary reports
parity_refits = 0 for RAPT-Enhanced on UGR'16, so parity refitting never fired.
The remaining difference between RAPT and RAPT-Enhanced on that stream is the
novelty-refit buffer size (1500 vs 500). This script isolates the two mechanisms
by rerunning the UGR'16 stream with a small ladder of controllers:

  base            RAPT, novelty buffer 500            (Table II 'RAPT')
  novelty1500     RAPT, novelty buffer 1500           (== RAPT-Enhanced with parity off)
  parity050       RAPT + parity refit, threshold 0.50 (the paper's setting)
  parity080       RAPT + parity refit, threshold 0.80
  parity090       RAPT + parity refit, threshold 0.90

Everything else (ensemble, buffer, anchor, seeds, window size, prefix) is
identical to the 9A harness.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp9a"))

from three_dataset_load import LOADERS, build_windows  # noqa: E402
from models_9a import make_class_anchor, make_train_buffer, create_base_ensemble  # noqa: E402
from rapt_9a import RAPTSystem  # noqa: E402
import exp9a_config as base  # noqa: E402

BUFFER = base.BUFFER_CAPACITY


class Variant(RAPTSystem):
    def __init__(self, novelty_n=500, parity=None, **kw):
        super().__init__(**kw)
        self.novelty_n = novelty_n
        self.parity = parity
        self.parity_refits = 0
        self._recent = []
        self._reused = False

    def _train_policy(self, window_id, X_buffer, y_buffer):
        ens = create_base_ensemble(seed=self.seed + window_id * 19)
        if X_buffer is not None and len(X_buffer) > 50:
            rx = np.array(X_buffer)[-self.novelty_n:]
            ry = np.array(y_buffer)[-self.novelty_n:]
            tx, ty = make_train_buffer(rx, ry, self.anchor_X, self.anchor_y,
                                       self.buffer_capacity)
            ens.fit(tx, ty)
        return ens

    def handle_regime_transition(self, new_regime_id, window_id, X_buffer=None, y_buffer=None):
        reused, cpu, wall = super().handle_regime_transition(
            new_regime_id, window_id, X_buffer, y_buffer)
        self._reused = reused
        self._recent = []
        return reused, cpu, wall

    def update(self, X_win, y_win, X_buffer=None, y_buffer=None):
        if self.parity is None:
            return 0.0
        acc = float(np.mean(self.active_ensemble.predict(X_win) == np.asarray(y_win)))
        self._recent.append(acc)
        if len(self._recent) > 3:
            self._recent.pop(0)
        if (self._reused and X_buffer is not None and len(self._recent) >= 3
                and np.mean(self._recent) < self.parity):
            ens = self._train_policy(0, X_buffer, y_buffer)
            self.active_ensemble = ens
            self.parity_refits += 1
            self._recent = []
        return 0.0


def load_stream():
    stream, sd = build_windows(LOADERS["UGR'16"]())
    feat = sd["feature_columns"]
    X_raw = stream[feat].values.astype(np.float64)
    y = stream["y"].values.astype(int)
    reg = stream["regime_id"].values
    wid = stream["window_id"].values
    n_init = sd["initial_train_windows"]
    init = wid < n_init
    mu = X_raw[init].mean(axis=0)
    s = np.where(X_raw[init].std(axis=0) < 1e-9, 1.0, X_raw[init].std(axis=0))
    X = (X_raw - mu) / s
    slices = {w: np.where(wid == w)[0] for w in range(n_init, sd["total_windows"])}
    return X, y, reg, wid, n_init, sd["total_windows"], slices


def f1(yt, yp):
    out = []
    for c in np.unique(np.concatenate([yt, yp])):
        tp = np.sum((yp == c) & (yt == c)); fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out))


def run(seed, **kw):
    X, y, reg, wid, n_init, n_win, slices = load_stream()
    init = wid < n_init
    X_init, y_init = X[init], y[init]
    ax, ay = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    sys_ = Variant(seed=seed, mode="full", enable_calibration=True,
                   anchor_X=ax, anchor_y=ay, buffer_capacity=BUFFER, **kw)
    prev = reg[np.where(wid == n_init - 1)[0][0]]
    sys_.fit_initial(prev, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    per_win = []
    yt, yp = [], []
    for w in range(n_init, n_win):
        idx = slices[w]
        r = reg[idx[0]]
        if r != prev:
            sys_.handle_regime_transition(r, w, X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
            prev = r
        p = sys_.predict(X[idx])
        per_win.append(f1(y[idx], p))
        yt.append(y[idx]); yp.append(p)
        buf_X.extend(X[idx]); buf_y.extend(y[idx])
        buf_X, buf_y = buf_X[-BUFFER:], buf_y[-BUFFER:]
        sys_.update(X[idx], y[idx], X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
    return float(np.mean(per_win)), sys_.parity_refits, sys_.reused_policy_count


VARIANTS = [("base (novelty 500)", dict(novelty_n=500, parity=None)),
            ("novelty1500 (= Enhanced, parity off)", dict(novelty_n=1500, parity=None)),
            ("parity 0.50 (paper setting)", dict(novelty_n=500, parity=0.50)),
            ("parity 0.90", dict(novelty_n=500, parity=0.90)),
            ("parity 0.95", dict(novelty_n=500, parity=0.95)),
            ("parity 0.97", dict(novelty_n=500, parity=0.97))]

print("E2 -- UGR'16 mechanism isolation\n")
print(f"{'variant':40s} {'seed42 F1':>10s} {'refits':>7s} {'reuses':>7s}")
for name, kw in VARIANTS:
    s, pr, ru = run(42, **kw)
    print(f"{name:40s} {s:10.4f} {pr:7d} {ru:7d}")

# exact reproduction check against the committed summary
import csv
committed = {}
for r in csv.DictReader(open(os.path.join(
        ROOT, "results", "experiment_9a_three", "raw", "summary_ugr16_full.csv"))):
    if r["method"] in ("RAPT", "RAPT-Enhanced"):
        committed.setdefault(r["method"], {})[int(r["seed"])] = float(r["macro_f1"])
print("\nReproduction check (mean over seeds 42-46):")
for name, kw in [("base (novelty 500)", dict(novelty_n=500, parity=None)),
                 ("novelty1500 (= Enhanced, parity off)", dict(novelty_n=1500, parity=None))]:
    vals = [run(s, **kw)[0] for s in (42, 43, 44, 45, 46)]
    ref = "RAPT" if kw["novelty_n"] == 500 else "RAPT-Enhanced"
    cv = np.mean(list(committed[ref].values()))
    print(f"  {name:40s} repro {np.mean(vals):.4f}  committed {ref} {cv:.4f}  "
          f"{'MATCH' if abs(np.mean(vals)-cv) < 1e-4 else 'DIFFER'}")

