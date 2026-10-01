"""E5 -- what does a safe variant actually cost?

E2 showed the parity-refit branch never fires on UGR'16 (rolling per-window
accuracy of a reused policy stays far above the 0.5 parity threshold), so the
'reuse with a performance check' mechanism is untested by the reported run. This
script measures a variant whose check does fire: a reused policy is refit on the
recent buffer whenever its rolling per-window accuracy falls below an absolute
floor. We sweep the floor and report macro-F1, refit count and adaptation CPU on
UGR'16, with 5G Campus as a control.
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


def macro_f1(yt, yp):
    out = []
    for c in np.unique(np.concatenate([yt, yp])):
        tp = np.sum((yp == c) & (yt == c)); fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out))


class SafeRAPT(RAPTSystem):
    """Reused policy refit when its rolling per-window accuracy < floor."""

    def __init__(self, floor=None, window=3, novelty_n=500, **kw):
        super().__init__(**kw)
        self.floor = floor
        self.win = window
        self.novelty_n = novelty_n
        self.refits = 0
        self._acc = []
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
        self._acc = []
        return reused, cpu, wall

    def update(self, X_win, y_win, X_buffer=None, y_buffer=None):
        if self.floor is None:
            return 0.0
        acc = float(np.mean(self.active_ensemble.predict(X_win) == np.asarray(y_win)))
        self._acc.append(acc)
        if len(self._acc) > self.win:
            self._acc.pop(0)
        if (self._reused and len(self._acc) >= self.win
                and np.mean(self._acc) < self.floor and X_buffer is not None):
            self.active_ensemble = self._train_policy(0, X_buffer, y_buffer)
            self.refits += 1
            self._acc = []
        return 0.0


def run(loader, seed, floor, novelty_n=500):
    stream, sd = build_windows(loader())
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
    X_init, y_init = X[init], y[init]
    ax, ay = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    sys_ = SafeRAPT(seed=seed, mode="full", enable_calibration=True,
                    anchor_X=ax, anchor_y=ay, buffer_capacity=BUFFER,
                    floor=floor, novelty_n=novelty_n)
    prev = reg[np.where(wid == n_init - 1)[0][0]]
    sys_.fit_initial(prev, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    per_win = []
    for w in range(n_init, sd["total_windows"]):
        idx = np.where(wid == w)[0]
        r = reg[idx[0]]
        if r != prev:
            sys_.handle_regime_transition(r, w, X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
            prev = r
        p = sys_.predict(X[idx])
        per_win.append(macro_f1(y[idx], p))
        buf_X.extend(X[idx]); buf_y.extend(y[idx])
        buf_X, buf_y = buf_X[-BUFFER:], buf_y[-BUFFER:]
        sys_.update(X[idx], y[idx], X_buffer=buf_X[-BUFFER:], y_buffer=buf_y[-BUFFER:])
    return float(np.mean(per_win)), sys_.refits, sys_.adaptation_cpu_time


SEEDS = [42, 43, 44, 45, 46]
for stream_name, loader in [("UGR'16", LOADERS["UGR'16"]),
                            ("5G Campus", LOADERS["5G Campus QoS"])]:
    print(f"\nE5 -- absolute-floor safe variant on {stream_name}")
    print(f"{'floor':>6s} {'F1 (mean)':>10s} {'refits':>7s} {'adapt CPU s':>12s}")
    for floor in [None, 0.60, 0.70, 0.80, 0.90, 0.95]:
        vals = [run(loader, s, floor) for s in SEEDS]
        f1 = np.mean([v[0] for v in vals])
        rf = np.mean([v[1] for v in vals])
        cpu = np.mean([v[2] for v in vals])
        tag = "base RAPT" if floor is None else ""
        print(f"{str(floor):>6s} {f1:10.4f} {rf:7.1f} {cpu:12.4f}  {tag}")
