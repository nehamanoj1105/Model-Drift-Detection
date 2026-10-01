"""v3 shared library.

Imports the existing (unchanged) 9A/9B pipelines and the v2 RAPT controller,
and adds:
  * a method registry that maps a method name to a driver factory, so every step
    runs through one code path;
  * provenance modes for RAPT (original / deferred / regime_pure);
  * pooled-macro-F1 as the primary metric, per-window macro-F1 as secondary
    (never mixed in one table);
  * block-bootstrap helpers and a CSV-only reporting contract.

Nothing here modifies existing files.
"""
import os
import sys
import json
import time
import hashlib
import platform
import datetime
import copy

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(FINAL, "..", ".."))
REVAL = os.path.join(ROOT, "results", "revalidation")
V2 = os.path.join(FINAL, "v2")
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
EXP9B = os.path.join(ROOT, "experiments", "exp9b")

for _p in (REVAL, V2, EXP9A, EXP9B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streams as S                       # noqa: E402
from metrics_pooled import pooled_metrics, pooled_binary_attack_metrics, window_confusion  # noqa: E402
from models_9a import create_base_ensemble, make_class_anchor, make_train_buffer  # noqa: E402
from evaluation_9a import window_metrics  # noqa: E402
from event_driven_9a import EventDrivenEnsemble  # noqa: E402
from drift_detectors_9a import DetectorAdaptiveModel  # noqa: E402
from rapt_9a import RAPTSystem, RAPTEnhancedSystem, RegimeCheckpoint  # noqa: E402
from run_v3 import RAPTV2, FRDriver, EventDrivenV2, DetectorAdaptiveModelV2, set_detector_params  # noqa: E402

V3 = HERE
RAW = os.path.join(V3, "raw")
FIGDIR = os.path.join(V3, "figures")
TABLEDIR = os.path.join(V3, "latex_tables")
CONFIG = os.path.join(V3, "config_frozen_v3.json")
GATES = os.path.join(V3, "gates.csv")

SEEDS = [42, 43, 44, 45, 46]
SEEDS_CHEAP = list(range(42, 62))
DATASETS = ["5G Campus QoS", "UGR'16", "NordicDat", "5G NR"]
SLUG = {"5G Campus QoS": "5g_campus", "UGR'16": "ugr16",
        "NordicDat": "nordicdat", "5G NR": "5g_nr"}

BUFFER_CAPACITY = 1000
CHEAP_TREES = 20
CHEAP_BUFFER = 300
FULL_TREES = 50
DETECTOR_PARAMS = {
    "ADWIN": {"delta": 0.002},
    "Page-Hinkley": {"min_instances": 30, "delta": 0.005, "threshold": 50, "alpha": 0.9999},
    "EDD": {"min_instances": 30, "warning_level": 2.0, "drift_level": 3.0},
    "EDMA": {"min_instances": 30, "alpha": 0.2, "k": 2.0},
}
DETECTORS = ["ADWIN", "Page-Hinkley", "EDD", "EDMA"]

# set by a caller before run_stream when Periodic-Cheap-Matched is requested
PERIODIC_MATCHED = 10


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (V3, RAW, FIGDIR, TABLEDIR):
        os.makedirs(d, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    with open(CONFIG) as f:
        return json.load(f)


def freeze_config():
    """Write the config hash next to the config (run once, before evaluation)."""
    ensure_dirs()
    p = CONFIG + ".sha256"
    with open(p, "w") as f:
        f.write(sha256(CONFIG) + "\n")
    return sha256(CONFIG)


def write_csv(df, name):
    ensure_dirs()
    p = os.path.join(V3, name)
    df.to_csv(p, index=False)
    return p


def write_raw(df, name):
    ensure_dirs()
    p = os.path.join(RAW, name)
    df.to_csv(p, index=False)
    return p


def write_tex(text, name):
    ensure_dirs()
    p = os.path.join(TABLEDIR, name)
    with open(p, "w") as f:
        f.write(text)
    return p


def gate(step, status, note):
    ensure_dirs()
    row = pd.DataFrame([{
        "gate": step, "status": status, "note": note,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"}])
    row.to_csv(GATES, mode="a", header=not os.path.exists(GATES), index=False)
    print(f"[gate {step}] {status}: {note}", flush=True)


def blocked(step, note):
    gate(step, "BLOCKED", note)
    raise SystemExit(f"BLOCKED at {step}: {note}")


# ---------------------------------------------------------------------------
# stream preparation (identical to the revalidation harness)
# ---------------------------------------------------------------------------
def prepare(stream_df, sd):
    feat = sd["feature_columns"]
    X_all = stream_df[feat].values.astype(np.float64)
    y_all = stream_df["y"].values.astype(int)
    regimes = stream_df["regime_id"].values
    n_init = sd["initial_train_windows"]
    n_windows = sd["total_windows"]
    wid = stream_df["window_id"].values
    init_mask = wid < n_init
    X_init_raw = X_all[init_mask]
    mu = X_init_raw.mean(axis=0)
    sdv = np.where(X_init_raw.std(axis=0) < 1e-9, 1.0, X_init_raw.std(axis=0))
    X = (X_all - mu) / sdv
    X_init = X[init_mask]
    y_init = y_all[init_mask]
    win_slices = {w: np.where(wid == w)[0] for w in range(n_init, n_windows)}
    return X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows


def regime_of(stream_df):
    return stream_df.groupby("window_id")["regime_id"].first()


def blocks_of(stream_df, n_blocks=None):
    """Contiguous window blocks for the bootstrap.

    UGR'16 uses day blocks derived from the regime label (WD_d / WE_d map to a
    day index); other streams use contiguous blocks of roughly equal size, with
    n_blocks = the number of distinct regimes (regime visits) unless overridden.
    """
    rs = regime_of(stream_df)
    wids = rs.index.to_numpy()
    labs = rs.to_numpy()
    if n_blocks is None:
        n_blocks = max(1, len(pd.unique(labs)))
    # contiguous split of the window index into n_blocks
    edges = np.linspace(0, len(wids), n_blocks + 1).astype(int)
    out = {}
    for b in range(n_blocks):
        out[b] = wids[edges[b]:edges[b + 1]]
    return out


# ---------------------------------------------------------------------------
# RAPT v3: provenance modes on top of RAPTV2
# ---------------------------------------------------------------------------
class RAPTV3(RAPTV2):
    """RAPTV2 with an explicit provenance policy for stored checkpoints.

    store_mode
    ----------
    original     : original behaviour (checkpoint trained on the buffer that
                   still ends in the previous regime, keyed by the new regime).
    deferred     : store after the new regime's first window labels are known;
                   checkpoint trained on the buffer including that window.
    regime_pure  : store after labels, trained on the new regime's own window
                   rows plus the class anchor (true provenance).
    """

    def __init__(self, *args, store_mode="original", **kwargs):
        super().__init__(*args, **kwargs)
        self.store_mode = store_mode

    def on_transition(self, new_regime_id, window_id, X_buffer, y_buffer,
                      train_windows, train_regimes):
        if self.store_mode != "original" and new_regime_id not in self.repository:
            self._pending = (new_regime_id, window_id, self.store_mode)
            self.current_regime_id = new_regime_id
            self._current_reused = False
            return False, 0.0, 0.0
        return super().on_transition(new_regime_id, window_id, X_buffer, y_buffer,
                                     train_windows, train_regimes)

    def update(self, w, X_win, y_win, X_buffer, y_buffer, train_windows, train_regimes):
        if getattr(self, "_pending", None) is not None:
            reg, win, mode = self._pending
            self._pending = None
            t0 = time.process_time()
            if mode == "regime_pure":
                Xb = np.asarray(X_win)
                yb = np.asarray(y_win)
                trX, trY = make_train_buffer(Xb, yb, self.anchor_X, self.anchor_y,
                                             max(len(Xb), 1))
                ens = create_base_ensemble(seed=self.seed + self.created_policy_count * 13,
                                           n_estimators=self.n_trees)
                ens.fit(trX, trY)
                n_used = len(trX)
            else:
                ens, n_used = self._train_policy(win, X_buffer, y_buffer)
            self.trees_trained_count += ens.get_num_trees()
            self._store(reg, ens, win, train_windows, train_regimes, n_used)
            self.active_ensemble = ens
            self.created_policy_count += 1
            self._recent_correct = []
            cpu = time.process_time() - t0
            self.adaptation_cpu_time += cpu
            return cpu, {"is_refresh": 1}
        return super().update(w, X_win, y_win, X_buffer, y_buffer,
                              train_windows, train_regimes)


# ---------------------------------------------------------------------------
# unified runner
# ---------------------------------------------------------------------------
def _rec(seed, method, w, regime, y_true, y_pred, adapt_cpu, pred_cpu,
         is_retrain, is_reuse, is_refresh, n_classes):
    m = window_metrics(y_true, y_pred)
    cm = window_confusion(y_true, y_pred, n_classes)
    return {
        "seed": seed, "method": method, "window_id": int(w), "regime_id": str(regime),
        "n_samples": cm["n_samples"],
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "precision": m["precision"], "recall": m["recall"],
        "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
        "cm_json": cm["cm_json"],
        "adaptation_cpu_sec": float(adapt_cpu),
        "prediction_cpu_sec": float(pred_cpu),
        "is_retrain": int(is_retrain), "is_reuse": int(is_reuse),
        "is_refresh": int(is_refresh),
    }


def _summary(seed, method, recs, init_cpu, adapt_cpu, retrains, trees_trained,
             trees_reused, reuse_events, refresh_events, total_wall, n_classes, extra=None):
    df = pd.DataFrame(recs)
    row = {
        "seed": seed, "method": method,
        "per_window_macro_f1": float(df["macro_f1"].mean()),
        "per_window_accuracy": float(df["accuracy"].mean()),
        "per_window_precision": float(df["precision"].mean()),
        "per_window_recall": float(df["recall"].mean()),
        "init_cpu_sec": float(init_cpu),
        "adaptation_cpu_sec": float(adapt_cpu),
        "total_cpu_sec": float(init_cpu + adapt_cpu),
        "prediction_cpu_sec": float(df["prediction_cpu_sec"].sum()),
        "total_runtime_sec": float(total_wall),
        "retrain_events": int(retrains),
        "reuse_events": int(reuse_events),
        "refresh_events": int(refresh_events),
        "trees_trained": int(trees_trained),
        "trees_reused": int(trees_reused),
    }
    if extra:
        row.update(extra)
    return row


def rapt_factory(method, seed, anchor_X, anchor_y, cfg):
    """Return (driver, refit_n) for a RAPT-family method name."""
    if method in ("RAPT", "RAPT-Enhanced"):
        # the ORIGINAL published implementation, imported unchanged
        if method == "RAPT":
            sys_ = RAPTSystem(seed=seed, mode="full", enable_calibration=True,
                              anchor_X=anchor_X, anchor_y=anchor_y,
                              buffer_capacity=BUFFER_CAPACITY)
            return sys_, 500, "orig"
        sys_ = RAPTEnhancedSystem(seed=seed, mode="full", enable_calibration=True,
                                  anchor_X=anchor_X, anchor_y=anchor_y,
                                  buffer_capacity=BUFFER_CAPACITY,
                                  novelty_refit_n=1500)
        return sys_, 1500, "orig_enh"

    base = dict(seed=seed, anchor_X=anchor_X, anchor_y=anchor_y,
                refit_n=BUFFER_CAPACITY)
    table = {
        "RAPT_v2": dict(store_mode="deferred"),
        "RAPT_v2_pure": dict(store_mode="regime_pure"),
        "RAPT-Enhanced_v2": dict(store_mode="deferred", parity_threshold=0.5),
        "RAPT-Enhanced_v2_pure": dict(store_mode="regime_pure", parity_threshold=0.5),
        "RAPT-Deferred": dict(deferred=True),
        "RAPT-Cheap": dict(store_mode="deferred", refresh_every=5,
                           refresh_trees=CHEAP_TREES, refresh_buffer=CHEAP_BUFFER),
        "RAPT-Cheap-Original": dict(refresh_every=5, refresh_trees=CHEAP_TREES,
                                    refresh_buffer=CHEAP_BUFFER),
        "RAPT-Cheap-Floor": dict(store_mode="deferred", refresh_every=1,
                                 refresh_mode="floor", refresh_trees=CHEAP_TREES,
                                 refresh_buffer=CHEAP_BUFFER),
        "RAPT-Cheap-WinEq": dict(store_mode="deferred", refresh_every=5,
                                 refresh_trees=CHEAP_TREES, refresh_buffer=CHEAP_BUFFER,
                                 refresh_counter_global=True),
    }
    if method not in table:
        raise ValueError(method)
    kw = dict(base)
    kw.update(table[method])
    return RAPTV3(**kw), BUFFER_CAPACITY, "v2"


def run_stream(stream_df, sd, seed, methods, label_delay=False):
    """Run `methods` on one stream for one seed. Returns (per_window, summary, pooled)."""
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = prepare(stream_df, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream_df["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    cfg = load_config()

    recs_all, summ_all = [], []
    yt_all, yp_all = {}, {}

    def append(buf_X, buf_y, w, idx, delay_from=None):
        src = idx if delay_from is None else win_slices[delay_from]
        buf_X.extend(X[src]); buf_y.extend(y_all[src])
        if len(buf_X) > BUFFER_CAPACITY:
            del buf_X[:-BUFFER_CAPACITY]; del buf_y[:-BUFFER_CAPACITY]

    def run_simple(method, driver, transition_trigger):
        t_w0 = time.perf_counter()
        init_cpu = driver.fit_initial(X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = init_regime
        recs, yt, yp = [], [], []
        pending = None
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            is_trans = reg != prev
            prev = reg
            adapt_cpu, did = driver.maybe_adapt(w, is_trans, buf_X, buf_y) \
                if transition_trigger else (0.0, False)
            t0 = time.process_time(); y_p = driver.predict(X[idx]); pc = time.process_time() - t0
            recs.append(_rec(seed, method, w, reg, y_all[idx], y_p, adapt_cpu, pc,
                             did, 0, 0, n_classes))
            yt.append(y_all[idx]); yp.append(y_p)
            if label_delay:
                if pending is not None:
                    append(buf_X, buf_y, w, pending, delay_from=pending)
                pending = w
            else:
                append(buf_X, buf_y, w, idx)
        summ_all.append(_summary(seed, method, recs, init_cpu,
                                 driver.adaptation_cpu_time, driver.retrain_events,
                                 driver.trees_trained_count, 0, 0, 0,
                                 time.perf_counter() - t_w0, n_classes))
        recs_all.extend(recs)
        yt_all[method] = np.concatenate(yt); yp_all[method] = np.concatenate(yp)

    for method in methods:
        if method == "Frozen":
            run_simple("Frozen", FRDriver(seed, trigger="none"), False)
        elif method == "Event-Driven":
            ed = EventDrivenEnsemble(seed=seed, buffer_capacity=BUFFER_CAPACITY,
                                     error_window_size=20, error_threshold_k=2.0,
                                     anchor_X=anchor_X, anchor_y=anchor_y)
            t_w0 = time.perf_counter()
            init_cpu = ed.fit_initial(X_init, y_init)
            if isinstance(init_cpu, tuple):
                init_cpu = init_cpu[0]
            recs, yt, yp = [], [], []
            for w in range(n_init, n_windows):
                idx = win_slices[w]
                reg = regimes[idx[0]]
                t0 = time.process_time(); y_p = ed.predict(X[idx]); pc = time.process_time() - t0
                err = 1.0 - float(np.mean(y_p == y_all[idx]))
                trig, adapt_cpu, _ = ed.update_and_adapt(X[idx], y_all[idx], err)
                recs.append(_rec(seed, "Event-Driven", w, reg, y_all[idx], y_p,
                                 adapt_cpu, pc, int(trig), 0, 0, n_classes))
                yt.append(y_all[idx]); yp.append(y_p)
            summ_all.append(_summary(seed, "Event-Driven", recs, init_cpu,
                                     ed.cumulative_cpu_time - init_cpu, ed.retrain_events,
                                     ed.total_trees_trained, 0, 0, 0,
                                     time.perf_counter() - t_w0, n_classes))
            recs_all.extend(recs)
            yt_all["Event-Driven"] = np.concatenate(yt); yp_all["Event-Driven"] = np.concatenate(yp)
        elif method in ("Full Retraining", "FR-Cheap", "Periodic-Cheap-5",
                        "Periodic-Cheap-10", "Periodic-Cheap-Matched"):
            if method == "Full Retraining":
                nt, trig, every = FULL_TREES, "regime", None
            elif method == "FR-Cheap":
                nt, trig, every = CHEAP_TREES, "regime", None
            elif method == "Periodic-Cheap-5":
                nt, trig, every = CHEAP_TREES, "periodic", 5
            elif method == "Periodic-Cheap-10":
                nt, trig, every = CHEAP_TREES, "periodic", 10
            else:
                nt, trig, every = CHEAP_TREES, "periodic", int(PERIODIC_MATCHED)
            run_simple(method, FRDriver(seed, n_trees=nt,
                                        buffer=CHEAP_BUFFER if nt == CHEAP_TREES else BUFFER_CAPACITY,
                                        trigger=trig, every=every,
                                        anchor_X=anchor_X, anchor_y=anchor_y), True)
        elif method == "Event-Driven-Cheap":
            ed = EventDrivenV2(seed, n_trees=CHEAP_TREES, buffer=CHEAP_BUFFER,
                               anchor_X=anchor_X, anchor_y=anchor_y)
            t_w0 = time.perf_counter()
            init_cpu = ed.fit_initial(X_init, y_init)
            buf_X, buf_y = list(X_init), list(y_init)
            recs, yt, yp = [], [], []
            pending = None
            for w in range(n_init, n_windows):
                idx = win_slices[w]
                reg = regimes[idx[0]]
                t0 = time.process_time(); y_p = ed.predict(X[idx]); pc = time.process_time() - t0
                err = 1.0 - float(np.mean(y_p == y_all[idx]))
                if label_delay:
                    if pending is not None:
                        append(buf_X, buf_y, w, pending, delay_from=pending)
                    pending = w
                else:
                    append(buf_X, buf_y, w, idx)
                adapt_cpu, trig = ed.update_and_adapt(X[idx], y_all[idx], err, buf_X, buf_y)
                recs.append(_rec(seed, method, w, reg, y_all[idx], y_p, adapt_cpu, pc,
                                 int(trig), 0, 0, n_classes))
                yt.append(y_all[idx]); yp.append(y_p)
            summ_all.append(_summary(seed, method, recs, init_cpu,
                                     ed.adaptation_cpu_time, ed.retrain_events,
                                     ed.trees_trained_count, 0, 0, 0,
                                     time.perf_counter() - t_w0, n_classes))
            recs_all.extend(recs)
            yt_all[method] = np.concatenate(yt); yp_all[method] = np.concatenate(yp)
        elif method in DETECTORS:
            dm = DetectorAdaptiveModelV2(method, seed=seed, buffer_capacity=BUFFER_CAPACITY,
                                         anchor_X=anchor_X, anchor_y=anchor_y,
                                         **DETECTOR_PARAMS[method])
            t_w0 = time.perf_counter()
            init_cpu, _ = dm.fit_initial(X_init, y_init)
            buf_X, buf_y = list(X_init), list(y_init)
            recs, yt, yp = [], [], []
            pending = None
            for w in range(n_init, n_windows):
                idx = win_slices[w]
                reg = regimes[idx[0]]
                t0 = time.process_time(); y_p = dm.predict(X[idx]); pc = time.process_time() - t0
                err = 1.0 - float(np.mean(y_p == y_all[idx]))
                if label_delay:
                    if pending is not None:
                        append(buf_X, buf_y, w, pending, delay_from=pending)
                    pending = w
                else:
                    append(buf_X, buf_y, w, idx)
                trig, did, adapt_cpu, _ = dm.update_and_adapt(w, X[idx], y_all[idx], err)
                recs.append(_rec(seed, method, w, reg, y_all[idx], y_p, adapt_cpu, pc,
                                 int(did), 0, 0, n_classes))
                yt.append(y_all[idx]); yp.append(y_p)
            summ_all.append(_summary(seed, method, recs, init_cpu,
                                     dm.cumulative_cpu_time - init_cpu, dm.retrain_events,
                                     dm.total_trees_trained, 0, 0, 0,
                                     time.perf_counter() - t_w0, n_classes,
                                     extra={"detected_events": dm.detected_events}))
            recs_all.extend(recs)
            yt_all[method] = np.concatenate(yt); yp_all[method] = np.concatenate(yp)
        elif method.startswith("RAPT"):
            sys_, refit_n, kind = rapt_factory(method, seed, anchor_X, anchor_y, cfg)
            t_w0 = time.perf_counter()
            init_ret = sys_.fit_initial(init_regime, X_init, y_init)
            init_cpu = init_ret[0] if isinstance(init_ret, tuple) else init_ret
            buf_X, buf_y = list(X_init), list(y_init)
            prev = init_regime
            recs, yt, yp = [], [], []
            pending = None
            wreg = {int(w): str(regimes[win_slices[w][0]]) for w in range(n_init, n_windows)}
            wreg[int(n_init - 1)] = str(init_regime)
            for w in range(n_init, n_windows):
                idx = win_slices[w]
                reg = regimes[idx[0]]
                wa, reused, refreshed = 0.0, 0, 0
                if reg != prev:
                    nbuf = len(buf_X)
                    nbuf_win = max(1, int(np.ceil(nbuf / max(1, sd["window_size"]))))
                    train_wids = [x for x in range(max(n_init, w - nbuf_win), w)]
                    train_regs = [wreg.get(x, "?") for x in train_wids]
                    if kind in ("orig", "orig_enh"):
                        ri, wa, _ = sys_.handle_regime_transition(
                            reg, w, X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
                    else:
                        ri, wa, _ = sys_.on_transition(
                            reg, w, buf_X[-refit_n:], buf_y[-refit_n:], train_wids, train_regs)
                    reused = int(ri)
                    prev = reg
                t0 = time.process_time(); y_p = sys_.predict(X[idx]); pc = time.process_time() - t0
                if label_delay:
                    if pending is not None:
                        append(buf_X, buf_y, w, pending, delay_from=pending)
                    pending = w
                else:
                    append(buf_X, buf_y, w, idx)
                if kind in ("orig", "orig_enh"):
                    wa += sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-refit_n:],
                                      y_buffer=buf_y[-refit_n:])
                else:
                    wa2, flags = sys_.update(w, X[idx], y_all[idx], buf_X, buf_y, [w], [reg])
                    wa += wa2
                    refreshed = int(flags.get("is_refresh", 0))
                recs.append(_rec(seed, method, w, reg, y_all[idx], y_p, wa, pc,
                                 0, reused, refreshed, n_classes))
                yt.append(y_all[idx]); yp.append(y_p)
            reuse_events = int(sum(r["is_reuse"] for r in recs))
            refresh_events = int(sum(r["is_refresh"] for r in recs))
            extra = {}
            if kind in ("orig", "orig_enh"):
                extra["parity_refits"] = getattr(sys_, "parity_refits", 0)
                extra["n_refits"] = int(getattr(sys_, "created_policy_count", 0) - 1)
            else:
                extra["parity_refits"] = sys_.parity_refits
                extra["gate_accept"] = sys_.gate_accept
                extra["gate_reject"] = sys_.gate_reject
                extra["n_refits"] = len(sys_.refit_rows)
            summ_all.append(_summary(seed, method, recs, init_cpu, sys_.adaptation_cpu_time,
                                     getattr(sys_, "created_policy_count", 0) - 1,
                                     sys_.trees_trained_count, sys_.trees_reused_count,
                                     reuse_events, refresh_events,
                                     time.perf_counter() - t_w0, n_classes, extra=extra))
            recs_all.extend(recs)
            yt_all[method] = np.concatenate(yt); yp_all[method] = np.concatenate(yp)
        else:
            raise ValueError(f"unknown method {method}")

    pw = pd.DataFrame(recs_all)
    sm = pd.DataFrame(summ_all)
    pooled_rows = []
    for method in yt_all:
        yt, yp = yt_all[method], yp_all[method]
        pm = pooled_metrics(yt, yp, n_classes)
        pm.update({"seed": seed, "method": method})
        if n_classes == 2:
            pm.update(pooled_binary_attack_metrics(yt, yp, attack_label=1, proba=None))
        pooled_rows.append(pm)
    pooled = pd.DataFrame(pooled_rows)
    return pw, sm, pooled


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def block_bootstrap_ci(per_window, method_a, method_b, blocks, metric="macro_f1",
                       n_boot=2000, seed=0):
    """Paired block bootstrap CI of mean(metric_a - metric_b) over windows.

    `per_window` must have columns window_id, method, <metric>. `blocks` maps a
    block id to an array of window ids.
    """
    a = per_window[per_window.method == method_a].set_index("window_id")[metric]
    b = per_window[per_window.method == method_b].set_index("window_id")[metric]
    common = a.index.intersection(b.index)
    a = a.loc[common]; b = b.loc[common]
    d = (a - b).to_numpy()
    wid = common.to_numpy()
    block_ids = []
    for bid, wids in blocks.items():
        sel = np.isin(wid, np.asarray(wids))
        if sel.any():
            block_ids.append(d[sel])
    if len(block_ids) < 2:
        return float("nan"), float("nan"), float("nan"), len(block_ids)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(block_ids), len(block_ids))
        means.append(np.mean(np.concatenate([block_ids[i] for i in pick])))
    means = np.asarray(means)
    return float(np.mean(d)), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), len(block_ids)


def paired_wilcoxon(x, y):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    if np.allclose(d, 0):
        return 1.0, 0.0
    try:
        s, p = wilcoxon(x, y)
        return float(p), float(np.mean(d))
    except Exception:
        return float("nan"), float(np.mean(d))


def cohen_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    s = np.std(d, ddof=1) if len(d) > 1 else 0.0
    return float(np.mean(d) / s) if s > 0 else float("nan")


def tost(x, y, margin):
    """Two one-sided tests for equivalence at +/- margin. Returns (p, equivalent)."""
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    try:
        _, p_low = wilcoxon(d, np.full_like(d, -margin), alternative="greater")
        _, p_high = wilcoxon(d, np.full_like(d, margin), alternative="less")
        p = max(p_low, p_high)
    except Exception:
        p = float("nan")
    return float(p), bool(p < 0.05)


def holm(pvals):
    """Holm-Bonferroni correction. Returns adjusted p-values in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def pareto_flags(df, f1_col="pooled_macro_f1", cpu_col="adaptation_cpu_sec"):
    """Return a boolean array: True where a row is on the F1-up / CPU-down frontier."""
    f1 = df[f1_col].to_numpy(float)
    cpu = df[cpu_col].to_numpy(float)
    flags = np.ones(len(df), bool)
    for i in range(len(df)):
        dominated = ((f1 >= f1[i]) & (cpu <= cpu[i]) &
                     ((f1 > f1[i]) | (cpu < cpu[i]))).any()
        flags[i] = not dominated
    return flags


def env_info():
    import sklearn, scipy, numpy
    return {
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "scipy": scipy.__version__,
    }
