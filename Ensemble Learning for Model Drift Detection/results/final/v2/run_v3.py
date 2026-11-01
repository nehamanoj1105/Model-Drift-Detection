"""Unified prequential runner (v2) for the revalidation.

Same protocol as the existing 9A/9B runners (chronological, prefix = first 20%
of windows, predict -> evaluate -> adapt, no future leakage) but:

  * writes per-window confusion counts (tp/fp/fn/tn + full matrix) to the raw CSV;
  * computes pooled metrics over all evaluation samples of a run;
  * supports the new cost-matched controls (FR-Cheap, Periodic-Cheap,
    Event-Driven-Cheap, RAPT-Cheap), RAPT-Deferred, gate-fix and label delay;
  * logs RAPT checkpoint provenance.

The base RAPT / RAPT-Enhanced classes are imported unchanged from the existing
9A module for the reproduction cross-check; the new variants use RAPTV2.
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RAW, ensure_dirs  # noqa: E402
from metrics_pooled import pooled_metrics, pooled_binary_attack_metrics, window_confusion  # noqa: E402
from detectors_v2 import DetectorAdaptiveModelV2  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP9A = os.path.join(ROOT, "experiments", "exp9a")
sys.path.insert(0, EXP9A)

from models_9a import create_base_ensemble, make_class_anchor, make_train_buffer  # noqa: E402
from evaluation_9a import window_metrics  # noqa: E402
from event_driven_9a import EventDrivenEnsemble  # noqa: E402
from rapt_9a import RAPTSystem, RAPTEnhancedSystem, RegimeCheckpoint  # noqa: E402

# --- frozen v2 configuration (see results/final/v2/config_frozen_v2.json) ---
with open(os.path.join(HERE, "config_frozen_v2.json")) as _f:
    import json as _json
    CFG = _json.load(_f)

BUFFER_CAPACITY = CFG["full_config"]["buffer_rows"]
FULL_TREES = CFG["full_config"]["trees"]
CHEAP_TREES = CFG["cheap_config"]["trees"]
CHEAP_BUFFER = CFG["cheap_config"]["buffer_rows"]
ERROR_WINDOW_SIZE = 20
ERROR_THRESHOLD_K = 2.0
DETECTORS = ["ADWIN", "EDD", "Page-Hinkley", "EDMA"]
# per-detector parameter overrides for sensitivity grids (defaults in config)
DETECTOR_PARAMS = {}


def set_detector_params(name, **params):
    DETECTOR_PARAMS[name] = params


# ---------------------------------------------------------------------------
# RAPT v2: original semantics + provenance logging + new options
# ---------------------------------------------------------------------------
class RAPTV2:
    """Mirror of the original RAPTSystem with optional extensions.

    Modes
    -----
    base            : original behaviour (reuse by regime id, novelty refit).
    deferred        : on a *new* regime, serve the previous policy for that
                      regime's first window; store the checkpoint only after
                      that window's labels are available.
    """

    def __init__(self, seed=42, n_trees=FULL_TREES, buffer=BUFFER_CAPACITY,
                 refit_n=None, deferred=False, refresh_every=None,
                 refresh_trees=None, refresh_buffer=None,
                 use_gate=False, gate_fix=False, gate_threshold=0.65,
                 parity_threshold=None, parity_window=3, enable_calibration=True,
                 rel_drop=None, refresh_mode="periodic",
                 anchor_X=None, anchor_y=None, refresh_counter_global=False):
        self.seed = seed
        self.n_trees = n_trees
        self.buffer = buffer
        self.refit_n = refit_n if refit_n is not None else buffer
        self.deferred = deferred
        self.refresh_every = refresh_every
        self.refresh_trees = refresh_trees if refresh_trees is not None else n_trees
        self.refresh_buffer = refresh_buffer if refresh_buffer is not None else buffer
        self.use_gate = use_gate
        self.gate_fix = gate_fix
        self.gate_threshold = gate_threshold
        self.parity_threshold = parity_threshold
        self.parity_window = parity_window
        self.enable_calibration = enable_calibration
        self.rel_drop = rel_drop
        self.refresh_mode = refresh_mode
        self.refresh_counter_global = refresh_counter_global
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y

        self.repository = {}
        self.current_regime_id = None
        self.active_ensemble = None
        self._recent_correct = []
        self._current_reused = False
        self._windows_since_refresh = 0
        self._acc_ema = None

        self.created_policy_count = 0
        self.reused_policy_count = 0
        self.adaptation_cpu_time = 0.0
        self.trees_trained_count = 0
        self.trees_reused_count = 0
        self.parity_refits = 0
        self.refresh_events = 0
        self.gate_accept = 0
        self.gate_reject = 0
        self.provenance = []          # checkpoint provenance rows
        self.refit_rows = []          # training rows used at each novelty refit
        self._pending = None          # (regime_id, window_id) awaiting labels

    # -- helpers ------------------------------------------------------------
    def _train_policy(self, window_id, X_buffer, y_buffer, n_trees=None, buf=None):
        ens = create_base_ensemble(seed=self.seed + window_id * 19,
                                   n_estimators=n_trees or self.n_trees)
        buf_rows = 0
        anchor_rows = 0
        if X_buffer is not None and y_buffer is not None and len(X_buffer) > 50:
            recent_X = np.asarray(X_buffer)[-self.refit_n:]
            recent_y = np.asarray(y_buffer)[-self.refit_n:]
            buf_rows = min(len(recent_X), buf if buf is not None else self.buffer)
            anchor_rows = len(self.anchor_X) if self.anchor_X is not None else 0
            train_X, train_y = make_train_buffer(
                recent_X, recent_y, self.anchor_X, self.anchor_y,
                buf if buf is not None else self.buffer)
            ens.fit(train_X, train_y)
            n_used = len(train_X)
        else:
            n_used = 0
        self.refit_rows.append({"buffer_rows": int(buf_rows),
                                "anchor_rows": int(anchor_rows),
                                "total_rows": int(n_used)})
        return ens, n_used

    def _fingerprint(self, X):
        X = np.asarray(X)
        return np.concatenate([X.mean(axis=0), X.std(axis=0)])

    def _cosine(self, a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _store(self, regime_id, ens, window_id, train_windows, train_regimes, n_used,
               fingerprint=None):
        ckpt = RegimeCheckpoint(regime_id, ens, list(ens.weights), window_id)
        if fingerprint is not None:
            ckpt.fingerprint = fingerprint
        self.repository[regime_id] = ckpt
        self.provenance.append({
            "regime_key": regime_id, "created_window": window_id,
            "train_window_ids": ";".join(str(x) for x in train_windows[:5]) + ("..." if len(train_windows) > 5 else ""),
            "train_regimes": ";".join(sorted(set(str(r) for r in train_regimes))),
            "train_regime_matches_key": int(len(set(str(r) for r in train_regimes)) == 1
                                            and str(train_regimes[0]) == str(regime_id)),
            "n_train_rows": n_used,
        })

    # -- lifecycle ----------------------------------------------------------
    def fit_initial(self, regime_id, X_init, y_init, window_id=0):
        t0 = time.process_time()
        ens = create_base_ensemble(seed=self.seed, n_estimators=self.n_trees)
        ens.fit(X_init, y_init)
        cpu = time.process_time() - t0
        self.active_ensemble = ens
        ckpt = RegimeCheckpoint(regime_id, ens, [0.5, 0.5], window_id)
        ckpt.fingerprint = self._fingerprint(X_init)
        self.repository[regime_id] = ckpt
        self.current_regime_id = regime_id
        self.created_policy_count += 1
        self.trees_trained_count += ens.get_num_trees()
        self.adaptation_cpu_time += cpu
        return cpu

    def on_transition(self, new_regime_id, window_id, X_buffer, y_buffer,
                      train_windows, train_regimes):
        """Called *before* predicting the first window of a new regime."""
        t0 = time.process_time()
        reused = False
        n_used = 0

        if self.deferred and new_regime_id not in self.repository:
            # serve the current (previous-regime) policy for this first window,
            # store the checkpoint after the window's labels arrive
            self._pending = (new_regime_id, window_id)
            self.current_regime_id = new_regime_id
            self._current_reused = False
            cpu = time.process_time() - t0
            self.adaptation_cpu_time += cpu
            return False, cpu, 0.0

        if new_regime_id in self.repository:
            ckpt = self.repository[new_regime_id]
            accept = True
            if self.use_gate:
                if self.gate_fix:
                    # gate the *incoming* regime's checkpoint on the fingerprint
                    # of the incoming window's features (available, unlabelled,
                    # before prediction).  Original semantics instead fingerprint
                    # the pre-transition buffer and compare with the nearest
                    # stored regime of ANY key.
                    fp_new = getattr(self, "_incoming_fp", None)
                    fp_key = self.repository[new_regime_id].fingerprint \
                        if hasattr(ckpt, "fingerprint") else None
                    if fp_new is not None and fp_key is not None:
                        accept = self._cosine(fp_new, fp_key) >= self.gate_threshold
                    else:
                        accept = False
                else:
                    fp_buf = self._fingerprint(X_buffer[-50:]) if len(X_buffer) else None
                    best = None
                    for ck in self.repository.values():
                        if getattr(ck, "fingerprint", None) is None or fp_buf is None:
                            continue
                        s = self._cosine(fp_buf, ck.fingerprint)
                        best = s if best is None else max(best, s)
                    accept = best is not None and best >= self.gate_threshold
            if accept:
                reused = True
                self.reused_policy_count += 1
                self.trees_reused_count += ckpt.ensemble.get_num_trees()
                self.active_ensemble = copy.copy(ckpt.ensemble)
                self.active_ensemble.weights = list(ckpt.weights)
                if self.enable_calibration and X_buffer is not None and len(X_buffer) >= 20:
                    Xb = np.asarray(X_buffer[-50:]); yb = np.asarray(y_buffer[-50:])
                    p_rf = self.active_ensemble.rf.predict_proba(Xb)
                    p_et = self.active_ensemble.et.predict_proba(Xb)
                    acc_rf = np.mean(np.argmax(p_rf, axis=1) == yb)
                    acc_et = np.mean(np.argmax(p_et, axis=1) == yb)
                    if acc_rf + acc_et > 0:
                        self.active_ensemble.weights = [acc_rf / (acc_rf + acc_et),
                                                        acc_et / (acc_rf + acc_et)]
                self.gate_accept += 1
                self.provenance.append({
                    "regime_key": new_regime_id, "created_window": ckpt.created_window,
                    "reused_at_window": window_id, "age_windows": window_id - ckpt.created_window,
                    "train_window_ids": "", "train_regimes": "",
                    "train_regime_matches_key": 1, "n_train_rows": 0, "event": "reuse",
                })
            else:
                self.gate_reject += 1
        if not reused:
            self.created_policy_count += 1
            ens, n_used = self._train_policy(window_id, X_buffer, y_buffer)
            self.trees_trained_count += ens.get_num_trees()
            # NOTE: the original controller stores the checkpoint under the new
            # regime key even though the buffer it was trained on ends in the
            # PREVIOUS regime (the incoming window's labels are not yet known).
            self._store(new_regime_id, ens, window_id, train_windows, train_regimes,
                        n_used, fingerprint=self._fingerprint(X_buffer))
            self.active_ensemble = ens

        self.current_regime_id = new_regime_id
        self._current_reused = reused
        self._recent_correct = []
        if not self.refresh_counter_global:
            self._windows_since_refresh = 0
        cpu = time.process_time() - t0
        self.adaptation_cpu_time += cpu
        return reused, cpu, 0.0

    def predict(self, X_win):
        return self.active_ensemble.predict(X_win)

    def predict_proba(self, X_win):
        return self.active_ensemble.predict_proba(X_win)

    def update(self, w, X_win, y_win, X_buffer, y_buffer, train_windows, train_regimes):
        """Post-prediction update."""
        t0 = time.process_time()
        cpu = 0.0

        # deferred storage: labels of the first window of a new regime are now known
        if self._pending is not None:
            reg, win = self._pending
            self._pending = None
            ens, n_used = self._train_policy(win, X_buffer, y_buffer)
            self.trees_trained_count += ens.get_num_trees()
            self._store(reg, ens, win, train_windows, train_regimes, n_used)
            self.active_ensemble = ens
            self.created_policy_count += 1
            self._recent_correct = []
            cpu += time.process_time() - t0
            self.adaptation_cpu_time += cpu
            return cpu, {"is_refresh": 1}

        # parity refit (RAPT-Enhanced semantics)
        if self.parity_threshold is not None:
            acc = float(np.mean(self.active_ensemble.predict(X_win) == np.asarray(y_win)))
            self._recent_correct.append(acc)
            if len(self._recent_correct) > self.parity_window:
                self._recent_correct.pop(0)
            if (self._current_reused and len(self._recent_correct) >= self.parity_window
                    and np.mean(self._recent_correct) < self.parity_threshold):
                ens, _ = self._train_policy(0, X_buffer, y_buffer)
                self.active_ensemble = ens
                self.parity_refits += 1
                self.trees_trained_count += ens.get_num_trees()
                self.repository[self.current_regime_id] = RegimeCheckpoint(
                    self.current_regime_id, ens, list(ens.weights), 0)
                self._recent_correct = []
                cpu += time.process_time() - t0
                self.adaptation_cpu_time += cpu
                return cpu, {"is_refresh": 1, "is_parity_refit": 1}

        # periodic / floor / evidence refresh
        do_refresh = False
        if self.refresh_every is not None:
            self._windows_since_refresh += 1
            if self.refresh_mode in ("evidence", "floor"):
                acc = float(np.mean(self.active_ensemble.predict(X_win) == np.asarray(y_win)))
                if self.refresh_mode == "floor":
                    do_refresh = acc < 0.97
                else:
                    if self._acc_ema is None:
                        self._acc_ema = acc
                    # reference: the pre-drift EMA level, retained across regime
                    # transitions (never reset) -> relative-drop trigger
                    if acc < self._acc_ema * (1.0 - (self.rel_drop or 0.10)):
                        do_refresh = True
                    self._acc_ema = 0.9 * self._acc_ema + 0.1 * acc
            elif self._windows_since_refresh >= self.refresh_every:
                do_refresh = True
        if do_refresh:
            ens, _ = self._train_policy(w, X_buffer, y_buffer,
                                        n_trees=self.refresh_trees,
                                        buf=self.refresh_buffer)
            self.active_ensemble = ens
            ckpt = RegimeCheckpoint(self.current_regime_id, ens, list(ens.weights), w)
            ckpt.fingerprint = self._fingerprint(X_buffer)
            self.repository[self.current_regime_id] = ckpt
            self.trees_trained_count += ens.get_num_trees()
            self.refresh_events += 1
            self._windows_since_refresh = 0
            cpu += time.process_time() - t0
            self.adaptation_cpu_time += cpu
            return cpu, {"is_refresh": 1}

        return cpu, {}

    def n_refresh(self):
        return self.refresh_events + self.parity_refits


# ---------------------------------------------------------------------------
# Simple drivers
# ---------------------------------------------------------------------------
class FRDriver:
    """Full retraining / cheap retraining, optionally periodic instead of
    regime-triggered."""

    def __init__(self, seed, n_trees=FULL_TREES, buffer=BUFFER_CAPACITY,
                 trigger="regime", every=None, anchor_X=None, anchor_y=None):
        self.seed = seed
        self.n_trees = n_trees
        self.buffer = buffer
        self.trigger = trigger
        self.every = every
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.ensemble = None
        self.adaptation_cpu_time = 0.0
        self.retrain_events = 0
        self.trees_trained_count = 0
        self._since = 0

    def fit_initial(self, X_init, y_init):
        t0 = time.process_time()
        self.ensemble = create_base_ensemble(seed=self.seed, n_estimators=self.n_trees)
        self.ensemble.fit(X_init, y_init)
        cpu = time.process_time() - t0
        self.trees_trained_count += self.ensemble.get_num_trees()
        self.adaptation_cpu_time += cpu
        return cpu

    def maybe_adapt(self, w, is_transition, buf_X, buf_y):
        do = (self.trigger == "regime" and is_transition)
        if self.trigger == "periodic":
            self._since += 1
            do = self._since >= self.every
        if not do:
            return 0.0, False
        trX, trY = make_train_buffer(np.asarray(buf_X), np.asarray(buf_y),
                                     self.anchor_X, self.anchor_y, self.buffer)
        self.retrain_events += 1
        t0 = time.process_time()
        self.ensemble = create_base_ensemble(seed=self.seed + self.retrain_events * 11,
                                             n_estimators=self.n_trees)
        self.ensemble.fit(trX, trY)
        cpu = time.process_time() - t0
        self.trees_trained_count += self.ensemble.get_num_trees()
        self.adaptation_cpu_time += cpu
        if self.trigger == "periodic":
            self._since = 0
        return cpu, True

    def predict(self, X_win):
        return self.ensemble.predict(X_win)


class EventDrivenV2:
    """Event-Driven with configurable refit size."""

    def __init__(self, seed, n_trees=FULL_TREES, buffer=BUFFER_CAPACITY,
                 error_window_size=ERROR_WINDOW_SIZE, k=ERROR_THRESHOLD_K,
                 anchor_X=None, anchor_y=None):
        self.seed = seed
        self.n_trees = n_trees
        self.buffer = buffer
        self.ews = error_window_size
        self.k = k
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.ensemble = None
        self.recent_errors = []
        self.adaptation_cpu_time = 0.0
        self.retrain_events = 0
        self.trees_trained_count = 0

    def fit_initial(self, X_init, y_init):
        t0 = time.process_time()
        self.ensemble = create_base_ensemble(seed=self.seed, n_estimators=self.n_trees)
        self.ensemble.fit(X_init, y_init)
        cpu = time.process_time() - t0
        self.trees_trained_count += self.ensemble.get_num_trees()
        self.adaptation_cpu_time += cpu
        return cpu

    def predict(self, X_win):
        return self.ensemble.predict(X_win)

    def update_and_adapt(self, X_win, y_win, win_error, buf_X, buf_y):
        self.recent_errors.append(win_error)
        if len(self.recent_errors) > self.ews:
            self.recent_errors.pop(0)
        triggered = False
        if len(self.recent_errors) >= 5:
            mu = np.mean(self.recent_errors[:-1])
            sd = np.std(self.recent_errors[:-1])
            if self.recent_errors[-1] > mu + self.k * max(sd, 0.05):
                triggered = True
        cpu = 0.0
        if triggered:
            trX, trY = make_train_buffer(np.asarray(buf_X), np.asarray(buf_y),
                                         self.anchor_X, self.anchor_y, self.buffer)
            self.retrain_events += 1
            t0 = time.process_time()
            self.ensemble = create_base_ensemble(seed=self.seed + self.retrain_events * 13,
                                                 n_estimators=self.n_trees)
            self.ensemble.fit(trX, trY)
            cpu = time.process_time() - t0
            self.trees_trained_count += self.ensemble.get_num_trees()
            self.adaptation_cpu_time += cpu
            self.recent_errors = [win_error]
        return cpu, triggered


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------
def _prepare(stream_df, sd):
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


def _rec(seed, method, w, regime, y_true, y_pred, adapt_cpu, pred_cpu,
         is_retrain, is_reuse, is_refresh, n_classes, proba=None):
    m = window_metrics(y_true, y_pred)
    cm = window_confusion(y_true, y_pred, n_classes)
    row = {
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
    return row


def _summary(seed, method, records, init_cpu, adapt_cpu, retrains, trees_trained,
             trees_reused, reuse_events, refresh_events, total_wall, n_classes,
             extra=None):
    df = pd.DataFrame(records)
    df = df[df["method"] == method]
    y_true = np.concatenate([np.repeat([], 0)] * 0 + [np.array([])]) if False else None
    row = {
        "seed": seed, "method": method,
        "per_window_macro_f1": float(df["macro_f1"].mean()),
        "per_window_accuracy": float(df["accuracy"].mean()),
        "per_window_precision": float(df["precision"].mean()),
        "per_window_recall": float(df["recall"].mean()),
        "init_cpu_sec": float(init_cpu),
        "adaptation_cpu_sec": float(adapt_cpu),
        "total_cpu_sec": float(init_cpu + adapt_cpu),
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


def run_seed_v2(stream_df, sd, seed, methods, label_delay=False,
                log_proba=False, provenance_out=None):
    """Run the requested methods on one stream for one seed.

    Returns (per_window_df, summary_df, pooled_df, proba_store).
    """
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = _prepare(stream_df, sd)
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream_df["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]

    records, summaries = [], []
    proba_store = {}

    def buffer_append(buf_X, buf_y, w, idx, delay_from=None):
        src = idx if delay_from is None else win_slices[delay_from]
        buf_X.extend(X[src]); buf_y.extend(y_all[src])
        if len(buf_X) > BUFFER_CAPACITY:
            del buf_X[:-BUFFER_CAPACITY]
            del buf_y[:-BUFFER_CAPACITY]

    def run_simple(method, driver, transition_trigger):
        t_w0 = time.perf_counter()
        init_cpu = driver.fit_initial(X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        prev = init_regime
        recs = []
        yt_all, yp_all, pr_all = [], [], []
        pending = None
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            is_trans = reg != prev
            prev = reg
            adapt_cpu, did_retrain = driver.maybe_adapt(w, is_trans, buf_X, buf_y) \
                if transition_trigger else (0.0, False)
            t0 = time.process_time()
            yp = driver.predict(X[idx])
            pc = time.process_time() - t0
            if log_proba:
                try:
                    pr_all.append(driver.ensemble.predict_proba(X[idx])[:, -1])
                except Exception:
                    pass
            recs.append(_rec(seed, method, w, reg, y_all[idx], yp, adapt_cpu, pc,
                             did_retrain, 0, 0, n_classes))
            yt_all.append(y_all[idx]); yp_all.append(yp)
            if label_delay:
                if pending is not None:
                    buffer_append(buf_X, buf_y, w, pending, delay_from=pending)
                pending = w
            else:
                buffer_append(buf_X, buf_y, w, idx)
        summaries.append(_summary(seed, method, recs, init_cpu,
                                  driver.adaptation_cpu_time - init_cpu,
                                  driver.retrain_events,
                                  driver.trees_trained_count, 0, 0, 0,
                                  time.perf_counter() - t_w0, n_classes))
        records.extend(recs)
        if log_proba and pr_all:
            proba_store[method] = (np.concatenate(yt_all), np.concatenate(yp_all),
                                   np.concatenate(pr_all))
        else:
            proba_store[method] = (np.concatenate(yt_all), np.concatenate(yp_all), None)

    # ---------------- Frozen ----------------
    if "Frozen" in methods:
        run_simple("Frozen", FRDriver(seed, trigger="none"), False)

    # ---------------- Event-Driven (+ cheap) ----------------
    if "Event-Driven" in methods:
        # use the ORIGINAL class unchanged for exact reproduction
        ed = EventDrivenEnsemble(seed=seed, buffer_capacity=BUFFER_CAPACITY,
                                 error_window_size=ERROR_WINDOW_SIZE,
                                 error_threshold_k=ERROR_THRESHOLD_K,
                                 anchor_X=anchor_X, anchor_y=anchor_y)
        t_w0 = time.perf_counter()
        init_ret = ed.fit_initial(X_init, y_init)
        init_cpu = init_ret[0] if isinstance(init_ret, tuple) else init_ret
        recs = []; yt_all, yp_all = [], []
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            t0 = time.process_time(); yp = ed.predict(X[idx]); pc = time.process_time() - t0
            err = 1.0 - float(np.mean(yp == y_all[idx]))
            trig, adapt_cpu, _ = ed.update_and_adapt(X[idx], y_all[idx], err)
            recs.append(_rec(seed, "Event-Driven", w, reg, y_all[idx], yp,
                             adapt_cpu, pc, int(trig), 0, 0, n_classes))
            yt_all.append(y_all[idx]); yp_all.append(yp)
        summaries.append(_summary(seed, "Event-Driven", recs, init_cpu,
                                  ed.cumulative_cpu_time - init_cpu, ed.retrain_events,
                                  ed.total_trees_trained, 0, 0, 0,
                                  time.perf_counter() - t_w0, n_classes))
        records.extend(recs)
        proba_store["Event-Driven"] = (np.concatenate(yt_all), np.concatenate(yp_all), None)

    if "Event-Driven-Cheap" in methods:
        t_w0 = time.perf_counter()
        ed = EventDrivenV2(seed, n_trees=CHEAP_TREES, buffer=CHEAP_BUFFER,
                           anchor_X=anchor_X, anchor_y=anchor_y)
        init_cpu = ed.fit_initial(X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        recs = []; yt_all, yp_all = [], []
        pending = None
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            t0 = time.process_time(); yp = ed.predict(X[idx]); pc = time.process_time() - t0
            err = 1.0 - float(np.mean(yp == y_all[idx]))
            if label_delay:
                if pending is not None:
                    buffer_append(buf_X, buf_y, w, pending, delay_from=pending)
                pending = w
            else:
                buffer_append(buf_X, buf_y, w, idx)
            adapt_cpu, trig = ed.update_and_adapt(X[idx], y_all[idx], err, buf_X, buf_y)
            recs.append(_rec(seed, "Event-Driven-Cheap", w, reg, y_all[idx], yp,
                             adapt_cpu, pc, int(trig), 0, 0, n_classes))
            yt_all.append(y_all[idx]); yp_all.append(yp)
        summaries.append(_summary(seed, "Event-Driven-Cheap", recs, init_cpu,
                                  ed.adaptation_cpu_time, ed.retrain_events,
                                  ed.trees_trained_count, 0, 0, 0,
                                  time.perf_counter() - t_w0, n_classes))
        records.extend(recs)
        proba_store["Event-Driven-Cheap"] = (np.concatenate(yt_all), np.concatenate(yp_all), None)

    # ---------------- Full retraining / cheap / periodic ----------------
    fr_specs = [("Full Retraining", FULL_TREES, "regime", None),
                ("FR-Cheap", CHEAP_TREES, "regime", None),
                ("Periodic-Cheap-5", CHEAP_TREES, "periodic", 5),
                ("Periodic-Cheap-10", CHEAP_TREES, "periodic", 10)]
    for method, nt, trig, every in fr_specs:
        if method not in methods:
            continue
        run_simple(method, FRDriver(seed, n_trees=nt,
                                    buffer=CHEAP_BUFFER if nt == CHEAP_TREES else BUFFER_CAPACITY,
                                    trigger=trig, every=every,
                                    anchor_X=anchor_X, anchor_y=anchor_y),
                   True)

    # ---------------- Detectors ----------------
    for det in DETECTORS:
        if det not in methods:
            continue
        dm = DetectorAdaptiveModelV2(det, seed=seed, buffer_capacity=BUFFER_CAPACITY,
                                     anchor_X=anchor_X, anchor_y=anchor_y,
                                     **DETECTOR_PARAMS.get(det, {}))
        t_w0 = time.perf_counter()
        init_cpu, _ = dm.fit_initial(X_init, y_init)
        buf_X, buf_y = list(X_init), list(y_init)
        recs = []; yt_all, yp_all = [], []
        pending = None
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            t0 = time.process_time(); yp = dm.predict(X[idx]); pc = time.process_time() - t0
            err = 1.0 - float(np.mean(yp == y_all[idx]))
            if label_delay:
                if pending is not None:
                    buffer_append(buf_X, buf_y, w, pending, delay_from=pending)
                pending = w
            else:
                buffer_append(buf_X, buf_y, w, idx)
            trig, did, adapt_cpu, _ = dm.update_and_adapt(w, X[idx], y_all[idx], err)
            recs.append(_rec(seed, det, w, reg, y_all[idx], yp, adapt_cpu, pc,
                             int(did), 0, 0, n_classes))
            yt_all.append(y_all[idx]); yp_all.append(yp)
        summaries.append(_summary(seed, det, recs, init_cpu,
                                  dm.cumulative_cpu_time - init_cpu, dm.retrain_events,
                                  dm.total_trees_trained, 0, 0, 0,
                                  time.perf_counter() - t_w0, n_classes,
                                  extra={"detected_events": dm.detected_events}))
        records.extend(recs)
        proba_store[det] = (np.concatenate(yt_all), np.concatenate(yp_all), None)

    # ---------------- RAPT family ----------------
    rapt_specs = {
        "RAPT": dict(cls="orig"),
        "RAPT-Enhanced": dict(cls="orig_enh"),
        "RAPT-Deferred": dict(cls="v2", deferred=True),
        "RAPT-Cheap": dict(cls="v2", refresh_every=5, refresh_trees=CHEAP_TREES,
                           refresh_buffer=CHEAP_BUFFER),
        "RAPT-GateFix": dict(cls="v2", use_gate=True, gate_fix=True),
        "RAPT-Evidence": dict(cls="v2", refresh_every=1, refresh_mode="evidence",
                              rel_drop=0.10, refresh_trees=CHEAP_TREES,
                              refresh_buffer=CHEAP_BUFFER),
        "RAPT-Floor": dict(cls="v2", refresh_every=1, refresh_mode="floor",
                           refresh_trees=CHEAP_TREES, refresh_buffer=CHEAP_BUFFER),
        "RAPT-Parity-Refit": dict(cls="v2", parity_threshold=0.5),
        # window-equivalent refresh: on streams where every window is a regime
        # transition (UGR'16) the per-regime counter is reset every window, so
        # the periodic refresh never fires. This variant keeps a GLOBAL counter
        # so "refresh every 5 windows" is actually applied.
        "RAPT-Cheap-WinEq": dict(cls="v2", refresh_every=5, refresh_trees=CHEAP_TREES,
                                 refresh_buffer=CHEAP_BUFFER, refresh_counter_global=True),
    }
    for method, spec in rapt_specs.items():
        if method not in methods:
            continue
        t_w0 = time.perf_counter()
        if spec["cls"] == "orig":
            sys_ = RAPTSystem(seed=seed, mode="full", enable_calibration=True,
                              anchor_X=anchor_X, anchor_y=anchor_y,
                              buffer_capacity=BUFFER_CAPACITY)
            refit_n = 500
        elif spec["cls"] == "orig_enh":
            sys_ = RAPTEnhancedSystem(seed=seed, mode="full", enable_calibration=True,
                                      anchor_X=anchor_X, anchor_y=anchor_y,
                                      buffer_capacity=BUFFER_CAPACITY,
                                      novelty_refit_n=1500)
            refit_n = 1500
        else:
            sys_ = RAPTV2(seed=seed, anchor_X=anchor_X, anchor_y=anchor_y,
                          refit_n=BUFFER_CAPACITY, **{k: v for k, v in spec.items()
                                                      if k != "cls"})
            refit_n = BUFFER_CAPACITY
        init_ret = sys_.fit_initial(init_regime, X_init, y_init)
        init_cpu = init_ret[0] if isinstance(init_ret, tuple) else init_ret
        buf_X, buf_y = list(X_init), list(y_init)
        prev = init_regime
        recs = []; yt_all, yp_all = [], []
        pending = None
        # window_id -> regime for provenance of buffered training data
        wreg = {int(w): str(regimes[win_slices[w][0]]) for w in range(n_init, n_windows)}
        wreg[int(n_init - 1)] = str(init_regime)
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            reg = regimes[idx[0]]
            wa, reused, refreshed = 0.0, 0, 0
            if reg != prev:
                # buffered samples cover the most recent windows before w
                nbuf = len(buf_X)
                nbuf_win = max(1, int(np.ceil(nbuf / max(1, sd["window_size"]))))
                train_wids = [x for x in range(max(n_init, w - nbuf_win), w)]
                train_regs = [wreg.get(x, "?") for x in train_wids]
                if isinstance(sys_, RAPTV2):
                    if getattr(sys_, "gate_fix", False):
                        sys_._incoming_fp = sys_._fingerprint(X[idx])
                    reused_i, wa, _ = sys_.on_transition(
                        reg, w, buf_X[-refit_n:], buf_y[-refit_n:],
                        train_wids, train_regs)
                else:
                    reused_i, wa, _ = sys_.handle_regime_transition(
                        reg, w, X_buffer=buf_X[-refit_n:], y_buffer=buf_y[-refit_n:])
                reused = int(reused_i)
                prev = reg
            t0 = time.process_time(); yp = sys_.predict(X[idx]); pc = time.process_time() - t0
            if label_delay:
                if pending is not None:
                    buffer_append(buf_X, buf_y, w, pending, delay_from=pending)
                pending = w
            else:
                buffer_append(buf_X, buf_y, w, idx)
            if isinstance(sys_, RAPTV2):
                wa2, flags = sys_.update(w, X[idx], y_all[idx], buf_X, buf_y,
                                         [w], [reg])
                wa += wa2
                refreshed = int(flags.get("is_refresh", 0))
            else:
                wa += sys_.update(X[idx], y_all[idx], X_buffer=buf_X[-refit_n:],
                                  y_buffer=buf_y[-refit_n:])
            recs.append(_rec(seed, method, w, reg, y_all[idx], yp, wa, pc,
                             0, reused, refreshed, n_classes))
            yt_all.append(y_all[idx]); yp_all.append(yp)
        reuse_events = int(sum(r["is_reuse"] for r in recs))
        refresh_events = int(sum(r["is_refresh"] for r in recs))
        extra = {}
        if isinstance(sys_, RAPTV2):
            extra["parity_refits"] = sys_.parity_refits
            extra["gate_accept"] = sys_.gate_accept
            extra["gate_reject"] = sys_.gate_reject
            extra["refit_n"] = refit_n
            extra["n_refits"] = len(sys_.refit_rows)
            if sys_.refit_rows:
                extra["refit_total_rows_max"] = int(max(r["total_rows"] for r in sys_.refit_rows))
                extra["refit_buffer_rows_max"] = int(max(r["buffer_rows"] for r in sys_.refit_rows))
                extra["refit_anchor_rows"] = int(sys_.refit_rows[-1]["anchor_rows"])
            else:
                extra["refit_total_rows_max"] = 0
                extra["refit_buffer_rows_max"] = 0
                extra["refit_anchor_rows"] = 0
            if provenance_out is not None and sys_.provenance:
                pd.DataFrame(sys_.provenance).to_csv(
                    provenance_out, mode="a", header=not os.path.exists(provenance_out),
                    index=False)
        else:
            extra["parity_refits"] = getattr(sys_, "parity_refits", 0)
            extra["refit_n"] = refit_n
            _anchor = len(anchor_X) if anchor_X is not None else 0
            _buf = min(refit_n, BUFFER_CAPACITY)
            extra["refit_buffer_rows_max"] = int(_buf)
            extra["refit_anchor_rows"] = int(_anchor)
            extra["refit_total_rows_max"] = int(_buf + _anchor)
            extra["n_refits"] = int(getattr(sys_, "created_policy_count", 0) - 1)
        summaries.append(_summary(seed, method, recs, init_cpu, sys_.adaptation_cpu_time,
                                  getattr(sys_, "created_policy_count", 0) - 1,
                                  sys_.trees_trained_count, sys_.trees_reused_count,
                                  reuse_events, refresh_events,
                                  time.perf_counter() - t_w0, n_classes, extra=extra))
        records.extend(recs)
        proba_store[method] = (np.concatenate(yt_all), np.concatenate(yp_all), None)

    pw = pd.DataFrame(records)
    sm = pd.DataFrame(summaries)

    # pooled metrics per method from the concatenated predictions
    pooled_rows = []
    for method, (yt, yp, pr) in proba_store.items():
        pm = pooled_metrics(yt, yp, n_classes)
        pm.update({"seed": seed, "method": method})
        if n_classes == 2:
            pm.update(pooled_binary_attack_metrics(yt, yp, attack_label=1, proba=pr))
        pooled_rows.append(pm)
    pooled = pd.DataFrame(pooled_rows)
    return pw, sm, pooled, proba_store
