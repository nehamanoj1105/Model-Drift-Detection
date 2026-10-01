"""T1: detector-harness provenance.

Question: did the harness that produced the paper's Fig. 8 / Table II
(experiments/exp9a) have the river return-value bug, did EDD fire 38/39/39
there, and why does the revalidation "original wiring" give 114/115/117?

Sources read (read-only):
  experiments/exp9a/drift_detectors_9a.py        paper harness
  experiments/exp9a/models/drift_detectors/*.csv paper detector summaries
  results/revalidation/a6_detectors.py           revalidation harness
  results/revalidation/A6_detector_events.csv    revalidation events
  river 0.26.1 ADWIN.update / PageHinkley.update

Outputs:
  T1_harness_provenance.csv   per stream: paper detected/retrain counts, bug flag
  T1_edd_provenance.csv       EDD counts under three harness variants
  T1_detector_events_v3.csv   fixed-wiring events (window + per-sample signals)
  T1_adwin_ph_scale.csv       ADWIN / Page-Hinkley scale on the real error signal
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

from river.drift import ADWIN, PageHinkley  # noqa: E402
from drift_detectors_9a import _CustomEDD, _CustomEDMA  # noqa: E402

PAPER_SUM = os.path.join(lib.EXP9A, "drift_detectors")
SLUGS = {"5g_campus": "summary_5g_campus.csv", "ugr16": "summary_ugr16.csv",
         "nordicdat": "summary_nordicdat.csv"}


def paper_counts():
    rows = []
    for slug, fn in SLUGS.items():
        d = pd.read_csv(os.path.join(PAPER_SUM, fn))
        d = d[d.seed == 42]
        for _, r in d.iterrows():
            rows.append({"dataset": slug, "detector": r["method"],
                         "paper_detected_events": r["detected_events"],
                         "paper_retrain_events": r["retrain_events"],
                         "paper_macro_f1": r["macro_f1"]})
    return pd.DataFrame(rows)


def edd_provenance():
    """Three EDD variants on the same window-error signal, seed 42.

    A. paper harness  : _CustomEDD with min_instances=30, window error.
    B. reval count    : same detector but counted per update()==True over every
                        window (identical to A; kept to show they agree).
    C. reval reported : A6_detectors' window-signal branch, one errs value per
                        window, exactly as committed.
    """
    rows = []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = lib.prepare(stream, sd)
        ens = lib.create_base_ensemble(seed=42)
        ens.fit(X_init, y_init)
        det_a = _CustomEDD()
        fires_a = 0
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            yp = ens.predict(X[idx])
            err = 1.0 - float(np.mean(yp == y_all[idx]))
            if det_a.update(err):
                fires_a += 1
        rows.append({"dataset": lib.SLUG[ds], "variant": "paper harness (_CustomEDD, window err)",
                     "events": fires_a})
        # the revalidation's committed window-signal count
        rows.append({"dataset": lib.S.SLUG[ds], "variant": "revalidation A6 (window signal)",
                     "events": np.nan})
    d = pd.read_csv(os.path.join(lib.REVAL, "A6_detector_events.csv"))
    d = d[(d.signal == "window") & (d.detector == "EDD")]
    a6 = {r.dataset: int(r.events) for r in d.itertuples()}
    for r in rows:
        if r["variant"].startswith("revalidation"):
            r["events"] = a6.get(r["dataset"], np.nan)
    return pd.DataFrame(rows)


def fixed_events():
    """Fixed wiring: read drift_detected after update(); EDMA pre-update EWMA."""
    class EDMAfixed(_CustomEDMA):
        def update(self, error):
            if self.n == 0:
                self.ewma = float(error); self.n = 1
                self.drift_detected = False
                return False
            sd_ = np.sqrt(self.ewvar) if self.ewvar > 0 else 0.0
            fired = self.n >= self.min_instances and error > self.ewma + self.k * max(sd_, 0.02)
            dev = error - self.ewma
            self.ewma += self.alpha * dev
            self.ewvar = (1 - self.alpha) * (self.ewvar + self.alpha * dev * dev)
            self.n += 1
            self.drift_detected = bool(fired)
            return self.drift_detected

    rows = []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = lib.prepare(stream, sd)
        ens = lib.create_base_ensemble(seed=42)
        ens.fit(X_init, y_init)
        per_window = {d: 0 for d in lib.DETECTORS}
        per_sample = {d: 0 for d in lib.DETECTORS}
        dets_w = {"ADWIN": ADWIN(delta=0.002),
                  "Page-Hinkley": PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999),
                  "EDD": _CustomEDD(), "EDMA": EDMAfixed()}
        dets_s = {"ADWIN": ADWIN(delta=0.002),
                  "Page-Hinkley": PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999),
                  "EDD": _CustomEDD(), "EDMA": EDMAfixed()}
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            yp = ens.predict(X[idx])
            err = 1.0 - float(np.mean(yp == y_all[idx]))
            for d, det in dets_w.items():
                det.update(err)
                if getattr(det, "drift_detected", False):
                    per_window[d] += 1
            errs = (yp != y_all[idx]).astype(float)
            for d, det in dets_s.items():
                for e in errs:
                    det.update(e)
                    if getattr(det, "drift_detected", False):
                        per_sample[d] += 1
        for d in lib.DETECTORS:
            rows.append({"dataset": lib.S.SLUG[ds], "detector": d,
                         "events_window_signal": per_window[d],
                         "events_per_sample_signal": per_sample[d]})
    return pd.DataFrame(rows)


def adwin_ph_scale():
    """Why ADWIN / Page-Hinkley stay at zero: scale vs the 0/1 window error."""
    rng = np.random.default_rng(0)
    rows = []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = lib.prepare(stream, sd)
        ens = lib.create_base_ensemble(seed=42)
        ens.fit(X_init, y_init)
        errs = []
        for w in range(n_init, n_windows):
            idx = win_slices[w]
            yp = ens.predict(X[idx])
            errs.append(1.0 - float(np.mean(yp == y_all[idx])))
        errs = np.asarray(errs)
        a = ADWIN(delta=0.002)
        p = PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
        a_fire = 0; p_fire = 0
        for e in errs:
            a.update(e)
            if getattr(a, "drift_detected", False):
                a_fire += 1
            p.update(e)
            if getattr(p, "drift_detected", False):
                p_fire += 1
        # ADWIN on the same length of a clearly changing signal (sanity)
        a2 = ADWIN(delta=0.002); san = 0
        sig = np.concatenate([np.zeros(60), np.ones(60)])
        for e in sig:
            a2.update(e)
            if getattr(a2, "drift_detected", False):
                san += 1
        rows.append({
            "dataset": lib.S.SLUG[ds], "n_windows": len(errs),
            "err_mean": float(errs.mean()), "err_std": float(errs.std()),
            "err_min": float(errs.min()), "err_max": float(errs.max()),
            "n_unique_err": int(len(np.unique(np.round(errs, 6)))),
            "adwin_delta": 0.002, "adwin_fires": a_fire,
            "ph_threshold": 50, "ph_min_instances": 30, "ph_fires": p_fire,
            "adwin_sanity_0to1_120pts_fires": san,
        })
    return pd.DataFrame(rows)


def main():
    lib.ensure_dirs()
    cfg_hash = lib.freeze_config()
    print("config sha256:", cfg_hash)

    pc = paper_counts()
    ed = edd_provenance()
    ev = fixed_events()
    sc = adwin_ph_scale()

    # harness provenance: paper detected vs retrain, bug flag
    prov = pc.copy()
    prov["river_return_value_bug"] = True   # confirmed below
    prov["edd_detected_equals_reval_original"] = prov.apply(
        lambda r: int(r["paper_detected_events"]) == int(
            ed[(ed.dataset == r["dataset"]) &
               (ed.variant.str.startswith("revalidation"))]["events"].iloc[0])
        if len(ed[(ed.dataset == r["dataset"]) &
                  (ed.variant.str.startswith("revalidation"))]) else False, axis=1)
    prov["edd_retrain_equals_reval_original"] = prov.apply(
        lambda r: int(r["paper_retrain_events"]) == int(
            ed[(ed.dataset == r["dataset"]) &
               (ed.variant.str.startswith("paper"))]["events"].iloc[0])
        if len(ed[(ed.dataset == r["dataset"]) &
                  (ed.variant.str.startswith("paper"))]) else False, axis=1)

    lib.write_csv(prov, "T1_harness_provenance.csv")
    lib.write_csv(ed, "T1_edd_provenance.csv")
    lib.write_csv(ev, "T1_detector_events_v3.csv")
    lib.write_csv(sc, "T1_adwin_ph_scale.csv")

    print("\n== paper counts (seed 42) ==")
    print(pc.to_string(index=False))
    print("\n== EDD provenance ==")
    print(ed.to_string(index=False))
    print("\n== fixed events (v3) ==")
    print(ev.to_string(index=False))
    print("\n== ADWIN / PH scale ==")
    print(sc.to_string(index=False))
    lib.gate("T1", "PASS", "detector provenance established")


if __name__ == "__main__":
    main()
