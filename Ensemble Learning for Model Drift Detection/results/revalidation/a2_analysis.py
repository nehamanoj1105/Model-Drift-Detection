"""A2: why reuse fails on UGR'16 -- provenance, deferred, oracle diagnostics,
and per-regime P(Y|X) stability.

Outputs:
  A2_checkpoint_provenance.csv
  A2_deferred_vs_base.csv
  A2_oracle_diagnostics.csv
  A2_regime_stability.csv
  A2_hypothesis_verdict.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RAW, SEEDS, write_csv, write_raw_csv  # noqa: E402
from streams import get_stream, ALL_DATASETS, SLUG  # noqa: E402
import run_stream_v2 as R  # noqa: E402
from metrics_pooled import pooled_metrics, window_confusion  # noqa: E402
from sklearn.metrics import f1_score


def _prepare(stream_df, sd):
    return R._prepare(stream_df, sd)


def run_provenance(stream_df, sd, seed):
    """Run base RAPTV2 with provenance logging; return provenance DataFrame."""
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = _prepare(stream_df, sd)
    from models_9a import make_class_anchor
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream_df["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    sys_ = R.RAPTV2(seed=seed, anchor_X=anchor_X, anchor_y=anchor_y,
                    refit_n=R.BUFFER_CAPACITY)
    sys_.fit_initial(init_regime, X_init, y_init)
    buf_X, buf_y = list(X_init), list(y_init)
    prev = init_regime
    wreg = {int(w): str(regimes[win_slices[w][0]]) for w in range(n_init, n_windows)}
    wreg[int(n_init - 1)] = str(init_regime)
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = regimes[idx[0]]
        if reg != prev:
            nbuf_win = max(1, int(np.ceil(len(buf_X) / max(1, sd["window_size"]))))
            train_wids = list(range(max(n_init, w - nbuf_win), w))
            train_regs = [wreg.get(x, "?") for x in train_wids]
            sys_.on_transition(reg, w, buf_X[-R.BUFFER_CAPACITY:], buf_y[-R.BUFFER_CAPACITY:],
                               train_wids, train_regs)
            prev = reg
        sys_.predict(X[idx])
        buf_X.extend(X[idx]); buf_y.extend(y_all[idx])
        if len(buf_X) > R.BUFFER_CAPACITY:
            del buf_X[:-R.BUFFER_CAPACITY]; del buf_y[:-R.BUFFER_CAPACITY]
        sys_.update(w, X[idx], y_all[idx], buf_X, buf_y, [w], [reg])
    prov = pd.DataFrame(sys_.provenance)
    if len(prov):
        prov.insert(0, "seed", seed)
    return prov, sys_


def _pooled_run(stream_df, sd, seed, methods):
    pw, sm, pooled, _ = R.run_seed_v2(stream_df, sd, seed, methods)
    return pooled


# ---------------------------------------------------------------------------
# Oracle diagnostics
# ---------------------------------------------------------------------------
class OracleSameRegime:
    """Oracle-A: on transition to a previously-seen regime, train on that
    regime's most recent previous visit (uses knowledge of past window->regime
    membership). Not deployable."""

    def __init__(self, seed, n_trees=R.FULL_TREES, anchor_X=None, anchor_y=None):
        self.seed = seed
        self.n_trees = n_trees
        self.anchor_X = anchor_X
        self.anchor_y = anchor_y
        self.ensemble = None
        self.adaptation_cpu_time = 0.0
        self.retrain_events = 0
        self.trees_trained_count = 0

    def fit_initial(self, X_init, y_init):
        t0 = time.process_time()
        from models_9a import create_base_ensemble
        self.ensemble = create_base_ensemble(seed=self.seed, n_estimators=self.n_trees)
        self.ensemble.fit(X_init, y_init)
        cpu = time.process_time() - t0
        self.trees_trained_count += self.ensemble.get_num_trees()
        self.adaptation_cpu_time += cpu
        return cpu

    def predict(self, X):
        return self.ensemble.predict(X)

    def adapt_on_regime_data(self, Xr, yr):
        from models_9a import create_base_ensemble
        t0 = time.process_time()
        self.ensemble = create_base_ensemble(seed=self.seed + self.retrain_events * 7,
                                             n_estimators=self.n_trees)
        self.ensemble.fit(Xr, yr)
        cpu = time.process_time() - t0
        self.retrain_events += 1
        self.trees_trained_count += self.ensemble.get_num_trees()
        self.adaptation_cpu_time += cpu
        return cpu


def run_oracle_same_regime(stream_df, sd, seed):
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = _prepare(stream_df, sd)
    from models_9a import make_class_anchor
    anchor_X, anchor_y = make_class_anchor(X_init, y_init, per_class=15, seed=seed)
    wid = stream_df["window_id"].values
    init_regime = regimes[np.where(wid == n_init - 1)[0][0]]
    # record window samples per regime (known a-priori -> oracle)
    wid_all = stream_df["window_id"].values
    all_slices = {w: np.where(wid_all == w)[0] for w in range(0, n_windows)}
    reg_windows = {}
    for w in range(0, n_windows):
        r = str(regimes[all_slices[w][0]])
        reg_windows.setdefault(r, []).append(w)

    drv = OracleSameRegime(seed, anchor_X=anchor_X, anchor_y=anchor_y)
    drv.fit_initial(X_init, y_init)
    prev = init_regime
    seen = {str(init_regime): [list(range(0, n_init))]}
    yt_all, yp_all = [], []
    for w in range(n_init, n_windows):
        idx = win_slices[w]
        reg = str(regimes[idx[0]])
        if reg != prev:
            prev_visits = seen.get(reg, [])
            if prev_visits:
                ws = prev_visits[-1]
                Xr = np.vstack([X[all_slices[x]] for x in ws])
                yr = np.concatenate([y_all[all_slices[x]] for x in ws])
                if len(Xr):
                    drv.adapt_on_regime_data(Xr, yr)
            seen.setdefault(reg, []).append([w])
            prev = reg
        else:
            if seen.get(reg):
                seen[reg][-1].append(w)
        yp = drv.predict(X[idx])
        yt_all.append(y_all[idx]); yp_all.append(yp)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    pm = pooled_metrics(yt, yp, n_classes)
    pm.update({"seed": seed, "method": "Oracle-SameRegime",
               "adaptation_cpu_sec": drv.adaptation_cpu_time,
               "retrain_events": drv.retrain_events})
    return pm


# ---------------------------------------------------------------------------
# P(Y|X) stability per regime between consecutive visits
# ---------------------------------------------------------------------------
def regime_stability(stream_df, sd, seed=42):
    from models_9a import create_base_ensemble
    n_classes = sd["n_classes"]
    X, y_all, regimes, X_init, y_init, win_slices, n_init, n_windows = _prepare(stream_df, sd)
    wid = stream_df["window_id"].values
    # contiguous visits of each regime in the evaluation range
    visits = []
    prev = None
    for w in range(n_init, n_windows):
        r = str(regimes[win_slices[w][0]])
        if r != prev:
            visits.append({"regime": r, "windows": [w]})
            prev = r
        else:
            visits[-1]["windows"].append(w)
    rows = []
    for i in range(1, len(visits)):
        v = visits[i]
        # previous visit of the same regime
        prev_same = None
        for j in range(i - 1, -1, -1):
            if visits[j]["regime"] == v["regime"]:
                prev_same = visits[j]
                break
        prev_contig = visits[i - 1]
        def stack(vv):
            idxs = [win_slices[w] for w in vv["windows"] if w in win_slices]
            if not idxs:
                return None, None
            return np.vstack([X[ix] for ix in idxs]), np.concatenate([y_all[ix] for ix in idxs])
        Xt, yt = stack(v)
        if Xt is None:
            continue
        row = {"regime": v["regime"], "visit_index": i,
               "n_test": len(yt), "test_positive_rate": float(np.mean(yt == 1))}
        if prev_same is not None:
            Xs, ys = stack(prev_same)
            if Xs is not None and len(np.unique(ys)) > 1:
                m = create_base_ensemble(seed=seed, n_estimators=50)
                m.fit(Xs, ys)
                yp = m.predict(Xt)
                row["f1_same_regime_prev_visit"] = float(f1_score(yt, yp, average="macro", zero_division=0))
                row["pos_rate_prev_visit"] = float(np.mean(ys == 1))
                # class-conditional feature shift (mean abs z of feature means)
                shift = []
                for c in np.unique(ys):
                    ms = Xs[ys == c].mean(axis=0)
                    mt = Xt[yt == c].mean(axis=0) if (yt == c).any() else ms
                    sdv = Xs.std(axis=0) + 1e-9
                    shift.append(np.mean(np.abs(mt - ms) / sdv))
                row["class_cond_feature_shift"] = float(np.mean(shift))
        if prev_contig is not None:
            Xc, yc = stack(prev_contig)
            if Xc is not None and len(np.unique(yc)) > 1:
                m = create_base_ensemble(seed=seed, n_estimators=50)
                m.fit(Xc, yc)
                yp = m.predict(Xt)
                row["f1_contiguous_prev_block"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    datasets = ALL_DATASETS
    prov_all, stab_all, oracle_all, pooled_all = [], [], [], []
    for ds in datasets:
        stream, sd = get_stream(ds)
        slug = SLUG[ds]
        print(f"\n### {ds}", flush=True)
        for seed in SEEDS:
            prov, sys_ = run_provenance(stream, sd, seed)
            if len(prov):
                prov.insert(0, "dataset", slug)
                prov_all.append(prov)
            # deferred vs base
            pl = _pooled_run(stream, sd, seed, ["RAPT", "RAPT-Deferred",
                                                "Full Retraining", "Frozen",
                                                "Event-Driven"])
            pl.insert(0, "dataset", slug)
            pooled_all.append(pl)
            # oracle
            try:
                o = run_oracle_same_regime(stream, sd, seed)
                o.update({"dataset": slug})
                oracle_all.append(o)
            except Exception as e:
                print("  oracle failed", e)
            print(f"   seed {seed} done", flush=True)
        # stability is deterministic per seed for training; use seed 42
        st = regime_stability(stream, sd, seed=42)
        if len(st):
            st.insert(0, "dataset", slug)
            stab_all.append(st)

    if prov_all:
        prov = pd.concat(prov_all, ignore_index=True)
        write_csv(prov, "A2_checkpoint_provenance.csv")
        # fraction mismatched
        chk = prov[prov.get("event").isna() if "event" in prov else prov.index == prov.index]
        created = prov[prov["train_regimes"].notna() & (prov["train_regimes"] != "")]
        mism = created[created["train_regime_matches_key"] == 0]
        print(f"\ncheckpoints created: {len(created)}, trained on a DIFFERENT regime: "
              f"{len(mism)} ({100*len(mism)/max(1,len(created)):.1f}%)")
    if pooled_all:
        write_csv(pd.concat(pooled_all, ignore_index=True), "A2_deferred_vs_base.csv")
    if oracle_all:
        write_csv(pd.DataFrame(oracle_all), "A2_oracle_diagnostics.csv")
    if stab_all:
        write_csv(pd.concat(stab_all, ignore_index=True), "A2_regime_stability.csv")


if __name__ == "__main__":
    main()
