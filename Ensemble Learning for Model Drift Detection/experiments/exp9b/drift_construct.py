"""
Controlled Drift Construction for Experiment 9B.

Builds, from the *existing* processed 9B stream (the second dataset already used
by Experiment 9B — 5G NR end-to-end latency, 499 windows, 4 regimes A/B/C/D):

  * Covariate / data drift   : P(X) changes, P(Y|X) held as constant as possible.
                               Implemented by mixing source-regime and target-regime
                               *feature* windows at a controlled ratio; labels are
                               never altered.

  * Concept drift            : P(Y|X) changes. Implemented by a deterministic,
                               seed-fixed reflection of the most predictive features
                               (X -> a + b - X) applied to the post-drift portion
                               only, which reverses the feature->label relationship.

  * Recurring concept drift  : A -> (concept drift) -> B -> A', where A' has the
                               same feature distribution as A but a changed P(Y|X).

All constructions are strictly chronological (source-before-target), leakage-free
(the transformations are functions of X only and are applied only after the drift
point), deterministic given a seed, and documented below.

Nothing here retrains or inspects the test labels to decide *when* or *how* to
transform: the drift point and severity are fixed by configuration.
"""

import numpy as np
import pandas as pd

from exp9b_drift_config import (
    INITIAL_TRAIN_FRACTION, CONCEPT_DRIFT_FEATURE_COUNT,
    CONCEPT_DRIFT_MAX_AFFECTED,
)
from preprocessing_9b import FEATURE_COLS


# --------------------------------------------------------------------------
# Stream loading
# --------------------------------------------------------------------------
def load_stream(stream_csv, stream_json):
    import json
    df = pd.read_csv(stream_csv)
    with open(stream_json) as f:
        stream_def = json.load(f)
    return df, stream_def


def initial_train_count(df, stream_def):
    """Reuse the existing 9B initial-training protocol (20% of the stream)."""
    n = stream_def.get("initial_train_windows")
    if n is None:
        n = int(len(df) * INITIAL_TRAIN_FRACTION)
    return int(n)


# --------------------------------------------------------------------------
# Regime / segment helpers
# --------------------------------------------------------------------------
def get_regime_windows(df, regime_id):
    return df[df["regime_id"] == regime_id].reset_index(drop=True)


def select_transition_segments(stream_def, min_windows=20, max_pairs=2):
    """
    Select source -> target regime transitions from the *existing* 9B segment
    layout. A pair is (src_regime, tgt_regime) with distinct regimes; we take the
    first occurrence of each transition type so the construction is deterministic.
    """
    segs = stream_def["segments"]
    pairs = []
    seen = set()
    for a, b in zip(segs[:-1], segs[1:]):
        key = (a["regime_id"], b["regime_id"])
        if a["regime_id"] == b["regime_id"]:
            continue
        if key in seen:
            continue
        if a["window_count"] < min_windows or b["window_count"] < min_windows:
            continue
        seen.add(key)
        pairs.append({
            "src_regime": a["regime_id"],
            "tgt_regime": b["regime_id"],
            "src_segment_index": a["segment_index"],
            "tgt_segment_index": b["segment_index"],
        })
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError("No valid source->target regime transitions found.")
    return pairs


# --------------------------------------------------------------------------
# Covariate drift (P(X) changes, labels untouched)
# --------------------------------------------------------------------------
def _interleaved_mask(n, k_target, rng):
    """
    Return a boolean mask of length n with exactly k_target True entries, placed
    by a fixed-seed random interleaving. Preserves chronology statistically (the
    target-regime samples are spread across the window rather than appended),
    while never touching labels.
    """
    mask = np.zeros(n, dtype=bool)
    if k_target <= 0:
        return mask
    if k_target >= n:
        mask[:] = True
        return mask
    idx = rng.choice(n, size=k_target, replace=False)
    mask[idx] = True
    return mask


def build_covariate_drift_stream(df, stream_def, level, seed, n_init, n_before=30, n_after=40):
    """
    Construct one covariate-drift stream for a given severity `level`.

    Layout per selected source->target transition:
      * (n_init + n_before) source-regime windows  -> initial training prefix +
        pre-drift reference (labels untouched), then
      * n_after windows of which round(level * n_after) are target-regime feature
        windows and the remainder are source-regime windows, interleaved with a
        fixed-seed permutation.

    Labels are copied verbatim from the source dataset; no relabeling occurs.
    P(X) therefore shifts toward the target regime while P(Y|X) is unchanged
    (each sample keeps the label it had in its own regime).

    Returns (df_stream, drift_points); drift_points are the global window indices
    at which the mixed post-drift block begins.
    """
    rng = np.random.default_rng(seed)
    pairs = select_transition_segments(stream_def)
    parts = []
    drift_points = []
    gid = 0
    for p in pairs:
        src = get_regime_windows(df, p["src_regime"])
        tgt = get_regime_windows(df, p["tgt_regime"])

        n_pre = n_init + n_before
        src_before = src.iloc[:n_pre].copy()
        src_pool = src.iloc[n_pre:n_pre + n_after].copy()
        tgt_pool = tgt.iloc[:n_after].copy()

        n_after_eff = min(n_after, len(src_pool))
        k_target = int(round(level * n_after_eff))
        k_target = min(k_target, len(tgt_pool), n_after_eff)
        k_source = n_after_eff - k_target

        mask = _interleaved_mask(n_after_eff, k_target, rng)

        tgt_it = iter(tgt_pool.itertuples(index=False))
        src_it = iter(src_pool.itertuples(index=False))
        after_records = []
        for is_tgt in mask:
            row = next(tgt_it) if is_tgt else next(src_it)
            rec = dict(zip(tgt_pool.columns, row))
            rec["_is_target"] = int(is_tgt)
            after_records.append(rec)
        # All post-drift windows share a single (drift) regime id so that
        # Full Retraining / RAPT register exactly ONE regime change at the drift
        # point, rather than reacting to the interleaved sample provenance.
        drift_regime_id = f"{p['tgt_regime']}_covdrift"

        # Rebuild with clean chronological ids.
        for r in src_before.to_dict("records"):
            rec = dict(r)
            rec["window_id"] = gid
            rec["regime_id"] = p["src_regime"]
            rec["source_regime_id"] = p["src_regime"]
            rec["target_regime_id"] = p["tgt_regime"]
            rec["drift_level"] = level
            rec["is_target_regime_sample"] = 0
            parts.append(rec)
            gid += 1
        for r in after_records:
            is_tgt = int(r.pop("_is_target"))
            rec = dict(r)
            rec["window_id"] = gid
            rec["regime_id"] = drift_regime_id
            rec["source_regime_id"] = p["src_regime"]
            rec["target_regime_id"] = p["tgt_regime"]
            rec["drift_level"] = level
            rec["is_target_regime_sample"] = is_tgt
            parts.append(rec)
            gid += 1
        drift_points.append(gid - n_after_eff)

    out = pd.DataFrame(parts).reset_index(drop=True)
    return out, drift_points


# --------------------------------------------------------------------------
# Concept drift (P(Y|X) changes)
# --------------------------------------------------------------------------
def rank_predictive_features(df, n_features=CONCEPT_DRIFT_FEATURE_COUNT):
    """
    Rank features by mutual information with the target on the *pre-drift* portion
    only (the initial training prefix), so the choice of which features to
    transform never uses post-drift labels. Returns the top-n feature names.
    """
    from sklearn.feature_selection import mutual_info_classif
    n_init = int(len(df) * INITIAL_TRAIN_FRACTION)
    ref = df.iloc[:n_init]
    X = ref[FEATURE_COLS].values
    y = ref["qos_target"].values
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    order = np.argsort(mi)[::-1]
    return [FEATURE_COLS[i] for i in order[:n_features]], {FEATURE_COLS[i]: float(mi[i]) for i in range(len(FEATURE_COLS))}


def apply_concept_transform(chunk, affected_features):
    """
    Deterministic rank-reversal of the affected features using a *single shared
    row permutation*, derived from the most predictive affected feature.

    Construction:
      1. Order the post-drift rows by the primary affected feature (ascending).
      2. Reverse that order to obtain a permutation pi.
      3. Apply the SAME permutation pi to every affected feature column.

    Properties:
      * Because pi is a row permutation applied to the affected columns, the
        *joint* distribution of the affected features is preserved exactly (same
        rows, reordered), and every per-feature marginal is preserved exactly
        (KS statistic = 0). The construction therefore does NOT shift P(X).
      * The pairing between the affected features and Y is reversed, so the
        conditional relationship P(Y|X) genuinely changes.
      * It is a pure, deterministic function of X, applied only after the drift
        point, and never reads the target.
      * `affected_features` must be non-empty; the primary feature is the first
        element.
    """
    out = chunk.copy()
    if not affected_features:
        return out
    n = len(out)
    primary = affected_features[0]
    col = out[primary].values.astype(float)
    order = np.argsort(col, kind="mergesort")
    perm = np.empty(n, dtype=int)
    perm[order] = order[::-1]
    for f in affected_features:
        out[f] = out[f].values[perm]
    return out


def select_affected_features(df, target_regime_id, level, ranked=None):
    """
    Choose which features to transform for a given severity.

    `level` is the fraction of the top-ranked predictive features that receive
    the reversal (ceil). Features that are constant within the target regime are
    skipped because the reversal would be a no-op there.
    """
    if ranked is None:
        ranked, _ = rank_predictive_features(df)
    tgt = get_regime_windows(df, target_regime_id)
    varying = [f for f in ranked if float(np.std(tgt[f].values)) > 1e-9]
    k = max(1, int(np.ceil(level * len(ranked))))
    k = min(k, CONCEPT_DRIFT_MAX_AFFECTED)
    return varying[:k]


def build_concept_drift_stream(df, stream_def, level, seed, n_init, n_before=30, n_after=50,
                               affected_features=None, source_regime=None):
    """
    Construct one *within-regime* concept-drift stream for a given severity.

    Rationale: concept drift is a change in P(Y|X). To isolate it from covariate
    drift we keep the regime (and hence the feature distribution) fixed and only
    change the feature->label relationship. Concretely:

      * windows [0, n_init)                 : source regime, untouched
                                              -> initial training prefix
      * windows [n_init, n_init+n_before)   : source regime, untouched
                                              -> pre-drift reference
      * windows [n_init+n_before, +n_after) : source regime, with the
                                              rank-reversal applied to the top-k
                                              predictive features

    `level` is the fraction of the top-ranked predictive features (ranked on the
    pre-drift prefix only) that receive the reversal (ceil). Because the reversal
    is a permutation of the observed values, each affected feature's marginal is
    preserved exactly (KS = 0) while its association with Y is reversed, i.e. the
    construction changes P(Y|X) and not P(X).

    The regime id is left unchanged, so regime-keyed adapters (Full Retraining,
    RAPT) do not receive an oracle boundary signal — they must cope with drift
    they cannot observe through the regime key. This is the purest concept-drift
    test and is reported as such.
    """
    ranked, _ = rank_predictive_features(df)
    if source_regime is None:
        source_regime = stream_def["segments"][0]["regime_id"]
    src = get_regime_windows(df, source_regime)
    if len(src) < n_init + n_before + n_after:
        n_after = max(10, len(src) - n_init - n_before)

    if affected_features is None:
        affected = [] if level <= 0 else select_affected_features(df, source_regime, level, ranked)
    else:
        affected = affected_features
    k = len(affected)

    pre = src.iloc[:n_init + n_before].copy()
    post = src.iloc[n_init + n_before:n_init + n_before + n_after].copy()
    post = apply_concept_transform(post, affected)

    parts = []
    gid = 0
    for chunk, is_after in ((pre, False), (post, True)):
        for r in chunk.to_dict("records"):
            rec = dict(r)
            rec["window_id"] = gid
            rec["regime_id"] = source_regime
            rec["source_regime_id"] = source_regime
            rec["drift_level"] = level
            rec["is_concept_drift"] = int(is_after)
            rec["concept_features_affected"] = k
            parts.append(rec)
            gid += 1

    out = pd.DataFrame(parts).reset_index(drop=True)
    drift_points = [n_init + n_before]
    return out, drift_points, list(affected)


def build_recurring_concept_stream(df, stream_def, level, seed, n_init,
                                   n_a=20, n_b=40, n_aprime=20):
    """
    A -> B -> A' recurring concept drift construction.

      * A  : source regime, unmodified (n_init + n_a windows; the leading n_init
             windows form the initial training prefix)
      * B  : a *different* regime, unmodified (n_b windows)   [regime change]
      * A' : the source regime AGAIN, under the *same regime id as A*, with the
             rank-reversal applied to the top-k predictive features (n_aprime
             windows). The feature distribution resembles A; P(Y|X) has changed.

    This is the key recurrence test: because A' reuses A's regime id, RAPT's
    repository lookup hits and it blindly reuses the stored A policy, even though
    the label relationship has changed. RAPT's reuse/non-reuse behaviour and the
    resulting performance are measured — the algorithm is deliberately NOT
    modified here.
    """
    pairs = select_transition_segments(stream_def, min_windows=1, max_pairs=1)
    p = pairs[0]
    A_id, B_id = p["src_regime"], p["tgt_regime"]
    A = get_regime_windows(df, A_id)
    B = get_regime_windows(df, B_id)
    if len(A) < n_init + n_a + n_aprime:
        n_a = max(5, len(A) - n_init - n_aprime)
    if len(B) < n_b:
        n_b = len(B)

    ranked, _ = rank_predictive_features(df)
    affected = select_affected_features(df, A_id, level, ranked)

    a_block = A.iloc[:n_init + n_a].copy()
    b_block = B.iloc[:n_b].copy()
    a_prime_block = A.iloc[n_init + n_a:n_init + n_a + n_aprime].copy()
    a_prime_block = apply_concept_transform(a_prime_block, affected)

    parts = []
    gid = 0
    for chunk, reg_id, phase, is_concept in (
        (a_block, A_id, "A", False),
        (b_block, B_id, "B", False),
        (a_prime_block, A_id, "A_prime", True),
    ):
        for r in chunk.to_dict("records"):
            rec = dict(r)
            rec["window_id"] = gid
            rec["regime_id"] = reg_id
            rec["phase"] = phase
            rec["drift_level"] = level
            rec["is_concept_drift"] = int(is_concept)
            parts.append(rec)
            gid += 1

    out = pd.DataFrame(parts).reset_index(drop=True)
    boundaries = {
        "A_end": n_init + n_a,
        "B_end": n_init + n_a + n_b,
        "A_prime_start": n_init + n_a + n_b,
    }
    return out, boundaries, list(affected)


# --------------------------------------------------------------------------
# Diagnostics: evidence that the concept construction changes P(Y|X)
# --------------------------------------------------------------------------
def class_conditional_stats(df, features):
    """Class-conditional feature mean/std before vs after the drift point."""
    before = df[df["is_concept_drift"] == 0]
    after = df[df["is_concept_drift"] == 1]
    rows = []
    for f in features:
        for cls in sorted(df["qos_target"].unique()):
            b = before[before["qos_target"] == cls][f]
            a = after[after["qos_target"] == cls][f]
            rows.append({
                "feature": f,
                "class": int(cls),
                "pre_mean": float(b.mean()) if len(b) else np.nan,
                "post_mean": float(a.mean()) if len(a) else np.nan,
                "pre_std": float(b.std()) if len(b) else np.nan,
                "post_std": float(a.std()) if len(a) else np.nan,
                "n_pre": int(len(b)),
                "n_post": int(len(a)),
            })
    return pd.DataFrame(rows)


def _mi(x, y, seed=0):
    from sklearn.feature_selection import mutual_info_classif
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return float(mutual_info_classif(x, y, discrete_features=False, random_state=seed)[0])


def feature_target_association(df, features):
    """Mutual information between each feature and Y, before vs after drift, plus
    the two-sample KS statistic between the pre- and post-drift feature values
    (a value of ~0 confirms the marginal P(X) is preserved)."""
    from scipy.stats import ks_2samp
    before = df[df["is_concept_drift"] == 0]
    after = df[df["is_concept_drift"] == 1]
    rows = []
    for f in features:
        rows.append({
            "feature": f,
            "mi_pre": _mi(before[f].values, before["qos_target"].values),
            "mi_post": _mi(after[f].values, after["qos_target"].values),
            "ks_pre_post": float(ks_2samp(before[f].values, after[f].values).statistic),
        })
    return pd.DataFrame(rows)


def model_based_py_change(df, n_init):
    """
    Direct evidence that P(Y|X) changed while P(X) did not: train the base
    ensemble on the pre-drift prefix and compare its Macro-F1 on a held-out
    pre-drift block vs the post-drift block. A large drop, together with KS ~ 0
    on the affected features, demonstrates a conditional (concept) change rather
    than a covariate change.
    """
    from sklearn.metrics import f1_score
    from models_9b import create_base_ensemble
    from preprocessing_9b import FEATURE_COLS
    pre = df[df["is_concept_drift"] == 0]
    post = df[df["is_concept_drift"] == 1]
    train = pre.iloc[:n_init]
    holdout = pre.iloc[n_init:]
    if len(holdout) == 0 or len(post) == 0:
        return None
    model = create_base_ensemble(seed=0)
    model.fit(train[FEATURE_COLS].values, train["qos_target"].values)
    f1_hold = f1_score(holdout["qos_target"], model.predict(holdout[FEATURE_COLS].values),
                       average="macro", zero_division=0)
    f1_post = f1_score(post["qos_target"], model.predict(post[FEATURE_COLS].values),
                       average="macro", zero_division=0)
    return {"f1_pre_holdout": float(f1_hold), "f1_post_drift": float(f1_post),
            "delta_f1": float(f1_post - f1_hold)}


def transform_invariance(df_plain, df_drift, features):
    """
    Evidence that the concept transform preserves P(X): compare the post-drift
    block of the *untransformed* stream with the post-drift block of the
    *transformed* stream. KS ~ 0 confirms the feature distribution is unchanged;
    the change in I(feature; Y) confirms the conditional relationship moved.
    """
    from scipy.stats import ks_2samp
    pre = df_plain[df_plain["is_concept_drift"] == 1]
    post = df_drift[df_drift["is_concept_drift"] == 1]
    rows = []
    for f in features:
        rows.append({
            "feature": f,
            "ks_plain_vs_transformed": float(ks_2samp(pre[f].values, post[f].values).statistic),
            "mi_plain": _mi(pre[f].values, pre["qos_target"].values),
            "mi_transformed": _mi(post[f].values, post["qos_target"].values),
        })
    return pd.DataFrame(rows)


def conditional_proba_estimate(df, feature, n_bins=4):
    """
    Empirical P(Y = BAD | feature-quantile-bin) before vs after drift.

    A visible change in these conditional distributions is direct evidence that
    P(Y|X) (not merely P(X)) has moved.
    """
    before = df[df["is_concept_drift"] == 0].copy()
    after = df[df["is_concept_drift"] == 1].copy()

    def _tables(sub):
        try:
            sub["_bin"] = pd.qcut(sub[feature], q=n_bins, duplicates="drop")
        except ValueError:
            return {}
        out = {}
        for b, g in sub.groupby("_bin", observed=True):
            p = (g["qos_target"] == 2).mean()  # P(BAD)
            out[str(b)] = {"p_bad": float(p), "n": int(len(g))}
        return out

    return {"feature": feature, "pre": _tables(before), "post": _tables(after)}
