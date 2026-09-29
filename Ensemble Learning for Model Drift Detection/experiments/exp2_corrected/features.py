"""
Source-Target Feature Engineering for Experiment 2 Corrected.

All features are derived exclusively from data available BEFORE a transfer decision.
No target-regime labels, no future transfer outcomes.

Feature groups:
  1. Distributional (mean/std/quantile differences, Wasserstein, JSD)
  2. Temporal (autocorrelation, trend, volatility differences)
  3. Correlation structure distance
  4. Historical policy quality (from checkpoint metadata)
"""

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from checkpoint_manager import RegimeCheckpointExp2


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-10 else default


def _jsd_1d(p_vals: np.ndarray, q_vals: np.ndarray, n_bins: int = 20) -> float:
    """Jensen-Shannon divergence between two 1D empirical distributions."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    if abs(hi - lo) < 1e-10:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(jensenshannon(p_hist, q_hist))


def compute_distributional_similarity(
    X_source: np.ndarray,
    X_target: np.ndarray,
) -> float:
    """
    Aggregate distributional similarity score in [0, 1].
    Higher = more similar distributions.

    Uses mean of per-feature (1 - normalised Wasserstein).
    """
    n_feat = X_source.shape[1]
    sims = []
    for f in range(n_feat):
        s_vals = X_source[:, f]
        t_vals = X_target[:, f]
        rng = max(abs(s_vals.max() - s_vals.min()),
                  abs(t_vals.max() - t_vals.min()), 1e-10)
        w_dist = wasserstein_distance(s_vals, t_vals)
        norm_w = w_dist / rng
        sims.append(max(0.0, 1.0 - norm_w))
    return float(np.mean(sims))


def compute_source_target_features(
    source_ckpt: 'RegimeCheckpointExp2',
    X_source_windows: np.ndarray,
    X_target_windows: np.ndarray,
    current_episode: int,
) -> np.ndarray:
    """
    Build the feature vector for (source, target) pair used by the probability model.

    All inputs must be from data STRICTLY available before the transfer decision.

    Parameters
    ----------
    source_ckpt : RegimeCheckpointExp2
        The candidate source checkpoint.
    X_source_windows : np.ndarray, shape (n_source_windows, n_features)
        Feature matrix for source regime windows (from checkpoint training data statistics).
    X_target_windows : np.ndarray, shape (n_target_windows, n_features)
        Feature matrix for target regime windows seen so far (pre-label, current telemetry).
    current_episode : int
        Current episode index (for computing checkpoint age).

    Returns
    -------
    np.ndarray of shape (n_meta_features,)
    """
    n_feat = X_source_windows.shape[1]
    features = []

    # ── 1. Distribution features ─────────────────────────────────────────────
    s_mean = X_source_windows.mean(axis=0)
    t_mean = X_target_windows.mean(axis=0)
    s_std  = X_source_windows.std(axis=0) + 1e-8
    t_std  = X_target_windows.std(axis=0) + 1e-8

    mean_diff = t_mean - s_mean
    std_diff  = t_std  - s_std

    # Normalised mean difference per feature (z-score w.r.t. source std)
    norm_mean_diff = mean_diff / s_std
    features.extend(norm_mean_diff.tolist())           # n_feat values

    # Std ratio
    std_ratio = t_std / s_std
    features.extend(std_ratio.tolist())                # n_feat values

    # Aggregate distributional similarity score
    sim_score = compute_distributional_similarity(X_source_windows, X_target_windows)
    features.append(sim_score)                         # 1 value

    # Per-feature Wasserstein (normalised)
    wass_per_feat = []
    for f in range(n_feat):
        rng = max(s_std[f], 1e-10) * 4
        w = wasserstein_distance(X_source_windows[:, f], X_target_windows[:, f])
        wass_per_feat.append(min(w / rng, 5.0))
    features.extend(wass_per_feat)                     # n_feat values

    # Aggregate Wasserstein
    features.append(float(np.mean(wass_per_feat)))    # 1 value

    # Jensen-Shannon divergence (mean over features)
    jsd_vals = [_jsd_1d(X_source_windows[:, f], X_target_windows[:, f]) for f in range(n_feat)]
    features.append(float(np.mean(jsd_vals)))          # 1 value

    # IQR differences
    q25_s = np.percentile(X_source_windows, 25, axis=0)
    q75_s = np.percentile(X_source_windows, 75, axis=0)
    q25_t = np.percentile(X_target_windows, 25, axis=0)
    q75_t = np.percentile(X_target_windows, 75, axis=0)
    iqr_diff = (q75_t - q25_t) - (q75_s - q25_s)
    norm_iqr_diff = iqr_diff / (q75_s - q25_s + 1e-8)
    features.append(float(np.mean(np.abs(norm_iqr_diff))))  # 1 value

    # ── 2. Temporal / autocorrelation features ───────────────────────────────
    def _acf_lag1(X_2d: np.ndarray) -> np.ndarray:
        """Per-feature lag-1 autocorrelation."""
        acf = []
        for f in range(X_2d.shape[1]):
            x = X_2d[:, f]
            if len(x) < 3:
                acf.append(0.0)
                continue
            mu = x.mean()
            denom = np.sum((x - mu) ** 2)
            if denom < 1e-12:
                acf.append(0.0)
                continue
            acf.append(float(np.sum((x[:-1] - mu) * (x[1:] - mu)) / denom))
        return np.array(acf)

    acf_s = _acf_lag1(X_source_windows)
    acf_t = _acf_lag1(X_target_windows)
    acf_diff = acf_t - acf_s
    features.append(float(np.mean(np.abs(acf_diff))))  # 1 value

    # Trend (linear slope ratio)
    def _trend(X_2d: np.ndarray) -> np.ndarray:
        t_idx = np.arange(len(X_2d))
        slopes = []
        for f in range(X_2d.shape[1]):
            if len(t_idx) < 2:
                slopes.append(0.0)
                continue
            cov = np.cov(t_idx.astype(float), X_2d[:, f])
            slope = cov[0, 1] / (cov[0, 0] + 1e-10)
            slopes.append(slope)
        return np.array(slopes)

    trend_s = _trend(X_source_windows)
    trend_t = _trend(X_target_windows)
    trend_diff = np.abs(trend_t - trend_s)
    features.append(float(np.mean(trend_diff)))         # 1 value

    # Volatility (rolling std difference)
    vol_s = np.array([X_source_windows[:, f].std() for f in range(n_feat)])
    vol_t = np.array([X_target_windows[:, f].std() for f in range(n_feat)])
    features.append(float(np.mean(np.abs(vol_t - vol_s))))  # 1 value

    # ── 3. Correlation structure distance ────────────────────────────────────
    if X_source_windows.shape[0] > 2 and X_target_windows.shape[0] > 2:
        corr_s = np.corrcoef(X_source_windows.T)
        corr_t = np.corrcoef(X_target_windows.T)
        corr_dist = float(np.linalg.norm(corr_s - corr_t, ord='fro'))
    else:
        corr_dist = 0.0
    features.append(corr_dist)                          # 1 value

    # ── 4. Historical policy quality features ────────────────────────────────
    features.append(source_ckpt.historical_f1)          # 1 value
    features.append(float(current_episode - source_ckpt.creation_episode))  # age
    features.append(float(source_ckpt.reuse_count))
    features.append(source_ckpt.transfer_success_rate)
    features.append(source_ckpt.negative_transfer_rate)
    features.append(float(source_ckpt.training_sample_count))

    # Weight imbalance (deviation from uniform 0.5/0.5)
    w_imbalance = abs(source_ckpt.weights[0] - 0.5)
    features.append(w_imbalance)                        # 1 value

    return np.array(features, dtype=np.float64)


def compute_source_target_features_batch(
    candidates: list,
    X_source_windows_dict: dict,
    X_target_windows: np.ndarray,
    current_episode: int,
) -> np.ndarray:
    """
    Compute meta-features for a list of candidate checkpoints simultaneously.

    Parameters
    ----------
    candidates : list of RegimeCheckpointExp2
    X_source_windows_dict : dict regime_id -> np.ndarray of window features
    X_target_windows : np.ndarray  (already observed target windows, no labels needed)
    current_episode : int

    Returns
    -------
    np.ndarray of shape (n_candidates, n_meta_features)
    """
    rows = []
    for ckpt in candidates:
        X_src = X_source_windows_dict.get(ckpt.regime_id)
        if X_src is None:
            # Fallback: use checkpoint stored statistics to approximate source distribution
            if ckpt.feature_mean is not None:
                # Reconstruct approximate source windows from stored mean/std
                rng = np.random.default_rng(seed=42)
                X_src = rng.normal(
                    loc=ckpt.feature_mean,
                    scale=ckpt.feature_std,
                    size=(max(50, len(X_target_windows)), len(ckpt.feature_mean)),
                )
            else:
                X_src = X_target_windows  # degenerate fallback -> similarity = 1.0
        feat = compute_source_target_features(
            source_ckpt=ckpt,
            X_source_windows=X_src,
            X_target_windows=X_target_windows,
            current_episode=current_episode,
        )
        rows.append(feat)
    return np.vstack(rows)


def get_feature_count(n_stream_features: int) -> int:
    """Return meta-feature dimension given number of stream features."""
    # n_feat * 2 (norm_mean_diff + std_ratio)
    # + n_feat (wass_per_feat)
    # + 6 aggregate scalars (sim, mean_wass, jsd, iqr, acf, trend, vol, corr_dist)
    # + 7 policy features
    return n_stream_features * 3 + 15
