"""
================================================================================
EXPERIMENT 5 — QUANTILE FINGERPRINTING & REGIME MEMORY
================================================================================
Implements distribution quantile fingerprinting, cosine similarity matching,
and regime memory lookup/update for regime-aware ensemble policy transfer.
================================================================================
"""

import numpy as np
from scipy import stats


def extract_quantile_fingerprint(X_window):
    """
    Extracts a compact 21-dimensional quantile fingerprint from a window of features.
    For each feature column:
      - Percentiles: 10th, 25th, 50th (median), 75th, 90th
      - Mean
      - Standard deviation
    """
    fp_components = []
    n_features = X_window.shape[1]

    for col_idx in range(n_features):
        col_vals = X_window[:, col_idx]
        p10, p25, p50, p75, p90 = np.percentile(col_vals, [10, 25, 50, 75, 90])
        mean_val = np.mean(col_vals)
        std_val = np.std(col_vals)
        fp_components.extend([p10, p25, p50, p75, p90, mean_val, std_val])

    fp = np.array(fp_components, dtype=np.float64)
    # L2 normalize fingerprint for robust cosine similarity
    norm = np.linalg.norm(fp)
    if norm > 1e-8:
        fp = fp / norm
    return fp


def compute_similarity(fp1, fp2, metric='cosine'):
    """
    Computes fingerprint similarity.
    Cosine similarity: dot product of L2-normalized vectors.
    """
    if metric == 'cosine':
        dot = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        sim = dot / (norm1 * norm2)
        return float(np.clip(sim, 0.0, 1.0))
    elif metric == 'rbf':
        dist_sq = np.sum((fp1 - fp2) ** 2)
        return float(np.exp(-dist_sq / 2.0))
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


class RegimeMemory:
    """
    Maintains discovered regimes, their quantile centroids, and regime-specific model weights.
    """

    def __init__(self, similarity_threshold=0.85, alpha=0.7, temperature=1.0, component_names=['RF', 'ET', 'GB']):
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.temperature = temperature
        self.component_names = component_names

        self.regimes = {}  # regime_id -> dict
        self.next_regime_id = 0
        self.total_queries = 0
        self.retrieval_count = 0
        self.similarity_history = []

    def query_regime(self, fp):
        """
        Queries memory with current window fingerprint.
        Returns:
          regime_id, max_sim, weights, is_retrieved
        """
        self.total_queries += 1

        if not self.regimes:
            # First regime discovery
            r_id = self.next_regime_id
            self.next_regime_id += 1
            default_weights = {c: 1.0 / len(self.component_names) for c in self.component_names}
            default_ema = {c: 0.85 for c in self.component_names}

            self.regimes[r_id] = {
                'id': r_id,
                'centroid': np.copy(fp),
                'weights': default_weights,
                'ema_scores': default_ema,
                'sample_count': 1,
                'retrieval_count': 0
            }
            self.similarity_history.append(1.0)
            return r_id, 1.0, default_weights, False

        # Compute similarities against all existing regime centroids
        best_id = None
        max_sim = -1.0

        for r_id, r_data in self.regimes.items():
            sim = compute_similarity(fp, r_data['centroid'])
            if sim > max_sim:
                max_sim = sim
                best_id = r_id

        self.similarity_history.append(max_sim)

        if max_sim >= self.similarity_threshold:
            # Re-use existing regime
            self.retrieval_count += 1
            self.regimes[best_id]['retrieval_count'] += 1
            return best_id, max_sim, dict(self.regimes[best_id]['weights']), True
        else:
            # Discover new regime
            r_id = self.next_regime_id
            self.next_regime_id += 1
            default_weights = {c: 1.0 / len(self.component_names) for c in self.component_names}
            default_ema = {c: 0.85 for c in self.component_names}

            self.regimes[r_id] = {
                'id': r_id,
                'centroid': np.copy(fp),
                'weights': default_weights,
                'ema_scores': default_ema,
                'sample_count': 1,
                'retrieval_count': 0
            }
            return r_id, max_sim, default_weights, False

    def update_regime(self, regime_id, fp, window_model_f1s):
        """
        Updates regime's centroid and model weights based on observed window performance.
        """
        r_data = self.regimes[regime_id]
        n = r_data['sample_count']

        # Update centroid running mean
        new_centroid = (n * r_data['centroid'] + fp) / (n + 1)
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-8:
            new_centroid = new_centroid / norm
        r_data['centroid'] = new_centroid
        r_data['sample_count'] = n + 1

        # Update EMA scores and compute Softmax weights
        for c in self.component_names:
            f1_val = window_model_f1s.get(c, 0.0)
            r_data['ema_scores'][c] = self.alpha * r_data['ema_scores'][c] + (1.0 - self.alpha) * f1_val

        # Softmax normalization over EMA scores
        scores = np.array([r_data['ema_scores'][c] for c in self.component_names])
        exp_scores = np.exp((scores - np.max(scores)) / self.temperature)
        probs = exp_scores / np.sum(exp_scores)

        for idx, c in enumerate(self.component_names):
            r_data['weights'][c] = float(probs[idx])
