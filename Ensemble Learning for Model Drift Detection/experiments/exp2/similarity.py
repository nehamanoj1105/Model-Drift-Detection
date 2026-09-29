"""
Similarity Metrics Engine for Experiment 2
Computes individual and combined distance / similarity metrics between source and target regime representations.
Includes Euclidean distance, Wasserstein distance, Jensen-Shannon divergence, Correlation distance, and Temporal distance.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon, euclidean

class RegimeSimilarityEngine:
    """
    Computes individual telemetry distance metrics and transparently combines them into S_combined.
    """
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def compute_distances(self, source_rep, target_rep, source_df=None, target_df=None):
        """
        Computes distance vector D between source and target representations.
        Returns dictionary of individual distances and combined distance/similarity.
        """
        distances = {}

        # 1. Euclidean distance on normalized distribution statistics (means and stds)
        s_stats = []
        t_stats = []
        for col in self.feature_cols:
            s_stats.extend([source_rep.get(f"{col}_mean", 0.0), source_rep.get(f"{col}_std", 0.0)])
            t_stats.extend([target_rep.get(f"{col}_mean", 0.0), target_rep.get(f"{col}_std", 0.0)])
        
        s_stats = np.array(s_stats, dtype=float)
        t_stats = np.array(t_stats, dtype=float)
        
        # Feature-wise scaling for euclidean distance
        stat_scale = np.std(np.vstack([s_stats, t_stats]), axis=0) + 1e-6
        norm_euclidean = float(np.linalg.norm((s_stats - t_stats) / stat_scale) / np.sqrt(len(s_stats)))
        distances['euclidean_distance'] = norm_euclidean

        # 2. Wasserstein distance (1D Wasserstein over telemetry feature distributions)
        if source_df is not None and target_df is not None:
            w1_dists = []
            for col in self.feature_cols:
                s_vals = source_df[col].values.astype(float)
                t_vals = target_df[col].values.astype(float)
                if len(s_vals) > 0 and len(t_vals) > 0:
                    w1 = stats.wasserstein_distance(s_vals, t_vals)
                    # Normalize by feature range
                    combined_std = np.std(np.concatenate([s_vals, t_vals])) + 1e-6
                    w1_dists.append(w1 / combined_std)
            distances['wasserstein_distance'] = float(np.mean(w1_dists)) if w1_dists else norm_euclidean
        else:
            # Fallback estimation using percentile distances if raw df slice not provided
            w1_est = []
            for col in self.feature_cols:
                p_s = np.array([source_rep.get(f"{col}_p10", 0), source_rep.get(f"{col}_p25", 0),
                               source_rep.get(f"{col}_median", 0), source_rep.get(f"{col}_p75", 0),
                               source_rep.get(f"{col}_p90", 0)])
                p_t = np.array([target_rep.get(f"{col}_p10", 0), target_rep.get(f"{col}_p25", 0),
                               target_rep.get(f"{col}_median", 0), target_rep.get(f"{col}_p75", 0),
                               target_rep.get(f"{col}_p90", 0)])
                w1_est.append(np.mean(np.abs(p_s - p_t)))
            distances['wasserstein_distance'] = float(np.mean(w1_est))

        # 3. Jensen-Shannon divergence
        if source_df is not None and target_df is not None:
            js_dists = []
            for col in self.feature_cols:
                s_vals = source_df[col].values.astype(float)
                t_vals = target_df[col].values.astype(float)
                if len(s_vals) > 0 and len(t_vals) > 0:
                    # Create 20-bin histograms across combined domain
                    min_val = min(s_vals.min(), t_vals.min())
                    max_val = max(s_vals.max(), t_vals.max())
                    if max_val - min_val < 1e-6:
                        js_dists.append(0.0)
                        continue
                    bins = np.linspace(min_val, max_val, 21)
                    hist_s, _ = np.histogram(s_vals, bins=bins, density=True)
                    hist_t, _ = np.histogram(t_vals, bins=bins, density=True)
                    # Convert density to probability mass
                    hist_s = hist_s + 1e-9
                    hist_t = hist_t + 1e-9
                    hist_s /= hist_s.sum()
                    hist_t /= hist_t.sum()
                    js_dists.append(float(jensenshannon(hist_s, hist_t)))
            distances['js_distance'] = float(np.mean(js_dists)) if js_dists else 0.0
        else:
            distances['js_distance'] = min(1.0, norm_euclidean / 2.0)

        # 4. Correlation structure distance (Frobenius norm between correlation vectors)
        s_corr = np.array(source_rep.get('correlation_vector', [0.0]), dtype=float)
        t_corr = np.array(target_rep.get('correlation_vector', [0.0]), dtype=float)
        distances['correlation_distance'] = float(np.linalg.norm(s_corr - t_corr) / np.sqrt(max(1, len(s_corr))))

        # 5. Temporal dynamics distance (autocorrelations and slopes)
        temp_diffs = []
        for col in self.feature_cols:
            s_ac = source_rep.get(f"{col}_lag1_ac", 0.0)
            t_ac = target_rep.get(f"{col}_lag1_ac", 0.0)
            s_slope = source_rep.get(f"{col}_slope", 0.0)
            t_slope = target_rep.get(f"{col}_slope", 0.0)
            temp_diffs.extend([abs(s_ac - t_ac), abs(s_slope - t_slope)])
        distances['temporal_distance'] = float(np.mean(temp_diffs))

        # Combined normalized distance
        weights = {
            'euclidean_distance': 0.25,
            'wasserstein_distance': 0.25,
            'js_distance': 0.20,
            'correlation_distance': 0.15,
            'temporal_distance': 0.15
        }
        
        combined_d = sum(weights[k] * distances[k] for k in weights)
        distances['combined_distance'] = float(combined_d)
        
        # Exponential similarity transformation: S in [0, 1]
        distances['combined_similarity'] = float(np.exp(-2.0 * combined_d))

        return distances


class SimilarityRanker:
    """
    Ranks candidate historical checkpoints based on similarity to current target regime.
    Provides methods for Similarity-Only selection and Similarity-Weighted policy blending.
    """
    def __init__(self, feature_cols):
        self.engine = RegimeSimilarityEngine(feature_cols)

    def rank_candidates(self, target_rep, candidate_checkpoints, target_df=None):
        """
        Calculates similarity metrics for all candidate checkpoints.
        Returns sorted list of (checkpoint, similarity_dict).
        """
        ranked = []
        for ckpt in candidate_checkpoints:
            source_rep = ckpt['representation']
            source_df = ckpt.get('df_slice', None)
            dists = self.engine.compute_distances(source_rep, target_rep, source_df=source_df, target_df=target_df)
            ranked.append({
                'checkpoint': ckpt,
                'metrics': dists,
                'similarity': dists['combined_similarity']
            })
        
        # Sort descending by similarity
        ranked.sort(key=lambda x: x['similarity'], reverse=True)
        return ranked

    def select_top1(self, target_rep, candidate_checkpoints, target_df=None):
        """
        Similarity-Only method: Selects single candidate with highest similarity.
        """
        if not candidate_checkpoints:
            return None, 0.0, {}
        ranked = self.rank_candidates(target_rep, candidate_checkpoints, target_df=target_df)
        best = ranked[0]
        return best['checkpoint'], best['similarity'], best['metrics']

    def compute_weighted_blend(self, target_rep, candidate_checkpoints, target_df=None, temperature=1.0):
        """
        Similarity-Weighted Transfer: Blends policy weights based on normalized similarity scores.
        """
        if not candidate_checkpoints:
            return None, {}
        ranked = self.rank_candidates(target_rep, candidate_checkpoints, target_df=target_df)
        
        sims = np.array([item['similarity'] for item in ranked], dtype=float)
        # Softmax / normalized weights
        exp_sims = np.exp(sims / max(1e-3, temperature))
        weights = exp_sims / np.sum(exp_sims)
        
        blended_rf_weight = sum(w * item['checkpoint']['weights'][0] for w, item in zip(weights, ranked))
        blended_et_weight = sum(w * item['checkpoint']['weights'][1] for w, item in zip(weights, ranked))
        
        # Renormalize blended ensemble weights
        total_w = blended_rf_weight + blended_et_weight
        if total_w > 0:
            blended_rf_weight /= total_w
            blended_et_weight /= total_w
        else:
            blended_rf_weight, blended_et_weight = 0.5, 0.5

        return {
            'weights': [blended_rf_weight, blended_et_weight],
            'ranked_candidates': ranked,
            'blending_weights': weights.tolist()
        }
