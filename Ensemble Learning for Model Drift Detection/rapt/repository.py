"""
================================================================================
RAPT STAGE 2 — AUTONOMOUS REGIME REPOSITORY & CONTINUOUS TRANSFER ENGINE
================================================================================
Implements:
  1. RegimeEntry: Encapsulates regime fingerprint, specialized models, and usage metadata.
  2. Continuous Transfer Engine: Computes kernel similarity weights:
       s_k = exp(-gamma * ||f_t - f_k||^2) / Sum_j exp(-gamma * ||f_t - f_j||^2)
     and synthesizes the online policy:
       P_RAPT(X) = Sum_k s_k * P_{E_k}(X)
  3. Autonomous Memory Management:
     - Novelty insertion when max similarity falls below novelty threshold tau_novelty
     - LRU (Least-Recently-Used) eviction capped at K <= 8 stored regimes
     - Eviction Cache Miss logging: detects when an evicted regime recurs, tracking
       the occurrence and quantifying lost transfer benefit.
================================================================================
"""

import numpy as np


class RegimeEntry:
    """
    Metadata container for a single stored regime in the repository.
    """
    def __init__(self, regime_id, name, fingerprint, models, weights, window_id):
        self.regime_id = regime_id
        self.name = name
        self.fingerprint = np.asarray(fingerprint, dtype=np.float32)
        self.models = models  # dict of base classifiers: {'RandomForest': ..., 'ExtraTrees': ..., 'GradientBoosting': ...}
        self.weights = weights
        self.created_window = window_id
        self.last_accessed_window = window_id
        self.access_count = 1

    def predict_proba(self, X):
        """Compute average soft-voting probability from this regime's specialized models."""
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        return np.mean(probs, axis=0)


class AutonomousRegimeRepository:
    """
    Autonomous Regime Repository with continuous policy synthesis and LRU eviction.
    """

    def __init__(self, max_capacity=8, gamma=15.0, novelty_threshold=0.65):
        self.max_capacity = max_capacity
        self.gamma = gamma
        self.novelty_threshold = novelty_threshold
        
        self.entries = {}           # regime_id -> RegimeEntry
        self.next_regime_id = 1
        
        # Eviction and regret tracking
        self.eviction_history = []  # List of evicted regime records
        self.eviction_cache_misses = [] # Occurrences where an evicted regime returned

    def size(self):
        return len(self.entries)

    def compute_similarity_weights(self, fp_current):
        """
        Compute continuous similarity weights via RBF kernel across all stored regimes:
          s_k = exp(-gamma * ||f_t - f_k||^2) / Sum_j exp(-gamma * ||f_t - f_j||^2)
          
        Returns:
          - weights_dict: dict of {regime_id: s_k}
          - max_similarity: float, highest similarity to any single stored regime
          - closest_id: regime_id of closest stored regime
        """
        if not self.entries:
            return {}, 0.0, None
            
        distances = {}
        raw_similarities = {}
        fp_cur = np.asarray(fp_current, dtype=np.float32)
        
        for r_id, entry in self.entries.items():
            dist_sq = float(np.sum((fp_cur - entry.fingerprint) ** 2))
            distances[r_id] = dist_sq
            sim = float(np.exp(-self.gamma * dist_sq))
            raw_similarities[r_id] = sim
            
        sum_sim = sum(raw_similarities.values())
        if sum_sim < 1e-12:
            # Uniform fallback if distances are exceptionally large
            n = len(self.entries)
            weights_dict = {r_id: 1.0 / n for r_id in self.entries}
            max_sim = 0.0
            closest_id = min(distances, key=distances.get)
        else:
            weights_dict = {r_id: sim / sum_sim for r_id, sim in raw_similarities.items()}
            closest_id = max(raw_similarities, key=raw_similarities.get)
            max_sim = raw_similarities[closest_id]
            
        return weights_dict, max_sim, closest_id

    def predict_synthesized_proba(self, X, weights_dict, current_window_id=None):
        """
        Synthesize online prediction probability via continuous policy transfer:
          P_RAPT(X) = Sum_k s_k * P_{E_k}(X)
        Also updates LRU access timestamps for contributing entries.
        """
        if not self.entries or not weights_dict:
            raise ValueError("Repository is empty; cannot predict.")
            
        p_synthesized = np.zeros(len(X), dtype=np.float64)
        
        for r_id, weight in weights_dict.items():
            if weight > 1e-4:
                entry = self.entries[r_id]
                p_synthesized += weight * entry.predict_proba(X)
                if current_window_id is not None:
                    entry.last_accessed_window = current_window_id
                    entry.access_count += 1
                    
        return np.clip(p_synthesized, 1e-5, 1.0 - 1e-5)

    def insert_regime(self, fingerprint, models, weights, window_id, name="regime"):
        """
        Insert newly trained regime policy into the repository.
        If capacity exceeds max_capacity, evicts the Least Recently Used (LRU) entry.
        """
        # A. Check if capacity is reached -> trigger LRU eviction
        if len(self.entries) >= self.max_capacity:
            # Evict entry with lowest last_accessed_window
            lru_id = min(self.entries, key=lambda k: (self.entries[k].last_accessed_window, -self.entries[k].created_window))
            evicted_entry = self.entries.pop(lru_id)
            
            eviction_record = {
                'regime_id': evicted_entry.regime_id,
                'name': evicted_entry.name,
                'fingerprint': evicted_entry.fingerprint,
                'created_window': evicted_entry.created_window,
                'evicted_window': window_id,
                'total_accesses': evicted_entry.access_count,
            }
            self.eviction_history.append(eviction_record)
            
        # B. Insert new regime entry
        r_id = self.next_regime_id
        self.next_regime_id += 1
        
        new_entry = RegimeEntry(
            regime_id=r_id,
            name=f"{name}_id{r_id}",
            fingerprint=fingerprint,
            models=models,
            weights=weights,
            window_id=window_id
        )
        self.entries[r_id] = new_entry
        return r_id

    def check_eviction_regret(self, fp_current, window_id, match_similarity_threshold=0.70):
        """
        Auditing check: determines if the current drift state matches an entry that
        was previously evicted from the repository.
        
        If a match is found, an Eviction Cache Miss is recorded.
        """
        if not self.eviction_history:
            return False, None
            
        fp_cur = np.asarray(fp_current, dtype=np.float32)
        for ev in self.eviction_history:
            dist_sq = float(np.sum((fp_cur - ev['fingerprint']) ** 2))
            sim = float(np.exp(-self.gamma * dist_sq))
            if sim >= match_similarity_threshold:
                miss_record = {
                    'window_id': window_id,
                    'evicted_regime_id': ev['regime_id'],
                    'evicted_name': ev['name'],
                    'similarity': sim,
                    'evicted_at_window': ev['evicted_window'],
                }
                self.eviction_cache_misses.append(miss_record)
                return True, miss_record
                
        return False, None
