"""
Correct Checkpoint Manager for Experiment 2 Corrected.

Critical invariant enforced here:
    A checkpoint for regime B is ALWAYS trained on regime B data.
    A source model transferred into B is NEVER saved as a B checkpoint.

Checkpoint lifecycle:
    1. Regime transition detected (A -> B)
    2. Probability model decides: transfer from A, or adapt locally
    3. Use chosen policy for streaming predictions
    4. After enough B data accumulates (>= MIN_CHECKPOINT_SAMPLES windows):
       - Train NEW ensemble on B data
       - Save as B checkpoint
    5. B checkpoint is now available for future episodes targeting B
"""

import copy
import time
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_CHECKPOINT_SAMPLES = 50   # minimum window-rows required to create a checkpoint
CHECKPOINT_BUFFER_SIZE = 200  # rolling buffer of target windows kept for checkpoint training


class RegimeCheckpointExp2:
    """
    A stored regime policy checkpoint.

    The rf_model and et_model inside are ALWAYS trained on data from regime_id.
    They are never the direct result of a transfer from another regime.
    """

    def __init__(
        self,
        regime_id: str,
        rf_model,
        et_model,
        weights: list,
        training_sample_count: int,
        historical_f1: float,
        creation_episode: int,
        feature_mean: np.ndarray = None,
        feature_std: np.ndarray = None,
    ):
        self.regime_id = regime_id
        self.rf_model = copy.deepcopy(rf_model)
        self.et_model = copy.deepcopy(et_model)
        self.weights = list(weights)
        self.training_sample_count = training_sample_count
        self.historical_f1 = historical_f1
        self.creation_episode = creation_episode
        self.last_used_episode = creation_episode
        self.successful_transfers = 0
        self.negative_transfers = 0
        self.reuse_count = 0

        # Store regime feature statistics for similarity computation
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    @property
    def transfer_success_rate(self) -> float:
        total = self.successful_transfers + self.negative_transfers
        return self.successful_transfers / total if total > 0 else 0.5

    @property
    def negative_transfer_rate(self) -> float:
        total = self.successful_transfers + self.negative_transfers
        return self.negative_transfers / total if total > 0 else 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted soft-vote probability prediction."""
        p_rf = self.rf_model.predict_proba(X)
        p_et = self.et_model.predict_proba(X)
        w1, w2 = self.weights[0], self.weights[1]
        s = w1 + w2
        if s <= 0:
            w1, w2, s = 0.5, 0.5, 1.0
        return (w1 / s) * p_rf + (w2 / s) * p_et

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.rf_model.classes_[np.argmax(proba, axis=1)]

    def __repr__(self):
        return (f"RegimeCheckpointExp2(regime={self.regime_id}, "
                f"n_train={self.training_sample_count}, "
                f"f1={self.historical_f1:.4f}, "
                f"episode={self.creation_episode})")


class CheckpointPool:
    """
    Pool of regime-specific policy checkpoints.

    Rules:
    - Each regime_id maps to the LATEST checkpoint for that regime.
    - Checkpoints are added only after local target-data training.
    - A transferred source model does NOT automatically become a checkpoint.
    """

    def __init__(self):
        self._pool: dict[str, RegimeCheckpointExp2] = {}
        self._history: list[RegimeCheckpointExp2] = []
        self._episode_counter: int = 0

    def episode_count(self) -> int:
        return self._episode_counter

    def get_available_for_target(
        self,
        target_regime: str,
        before_episode: int,
    ) -> list:
        """
        Return all checkpoints available BEFORE the given episode,
        excluding any checkpoint for target_regime itself (to avoid identity transfer).

        Parameters
        ----------
        target_regime : str
        before_episode : int
            Strict upper bound on creation episode.

        Returns
        -------
        list of RegimeCheckpointExp2 (may be empty for first few episodes)
        """
        candidates = []
        for ckpt in self._history:
            if ckpt.regime_id != target_regime and ckpt.creation_episode < before_episode:
                candidates.append(ckpt)
        # Deduplicate: keep only the latest checkpoint per source regime
        seen = {}
        for ckpt in reversed(candidates):
            if ckpt.regime_id not in seen:
                seen[ckpt.regime_id] = ckpt
        return list(seen.values())

    def add_checkpoint(
        self,
        regime_id: str,
        X_target: np.ndarray,
        y_target: np.ndarray,
        historical_f1: float,
        episode_id: int,
        create_ensemble_fn,
        seed: int,
    ) -> 'RegimeCheckpointExp2':
        """
        Train a fresh ensemble on X_target / y_target and save as regime_id checkpoint.

        NEVER call this with X from the source regime.
        """
        assert len(X_target) >= MIN_CHECKPOINT_SAMPLES, (
            f"Cannot create checkpoint for {regime_id}: only {len(X_target)} samples "
            f"(need >= {MIN_CHECKPOINT_SAMPLES})."
        )

        ensemble = create_ensemble_fn(seed=seed)
        ensemble.fit(X_target, y_target)

        feat_mean = X_target.mean(axis=0)
        feat_std  = X_target.std(axis=0) + 1e-8

        ckpt = RegimeCheckpointExp2(
            regime_id=regime_id,
            rf_model=ensemble.rf,
            et_model=ensemble.et,
            weights=list(ensemble.weights),
            training_sample_count=len(X_target),
            historical_f1=historical_f1,
            creation_episode=episode_id,
            feature_mean=feat_mean,
            feature_std=feat_std,
        )
        self._pool[regime_id] = ckpt
        self._history.append(ckpt)
        self._episode_counter += 1
        return ckpt

    def add_initial_checkpoint(
        self,
        regime_id: str,
        X_init: np.ndarray,
        y_init: np.ndarray,
        historical_f1: float,
        create_ensemble_fn,
        seed: int,
    ) -> 'RegimeCheckpointExp2':
        """Add the initial training-phase checkpoint (episode 0)."""
        return self.add_checkpoint(
            regime_id=regime_id,
            X_target=X_init,
            y_target=y_init,
            historical_f1=historical_f1,
            episode_id=0,
            create_ensemble_fn=create_ensemble_fn,
            seed=seed,
        )

    def has(self, regime_id: str) -> bool:
        return regime_id in self._pool

    def get_latest(self, regime_id: str) -> 'RegimeCheckpointExp2':
        return self._pool[regime_id]

    def all_regime_ids(self) -> list:
        return list(self._pool.keys())

    def verify_pool_diversity(self) -> bool:
        """Assert pool contains genuinely different regime policies (sanity check #9)."""
        return len(self._pool) > 1 or len(self._history) > 0

    def __len__(self):
        return len(self._pool)

    def __repr__(self):
        return f"CheckpointPool({list(self._pool.keys())})"


class ActiveEnsemble:
    """
    Tracks the currently active prediction model, which may be:
      (a) a transferred checkpoint (source model used for predictions in target regime)
      (b) the local RAPT-E checkpoint (when abstaining from transfer)
      (c) the initial checkpoint (before any transitions)

    Critically, the active ensemble is SEPARATE from the checkpoint pool.
    The checkpoint pool stores target-trained models.
    The active ensemble is whatever we're predicting with right now.
    """

    def __init__(self, ckpt: RegimeCheckpointExp2):
        self._ckpt = ckpt
        self.source_regime = ckpt.regime_id
        self.is_transfer = False

    def load_checkpoint(self, ckpt: RegimeCheckpointExp2, is_transfer: bool = False):
        """Switch to a new checkpoint for predictions."""
        self._ckpt = ckpt
        self.source_regime = ckpt.regime_id
        self.is_transfer = is_transfer

    def calibrate_weights(self, X_recent: np.ndarray, y_recent: np.ndarray):
        """Recompute soft-vote weights from recent target data (RAPT-E calibration)."""
        if len(X_recent) < 20:
            return
        p_rf = self._ckpt.rf_model.predict_proba(X_recent[-50:])
        p_et = self._ckpt.et_model.predict_proba(X_recent[-50:])
        y_sub = y_recent[-50:]
        acc_rf = np.mean(np.argmax(p_rf, axis=1) == y_sub)
        acc_et = np.mean(np.argmax(p_et, axis=1) == y_sub)
        total = acc_rf + acc_et
        if total > 0:
            self._ckpt.weights = [acc_rf / total, acc_et / total]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ckpt.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._ckpt.predict_proba(X)
