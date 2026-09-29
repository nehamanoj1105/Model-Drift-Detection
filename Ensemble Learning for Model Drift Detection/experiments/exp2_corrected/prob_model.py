"""
Online Calibrated Transferability Estimator for Experiment 2 Corrected.

Uses rolling-origin logistic regression trained only on STRICTLY past transfer episodes.

Key correctness guarantee:
    For each prediction at episode t, the model is trained on episodes 0..t-1 only.
    assert max(training_episode_ids) < current_episode_id  (enforced inside)

Cold-start heuristic (no fitted model yet):
    prob = 0.5 (uninformative prior)
    Not the similarity-weighted heuristic from the original buggy Exp2.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

warnings.filterwarnings('ignore')

# Minimum number of positive examples before the model is trained
MIN_POSITIVE_EXAMPLES = 3
# Minimum total episodes before any fitting attempt
MIN_EPISODES_TO_FIT = 6
# Transfer positive threshold
DELTA_F1_POSITIVE = +0.005
DELTA_F1_NEGATIVE = -0.005


class OnlineTransferabilityEstimator:
    """
    Online calibrated binary logistic regression for transfer probability prediction.

    Label definition:
        1 (positive)  : delta_f1 > +0.005
        0 (not-positive) : delta_f1 <= +0.005  (includes neutral and negative)

    Rolling-origin protocol:
        To predict at episode k, model is fit on episodes 0..k-1.
    """

    def __init__(self, tau: float = 0.60, seed: int = 42):
        self.tau = tau
        self.seed = seed
        self.is_fitted = False
        self._model = None
        self._scaler = StandardScaler()
        self._scaler_fitted = False

        # Record of all past (episode_id, features, label)
        self._episode_ids: list[int] = []
        self._X_history: list[np.ndarray] = []
        self._y_history: list[int] = []

        # Rolling-origin evaluation buffer
        self._eval_records: list[dict] = []

    def record_outcome(
        self,
        episode_id: int,
        features: np.ndarray,
        delta_f1: float,
        source_regime: str,
        target_regime: str,
        similarity: float,
    ):
        """
        Record a completed transfer episode outcome.
        Call this AFTER observing actual labels (post-prediction).
        """
        label = 1 if delta_f1 > DELTA_F1_POSITIVE else 0
        self._episode_ids.append(episode_id)
        self._X_history.append(features.copy())
        self._y_history.append(label)
        self._eval_records.append({
            'episode_id': episode_id,
            'source_regime': source_regime,
            'target_regime': target_regime,
            'delta_f1': delta_f1,
            'label': label,
            'similarity': similarity,
        })

    def _refit(self, current_episode_id: int):
        """
        Refit the logistic regression on all episodes strictly before current_episode_id.
        """
        if not self._episode_ids:
            self.is_fitted = False
            return

        # Strict anti-leakage filter
        valid_mask = [eid < current_episode_id for eid in self._episode_ids]
        if not any(valid_mask):
            self.is_fitted = False
            return

        X_train = np.array([x for x, m in zip(self._X_history, valid_mask) if m])
        y_train = np.array([y for y, m in zip(self._y_history, valid_mask) if m])
        eps_train = [e for e, m in zip(self._episode_ids, valid_mask) if m]

        # Temporal anti-leakage assertion
        if eps_train:
            assert max(eps_train) < current_episode_id, (
                f"Leakage: max training episode {max(eps_train)} >= current {current_episode_id}"
            )

        n_pos = int(y_train.sum())
        n_classes = len(np.unique(y_train))

        if n_pos < MIN_POSITIVE_EXAMPLES or n_classes < 2 or len(y_train) < MIN_EPISODES_TO_FIT:
            self.is_fitted = False
            return

        # Fit scaler on training data only
        if not self._scaler_fitted or len(X_train) > 0:
            self._scaler = StandardScaler()
            self._scaler.fit(X_train)
            self._scaler_fitted = True

        X_scaled = self._scaler.transform(X_train)

        try:
            base_lr = LogisticRegression(
                random_state=self.seed,
                max_iter=500,
                class_weight='balanced',
                C=0.1,
                solver='lbfgs',
            )
            # Use Platt calibration for probability calibration
            if len(X_scaled) >= 10:
                cv = min(3, n_pos)  # cross-val folds capped by positive count
                self._model = CalibratedClassifierCV(base_lr, cv=cv, method='sigmoid')
            else:
                self._model = base_lr
            self._model.fit(X_scaled, y_train)
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def predict_proba_positive(
        self,
        features: np.ndarray,
        current_episode_id: int,
    ) -> float:
        """
        Predict P(positive transfer | features) for a candidate source.

        Parameters
        ----------
        features : np.ndarray, shape (n_meta_features,)
        current_episode_id : int  used to ensure no leakage during refit

        Returns
        -------
        float in [0, 1]
        """
        self._refit(current_episode_id)

        if not self.is_fitted or self._model is None:
            # Cold-start: return uninformative prior
            return 0.5

        X = features.reshape(1, -1)
        if self._scaler_fitted:
            X = self._scaler.transform(X)

        try:
            proba = self._model.predict_proba(X)[0]
            # Find index of positive class (label=1)
            if hasattr(self._model, 'classes_'):
                classes = list(self._model.classes_)
            else:
                classes = [0, 1]
            if 1 in classes:
                return float(proba[classes.index(1)])
            return float(proba[-1])
        except Exception:
            return 0.5

    def should_transfer(self, prob_positive: float) -> bool:
        """Returns True if probability exceeds tau threshold."""
        return prob_positive >= self.tau

    def get_label_distribution(self) -> dict:
        """Return count of positive / neutral / negative transfer episodes."""
        if not self._y_history:
            return {'positive': 0, 'not_positive': 0, 'total': 0}
        y = np.array(self._y_history)
        return {
            'positive': int(y.sum()),
            'not_positive': int((y == 0).sum()),
            'total': len(y),
        }

    def evaluate_held_out(self) -> dict:
        """
        Rolling-origin evaluation of probability model on held-out episodes.

        For each episode k (from the first episode with enough data):
          - Train on episodes 0..k-1
          - Predict on episode k
          - Compute AUROC, AUPRC, Brier

        Returns dict with aggregate metrics.
        """
        if len(self._episode_ids) < MIN_EPISODES_TO_FIT + 2:
            return {
                'auroc': float('nan'), 'auprc': float('nan'),
                'brier': float('nan'), 'n_eval': 0,
                'n_positive_eval': 0,
            }

        preds, labels = [], []
        for k in range(MIN_EPISODES_TO_FIT, len(self._episode_ids)):
            eps_k = self._episode_ids[k]
            # Build training set: episodes strictly before k
            valid = [i for i, e in enumerate(self._episode_ids) if e < eps_k]
            if not valid:
                continue
            X_tr = np.array([self._X_history[i] for i in valid])
            y_tr = np.array([self._y_history[i] for i in valid])
            if len(np.unique(y_tr)) < 2 or y_tr.sum() < MIN_POSITIVE_EXAMPLES:
                continue

            try:
                sc = StandardScaler().fit(X_tr)
                X_tr_sc = sc.transform(X_tr)
                X_ev_sc = sc.transform(self._X_history[k].reshape(1, -1))
                lr = LogisticRegression(
                    max_iter=500, class_weight='balanced', C=0.1,
                    random_state=self.seed, solver='lbfgs',
                ).fit(X_tr_sc, y_tr)
                proba = lr.predict_proba(X_ev_sc)[0]
                classes = list(lr.classes_)
                p_pos = proba[classes.index(1)] if 1 in classes else proba[-1]
                preds.append(float(p_pos))
                labels.append(int(self._y_history[k]))
            except Exception:
                continue

        if len(labels) < 2 or len(np.unique(labels)) < 2:
            return {
                'auroc': float('nan'), 'auprc': float('nan'),
                'brier': float('nan'), 'n_eval': len(labels),
                'n_positive_eval': int(np.array(labels).sum()) if labels else 0,
            }

        preds_arr = np.array(preds)
        labels_arr = np.array(labels)

        try:
            auroc = float(roc_auc_score(labels_arr, preds_arr))
        except Exception:
            auroc = float('nan')
        try:
            auprc = float(average_precision_score(labels_arr, preds_arr))
        except Exception:
            auprc = float('nan')
        try:
            brier = float(brier_score_loss(labels_arr, preds_arr))
        except Exception:
            brier = float('nan')

        # ECE (10 bins)
        ece = _expected_calibration_error(preds_arr, labels_arr, n_bins=10)

        return {
            'auroc': auroc,
            'auprc': auprc,
            'brier': brier,
            'ece': ece,
            'n_eval': len(labels),
            'n_positive_eval': int(labels_arr.sum()),
            'predictions': preds_arr.tolist(),
            'true_labels': labels_arr.tolist(),
        }

    @property
    def eval_records(self):
        return list(self._eval_records)


def _expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_frac = mask.sum() / n
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += bin_frac * abs(bin_conf - bin_acc)
    return float(ece)
