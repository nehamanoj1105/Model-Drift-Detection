"""
src/estimator.py - Online logistic regression estimator for probability-guided transfer.

DESIGN (pre-registered, do not change after seeing results):
- sklearn LogisticRegression, class_weight='balanced', C=1, max_iter=300
- Features standardized with running statistics (fit on all training rows so far)
- At decision t: fit on all candidate rows from decisions STRICTLY before t
- Min 50 rows with both classes before fitting; else output historical positive rate
- Platt scaling once 200 out-of-sample rows with both classes available
- Store every out-of-sample score with its label for AUROC computation
- Online tau selection: from grid {0.20,...,0.90 step 0.05} + abstain-always
  choose value maximizing mean realized dF1 over earlier decisions
  require >= 20 earlier decisions; else abstain
  break ties toward larger tau

LEAKAGE PREVENTION:
- record_outcome() must be called AFTER interval I_t labels are observed
- predict() uses only history from decisions before current t
- No feature uses I_t outcomes
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Optional, Tuple


TAU_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
TAU_ABSTAIN = float('inf')  # Always abstain option


class OnlineEstimator:
    """
    Online rolling-origin logistic regression for transfer probability prediction.
    
    All state is updated strictly causally (online in the streaming loop).
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 300,
                 class_weight: str = 'balanced', seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed
        
        # History: list of (decision_t, feature_vector, label, direct_df1, oos_score)
        self._history: List[Dict] = []
        self._oos_scores: List[float] = []  # out-of-sample scores (predicted at that t)
        self._oos_labels: List[int] = []    # corresponding labels
        self._oos_df1s: List[float] = []    # corresponding direct_df1 values
        self._oos_decision_ts: List[int] = []  # corresponding decision t values
        
        self._model: Optional[LogisticRegression] = None
        self._scaler = StandardScaler()
        self._scaler_fitted = False
        self._platt_model = None
        self._platt_fitted = False
        
        self._n_fitted = 0  # decisions where estimator was fitted
        self._n_unfitted = 0  # decisions where it was not
        self._n_positive_labels = 0
        self._n_total_labels = 0
        
        # Tau state
        self._tau_history: List[Dict] = []  # {tau, chosen_cand_df1} per decision
    
    def record_outcome(
        self,
        decision_t: int,
        features: np.ndarray,
        label: int,         # 1=positive, 0=not-positive
        direct_df1: float,  # raw dF1 for this candidate at this decision
    ):
        """
        Record outcome AFTER I_t labels are observed (called in streaming loop after interval).
        This is strictly causal: predict() at decision_t must be called before record_outcome().
        """
        entry = {
            'decision_t': decision_t,
            'features': np.array(features, dtype=np.float64),
            'label': int(label),
            'direct_df1': float(direct_df1),
        }
        self._history.append(entry)
        self._n_total_labels += 1
        if label == 1:
            self._n_positive_labels += 1
    
    def _get_train_data_before(self, current_t: int):
        """Get all training data from decisions strictly before current_t."""
        entries = [e for e in self._history if e['decision_t'] < current_t]
        if not entries:
            return None, None
        
        X = np.array([e['features'] for e in entries])
        y = np.array([e['label'] for e in entries])
        return X, y
    
    def predict(self, features: np.ndarray, current_t: int) -> Tuple[float, bool]:
        """
        Predict P(positive transfer | features) using history before current_t.
        
        Returns: (prob, is_fitted)
        prob: in [0, 1]
        is_fitted: True if model was used (not fallback)
        
        LEAKAGE CHECK: Only uses history with decision_t < current_t.
        """
        X_train, y_train = self._get_train_data_before(current_t)
        
        if X_train is None or len(X_train) < 50:
            self._n_unfitted += 1
            # Return historical positive rate
            if self._n_total_labels > 0:
                rate = self._n_positive_labels / self._n_total_labels
            else:
                rate = 0.5
            return float(rate), False
        
        n_pos = int(y_train.sum())
        n_classes = len(np.unique(y_train))
        
        if n_classes < 2 or n_pos < 3:
            self._n_unfitted += 1
            rate = n_pos / len(y_train) if len(y_train) > 0 else 0.5
            return float(rate), False
        
        # Fit scaler on training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        try:
            model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.seed,
                solver='lbfgs',
            )
            model.fit(X_scaled, y_train)
        except Exception:
            self._n_unfitted += 1
            return 0.5, False
        
        # Check if Platt scaling is available
        # Count OOS rows with both classes
        if len(self._oos_labels) >= 200:
            oos_labels_arr = np.array(self._oos_labels)
            if len(np.unique(oos_labels_arr)) >= 2 and not self._platt_fitted:
                try:
                    oos_scores_arr = np.array(self._oos_scores)
                    # Fit Platt scaling
                    from sklearn.calibration import _SigmoidCalibration
                    platt = _SigmoidCalibration()
                    platt.fit(oos_scores_arr.reshape(-1, 1), oos_labels_arr)
                    self._platt_model = platt
                    self._platt_fitted = True
                except Exception:
                    pass
        
        # Predict
        x_new = scaler.transform(np.array(features, dtype=np.float64).reshape(1, -1))
        
        try:
            classes_list = list(model.classes_)
            proba = model.predict_proba(x_new)[0]
            if 1 in classes_list:
                raw_prob = float(proba[classes_list.index(1)])
            else:
                raw_prob = float(proba[-1])
        except Exception:
            self._n_unfitted += 1
            return 0.5, False
        
        # Apply Platt scaling if available
        if self._platt_fitted and self._platt_model is not None:
            try:
                # Raw score from LR
                raw_score = model.decision_function(x_new)[0]
                cal_prob = self._platt_model.predict(np.array([[raw_score]]))[0]
                raw_prob = float(np.clip(cal_prob, 0.0, 1.0))
            except Exception:
                pass
        
        self._n_fitted += 1
        return float(np.clip(raw_prob, 0.0, 1.0)), True

    def predict_batch(self, features_matrix: np.ndarray, current_t: int) -> Tuple[np.ndarray, bool]:
        """
        Predict probabilities for all candidate rows at decision current_t simultaneously.
        Fits LogisticRegression ONCE for decision current_t, then predicts for all candidate rows.
        
        Returns: (probs_array, is_fitted)
        """
        n_cands = len(features_matrix)
        if n_cands == 0:
            return np.array([], dtype=float), False
            
        X_train, y_train = self._get_train_data_before(current_t)
        
        if X_train is None or len(X_train) < 50:
            self._n_unfitted += n_cands
            rate = self._n_positive_labels / max(1, self._n_total_labels) if self._n_total_labels > 0 else 0.5
            return np.full(n_cands, float(rate)), False
            
        n_pos = int(y_train.sum())
        n_classes = len(np.unique(y_train))
        
        if n_classes < 2 or n_pos < 3:
            self._n_unfitted += n_cands
            rate = n_pos / len(y_train) if len(y_train) > 0 else 0.5
            return np.full(n_cands, float(rate)), False
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        try:
            model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.seed,
                solver='lbfgs',
            )
            model.fit(X_scaled, y_train)
        except Exception:
            self._n_unfitted += n_cands
            return np.full(n_cands, 0.5), False
            
        # Check Platt scaling availability
        if len(self._oos_labels) >= 200:
            oos_labels_arr = np.array(self._oos_labels)
            if len(np.unique(oos_labels_arr)) >= 2 and not self._platt_fitted:
                try:
                    oos_scores_arr = np.array(self._oos_scores)
                    from sklearn.calibration import _SigmoidCalibration
                    platt = _SigmoidCalibration()
                    platt.fit(oos_scores_arr.reshape(-1, 1), oos_labels_arr)
                    self._platt_model = platt
                    self._platt_fitted = True
                except Exception:
                    pass
                    
        X_cands_scaled = scaler.transform(np.array(features_matrix, dtype=np.float64))
        classes_list = list(model.classes_)
        probas = model.predict_proba(X_cands_scaled)
        
        if 1 in classes_list:
            pos_idx = classes_list.index(1)
            raw_probs = probas[:, pos_idx]
        else:
            raw_probs = probas[:, -1]
            
        if self._platt_fitted and self._platt_model is not None:
            try:
                scores = model.decision_function(X_cands_scaled)
                cal_probs = self._platt_model.predict(scores.reshape(-1, 1))
                raw_probs = np.clip(cal_probs, 0.0, 1.0)
            except Exception:
                pass
                
        self._n_fitted += n_cands
        return np.clip(raw_probs, 0.0, 1.0), True
    
    def record_oos_score(self, score: float, label: int, decision_t: int, df1: float):
        """Record an out-of-sample score for AUROC and Platt computation."""
        self._oos_scores.append(float(score))
        self._oos_labels.append(int(label))
        self._oos_decision_ts.append(decision_t)
        self._oos_df1s.append(float(df1))
    
    def choose_tau_online(self, tau_history: List[Dict]) -> float:
        """
        Choose tau online: from grid pick value maximizing mean realized dF1 over
        earlier decisions where we transferred. Require >= 20 earlier decisions.
        Break ties toward larger tau. Add abstain-always as option.
        
        tau_history: list of {tau_tried, chosen_cand_df1, did_transfer} per prior decision
        Returns: chosen tau (inf = always abstain)
        """
        if len(tau_history) < 20:
            return TAU_ABSTAIN  # Abstain until enough history
        
        # For each tau in grid, compute mean dF1 on decisions where P_max >= tau (transfer)
        # vs abstain (use local, df1 = 0 vs local)
        # We want to find tau that maximizes mean realized dF1
        
        best_tau = TAU_ABSTAIN
        best_mean_df1 = -float('inf')
        
        # Option: abstain always => mean df1 = 0 (we use local)
        abstain_df1 = 0.0  # abstaining gives 0 gain over local
        
        # For each tau in grid (ascending, break ties toward larger)
        for tau in sorted(TAU_GRID, reverse=True):  # reverse so larger tau wins ties
            transferred_df1s = []
            for entry in tau_history:
                if entry.get('p_max', 0.0) >= tau:
                    # Would have transferred: df1 = chosen_cand_df1
                    transferred_df1s.append(entry.get('chosen_cand_df1', 0.0))
                else:
                    # Would have abstained: df1 = 0 (local is reference)
                    transferred_df1s.append(0.0)
            
            if not transferred_df1s:
                continue
            
            mean_df1 = float(np.mean(transferred_df1s))
            if mean_df1 > best_mean_df1:
                best_mean_df1 = mean_df1
                best_tau = tau
        
        # Check if abstain always is better
        if abstain_df1 > best_mean_df1:
            best_tau = TAU_ABSTAIN
        
        return best_tau
    
    def get_auroc(self, test_only_decisions: Optional[List[int]] = None) -> Optional[float]:
        """Compute AUROC on stored OOS scores. Returns None if insufficient data."""
        if test_only_decisions is not None:
            mask = [t in test_only_decisions for t in self._oos_decision_ts]
            scores = [s for s, m in zip(self._oos_scores, mask) if m]
            labels = [l for l, m in zip(self._oos_labels, mask) if m]
        else:
            scores = self._oos_scores
            labels = self._oos_labels
        
        if len(labels) < 10:
            return None
        
        score_arr = np.array(scores)
        label_arr = np.array(labels)
        
        # K5: Check score std > 1e-6
        if score_arr.std() < 1e-6:
            return None  # Dead estimator, not valid
        
        if len(np.unique(label_arr)) < 2:
            return None
        
        try:
            return float(roc_auc_score(label_arr, score_arr))
        except Exception:
            return None
    
    def get_positive_rate(self) -> float:
        """Historical positive label rate."""
        if self._n_total_labels == 0:
            return 0.0
        return self._n_positive_labels / self._n_total_labels
    
    def get_fit_stats(self) -> Dict:
        return {
            'n_fitted': self._n_fitted,
            'n_unfitted': self._n_unfitted,
            'n_total_labels': self._n_total_labels,
            'n_positive_labels': self._n_positive_labels,
            'n_oos_scores': len(self._oos_scores),
        }
