"""
Dynamic Oracle Transfer implementation for exp2_final.
Evaluates all available historical candidate checkpoints per episode on true target labels
and selects the true empirical upper bound checkpoint per window/episode.
"""

import numpy as np
from sklearn.metrics import f1_score

class DynamicOracleSelector:
    """
    Dynamic Oracle model tracker that selects the candidate checkpoint producing
    the highest Macro F1 score on target evaluation data.
    """
    def __init__(self):
        self.best_checkpoint = None

    def evaluate_and_select(self, candidates: list, X_eval: np.ndarray, y_eval: np.ndarray, local_model=None):
        """
        Evaluates local model and all candidate models on target evaluation data (X_eval, y_eval).
        Selects the single best checkpoint.
        """
        best_f1 = -1.0
        best_cand = local_model
        
        models_to_test = list(candidates)
        if local_model is not None:
            models_to_test.append(local_model)
            
        for cand in models_to_test:
            y_pred = cand.predict(X_eval)
            score = f1_score(y_eval, y_pred, average='macro', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_cand = cand
                
        self.best_checkpoint = best_cand
        return self.best_checkpoint, best_f1
