"""
Regime Representation Extractor for Experiment 2
Computes online compact feature representations from telemetry streams and model state.
Strictly zero-look-ahead: uses only available pre-transfer telemetry.
"""

import numpy as np
import pandas as pd
from scipy import stats

class RegimeRepresentationExtractor:
    """
    Extracts distribution statistics, temporal characteristics, correlation structure,
    and model-behavior metadata from a window/segment of telemetry.
    """
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def extract_telemetry_features(self, df_win):
        """
        Extracts distribution, temporal, and correlation features from telemetry dataframe slice.
        """
        rep = {}
        
        # 1. Distribution Statistics for each feature column
        for col in self.feature_cols:
            vals = df_win[col].values.astype(float)
            if len(vals) == 0:
                vals = np.array([0.0])
                
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            med_v = float(np.median(vals))
            p10_v = float(np.percentile(vals, 10))
            p25_v = float(np.percentile(vals, 25))
            p75_v = float(np.percentile(vals, 75))
            p90_v = float(np.percentile(vals, 90))
            p95_v = float(np.percentile(vals, 95))
            iqr_v = float(p75_v - p25_v)
            
            rep[f"{col}_mean"] = mean_v
            rep[f"{col}_std"] = std_v
            rep[f"{col}_median"] = med_v
            rep[f"{col}_p10"] = p10_v
            rep[f"{col}_p25"] = p25_v
            rep[f"{col}_p75"] = p75_v
            rep[f"{col}_p90"] = p90_v
            rep[f"{col}_p95"] = p95_v
            rep[f"{col}_iqr"] = iqr_v
            
            # 2. Temporal Characteristics
            # Lag-1 Autocorrelation
            if len(vals) > 1 and std_v > 1e-9:
                lag1_ac = float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
                if np.isnan(lag1_ac):
                    lag1_ac = 0.0
            else:
                lag1_ac = 0.0
            rep[f"{col}_lag1_ac"] = lag1_ac
            
            # Trend / Slope over sequence
            if len(vals) > 2:
                x_seq = np.arange(len(vals))
                slope, _, _, _, _ = stats.linregress(x_seq, vals)
                slope_v = float(slope) if not np.isnan(slope) else 0.0
            else:
                slope_v = 0.0
            rep[f"{col}_slope"] = slope_v
            
            # Volatility (coefficient of variation)
            rep[f"{col}_volatility"] = float(std_v / (abs(mean_v) + 1e-6))
            
        # 3. Correlation Structure between Telemetry Variables
        df_sub = df_win[self.feature_cols].copy()
        corr_matrix = df_sub.corr().fillna(0.0).values
        n_feats = len(self.feature_cols)
        corr_feats = []
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                corr_feats.append(float(corr_matrix[i, j]))
        rep['correlation_vector'] = corr_feats
        
        return rep

    def combine_with_model_metadata(self, telemetry_rep, model_state, window_id):
        """
        Appends model behavior metadata (historical performance, weights, age, disagreement).
        """
        full_rep = dict(telemetry_rep)
        
        full_rep['model_weights_rf'] = float(model_state.get('weights', [0.5, 0.5])[0])
        full_rep['model_weights_et'] = float(model_state.get('weights', [0.5, 0.5])[1])
        full_rep['historical_f1'] = float(model_state.get('historical_f1', 1.0))
        full_rep['historical_acc'] = float(model_state.get('historical_acc', 1.0))
        full_rep['successful_transfers'] = int(model_state.get('successful_transfers', 0))
        full_rep['failed_transfers'] = int(model_state.get('failed_transfers', 0))
        full_rep['observation_count'] = int(model_state.get('observation_count', 1))
        
        created_w = model_state.get('created_window', 0)
        full_rep['checkpoint_age'] = int(window_id - created_w)
        
        total_attempts = full_rep['successful_transfers'] + full_rep['failed_transfers']
        full_rep['transfer_success_rate'] = float(full_rep['successful_transfers'] / (total_attempts + 1e-6)) if total_attempts > 0 else 0.5
        
        full_rep['model_disagreement'] = float(model_state.get('disagreement', 0.0))
        full_rep['prediction_entropy'] = float(model_state.get('entropy', 0.0))
        
        return full_rep
