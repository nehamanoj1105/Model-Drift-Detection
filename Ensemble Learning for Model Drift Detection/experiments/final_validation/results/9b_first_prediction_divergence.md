# 9B First Prediction Divergence Analysis

## 1. Divergence Location
- **Window Index**: `315`
- **Window ID**: `315`
- **Regime ID**: `regime_B`
- **True Label**: `1`
- **Ground Truth Prediction (exp9b)**: `1`
- **Buggy Final Validation Prediction**: `0`

## 2. Model Repository State at Divergence
- **Ground Truth Repository Regimes**: `['regime_A', 'regime_B', 'regime_C', 'regime_D']`
- **Buggy Final Validation Repository Regimes**: `['regime_A', 'regime_C', 'regime_D', 'regime_B']`

## 3. Root Cause Analysis
In `final_validation.py`, line 159 initialized `prev_regime` to `df_stream.iloc[n_init]['regime_id']` (`regime_B`), causing the pipeline to miss the transition from `regime_A` to `regime_B` at streaming index `99` (window 99).
As a result, `regime_B` was never stored in the RAPT policy repository during initial streaming. When `regime_B` recurred at window 300, ground truth reused the stored policy (achieving Macro F1 ~0.889447), whereas buggy `final_validation.py` treated `regime_B` as unobserved and fit a new policy from scratch (yielding degraded Macro F1 ~0.8699).
