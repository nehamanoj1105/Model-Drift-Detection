# FIRST DIVERGENCE DIAGNOSTIC REPORT

**First Divergent Window**: Window 305
**Regime at Divergence**: B
**True Label**: 2
**Original RAPT Prediction**: 2 (Correct: 1)
**RAPT-v2 Prediction**: 1 (Correct: 0)

## Root Causes Identified:

1. **Prefix Window Shift (`n_init`)**:
   - Original RAPT initialized on windows `0..199` (Regime A only, 200 windows).
   - RAPT-v2 initialized on windows `0..299` (300 windows), blurring Regime A and Regime B into Checkpoint A.
   - RAPT-v2 completely missed the first regime transition at window 200.

2. **Checkpoint Buffer Truncation (`X_buffer`)**:
   - Original RAPT trained new regime checkpoints on `hist_X[-500:]` (500 samples).
   - RAPT-v2 trained new regime checkpoints on `hist_X[-50:]` (50 samples), resulting in severely undertrained models.

3. **Weight Trajectory Discrepancy**:
   - Original RAPT weights at divergence: RF=0.5000, ET=0.5000
   - RAPT-v2 weights at divergence: RF=0.5000, ET=0.5000
