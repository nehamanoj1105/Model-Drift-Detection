# ENSEMBLE WEIGHT SENSITIVITY TEST

**Test Dataset**: Windows 200..399 (200 test samples)
**RF Only [1.0, 0.0] vs ET Only [0.0, 1.0] Disagreement Rate**: 6.50%
**RF Only [1.0, 0.0] vs Balanced [0.5, 0.5] Disagreement Rate**: 0.50%

## Conclusion:
Ensemble weights **DO** mathematically affect predictions when RF and ET predictions disagree.
In RAPT-v2, the reason RAPT-A, B, C, D, E produced identical F1 to RAPT-v1 was because the weight adaptation logic resulted in `[0.5, 0.5]` due to equal performance on the tiny 50-sample buffer, preventing any weight movement from occurring.
