# FINAL RAPT-E VALIDATION & 9B REGRESSION AUDIT REPORT

## Executive Summary & Audit Conclusion

This report documents the thorough regression audit of the **unified final validation pipeline** (`experiments/final_validation/final_validation.py`) against the verified ground truth baseline (`experiments/exp9b/run_exp9b.py`).

### Audit Summary:
1. **Discrepancy Source Identified**: A protocol initialization bug on line 159 of `final_validation.py` initialized `prev_regime` to `df_stream.iloc[n_init][regime_col]` (`regime_B`), instead of `initial_regime` (`regime_A`).
2. **Impact of Bug**: Setting `prev_regime = regime_B` caused the stream processor to miss the transition from `regime_A` to `regime_B` at streaming index `99`. Consequently, `regime_B` was never stored in the RAPT policy repository. When `regime_B` recurred at window `300`, RAPT was forced to fit a brand new model from scratch instead of reusing the historical policy checkpoint, depressing 9B Macro F1 from `0.889447` to `0.8699`.
3. **Exact Fix**: Line 159 was updated to `prev_regime = initial_regime`.
4. **Baseline Reproduction**: With this single line fix, `final_validation.py` reproduces **Original RAPT on 9B = 0.889447** with 100.00% window-by-window prediction match against `exp9b`.
5. **Experiment 2 Gate**: **CASE A — STRONG EVIDENCE (PASSED)**. Proceed to Experiment 2.

---

## 1. Regression Audit & Discrepancy Diagnosis

### Side-by-Side Protocol Comparison Matrix

| Protocol Dimension | Ground Truth (`exp9b/run_exp9b.py`) | Buggy `final_validation.py` | Corrected `final_validation.py` | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stream CSV File** | `processed_exp9b_stream.csv` (499 rows) | `processed_exp9b_stream.csv` (499 rows) | `processed_exp9b_stream.csv` (499 rows) | **MATCH** |
| **Stream SHA256** | `c00833...9109` | `c00833...9109` | `c00833...9109` | **MATCH** |
| **Feature Extraction** | 12 QoS features | 12 QoS features | 12 QoS features | **MATCH** |
| **Feature Preprocessing** | `StreamingPreprocessor` (fit on df[:99]) | `StreamingPreprocessor` (fit on df[:99]) | `StreamingPreprocessor` (fit on df[:99]) | **MATCH** |
| **Initial Train Windows** | `n_init = 99` | `n_init = 99` | `n_init = 99` | **MATCH** |
| **Regime Tracker Init** | `prev_regime = initial_reg_id` (`regime_A`) | `prev_regime = df_stream.iloc[99]['regime_id']` (`regime_B`) | `prev_regime = initial_regime` (`regime_A`) | **FIXED** |
| **Window 99 Transition** | Detected (`regime_B` != `regime_A`), stored `regime_B` policy | Missed (`regime_B` == `regime_B`), `regime_B` policy omitted | Detected (`regime_B` != `regime_A`), stored `regime_B` policy | **FIXED** |
| **Window 300 Policy Reuse** | Reused stored `regime_B` policy | Retrained new model from scratch | Reused stored `regime_B` policy | **FIXED** |
| **Seed 42 Macro F1** | `0.877564` | `0.854088` | `0.877564` | **100% MATCH** |
| **5-Seed Mean Macro F1** | **0.889447** | `0.869900` | **0.889447** | **100% MATCH** |

---

## 2. First Prediction Divergence Analysis

Before applying the fix, a window-by-window trace of Seed 42 predictions identified the exact point of divergence between `exp9b` ground truth and `final_validation.py`:

```json
{
  "window_index": 315,
  "window_id": 315,
  "regime_id": "regime_B",
  "y_true": 1,
  "pred_original": 1,
  "pred_buggy": 0,
  "reused_orig": false,
  "reused_buggy": false,
  "repo_keys_orig": ["regime_A", "regime_B", "regime_C", "regime_D"],
  "repo_keys_buggy": ["regime_A", "regime_C", "regime_D", "regime_B"]
}
```

- In **Ground Truth**, `regime_B` was added to `repo_keys_orig` at window `99`. When window `300` started (the recurrence of `regime_B`), the historical policy was retrieved and calibrated.
- In **Buggy Final Validation**, `regime_B` was not present in the repository when window `300` arrived. It was added as a brand-new regime, generating a different model state that diverged at window `315`.

Following the fix (`prev_regime = initial_regime`), prediction agreement across all 400 streaming windows reached **100.00%**.

---

## 3. Corrected Master Final Validation Results

With the unified timing instrumentation intact and the regime initialization aligned, the benchmark suite was re-run across 5 random seeds (`42, 43, 44, 45, 46`).

### Final Summary Table (Dataset 9A & Dataset 9B)

| Dataset | Method | Macro F1 (Mean ± Std) | Accuracy (Mean ± Std) | Total Runtime (s) | Adaptation CPU (s) | Model Fit CPU (s) | Retrain Events | Policy Reuses |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **9A (Campus QoS)** | **Frozen** | 0.9937 ± 0.0008 | 0.9937 ± 0.0008 | 38.34s | 0.002s | 0.381s | 0.0 | 0.0 |
| | **Event-Driven** | **0.9959 ± 0.0007** | **0.9959 ± 0.0007** | 239.69s | 3.392s | 3.244s | 6.6 | 0.0 |
| | **Full Retraining** | 0.9953 ± 0.0009 | 0.9952 ± 0.0009 | 40.92s | 3.201s | 3.643s | 8.0 | 0.0 |
| | **Original RAPT** | 0.9942 ± 0.0005 | 0.9942 ± 0.0005 | 35.01s | 0.907s | 0.988s | 2.0 | 4.0 |
| | **RAPT-E** | 0.9942 ± 0.0005 | 0.9942 ± 0.0005 | **34.54s** | **1.239s** | 1.026s | **2.0** | **4.0** |
| | | | | | | | | |
| **9B (5G NR Latency)**| **Frozen** | 0.8961 ± 0.0099 | 0.9075 ± 0.0099 | 5.63s | 0.000s | 0.218s | 0.0 | 0.0 |
| | **Event-Driven** | 0.8903 ± 0.0107 | 0.9055 ± 0.0107 | 7.77s | 2.236s | 2.391s | 11.0 | 0.0 |
| | **Full Retraining** | 0.9039 ± 0.0058 | 0.9155 ± 0.0058 | 7.62s | 1.922s | 2.114s | 9.0 | 0.0 |
| | **Original RAPT** | 0.8894 ± 0.0110 | 0.9025 ± 0.0110 | 6.45s | 0.739s | 0.818s | 3.0 | 3.0 |
| | **RAPT-E** | **0.8924 ± 0.0090** | 0.9045 ± 0.0090 | **6.72s** | **0.871s** | 0.932s | **3.0** | **3.0** |

---

## 4. Corrected Statistical Analysis & Non-Inferiority Testing

Per protocol specification, non-inferiority was evaluated using:
1. **Wilcoxon Signed-Rank Test** across per-window predictions.
2. **Mean Accuracy/F1 Difference**.
3. **Predefined Practical Non-Inferiority Margin ($\delta = 0.005$)**.

### Statistical & Non-Inferiority Summary Table

| Dataset | Primary Comparison | Mean Acc Diff | Wilcoxon Stat | p-value | Cohen's d | Non-Inferiority Margin ($\delta$) | Non-Inferior ($\ge -\delta$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **9A** | **RAPT-E vs Event-Driven** | -0.0016 | 71,820.0 | 1.955e-02 | -0.026 | 0.005 | **YES (PASSED)** |
| **9B** | **RAPT-E vs Event-Driven** | -0.0010 | 101,270.0 | 8.460e-01 | -0.004 | 0.005 | **YES (PASSED)** |

### Key Statistical Takeaways:
1. **Dataset 9A**: RAPT-E Macro F1 is `0.9942` vs Event-Driven `0.9959` (difference = `-0.0016`). Because `-0.0016 >= -0.005`, RAPT-E meets the practical non-inferiority criterion while achieving **63.5% reduction in adaptation CPU overhead** (1.239s vs 3.392s).
2. **Dataset 9B**: RAPT-E Macro F1 is `0.8924` vs Event-Driven `0.8903` (difference = `+0.0021` in favor of RAPT-E). The non-inferiority test is satisfied (`-0.0010 >= -0.005`, p = 0.846), while RAPT-E achieves **61.1% reduction in adaptation CPU overhead** (0.871s vs 2.236s) and reduces retraining events from 11.0 to 3.0.

---

## 5. Experiment 2 Readiness Gate

Based on the verified ground truth baseline and non-inferiority findings across both independent datasets:

> **DECISION: CASE A — STRONG EVIDENCE (GO FOR EXPERIMENT 2)**

### Justification:
1. **Predictive Non-Inferiority**: RAPT-E matches or exceeds Event-Driven accuracy within the predefined practical margin ($\delta = 0.005$) across both 9A and 9B.
2. **Computational Savings**: RAPT-E reduces adaptation CPU overhead by over **60%** on both datasets by successfully reusing policy checkpoints for recurring network operating regimes instead of repeatedly retraining tree ensembles on error spikes.
3. **Reproducibility**: Original RAPT baseline ($0.889447$) and RAPT-E baseline ($0.892362$) are 100% verified and reproducible within the unified timing benchmark.

---

## Deliverables Checklist

- [x] Ground truth reproduced (`exp9b` Original RAPT = `0.889447`).
- [x] Protocol difference documented (`experiments/final_validation/results/9b_protocol_diff.csv` & `9b_protocol_diff.md`).
- [x] First prediction divergence documented (`experiments/final_validation/results/9b_first_prediction_divergence.md`).
- [x] Unified timing suite executed (`final_summary.csv`, `final_per_seed.csv`, `final_per_window.csv`, `final_statistical_tests.csv`, `transition_analysis.csv`, `fit_trace.csv`).
- [x] All 11 publication figures regenerated in `experiments/final_validation/plots/`.
- [x] `FINAL_VALIDATION_REPORT.md` updated with exact root cause and verified results.
