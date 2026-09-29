# [SUPERSEDED / INVALIDATED] Experiment 2 Corrected -- Results Report

> [!CAUTION]
> **SUPERSEDED NOTICE:** This report and all results contained herein were **INVALIDATED** by the Phase 0 audit (`experiments/exp2_final/audit/phase0_findings.md`).
> Key flaws: (1) Post-run outcome recording, (2) Unfitted probability model collapsing to RAPT-E, (3) Oracle model locked to Frozen, (4) Similarity-Weighted collapsing to argmax.
> All results in this report MUST NOT be cited or used as scientific evidence. Refer to `experiments/exp2_final/` for the corrected study.

**Generated:** 2026-09-26 14:08:22
**Seeds:** [42, 43, 44]
**Streams evaluated:** ['9A_Original', '9A_Enriched', '9B_Original', '9B_Enriched']

---

## Executive Summary

Experiment 2 Corrected implements probabilistic transferability estimation for
RAPT-E regime policies, using a rolling-origin calibrated logistic regression.
All results are computed on the same validated streaming datasets as Experiment 1.

---

## Sanity Check Gates

| Stream | Check | Pass | Value | Expected Range |
|--------|-------|------|-------|----------------|
| 9A_Original | check_Frozen_reproduction | PASS | 0.9937 | (0.99, 0.998) |
| 9A_Original | check_Event_Driven_reproduction | PASS | 0.9958 | (0.99, 0.999) |
| 9A_Original | check_RAPT_E_reproduction | PASS | 0.9925 | (0.99, 0.998) |
| 9A_Original | check_5_transfer_label_diversity | FAIL FAIL | CASE D: insufficient diversity |  |
| 9A_Original | check_9_pool_diversity | PASS | OK |  |
| 9A_Enriched | check_Frozen_reproduction | PASS | 0.9967 | (0.99, 0.998) |
| 9A_Enriched | check_Event_Driven_reproduction | PASS | 0.9979 | (0.99, 0.999) |
| 9A_Enriched | check_RAPT_E_reproduction | PASS | 0.9964 | (0.99, 0.998) |
| 9A_Enriched | check_5_transfer_label_diversity | FAIL FAIL | CASE D: insufficient diversity |  |
| 9A_Enriched | check_9_pool_diversity | PASS | OK |  |
| 9B_Original | check_Frozen_reproduction | PASS | 0.9042 | (0.85, 0.93) |
| 9B_Original | check_Event_Driven_reproduction | PASS | 0.9042 | (0.85, 0.93) |
| 9B_Original | check_RAPT_E_reproduction | FAIL FAIL | 0.9350 | (0.85, 0.93) |
| 9B_Original | check_5_transfer_label_diversity | PASS | OK |  |
| 9B_Original | check_9_pool_diversity | PASS | OK |  |
| 9B_Enriched | check_Frozen_reproduction | PASS | 0.9162 | (0.85, 0.93) |
| 9B_Enriched | check_Event_Driven_reproduction | PASS | 0.9116 | (0.85, 0.93) |
| 9B_Enriched | check_RAPT_E_reproduction | FAIL FAIL | 0.9370 | (0.85, 0.93) |
| 9B_Enriched | check_5_transfer_label_diversity | PASS | OK |  |
| 9B_Enriched | check_9_pool_diversity | PASS | OK |  |

## Results -- 9A_Original

| Method | F1 Mean | F1 Std | CI95 | Adapt CPU (s) |
|--------|---------|--------|------|---------------|
| Event-Driven              | 0.9958 | 0.0010 | [0.9950, 0.9968] | 1.40 |
| Random-Historical         | 0.9940 | 0.0007 | [0.9932, 0.9944] | 0.00 |
| Frozen                    | 0.9937 | 0.0011 | [0.9926, 0.9944] | 0.00 |
| Similarity-Weighted       | 0.9937 | 0.0011 | [0.9926, 0.9944] | 0.00 |
| Similarity-Only           | 0.9937 | 0.0011 | [0.9926, 0.9944] | 0.00 |
| Oracle                    | 0.9937 | 0.0011 | [0.9926, 0.9944] | 0.00 |
| Hist-Reliability          | 0.9935 | 0.0004 | [0.9932, 0.9937] | 0.00 |
| RAPT-E                    | 0.9925 | 0.0006 | [0.9919, 0.9931] | 0.35 |
| Probability-Guided        | 0.9925 | 0.0006 | [0.9919, 0.9931] | 0.00 |

### Probability Model Quality

| Metric | Mean | Std |
|--------|------|-----|
| AUROC  | nan | nan |
| Brier  | nan | nan |
| Eval episodes | 0.0 | -- |

## Results -- 9A_Enriched

| Method | F1 Mean | F1 Std | CI95 | Adapt CPU (s) |
|--------|---------|--------|------|---------------|
| Event-Driven              | 0.9979 | 0.0004 | [0.9975, 0.9982] | 2.59 |
| Random-Historical         | 0.9969 | 0.0006 | [0.9962, 0.9973] | 0.00 |
| Hist-Reliability          | 0.9967 | 0.0003 | [0.9964, 0.9969] | 0.00 |
| Similarity-Weighted       | 0.9967 | 0.0007 | [0.9963, 0.9974] | 0.00 |
| Similarity-Only           | 0.9967 | 0.0007 | [0.9963, 0.9974] | 0.00 |
| Frozen                    | 0.9967 | 0.0008 | [0.9960, 0.9975] | 0.00 |
| Oracle                    | 0.9967 | 0.0008 | [0.9960, 0.9975] | 0.00 |
| RAPT-E                    | 0.9964 | 0.0003 | [0.9962, 0.9967] | 0.30 |
| Probability-Guided        | 0.9964 | 0.0003 | [0.9962, 0.9967] | 0.00 |

### Probability Model Quality

| Metric | Mean | Std |
|--------|------|-----|
| AUROC  | nan | nan |
| Brier  | nan | nan |
| Eval episodes | 0.0 | -- |

## Results -- 9B_Original

| Method | F1 Mean | F1 Std | CI95 | Adapt CPU (s) |
|--------|---------|--------|------|---------------|
| RAPT-E                    | 0.9350 | 0.0043 | [0.9325, 0.9396] | 0.53 |
| Probability-Guided        | 0.9350 | 0.0043 | [0.9325, 0.9396] | 0.00 |
| Similarity-Weighted       | 0.9067 | 0.0038 | [0.9027, 0.9099] | 0.00 |
| Similarity-Only           | 0.9067 | 0.0038 | [0.9027, 0.9099] | 0.00 |
| Hist-Reliability          | 0.9058 | 0.0063 | [0.9002, 0.9121] | 0.00 |
| Frozen                    | 0.9042 | 0.0029 | [0.9025, 0.9073] | 0.00 |
| Event-Driven              | 0.9042 | 0.0076 | [0.8977, 0.9120] | 2.26 |
| Oracle                    | 0.9042 | 0.0029 | [0.9025, 0.9073] | 0.00 |
| Random-Historical         | 0.9008 | 0.0080 | [0.8951, 0.9094] | 0.00 |

### Probability Model Quality

| Metric | Mean | Std |
|--------|------|-----|
| AUROC  | nan | nan |
| Brier  | nan | nan |
| Eval episodes | 0.0 | -- |

## Results -- 9B_Enriched

| Method | F1 Mean | F1 Std | CI95 | Adapt CPU (s) |
|--------|---------|--------|------|---------------|
| RAPT-E                    | 0.9370 | 0.0070 | [0.9308, 0.9440] | 0.84 |
| Probability-Guided        | 0.9370 | 0.0070 | [0.9308, 0.9440] | 0.00 |
| Frozen                    | 0.9162 | 0.0098 | [0.9074, 0.9258] | 0.00 |
| Oracle                    | 0.9162 | 0.0098 | [0.9074, 0.9258] | 0.00 |
| Similarity-Only           | 0.9162 | 0.0063 | [0.9101, 0.9219] | 0.00 |
| Similarity-Weighted       | 0.9162 | 0.0063 | [0.9101, 0.9219] | 0.00 |
| Random-Historical         | 0.9134 | 0.0089 | [0.9071, 0.9229] | 0.00 |
| Event-Driven              | 0.9116 | 0.0035 | [0.9085, 0.9151] | 3.09 |
| Hist-Reliability          | 0.9116 | 0.0063 | [0.9058, 0.9177] | 0.00 |

### Probability Model Quality

| Metric | Mean | Std |
|--------|------|-----|
| AUROC  | nan | nan |
| Brier  | nan | nan |
| Eval episodes | 0.0 | -- |

---

## Scientific Validity

> - No label leakage: step 9 (outcome) strictly precedes step 10 (history update).
> - Checkpoint pool invariant: target checkpoints trained on target data only.
> - Rolling-origin: probability model at episode k trained on episodes 0..k-1.
> - Oracle lower bounded by similarity-only (verified per episode).
