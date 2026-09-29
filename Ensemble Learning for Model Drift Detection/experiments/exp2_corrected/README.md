# Experiment 2 Corrected — README

## Overview

This directory contains a clean, scientifically valid implementation of Experiment 2:
**Probabilistic Regime Transfer** for RAPT-E on 5G streaming datasets.

It was created after the Experiment 2 Audit (`experiments/exp2_audit/`) identified four
confirmed bugs in the original `experiments/exp2/` implementation:

| Bug | Description |
|-----|-------------|
| B1 | 1-row-per-window preprocessing (loaded malformed CSV) |
| B2 | Checkpoint pool contamination (source model saved as target checkpoint) |
| B3 | No positive transfer labels (probability model collapsed to cold-start) |
| B4 | Oracle implementation invalid (Oracle < Similarity-Only in some episodes) |

## Files

| File | Purpose |
|------|---------|
| `stream_builder.py` | Rebuilds 9A/9B streams from raw CSVs using validated Exp1 pipelines |
| `checkpoint_manager.py` | Checkpoint pool with strict target-trained-only invariant |
| `features.py` | Source-target meta-features (distribution, temporal, correlation, policy) |
| `prob_model.py` | Online rolling-origin calibrated logistic regression |
| `transfer_episodes.py` | Transfer episode construction, oracle, baseline per-target |
| `exp2_corrected.py` | Main streaming loop (correct 11-step chronology) |
| `evaluation.py` | Aggregation, NTR, Wilcoxon, probability quality, sanity gates |
| `plots.py` | 13 publication figures |
| `run_exp2_corrected.py` | Orchestration CLI entry point |

## Correctness Guarantees

1. **No label leakage**: step 9 (observe outcome) strictly precedes step 10 (update history).
2. **Checkpoint pool invariant**: `add_checkpoint()` ALWAYS trains on target data;
   the source model from a transfer is never saved as the target checkpoint.
3. **Rolling-origin**: probability model at episode `k` is trained exclusively on
   episodes `0..k-1`. Enforced by assertion.
4. **Oracle invariant**: Oracle F1 ≥ Similarity-Only F1 per episode (asserted at runtime).
5. **Baseline gate**: Script exits with code 1 if Frozen / Event-Driven / RAPT-E F1
   lies outside the validated Exp1 ranges before running any Exp2 analysis.

## Running

```bash
cd "experiments/exp2_corrected"

python run_exp2_corrected.py \
    --data_9a "../../data/ntnu_owd_Packets_with_IATs.csv" \
    --data_9b_dir "../../data/zenodo_5g_nr/" \
    --seeds 42 43 44 \
    --streams 9A_Original 9A_Enriched 9B_Original 9B_Enriched
```

### Minimal test (fast, 1 seed, 1 stream)

```bash
python run_exp2_corrected.py \
    --data_9a "../../data/ntnu_owd_Packets_with_IATs.csv" \
    --data_9b_dir "../../data/zenodo_5g_nr/" \
    --seeds 42 \
    --streams 9A_Original
```

## Outputs

| Path | Contents |
|------|---------|
| `results/<stream>_summary.csv` | Per-method mean F1, std, CI95, CPU |
| `results/<stream>_candidate_records.csv` | Per-candidate transfer records |
| `results/<stream>_ntr.csv` | Negative transfer rate analysis |
| `results/<stream>_prob_quality.csv` | AUROC, AUPRC, Brier, ECE |
| `results/<stream>_wilcoxon.csv` | Paired Wilcoxon tests |
| `results/<stream>_oracle_comparison.csv` | Oracle vs methods |
| `results/all_streams_summary.csv` | Combined cross-stream table |
| `plots/` | 13 publication figures per stream |
| `EXP2_CORRECTED_REPORT.md` | Full formatted report |

## Methods Compared

| Method | Description |
|--------|-------------|
| Frozen | No adaptation after initial training |
| Event-Driven | Error-spike triggered retraining (Exp1 baseline) |
| RAPT-E | Historical policy reuse (Exp1 baseline) |
| Similarity-Only | Transfer to most distributionally similar source |
| Similarity-Weighted | Similarity-weighted transfer |
| Hist-Reliability | Transfer to historically most reliable source |
| Random-Historical | Uniform random source selection |
| **Probability-Guided** | **Transfer only when P(positive) ≥ τ = 0.60** |
| Oracle | Best actual transfer (true upper bound) |

## Scientific Protocol

The online decision chronology (per window):

```
1. Observe unlabeled telemetry X_w
2. Build target representation
3. Identify candidates: pool.get_available_for_target(before_episode=k)
4. Compute distributional features (source vs target)
5. Probability model predicts P(positive | features)   [trained on k-1 episodes]
6. Decide: transfer if P ≥ τ, else use RAPT-E local
7. Make predictions with chosen policy
8. Observe labels y_w
9. Compute delta_f1 = transfer_f1 - baseline_f1       [ONLY NOW]
10. Update transferability history                      [ONLY NOW]
11. Build target checkpoint after >= 50 target windows  [target data only]
```
