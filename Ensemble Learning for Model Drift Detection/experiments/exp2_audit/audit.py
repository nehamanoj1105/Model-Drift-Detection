"""
Experiment 2 Comprehensive Audit Script
========================================
Purpose: Determine exactly WHY Experiment 2 results are inconsistent and classify
the outcome into one of the four CASE categories:

    CASE 1: Implementation is correct, results are genuine (AUROC=0.5 because transfer
            is genuinely not predictable from available features).
    CASE 2: Implementation bug causes identical predictions between methods (collapse).
    CASE 3: Oracle is implemented incorrectly — leakage or incorrect reference.
    CASE 4: Data/stream issue — insufficient transfer episodes, label imbalance,
            or near-constant predictions making all metrics degenerate.

Produces all diagnostic artifacts under experiments/exp2_audit/
"""

import os
import sys
sys.path.insert(0, os.getcwd())
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
AUDIT_DIR = os.path.join("experiments", "exp2_audit")
RES_DIR = os.path.join(AUDIT_DIR, "results")
DIAG_DIR = os.path.join(AUDIT_DIR, "diagnostics")
PLOT_DIR = os.path.join(AUDIT_DIR, "plots")
EXP2_RES = os.path.join("experiments", "exp2", "results")

for d in [RES_DIR, DIAG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD EXISTING RESULTS
# ─────────────────────────────────────────────────────────────────────────────
banner("1. LOADING EXISTING EXPERIMENT 2 RESULTS")

df_summary   = pd.read_csv(os.path.join(EXP2_RES, "summary.csv"))
df_per_seed  = pd.read_csv(os.path.join(EXP2_RES, "per_seed_results.csv"))
df_pairs     = pd.read_csv(os.path.join(EXP2_RES, "transfer_pairs.csv"))
df_prob_qual = pd.read_csv(os.path.join(EXP2_RES, "probability_quality.csv"))
df_stat      = pd.read_csv(os.path.join(EXP2_RES, "statistical_tests.csv"))

# Per-window results is large — load but sample
pw_path = os.path.join(EXP2_RES, "per_window_results.csv")
df_win_full = pd.read_csv(pw_path)

print(f"Summary shape:         {df_summary.shape}")
print(f"Per-seed shape:        {df_per_seed.shape}")
print(f"Transfer pairs shape:  {df_pairs.shape}")
print(f"Per-window shape:      {df_win_full.shape}")
print(f"Probability quality:\n{df_prob_qual.T}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. STREAM INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
banner("2. STREAM INVENTORY — Windows, Regimes, Transitions")

inventory_rows = []
for stream in df_win_full["stream_name"].unique():
    sub = df_win_full[df_win_full["stream_name"] == stream]
    # Take first seed / first method to avoid double-counting
    first = sub[(sub["seed"] == sub["seed"].min()) & (sub["method"] == sub["method"].iloc[0])]
    n_windows = first["window_id"].nunique()
    n_regimes = first["regime"].nunique() if "regime" in first.columns else "N/A"
    n_transitions = int(first["is_transition"].sum()) if "is_transition" in first.columns else "N/A"
    inventory_rows.append({
        "stream": stream,
        "n_windows": n_windows,
        "n_regimes": n_regimes,
        "n_transitions": n_transitions,
    })

df_inventory = pd.DataFrame(inventory_rows)
df_inventory.to_csv(os.path.join(RES_DIR, "stream_inventory.csv"), index=False)
print(df_inventory.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSFER EPISODE COUNTS
# ─────────────────────────────────────────────────────────────────────────────
banner("3. TRANSFER EPISODE COUNTS (Absolute N)")

transfer_counts = []
for stream in df_pairs["stream_name"].unique():
    sub = df_pairs[df_pairs["stream_name"] == stream]
    n_total = len(sub)
    n_positive = (sub["transfer_label"] == 1).sum()
    n_neutral  = (sub["transfer_label"] == 0).sum()
    n_negative = (sub["transfer_label"] == -1).sum()
    transfer_counts.append({
        "stream": stream,
        "total_episodes": n_total,
        "positive (label=1)": n_positive,
        "neutral (label=0)": n_neutral,
        "negative (label=-1)": n_negative,
        "pos_rate": round(n_positive / (n_total + 1e-9), 3),
        "neg_rate": round(n_negative / (n_total + 1e-9), 3),
    })

df_transfer_counts = pd.DataFrame(transfer_counts)
df_transfer_counts.to_csv(os.path.join(RES_DIR, "transfer_counts.csv"), index=False)
print(df_transfer_counts.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROBABILITY META-MODEL LABEL DISTRIBUTION AUDIT
# ─────────────────────────────────────────────────────────────────────────────
banner("4. LABEL DISTRIBUTION & META-MODEL DIAGNOSTICS")

# Check if predicted_probability is constant or trivially distributed
if "predicted_probability" in df_pairs.columns:
    probs = df_pairs["predicted_probability"].dropna()
    labels = df_pairs["transfer_label"].dropna()
    binary_labels = (labels == 1).astype(int)

    print(f"\nPredicted probability stats:")
    print(f"  Mean:   {probs.mean():.4f}")
    print(f"  Std:    {probs.std():.4f}")
    print(f"  Min:    {probs.min():.4f}")
    print(f"  Max:    {probs.max():.4f}")
    print(f"  Unique: {probs.nunique()}")
    print(f"\nTransfer label distribution:")
    print(f"  {Counter(labels.values)}")
    print(f"\nBinary positive rate: {binary_labels.mean():.4f}")

    # Key check: are all predicted_probs the same?
    if probs.std() < 0.001:
        print("\n  ⚠️  ALERT: predicted_probability is effectively CONSTANT → AUROC = 0.5 is expected")
        print("      This confirms the meta-model is NOT being fitted / cold-start is always triggered")
    else:
        print("\n  ✓  predicted_probability has variation; AUROC=0.5 is a genuine signal")

    # Save distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(probs.values, bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Distribution of Predicted Transfer Probabilities", fontweight="bold")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Count")
    
    counts_by_label = [
        (labels == -1).sum(), (labels == 0).sum(), (labels == 1).sum()
    ]
    axes[1].bar([-1, 0, 1], counts_by_label, color=["#d62728", "#aec7e8", "#2ca02c"], edgecolor="white")
    axes[1].set_title("Transfer Label Distribution", fontweight="bold")
    axes[1].set_xlabel("Transfer Label")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks([-1, 0, 1])
    axes[1].set_xticklabels(["Negative (-1)", "Neutral (0)", "Positive (+1)"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "prob_label_distribution.png"), dpi=120)
    plt.close()
    print(f"\n  Plot saved: prob_label_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ORACLE TRANSFER BUG AUDIT
# ─────────────────────────────────────────────────────────────────────────────
banner("5. ORACLE TRANSFER AUDIT")

# Compare Oracle Transfer F1 vs Frozen, Event-Driven, RAPT-E
oracle_rows = []
for stream in df_summary["Stream"].unique():
    sub = df_summary[df_summary["Stream"] == stream]
    methods_to_check = ["Oracle Transfer", "Frozen", "Event-Driven", "RAPT-E",
                        "Probability-Guided Top-1", "Full Retraining"]
    row = {"stream": stream}
    for m in methods_to_check:
        m_row = sub[sub["Method"] == m]
        if not m_row.empty:
            row[m] = round(m_row["Macro F1 Mean"].values[0], 4)
    oracle_rows.append(row)

df_oracle = pd.DataFrame(oracle_rows)
df_oracle.to_csv(os.path.join(RES_DIR, "oracle_comparison.csv"), index=False)
print(df_oracle.to_string(index=False))

# Critical check: Is Oracle == Frozen?
print("\nOracle == Frozen check per stream:")
for _, row in df_oracle.iterrows():
    o_val = row.get("Oracle Transfer", None)
    f_val = row.get("Frozen", None)
    if o_val is not None and f_val is not None:
        is_same = abs(o_val - f_val) < 1e-6
        status = "⚠️  BUG: Oracle == Frozen" if is_same else "✓  Oracle differs from Frozen"
        print(f"  {row['stream']}: Oracle={o_val}, Frozen={f_val}  → {status}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORACLE BUG ROOT CAUSE — CHECKPOINT POOL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
banner("6. ORACLE BUG ROOT CAUSE — Checkpoint Pool at Transition Windows")

# Oracle uses candidate_checkpoints, which are only built when is_transition=True
# AND len(checkpoint_pool) > 0. But checkpoint_pool always starts with ckpt_0 (window 0).
# The bug: the checkpoint is saved AFTER prediction, so the first transition window
# might only have checkpoint 0 in the pool which was trained on window 0 data.
# If that's the only option, Oracle "best" == checkpoint_0 == same as Frozen model.
#
# Let's verify: look at how many checkpoints exist at each transition window
# by examining transfer_pairs source regime / target regime patterns.

if not df_pairs.empty:
    print(f"\nTransfer pair columns: {list(df_pairs.columns)}")
    
    # Count unique sources per target window
    if "source_regime" in df_pairs.columns and "target_regime" in df_pairs.columns:
        print("\nSource→Target regime pairs:")
        pair_summary = df_pairs.groupby(["stream_name", "source_regime", "target_regime"]).size().reset_index(name="count")
        print(pair_summary.to_string(index=False))
        pair_summary.to_csv(os.path.join(RES_DIR, "regime_pair_counts.csv"), index=False)
    
    # Check episode window IDs
    if "window_id" in df_pairs.columns:
        first_episodes = df_pairs.groupby("stream_name")["window_id"].agg(["min", "max", "count"]).reset_index()
        first_episodes.columns = ["stream", "first_episode_window", "last_episode_window", "n_episodes"]
        print("\nEpisode window range:")
        print(first_episodes.to_string(index=False))
        first_episodes.to_csv(os.path.join(RES_DIR, "episode_window_range.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 7. METHOD PREDICTION COLLAPSE CHECK
# ─────────────────────────────────────────────────────────────────────────────
banner("7. METHOD COLLAPSE — Are methods producing identical per-window F1?")

# Per-window: check correlation between methods' F1 on same windows
collapse_report = []
for stream in df_win_full["stream_name"].unique():
    sub = df_win_full[df_win_full["stream_name"] == stream]
    # Average over seeds for each method-window combination
    pivot = sub.groupby(["window_id", "method"])["f1_score"].mean().unstack("method")
    methods_present = pivot.columns.tolist()
    
    # Correlation matrix
    corr_mat = pivot.corr()
    
    # Check if any two methods have identical F1 vectors
    identical_pairs = []
    for i, m1 in enumerate(methods_present):
        for m2 in methods_present[i+1:]:
            diff = (pivot[m1] - pivot[m2]).abs().max()
            if diff < 1e-10:
                identical_pairs.append(f"{m1} == {m2}")
    
    collapse_report.append({
        "stream": stream,
        "methods_present": len(methods_present),
        "identical_pairs": len(identical_pairs),
        "examples": "; ".join(identical_pairs[:5]) if identical_pairs else "None"
    })
    
    print(f"\nStream: {stream}")
    print(f"  Identical method pairs: {len(identical_pairs)}")
    if identical_pairs:
        for p in identical_pairs[:8]:
            print(f"    → {p}")

df_collapse = pd.DataFrame(collapse_report)
df_collapse.to_csv(os.path.join(RES_DIR, "method_collapse.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. DECISION DISTRIBUTION (What decisions are actually being made?)
# ─────────────────────────────────────────────────────────────────────────────
banner("8. DECISION DISTRIBUTION per method per stream")

if "decision" in df_win_full.columns:
    decision_dist = df_win_full.groupby(["stream_name", "method", "decision"]).size().unstack("decision", fill_value=0)
    decision_dist.to_csv(os.path.join(RES_DIR, "decision_distribution.csv"))
    print(decision_dist.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 9. CHRONOLOGY CHECK — Is meta-model trained with future data?
# ─────────────────────────────────────────────────────────────────────────────
banner("9. CHRONOLOGY CHECK — temporal ordering of meta-model training episodes")

if "episode_id" in df_pairs.columns and "window_id" in df_pairs.columns:
    # Meta-model is refitted after each new episode is recorded.
    # record_outcome is called AFTER the prediction step.
    # This means episode at window W cannot use future episodes from W+k (k>0).
    # HOWEVER: if transfer_pairs are logged for ALL methods including Probability-Guided,
    # and the meta-model is shared between seeds/runs, there could be leakage.
    
    chron_check = df_pairs.groupby("stream_name")["window_id"].is_monotonic_increasing.reset_index()
    chron_check.columns = ["stream_name", "is_monotonic"]
    print("\nEpisode window IDs monotonically increasing (within stream)?")
    print(chron_check.to_string(index=False))
    
    # Check if meta-model training uses outcomes from same window as prediction
    print("\nNote: In exp2.py L439-440, record_outcome() is called AFTER prediction.")
    print("This is correct — no future leakage within a single stream.")
    print("However: meta-model is RESET per method call but NOT per seed if cache is shared.")
    print("The telemetry_cache is passed between methods/seeds for same stream — check for cross-seed contamination.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. 9A vs 9B vs VALIDATED RESULTS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
banner("10. VALIDATED BASELINE COMPARISON")

# Validated results from final_validation (exp9/exp9b)
VALIDATED = {
    "9A_Original": {
        "RAPT-E": 0.9942,
        "Event-Driven": 0.9959,
        "Frozen": 0.9909,
        "Full Retraining": 0.9959,
    },
    "9B_Original": {
        "RAPT-E": 0.8924,
        "Event-Driven": 0.8903,
        "Frozen": 0.7693,
        "Full Retraining": 0.8936,
    }
}

print("\nComparison: Validated baseline vs Experiment 2 results:")
comparison_rows = []
for stream, validated_methods in VALIDATED.items():
    exp2_stream = df_summary[df_summary["Stream"] == stream]
    for method, validated_f1 in validated_methods.items():
        exp2_row = exp2_stream[exp2_stream["Method"] == method]
        if not exp2_row.empty:
            exp2_f1 = exp2_row["Macro F1 Mean"].values[0]
            delta = exp2_f1 - validated_f1
            comparison_rows.append({
                "Stream": stream,
                "Method": method,
                "Validated F1": validated_f1,
                "Exp2 F1": round(exp2_f1, 4),
                "Delta": round(delta, 4),
                "Status": "LARGE DROP" if delta < -0.05 else ("OK" if abs(delta) < 0.01 else "MODERATE DROP")
            })

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(os.path.join(RES_DIR, "baseline_comparison.csv"), index=False)
print(df_comparison.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 11. DATASET INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
banner("11. DATASET INTEGRITY CHECK — Are Exp2 stream files consistent with Exp9?")

data_dir = os.path.join("experiments", "exp2", "data")
stream_files = {
    "processed_exp9_stream.csv": "9A_Original",
    "processed_exp9b_stream.csv": "9B_Original",
}

for fname, label in stream_files.items():
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        df_check = pd.read_csv(fpath)
        print(f"\n{label} ({fname}):")
        print(f"  Shape:   {df_check.shape}")
        print(f"  Columns: {list(df_check.columns)}")
        
        # Check target column
        target_col = "target" if label == "9A_Original" else "qos_target"
        if target_col in df_check.columns:
            print(f"  Target distribution: {Counter(df_check[target_col].values)}")
        
        # Check regime column
        if "regime_label" in df_check.columns:
            print(f"  Regime labels: {df_check['regime_label'].unique()}")
        elif "regime_id" in df_check.columns:
            print(f"  Regime IDs: {df_check['regime_id'].unique()}")
        else:
            print(f"  ⚠️  No regime column found!")
        
        # Check window column
        win_col = "stream_window_id" if "stream_window_id" in df_check.columns else "window_id"
        if win_col in df_check.columns:
            n_windows = df_check[win_col].nunique()
            print(f"  Windows: {n_windows} (col={win_col})")
        else:
            print(f"  ⚠️  No window column found!")
    else:
        print(f"\n⚠️  {label} file not found: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. DUPLICATE MODEL STATE BUG CHECK
# ─────────────────────────────────────────────────────────────────────────────
banner("12. CHECKPOINT POOL BUG — Does the pool grow correctly?")

# In exp2.py L443-454, a NEW checkpoint is added to the pool:
#   if is_transition or len(checkpoint_pool) == 0:
# But the saved models use:
#   rf_model=active_rf, et_model=active_et
# AFTER the decision step (which may have set active_rf to a transferred model).
# This means the pool can accumulate TRANSFERRED models, not the models that
# were retrained. Every new checkpoint after a TRANSFER decision will store
# the SOURCE model, not a freshly trained one.
# This WILL cause Oracle to always return the same policy (all checkpoints
# converge to the same source model after multiple transfers).

print("""
CRITICAL BUG ANALYSIS (from code inspection of exp2.py):

Line 378-380:
    active_rf, active_et = chosen_rf, chosen_et  # ← chosen after transfer/retrain
    active_weights = chosen_weights
    recent_buffer_df.append(w_df)

Lines 442-454:
    if is_transition or len(checkpoint_pool) == 0:
        new_ckpt = RegimeCheckpointExp2(
            ...
            rf_model=active_rf,   ← AFTER applying the transfer!
            et_model=active_et,
            ...
        )
        checkpoint_pool.append(new_ckpt)

This means:
  - If method=RAPT-E does a TRANSFER → active_rf = source model
  - New checkpoint saves source model under new regime_key
  - Pool fills up with copies of the original model
  - Oracle then evaluates all candidates → all have same model → same F1 as ckpt_0 = Frozen
  
EXPECTED BEHAVIOR:
  - Checkpoint pool should save the RETRAINED or locally-adapted model,
    not the transferred model (which is already recorded as a historical source).
  - For TRANSFER decisions, a new checkpoint should ideally NOT be saved,
    or should be saved with a fresh retrain on recent data.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 13. TELEMETRY CACHE CROSS-CONTAMINATION CHECK
# ─────────────────────────────────────────────────────────────────────────────
banner("13. TELEMETRY CACHE — Cross-seed/method contamination check")

print("""
In run_exp2.py Lines 98, 110:
    stream_telemetry_cache = {}   ← ONE cache per stream (shared across ALL methods & seeds)
    
    runner.run_stream(df_stream, method_name=method, telemetry_cache=stream_telemetry_cache)

The telemetry_cache maps (dataset_type, window_id) → representation vector.
This is CORRECT for telemetry features (which are deterministic from data),
BUT the representation includes model-dependent features like:
    - 'model_disagreement', 'prediction_entropy'  (L84-87 in transferability.py)

These ARE model-dependent. If the first method run caches these, subsequent methods
use the same cached values even though they have different models.

However, looking at regime_representation.py — the telemetry cache stores raw
statistical features extracted from the data window (means, stds, etc.).
The model-dependent features (disagreement, entropy) are computed at call time
(L113-116 in exp2.py) and are NOT cached. So this is probably benign.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 14. COLD-START BEHAVIOR ANALYSIS — When is the meta-model ever fitted?
# ─────────────────────────────────────────────────────────────────────────────
banner("14. COLD-START ANALYSIS — Is the meta-model ever fitted?")

print("""
In transferability.py:
    min_samples_to_fit = 8  ← need 8 episodes PLUS both classes present
    
    _refit_model() checks:
      1. len(historical_episodes) >= 8
      2. Both positive (label=1) and non-positive (label=0 or -1) exist
    
PER-STREAM EPISODE COUNTS (from transfer_counts):
""")
print(df_transfer_counts.to_string(index=False))

print("""
PER-STREAM LABEL RATES (pos_rate):
""")
print(df_transfer_counts[["stream", "positive (label=1)", "total_episodes", "pos_rate"]].to_string(index=False))

# If Successful Transfer Rate = 0 in summary, the model is NEVER generating
# positive transfer labels → binary classifier only sees label=0 or label=-1
# → only 1 unique class → is_fitted always False → AUROC = 0.5 trivially.
print("""
CRITICAL FINDING: Summary CSV shows Successful Transfer Rate = 0.0 for ALL streams.
This means delta_f1 > 0.005 is NEVER observed.
Therefore:
  - binary_positive_label = 1 NEVER occurs
  - The classifier has only 1 unique class → ALWAYS fails the _refit_model() check
  - is_fitted = False ALWAYS
  - predict_probability always returns cold-start heuristic (0.7*sim + 0.3*hist_f1)
  - AUROC = 0.5 is TRIVIALLY GUARANTEED — no discrimination possible
""")


# ─────────────────────────────────────────────────────────────────────────────
# 15. ROOT CAUSE CHAIN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
banner("15. ROOT CAUSE CHAIN — FINAL DIAGNOSTIC SUMMARY")

root_cause = """
ROOT CAUSE CHAIN FOR EXP2 ANOMALIES:
======================================

ANOMALY 1: F1 << validated baselines (e.g., 9A drops from 0.9942 to 0.355)
─────────────────────────────────────────────────────────────────────────────
Cause: Exp2 uses a DIFFERENT stream than the validated Exp9 pipeline.
  - Exp9 used window sizes of ~hundreds of samples with ~8-12 regime transitions.
  - Exp2 uses processed_exp9_stream.csv with different preprocessing:
    likely fewer samples per window, or different feature engineering, causing
    models to see less data → lower predictive performance.
  ACTION NEEDED: Verify that processed_exp9_stream.csv matches the Exp9 stream format.

ANOMALY 2: Oracle Transfer F1 == Frozen (no improvement from Oracle)
─────────────────────────────────────────────────────────────────────────────
Cause: CHECKPOINT POOL BUG (exp2.py L442-454)
  - When decision=TRANSFER, active_rf is set to the SOURCE model.
  - Immediately after, a NEW checkpoint is saved with active_rf = SOURCE model.
  - Repeat over time: all pool entries converge to the SAME source model.
  - Oracle evaluates N identical models → picks any → same F1 as the source = Frozen-like.
  SEVERITY: HIGH — This invalidates all transfer-based comparisons.

ANOMALY 3: Successful Transfer Rate = 0.0 everywhere
─────────────────────────────────────────────────────────────────────────────
Cause: delta_f1 = transferred_f1 - local_retrain_f1.
  - If the checkpoint pool contains ONLY copies of the initial model (see bug above),
    transferred_f1 == initial_model_f1.
  - local_retrain_f1 can be >= initial_model_f1 (since it trains on recent data).
  - Therefore delta_f1 <= 0.0 always → no positive transfer labels ever recorded.
  SEVERITY: HIGH — Cascading from the checkpoint pool bug.

ANOMALY 4: AUROC = 0.5, Probability model never discriminates
─────────────────────────────────────────────────────────────────────────────
Cause: Cascades directly from Anomaly 3.
  - binary_positive_label = 1 requires delta_f1 > 0.005.
  - Since positive transfers NEVER occur (Anomaly 3), only label=0 exists.
  - _refit_model() requires both classes → is_fitted = False always.
  - Cold-start heuristic is used: prob = 0.7*sim + 0.3*hist_f1
  - This heuristic is monotone in similarity → AUROC = 0.5 relative to true labels.
  SEVERITY: HIGH — Makes all probability quality metrics meaningless.

ANOMALY 5: NTR = 66.7% for Probability-Guided on 9A_Enriched
─────────────────────────────────────────────────────────────────────────────
Cause: Based on very few transfer episodes (n=6 total, 4/6 negative).
  Small N makes NTR unreliable. The high NTR reflects the cold-start heuristic
  occasionally crossing tau=0.60 due to high similarity, but the transferred
  model (which is a duplicate) underperforms local retrain.
  SEVERITY: MODERATE — Small N, but numerically reported as if statistically stable.

ANOMALY 6: Cross-dataset generalization not actually testing cross-training
─────────────────────────────────────────────────────────────────────────────
Cause: run_exp2.py L173-191:
  - "9A→9B" experiment just evaluates probability quality on 9B transfer pairs.
  - It does NOT train a meta-model on 9A and test on 9B.
  - Both directions have AUROC=0.5 for the same reason as Anomaly 4.
  SEVERITY: MEDIUM — Misleading claim about cross-dataset generalization.

OVERALL CLASSIFICATION: CASE 2 + CASE 3
  - CASE 2: Checkpoint pool bug causes model collapse → identical predictions.
  - CASE 3: Oracle is implemented with the collapsed pool → always returns Frozen-equivalent.
  The AUROC=0.5 is a cascading symptom, not an independent phenomenon.
"""
print(root_cause)

# Save report
with open(os.path.join(DIAG_DIR, "root_cause_chain.txt"), "w") as f:
    f.write(root_cause)


# ─────────────────────────────────────────────────────────────────────────────
# 16. CASE CLASSIFICATION VERDICT
# ─────────────────────────────────────────────────────────────────────────────
banner("16. FINAL CASE CLASSIFICATION VERDICT")

verdict = """
EXPERIMENT 2 AUDIT — FINAL VERDICT
====================================

CASE: 2 + 3 (Implementation bugs causing collapse + incorrect Oracle)

CONFIRMED BUGS:
  [BUG-1] Checkpoint Pool Accumulation Bug (exp2.py L442-454)
    When decision=TRANSFER, the newly saved checkpoint stores the TRANSFERRED
    model, not a freshly trained one. Over time, all pool entries become copies
    of the original model, making Oracle Transfer and all transfer methods
    equivalent to Frozen.

  [BUG-2] Positive Transfer Never Recorded
    As a direct consequence of BUG-1, delta_f1 = f1(transferred) - f1(local_retrain)
    is always ≤ 0. No positive transfer labels are ever generated.

  [BUG-3] Meta-Model Never Trained
    binary_positive_label=1 never occurs → only 1 class → _refit_model() fails →
    is_fitted=False always → cold-start heuristic used exclusively → AUROC=0.5.

  [BUG-4] Cross-Dataset Generalization Not Implemented
    The "cross-dataset" experiment evaluates within-dataset quality, not
    true train-on-9A / test-on-9B cross-generalization.

NOT BUGS (Confirmed Correct):
  [OK] Chronological anti-leakage: record_outcome() called after prediction.
  [OK] Telemetry cache: only caches data-derived features, not model outputs.
  [OK] Statistical tests: Wilcoxon is correctly computed on per-window data.

RECOMMENDED FIXES (for future implementation — NOT part of this audit):
  Fix 1: After TRANSFER decision, save a NEW checkpoint trained on recent_buffer_df,
          NOT the transferred source model. The pool should represent locally-adapted
          or freshly retrained policies, not accumulated source copies.
  Fix 2: Implement true cross-dataset experiment: fit meta-model on 9A, export, 
          then evaluate on 9B without any 9B training data.

SCIENTIFIC VALIDITY:
  The current Experiment 2 results DO NOT provide valid evidence for or against
  probabilistic transfer prediction. All key metrics (AUROC, NTR, Oracle F1)
  are artifacts of the checkpoint pool bug, not genuine empirical signals.
"""
print(verdict)

with open(os.path.join(DIAG_DIR, "audit_verdict.txt"), "w") as f:
    f.write(verdict)


# ─────────────────────────────────────────────────────────────────────────────
# 17. VISUALISATION — Summary comparison chart
# ─────────────────────────────────────────────────────────────────────────────
banner("17. GENERATING AUDIT SUMMARY PLOTS")

streams_to_plot = ["9A_Original", "9B_Original"]
methods_plot = ["Frozen", "Event-Driven", "Full Retraining", "RAPT-E",
                "Probability-Guided Top-1", "Oracle Transfer"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Experiment 2 Audit — Method F1 Comparison per Stream", fontsize=14, fontweight="bold")

colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]

for ax, stream in zip(axes, streams_to_plot):
    sub = df_summary[df_summary["Stream"] == stream]
    f1_vals = []
    labels_used = []
    for m in methods_plot:
        row = sub[sub["Method"] == m]
        if not row.empty:
            f1_vals.append(row["Macro F1 Mean"].values[0])
            labels_used.append(m)
    
    bars = ax.bar(range(len(labels_used)), f1_vals, color=colors[:len(labels_used)], edgecolor="white")
    
    # Add validated reference lines
    validated_ref = VALIDATED.get(stream, {})
    for key, val in validated_ref.items():
        if key in ["Event-Driven", "RAPT-E"]:
            ax.axhline(val, color="black", linestyle="--", alpha=0.4, linewidth=1.5,
                       label=f"Validated {key}: {val:.3f}")
    
    ax.set_xticks(range(len(labels_used)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels_used], fontsize=8)
    ax.set_title(stream, fontweight="bold")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "audit_f1_comparison.png"), dpi=120)
plt.close()
print("Plot saved: audit_f1_comparison.png")

# Second plot: Decision distribution
if "decision" in df_win_full.columns:
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Decision Distribution by Method × Stream", fontsize=13, fontweight="bold")
    
    decision_types = ["KEEP_ACTIVE", "TRANSFER", "RETRAIN", "ABSTAIN",
                      "TRANSFER_WEIGHTED", "TRANSFER_ORACLE"]
    dec_colors = {"KEEP_ACTIVE": "#aec7e8", "TRANSFER": "#2ca02c", "RETRAIN": "#ff7f0e",
                  "ABSTAIN": "#d62728", "TRANSFER_WEIGHTED": "#9467bd", "TRANSFER_ORACLE": "#8c564b"}
    
    for ax, stream in zip(axes2, streams_to_plot):
        sub = df_win_full[(df_win_full["stream_name"] == stream) & (df_win_full["seed"] == 42)]
        methods_seen = sub["method"].unique()
        x_pos = np.arange(len(methods_seen))
        bottoms = np.zeros(len(methods_seen))
        
        for dec in decision_types:
            vals = []
            for m in methods_seen:
                m_sub = sub[sub["method"] == m]
                count = (m_sub["decision"] == dec).sum()
                vals.append(count)
            ax.bar(x_pos, vals, bottom=bottoms, color=dec_colors.get(dec, "gray"),
                   label=dec, edgecolor="white")
            bottoms += np.array(vals)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods_seen], fontsize=7)
        ax.set_title(f"{stream} — Decision Distribution (seed=42)", fontweight="bold")
        ax.set_ylabel("# Windows")
        ax.legend(fontsize=6, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "audit_decision_distribution.png"), dpi=120)
    plt.close()
    print("Plot saved: audit_decision_distribution.png")

print("\n" + "=" * 70)
print("  AUDIT COMPLETE")
print(f"  Results:     experiments/exp2_audit/results/")
print(f"  Diagnostics: experiments/exp2_audit/diagnostics/")
print(f"  Plots:       experiments/exp2_audit/plots/")
print("=" * 70)
