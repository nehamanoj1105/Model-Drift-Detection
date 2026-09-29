"""
Evaluation Engine for Experiment 2
Computes comprehensive probability calibration statistics (AUROC, AUPRC, Brier Score, ECE),
Negative Transfer Rates (NTR), Oracle Headroom, and Paired Wilcoxon Signed-Rank Hypothesis Tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for probability predictions.
    """
    probs = np.clip(np.array(probs), 0.0, 1.0)
    labels = np.array(labels, dtype=int)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)
    
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Elements in bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
            
    return float(ece)

def evaluate_probability_quality(df_transfer_pairs):
    """
    Evaluates discrimination and calibration of similarity score vs probability model.
    """
    if df_transfer_pairs.empty or len(df_transfer_pairs) < 3:
        return {
            'sim_auroc': 0.5, 'sim_auprc': 0.5, 'sim_brier': 0.25, 'sim_ece': 0.0,
            'prob_auroc': 0.5, 'prob_auprc': 0.5, 'prob_brier': 0.25, 'prob_ece': 0.0
        }

    # Binary ground truth: +1 transfer vs non-positive (0 or -1)
    y_true = (df_transfer_pairs['transfer_label'] > 0).astype(int).values
    sim_scores = df_transfer_pairs['combined_similarity'].values
    prob_preds = df_transfer_pairs['predicted_probability'].values

    # Check if multiple classes present
    if len(np.unique(y_true)) < 2:
        return {
            'sim_auroc': 0.5, 'sim_auprc': 0.5, 'sim_brier': float(brier_score_loss(y_true, sim_scores)), 'sim_ece': compute_ece(sim_scores, y_true),
            'prob_auroc': 0.5, 'prob_auprc': 0.5, 'prob_brier': float(brier_score_loss(y_true, prob_preds)), 'prob_ece': compute_ece(prob_preds, y_true)
        }

    # Similarity metrics
    sim_auroc = float(roc_auc_score(y_true, sim_scores))
    p_sim, r_sim, _ = precision_recall_curve(y_true, sim_scores)
    sim_auprc = float(auc(r_sim, p_sim))
    sim_brier = float(brier_score_loss(y_true, sim_scores))
    sim_ece = compute_ece(sim_scores, y_true)

    # Probability model metrics
    prob_auroc = float(roc_auc_score(y_true, prob_preds))
    p_pr, r_pr, _ = precision_recall_curve(y_true, prob_preds)
    prob_auprc = float(auc(r_pr, p_pr))
    prob_brier = float(brier_score_loss(y_true, prob_preds))
    prob_ece = compute_ece(prob_preds, y_true)

    return {
        'sim_auroc': sim_auroc,
        'sim_auprc': sim_auprc,
        'sim_brier': sim_brier,
        'sim_ece': sim_ece,
        'prob_auroc': prob_auroc,
        'prob_auprc': prob_auprc,
        'prob_brier': prob_brier,
        'prob_ece': prob_ece
    }

def run_statistical_tests(df_per_window):
    """
    Runs paired Wilcoxon signed-rank tests across streaming windows for key pairwise comparisons.
    """
    methods = df_per_window['method'].unique()
    test_results = []

    pairs_to_test = [
        ('Probability-Guided Top-1', 'RAPT-E'),
        ('Probability-Guided Top-1', 'Similarity-Only'),
        ('Probability-Guided Top-1', 'Event-Driven'),
        ('Probability-Guided Top-1', 'Frozen'),
        ('Probability-Guided Top-1', 'Oracle Transfer')
    ]

    for m1, m2 in pairs_to_test:
        if m1 in methods and m2 in methods:
            f1_m1 = df_per_window[df_per_window['method'] == m1]['f1_score'].values
            f1_m2 = df_per_window[df_per_window['method'] == m2]['f1_score'].values

            min_len = min(len(f1_m1), len(f1_m2))
            if min_len > 5:
                f1_m1 = f1_m1[:min_len]
                f1_m2 = f1_m2[:min_len]
                diffs = f1_m1 - f1_m2

                # If non-zero differences exist
                if np.sum(np.abs(diffs) > 1e-9) > 0:
                    stat_val, p_val = stats.wilcoxon(diffs)
                    stat_val, p_val = float(stat_val), float(p_val)
                else:
                    stat_val, p_val = 0.0, 1.0

                mean_diff = float(np.mean(diffs))
                win_rate = float(np.mean(diffs > 0))

                test_results.append({
                    'comparison': f"{m1} vs {m2}",
                    'method_1': m1,
                    'method_2': m2,
                    'mean_diff_f1': mean_diff,
                    'win_rate_m1': win_rate,
                    'wilcoxon_stat': stat_val,
                    'p_value': p_val,
                    'significant_p05': p_val < 0.05
                })

    return pd.DataFrame(test_results)
