"""
Plots Module for Experiment 2 Corrected.
Generates all 13 required publication figures.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

PALETTE = {
    'Frozen':              '#6B7280',
    'Event-Driven':        '#3B82F6',
    'RAPT-E':              '#10B981',
    'Similarity-Only':     '#F59E0B',
    'Similarity-Weighted': '#F97316',
    'Hist-Reliability':    '#8B5CF6',
    'Random-Historical':   '#EC4899',
    'Probability-Guided':  '#EF4444',
    'Oracle':              '#1D4ED8',
}
METHODS_ORDER = [
    'Frozen', 'Event-Driven', 'RAPT-E',
    'Similarity-Only', 'Probability-Guided', 'Oracle',
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})


def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# ── Figure 1: Similarity vs ΔF1 ───────────────────────────────────────────────
def plot_similarity_vs_delta_f1(candidate_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'positive': '#10B981', 'neutral': '#F59E0B', 'negative': '#EF4444'}
    for label, grp in candidate_df.groupby('transfer_label'):
        ax.scatter(grp['similarity'], grp['delta_f1'],
                   c=colors.get(label, '#6B7280'), alpha=0.5, s=20, label=label)
    ax.axhline(0.005, ls='--', lw=0.8, color='gray', label='+/-0.005 threshold')
    ax.axhline(-0.005, ls='--', lw=0.8, color='gray')
    ax.set_xlabel('Distributional Similarity (Source -> Target)')
    ax.set_ylabel('ΔF1 (Transfer - Baseline)')
    ax.set_title('Fig 1 -- Similarity vs Actual ΔF1')
    ax.legend()
    _save(fig, out_path)


# ── Figure 2: Probability vs ΔF1 ─────────────────────────────────────────────
def plot_probability_vs_delta_f1(prob_df: pd.DataFrame, out_path: str):
    if prob_df is None or 'probability' not in prob_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'positive': '#10B981', 'neutral': '#F59E0B', 'negative': '#EF4444'}
    for label, grp in prob_df.groupby('transfer_label'):
        ax.scatter(grp['probability'], grp['delta_f1'],
                   c=colors.get(label, '#6B7280'), alpha=0.5, s=20, label=label)
    ax.axhline(0.005, ls='--', lw=0.8, color='gray')
    ax.axhline(-0.005, ls='--', lw=0.8, color='gray')
    ax.axvline(0.60, ls=':', lw=1.0, color='#1D4ED8', label='τ = 0.60')
    ax.set_xlabel('P(Positive Transfer)')
    ax.set_ylabel('ΔF1 (Transfer - Baseline)')
    ax.set_title('Fig 2 -- Predicted Probability vs Actual ΔF1')
    ax.legend()
    _save(fig, out_path)


# ── Figure 3: Held-out ROC ────────────────────────────────────────────────────
def plot_roc(prob_quality_list: list, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random (AUC = 0.50)')
    for i, pq in enumerate(prob_quality_list):
        preds = pq.get('predictions')
        labels = pq.get('true_labels')
        if preds and labels and len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, preds)
            auc = pq.get('auroc', 0)
            ax.plot(fpr, tpr, alpha=0.5, label=f'Seed {i} (AUC={auc:.3f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Fig 3 -- Held-Out ROC ({stream_name})')
    ax.legend(loc='lower right', fontsize=7)
    _save(fig, out_path)


# ── Figure 4: Held-out Precision-Recall ───────────────────────────────────────
def plot_precision_recall(prob_quality_list: list, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, pq in enumerate(prob_quality_list):
        preds = pq.get('predictions')
        labels = pq.get('true_labels')
        if preds and labels and len(set(labels)) > 1:
            prec, rec, _ = precision_recall_curve(labels, preds)
            auprc = pq.get('auprc', 0)
            ax.plot(rec, prec, alpha=0.5, label=f'Seed {i} (AUPRC={auprc:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Fig 4 -- Held-Out Precision-Recall ({stream_name})')
    ax.legend(fontsize=7)
    _save(fig, out_path)


# ── Figure 5: Calibration Curve ───────────────────────────────────────────────
def plot_calibration(prob_quality_list: list, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfectly calibrated')
    for i, pq in enumerate(prob_quality_list):
        preds = pq.get('predictions')
        labels = pq.get('true_labels')
        if preds and labels and len(set(labels)) > 1 and sum(labels) > 0:
            try:
                frac_pos, mean_pred = calibration_curve(labels, preds, n_bins=5)
                ece = pq.get('ece', float('nan'))
                ax.plot(mean_pred, frac_pos, 's-', alpha=0.7,
                        label=f'Seed {i} (ECE={ece:.3f})')
            except Exception:
                pass
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Fig 5 -- Calibration Curve ({stream_name})')
    ax.legend(fontsize=7)
    _save(fig, out_path)


# ── Figure 6: Negative-Transfer Counts ───────────────────────────────────────
def plot_negative_transfer_counts(ntr_by_method: dict, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(ntr_by_method.keys())
    neg_counts = [ntr_by_method[m].get('negative', 0) for m in methods]
    total_counts = [ntr_by_method[m].get('total', 1) for m in methods]
    ntr_rates = [n / max(t, 1) for n, t in zip(neg_counts, total_counts)]
    colors = [PALETTE.get(m, '#6B7280') for m in methods]
    bars = ax.bar(methods, ntr_rates, color=colors)
    ax.set_ylabel('Negative Transfer Rate')
    ax.set_title(f'Fig 6 -- Negative Transfer Rate by Method ({stream_name})')
    ax.set_ylim(0, max(ntr_rates) * 1.3 + 0.01)
    for bar, rate in zip(bars, ntr_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=25, ha='right')
    _save(fig, out_path)


# ── Figure 7: Transfer Probability by Outcome ─────────────────────────────────
def plot_prob_by_outcome(prob_df: pd.DataFrame, out_path: str):
    if prob_df is None or 'probability' not in prob_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'positive': '#10B981', 'neutral': '#F59E0B', 'negative': '#EF4444'}
    labels_order = ['negative', 'neutral', 'positive']
    data_by_label = [
        prob_df[prob_df['transfer_label'] == lbl]['probability'].values
        for lbl in labels_order
    ]
    parts = ax.violinplot(
        [d for d in data_by_label if len(d) > 0],
        positions=range(len([d for d in data_by_label if len(d) > 0])),
        showmedians=True,
    )
    ax.set_xticks(range(len([d for d in data_by_label if len(d) > 0])))
    ax.set_xticklabels([lbl for lbl, d in zip(labels_order, data_by_label) if len(d) > 0])
    ax.set_ylabel('P(Positive Transfer)')
    ax.set_title('Fig 7 -- Transfer Probability by Actual Outcome')
    _save(fig, out_path)


# ── Figure 8: Similarity-Only vs Probability-Guided Decisions ─────────────────
def plot_decision_comparison(candidate_df: pd.DataFrame, out_path: str):
    if candidate_df is None or candidate_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (x_col, title) in zip(axes, [
        ('similarity', 'Similarity-Only decisions'),
        ('probability', 'Probability-Guided decisions'),
    ]):
        if x_col not in candidate_df.columns:
            ax.text(0.5, 0.5, f'{x_col} not available', transform=ax.transAxes, ha='center')
            continue
        colors = {'positive': '#10B981', 'neutral': '#F59E0B', 'negative': '#EF4444'}
        for lbl, grp in candidate_df.groupby('transfer_label'):
            ax.scatter(grp[x_col], grp['delta_f1'], alpha=0.4, s=15,
                       c=colors.get(lbl, 'gray'), label=lbl)
        ax.axhline(0.005, ls='--', lw=0.8, color='gray')
        ax.axhline(-0.005, ls='--', lw=0.8, color='gray')
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel('ΔF1')
        ax.set_title(f'Fig 8 -- {title}')
        ax.legend(fontsize=7)
    fig.suptitle('Fig 8 -- Similarity vs Probability Decision Landscape', y=1.02)
    _save(fig, out_path)


# ── Figure 9: Oracle vs Learned Selectors ─────────────────────────────────────
def plot_oracle_vs_selectors(oracle_df: pd.DataFrame, stream_name: str, out_path: str):
    if oracle_df is None or oracle_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = oracle_df['method'].tolist()
    method_f1 = oracle_df['method_f1_mean'].tolist()
    oracle_f1 = oracle_df['oracle_f1_mean'].tolist()
    x = range(len(methods))
    ax.bar(x, method_f1, color=[PALETTE.get(m, '#6B7280') for m in methods], alpha=0.8, label='Method F1')
    ax.plot(x, oracle_f1, 'k^--', ms=7, label='Oracle F1')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel('Mean F1 (Macro)')
    ax.set_title(f'Fig 9 -- Oracle vs Learned Selectors ({stream_name})')
    ax.legend()
    _save(fig, out_path)


# ── Figure 10: F1 by Method ────────────────────────────────────────────────────
def plot_f1_by_method(summary_df: pd.DataFrame, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = [m for m in METHODS_ORDER if m in summary_df['method'].values]
    f1_means = [float(summary_df[summary_df['method'] == m]['f1_mean'].values[0]) for m in methods]
    f1_stds  = [float(summary_df[summary_df['method'] == m]['f1_std'].values[0]) for m in methods]
    x = range(len(methods))
    bars = ax.bar(x, f1_means, yerr=f1_stds,
                  color=[PALETTE.get(m, '#6B7280') for m in methods],
                  capsize=4, error_kw={'lw': 1.5})
    ylo = max(0, min(f1_means) - max(f1_stds) - 0.02)
    ax.set_ylim(ylo, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel('Mean F1 (Macro)')
    ax.set_title(f'Fig 10 -- F1 by Method ({stream_name})')
    for bar, val in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    _save(fig, out_path)


# ── Figure 11: Adaptation CPU ─────────────────────────────────────────────────
def plot_adaptation_cpu(summary_df: pd.DataFrame, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = [m for m in METHODS_ORDER if m in summary_df['method'].values]
    cpu_vals = [float(summary_df[summary_df['method'] == m]['adapt_cpu_mean'].values[0])
                for m in methods]
    cpu_stds = [float(summary_df[summary_df['method'] == m]['adapt_cpu_std'].values[0])
                for m in methods]
    x = range(len(methods))
    ax.bar(x, cpu_vals, yerr=cpu_stds,
           color=[PALETTE.get(m, '#6B7280') for m in methods],
           capsize=4, error_kw={'lw': 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel('Adaptation CPU (s)')
    ax.set_title(f'Fig 11 -- Adaptation CPU by Method ({stream_name})')
    _save(fig, out_path)


# ── Figure 12: F1 vs CPU Pareto ───────────────────────────────────────────────
def plot_pareto(summary_df: pd.DataFrame, stream_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in summary_df.iterrows():
        m = row['method']
        if m not in METHODS_ORDER:
            continue
        color = PALETTE.get(m, '#6B7280')
        ax.scatter(row['adapt_cpu_mean'], row['f1_mean'],
                   c=color, s=80, zorder=3, edgecolors='k', lw=0.5)
        ax.annotate(m, (row['adapt_cpu_mean'], row['f1_mean']),
                    fontsize=7, textcoords='offset points', xytext=(5, 3))
    ax.set_xlabel('Adaptation CPU (s)')
    ax.set_ylabel('Mean F1 (Macro)')
    ax.set_title(f'Fig 12 -- F1 vs CPU Pareto ({stream_name})')
    _save(fig, out_path)


# ── Figure 13: Cross-Dataset Calibration ──────────────────────────────────────
def plot_cross_dataset_calibration(cross_results: dict, out_path: str):
    if not cross_results:
        return
    streams = list(cross_results.keys())
    auroc_vals = [cross_results[s].get('auroc', float('nan')) for s in streams]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#3B82F6' if 'Original' in s else '#10B981' for s in streams]
    ax.bar(streams, auroc_vals, color=colors)
    ax.axhline(0.5, ls='--', lw=0.8, color='gray', label='Random baseline (0.5)')
    ax.set_ylabel('AUROC (Held-Out)')
    ax.set_title('Fig 13 -- Cross-Dataset Probability Calibration')
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.xticks(rotation=20, ha='right')
    _save(fig, out_path)


def generate_all_plots(
    summary_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    prob_quality_list: list,
    oracle_df: pd.DataFrame,
    ntr_by_method: dict,
    cross_results: dict,
    stream_name: str,
    plots_dir: str,
):
    """Generate all 13 publication figures for one stream."""
    os.makedirs(plots_dir, exist_ok=True)
    prefix = os.path.join(plots_dir, stream_name.replace(' ', '_'))

    print(f"\nGenerating plots for {stream_name}...")

    # Add probability column to candidate_df if available
    prob_df = None
    if candidate_df is not None and not candidate_df.empty:
        prob_df = candidate_df.copy()

    plot_similarity_vs_delta_f1(candidate_df, f"{prefix}_fig1_sim_vs_delta_f1.png")
    plot_probability_vs_delta_f1(prob_df, f"{prefix}_fig2_prob_vs_delta_f1.png")
    plot_roc(prob_quality_list, stream_name, f"{prefix}_fig3_roc.png")
    plot_precision_recall(prob_quality_list, stream_name, f"{prefix}_fig4_pr_curve.png")
    plot_calibration(prob_quality_list, stream_name, f"{prefix}_fig5_calibration.png")
    plot_negative_transfer_counts(ntr_by_method, stream_name, f"{prefix}_fig6_ntr.png")
    plot_prob_by_outcome(prob_df, f"{prefix}_fig7_prob_by_outcome.png")
    plot_decision_comparison(candidate_df, f"{prefix}_fig8_decision_comparison.png")
    plot_oracle_vs_selectors(oracle_df, stream_name, f"{prefix}_fig9_oracle.png")
    plot_f1_by_method(summary_df, stream_name, f"{prefix}_fig10_f1_by_method.png")
    plot_adaptation_cpu(summary_df, stream_name, f"{prefix}_fig11_adapt_cpu.png")
    plot_pareto(summary_df, stream_name, f"{prefix}_fig12_pareto.png")
    plot_cross_dataset_calibration(cross_results, f"{prefix}_fig13_cross_dataset.png")
