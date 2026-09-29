"""
================================================================================
EXPERIMENT 5 — PUBLICATION-QUALITY FIGURE GENERATOR
================================================================================
Generates all 7 required figures plus the similarity threshold ablation plot:
  - Figure 1: Experiment Architecture Diagram
  - Figure 2: F1 Comparison (Bar chart with error bars)
  - Figure 3: Accuracy Comparison (Bar chart with error bars)
  - Figure 4: Ensemble Weights Over Time (Line plot with regime boundaries)
  - Figure 5: Discovered Regime Map (Categorical timeline step plot)
  - Figure 6: Regime Similarity vs Policy Transfer Performance
  - Figure 7: Accuracy vs Adaptation CPU Compute Trade-Off (Pareto plot)
  - Figure 8: Similarity Threshold Sensitivity Ablation Plot
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

COLOR_PALETTE = {
    'Fixed Ensemble': '#4C72B0',           # Muted Blue
    'Global Adaptive Ensemble': '#DD8452', # Warm Orange
    'Regime-Aware Ensemble': '#55A868',    # Forest Green
    'Oracle Regime Weights': '#C44E52',    # Crimson
    'RF': '#1f77b4',
    'ET': '#ff7f0e',
    'GB': '#2ca02c'
}


def plot_fig1_architecture(save_dir):
    """Figure 1 — Experiment Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis('off')

    # Draw boxes
    boxes = [
        ("Streaming Data\nWindow X_t (500 samples)", 0.05, 0.45, 0.18, 0.25, "#EAEAEA"),
        ("Quantile Fingerprint\nExtract 21-dim (10, 25, 50, 75, 90, μ, σ)", 0.27, 0.45, 0.20, 0.25, "#D0E1F9"),
        ("Regime Memory\nLookup / Sim Check (τ)", 0.51, 0.45, 0.18, 0.25, "#D5E8D4"),
        ("Regime Weights\n[w_RF, w_ET, w_GB]", 0.73, 0.45, 0.15, 0.25, "#FFF2CC"),
        ("Base Models\nRF (50) + ET (50) + GB (50)", 0.51, 0.10, 0.20, 0.20, "#F8CECC"),
        ("Ensemble Prediction\nŷ = I(Σ w_m P_m >= 0.5)", 0.75, 0.10, 0.20, 0.20, "#E1D5E7")
    ]

    for label, x, y, w, h, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)

    # Draw arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#333333")
    ax.annotate("", xy=(0.27, 0.575), xytext=(0.23, 0.575), arrowprops=arrow_props)
    ax.annotate("", xy=(0.51, 0.575), xytext=(0.47, 0.575), arrowprops=arrow_props)
    ax.annotate("", xy=(0.73, 0.575), xytext=(0.69, 0.575), arrowprops=arrow_props)

    ax.annotate("", xy=(0.61, 0.30), xytext=(0.61, 0.45), arrowprops=arrow_props)
    ax.annotate("", xy=(0.75, 0.20), xytext=(0.71, 0.20), arrowprops=arrow_props)
    ax.annotate("", xy=(0.825, 0.30), xytext=(0.825, 0.45), arrowprops=arrow_props)

    # Annotate comparison paths
    ax.text(0.14, 0.82, "Baseline 1: Fixed Weights (1/3, 1/3, 1/3)", fontsize=11, fontweight='bold', color="#4C72B0")
    ax.text(0.14, 0.75, "Baseline 2: Global Adaptive Weights (Global EMA Softmax)", fontsize=11, fontweight='bold', color="#DD8452")
    ax.text(0.14, 0.03, "Prequential Evaluation (Test-Then-Train) | Zero Data Leakage", fontsize=11, fontstyle='italic', color="#555555")

    plt.title("Figure 1: Experiment 5 Architecture — Regime-Aware Ensemble Weighting", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_architecture.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'fig1_architecture.svg'), dpi=300)
    plt.close()


def plot_fig2_f1_comparison(df_summary, save_dir):
    """Figure 2 — F1 Score Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    agg = df_summary.groupby('method')['f1'].agg(['mean', 'std']).reset_index()
    order = ['Fixed Ensemble', 'Global Adaptive Ensemble', 'Regime-Aware Ensemble', 'Oracle Regime Weights']
    agg['method'] = pd.Categorical(agg['method'], categories=order, ordered=True)
    agg = agg.sort_values('method')

    colors = [COLOR_PALETTE.get(m, '#333333') for m in agg['method']]
    bars = ax.bar(agg['method'], agg['mean'], yerr=agg['std'], capsize=5, color=colors, edgecolor='black', alpha=0.85, width=0.55)

    for bar, mean_val in zip(bars, agg['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean_val + 0.01, f"{mean_val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel("Macro F1 Score", fontweight='bold')
    ax.set_title("Figure 2: Streaming Macro F1 Score Comparison (Mean ± Std)", fontweight='bold', pad=12)
    ax.set_ylim(0.0, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_f1_comparison.png'), dpi=300)
    plt.close()


def plot_fig3_accuracy_comparison(df_summary, save_dir):
    """Figure 3 — Accuracy Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    agg = df_summary.groupby('method')['accuracy'].agg(['mean', 'std']).reset_index()
    order = ['Fixed Ensemble', 'Global Adaptive Ensemble', 'Regime-Aware Ensemble', 'Oracle Regime Weights']
    agg['method'] = pd.Categorical(agg['method'], categories=order, ordered=True)
    agg = agg.sort_values('method')

    colors = [COLOR_PALETTE.get(m, '#333333') for m in agg['method']]
    bars = ax.bar(agg['method'], agg['mean'], yerr=agg['std'], capsize=5, color=colors, edgecolor='black', alpha=0.85, width=0.55)

    for bar, mean_val in zip(bars, agg['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean_val + 0.01, f"{mean_val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel("Classification Accuracy", fontweight='bold')
    ax.set_title("Figure 3: Streaming Accuracy Comparison (Mean ± Std)", fontweight='bold', pad=12)
    ax.set_ylim(0.0, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_accuracy_comparison.png'), dpi=300)
    plt.close()


def plot_fig4_weights_over_time(df_weights, save_dir):
    """Figure 4 — Ensemble Weights Over Time for Regime-Aware Method."""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    df_ra = df_weights[df_weights['method'] == 'Regime-Aware Ensemble']
    if df_ra.empty:
        df_ra = df_weights

    # Average weights across seeds per window
    w_avg = df_ra.groupby('window_id')[['w_rf', 'w_et', 'w_gb']].mean().reset_index()

    ax.plot(w_avg['window_id'], w_avg['w_rf'], label='Random Forest (RF)', color=COLOR_PALETTE['RF'], lw=2.2, marker='o', ms=4)
    ax.plot(w_avg['window_id'], w_avg['w_et'], label='Extra Trees (ET)', color=COLOR_PALETTE['ET'], lw=2.2, marker='s', ms=4)
    ax.plot(w_avg['window_id'], w_avg['w_gb'], label='Gradient Boosting (GB)', color=COLOR_PALETTE['GB'], lw=2.2, marker='^', ms=4)

    # Vertical dashed lines for major regime wave boundaries
    ax.axvline(x=11, color='gray', linestyle='--', alpha=0.7, label='Regime Transition (DDoS->Password)')
    ax.axvline(x=21, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=28, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=38, color='purple', linestyle=':', alpha=0.8, label='Regime Recurrence Wave 2')

    ax.set_xlabel("Streaming Window Index (500 samples/window)", fontweight='bold')
    ax.set_ylabel("Ensemble Weight", fontweight='bold')
    ax.set_title("Figure 4: Component Model Ensemble Weights Over Time (Regime-Aware)", fontweight='bold', pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_weights_over_time.png'), dpi=300)
    plt.close()


def plot_fig5_regime_map(df_window, save_dir):
    """Figure 5 — Discovered Regime Map over Time."""
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)

    df_ra = df_window[df_window['method'] == 'Regime-Aware Ensemble']
    if df_ra.empty:
        df_ra = df_window

    r_avg = df_ra.groupby('window_id')['regime_id'].apply(lambda x: x.mode()[0]).reset_index()

    ax.step(r_avg['window_id'], r_avg['regime_id'], where='post', color='#2b5c8f', lw=2.5)
    ax.scatter(r_avg['window_id'], r_avg['regime_id'], color='#c44e52', s=35, zorder=4, label='Discovered / Retrieved Regime ID')

    ax.set_xlabel("Streaming Window Index", fontweight='bold')
    ax.set_ylabel("Assigned / Retrieved Regime ID", fontweight='bold')
    ax.set_title("Figure 5: Discovered Operational Regime Map Over 50,000-Sample Stream", fontweight='bold', pad=12)
    ax.set_yticks(sorted(r_avg['regime_id'].unique()))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_regime_map.png'), dpi=300)
    plt.close()


def plot_fig6_similarity_vs_transfer(df_window, save_dir):
    """Figure 6 — Regime Similarity vs Policy Transfer Performance."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    df_ra = df_window[df_window['method'] == 'Regime-Aware Ensemble'].copy()
    if df_ra.empty:
        df_ra = df_window.copy()

    # Filter retrieved windows (where similarity matching occurred)
    retrieved = df_ra[df_ra['is_retrieved'] == True]
    if retrieved.empty:
        retrieved = df_ra

    x_vals = retrieved['regime_similarity'].values
    y_vals = retrieved['f1'].values

    ax.scatter(x_vals, y_vals, color='#55A868', alpha=0.6, s=40, label='Window Transferred F1')

    if len(x_vals) > 1 and np.std(x_vals) > 1e-5:
        m_slope, b_intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
        y_line = m_slope * x_line + b_intercept
        ax.plot(x_line, y_line, color='#1b4f24', lw=2.5, label=f'Linear Trend (slope={m_slope:.2f})')

    ax.set_xlabel("Quantile Fingerprint Cosine Similarity to Retained Regime", fontweight='bold')
    ax.set_ylabel("Transferred Ensemble Window Macro F1", fontweight='bold')
    ax.set_title("Figure 6: Quantile Similarity vs. Transferred Policy Performance", fontweight='bold', pad=12)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_similarity_vs_transfer.png'), dpi=300)
    plt.close()


def plot_fig7_accuracy_compute_tradeoff(df_summary, save_dir):
    """Figure 7 — Accuracy vs Compute Trade-Off (Pareto Plot)."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    agg = df_summary.groupby('method').agg({
        'f1': 'mean',
        'adaptation_cpu_s': 'mean',
        'unique_regimes': 'mean'
    }).reset_index()

    for _, row in agg.iterrows():
        m_name = row['method']
        color = COLOR_PALETTE.get(m_name, '#333333')
        ax.scatter(row['adaptation_cpu_s'], row['f1'], s=160, color=color, edgecolor='black', alpha=0.9, label=m_name)
        ax.annotate(
            m_name,
            (row['adaptation_cpu_s'], row['f1']),
            xytext=(10, 5), textcoords='offset points',
            fontweight='bold', fontsize=10
        )

    ax.set_xlabel("Adaptation CPU Time (seconds)", fontweight='bold')
    ax.set_ylabel("Macro F1 Score", fontweight='bold')
    ax.set_title("Figure 7: Macro F1 vs. Adaptation Compute Trade-Off (Pareto Frontier)", fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_accuracy_compute_tradeoff.png'), dpi=300)
    plt.close()


def plot_fig8_threshold_ablation(df_ablation, save_dir):
    """Figure 8 — Similarity Threshold Sensitivity Ablation Plot."""
    if df_ablation.empty:
        return

    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)

    agg = df_ablation.groupby('similarity_threshold').agg({
        'f1': ['mean', 'std'],
        'unique_regimes': 'mean',
        'regime_retrievals': 'mean'
    }).reset_index()

    thresholds = agg['similarity_threshold']
    f1_means = agg[('f1', 'mean')]
    f1_stds = agg[('f1', 'std')]
    reg_means = agg[('unique_regimes', 'mean')]

    color1 = '#55A868'
    ax1.errorbar(thresholds, f1_means, yerr=f1_stds, fmt='-o', color=color1, lw=2.5, capsize=4, label='Regime-Aware F1 Score')
    ax1.set_xlabel("Similarity Threshold τ", fontweight='bold')
    ax1.set_ylabel("Macro F1 Score", fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#C44E52'
    ax2.plot(thresholds, reg_means, '--s', color=color2, lw=2.0, ms=6, label='Discovered Regimes')
    ax2.set_ylabel("Number of Discovered Regimes", fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Figure 8: Sensitivity to Similarity Threshold τ (Ablation Study)", fontweight='bold', pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_threshold_ablation.png'), dpi=300)
    plt.close()


def generate_all_experiment5_figures(df_summary, df_window, df_weights, df_ablation, save_dir):
    """Generates all requested publication-quality figures."""
    os.makedirs(save_dir, exist_ok=True)

    plot_fig1_architecture(save_dir)
    plot_fig2_f1_comparison(df_summary, save_dir)
    plot_fig3_accuracy_comparison(df_summary, save_dir)
    plot_fig4_weights_over_time(df_weights, save_dir)
    plot_fig5_regime_map(df_window, save_dir)
    plot_fig6_similarity_vs_transfer(df_window, save_dir)
    plot_fig7_accuracy_compute_tradeoff(df_summary, save_dir)
    plot_fig8_threshold_ablation(df_ablation, save_dir)
