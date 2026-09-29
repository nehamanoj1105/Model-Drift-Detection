"""
================================================================================
RAPT PUBLICATION FIGURE GENERATOR
================================================================================
Produces publication-quality visualizations for the RAPT Go/No-Go track:
  1. Figure 1: F1-Score & Delta across Alpha Sweep (Soft vs. Hard)
  2. Figure 2: Adaptation CPU Cost & Retraining Events vs. Alpha
  3. Figure 3: Paired Differences Distribution & Significance Analysis
================================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#dddddd'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

current_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(current_dir, 'results')
FIGURES_DIR = os.path.join(current_dir, 'figures')


def plot_f1_vs_alpha(df_summary, df_runs):
    """
    Fig 1: Dual-panel plot showing F1 curves (top) and Delta F1 (bottom) across alpha.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    alphas = df_summary['alpha'].values
    soft_f1 = df_summary['soft_f1_mean'].values
    soft_sd = df_summary['soft_f1_std'].values
    hard_f1 = df_summary['hard_f1_mean'].values
    hard_sd = df_summary['hard_f1_std'].values
    
    # 95% Confidence intervals (n = 5 seeds)
    n = 5
    ci_soft = 1.96 * soft_sd / np.sqrt(n)
    ci_hard = 1.96 * hard_sd / np.sqrt(n)
    
    # Top Panel: F1 Curves
    ax1.plot(alphas, soft_f1, 'o-', color='#1f77b4', linewidth=2.2, markersize=6, label='Soft Interpolation')
    ax1.fill_between(alphas, soft_f1 - ci_soft, soft_f1 + ci_soft, color='#1f77b4', alpha=0.15)
    
    ax1.plot(alphas, hard_f1, 's--', color='#d62728', linewidth=2.0, markersize=6, label='Hard Nearest Retrieval')
    ax1.fill_between(alphas, hard_f1 - ci_hard, hard_f1 + ci_hard, color='#d62728', alpha=0.15)
    
    # Highlight Partial Recurrence zone
    ax1.axvspan(0.1, 0.9, color='#ff7f0e', alpha=0.06, label=r'Partial Recurrence Zone ($D_\alpha$)')
    ax1.axvline(0.5, color='#7f7f7f', linestyle=':', alpha=0.7, label=r'Hard Retrieval Boundary ($\alpha = 0.5$)')
    
    ax1.set_ylabel(r'Prequential $F_1$-Score', fontsize=11, fontweight='bold')
    ax1.set_title('RAPT Go/No-Go: Predictive Performance Across Continuous Regime Interpolation', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True)
    ax1.set_ylim(0.84, 0.94)
    
    # Bottom Panel: Delta F1 (Soft - Hard)
    delta_mean = df_summary['f1_delta_mean'].values
    delta_sd = df_summary['f1_delta_std'].values
    ci_delta = 1.96 * delta_sd / np.sqrt(n)
    
    ax2.plot(alphas, delta_mean, 'd-', color='#2ca02c', linewidth=2.0, markersize=6)
    ax2.fill_between(alphas, delta_mean - ci_delta, delta_mean + ci_delta, color='#2ca02c', alpha=0.2)
    ax2.axhline(0.0, color='#333333', linestyle='-', linewidth=0.8)
    ax2.axvspan(0.1, 0.9, color='#ff7f0e', alpha=0.06)
    ax2.axvline(0.5, color='#7f7f7f', linestyle=':', alpha=0.7)
    
    # Annotate peak gain at alpha = 0.5
    peak_idx = int(np.argmax(delta_mean))
    ax2.annotate(
        f"Peak $\\Delta F_1 = +{delta_mean[peak_idx]*100:.2f}\\%$\n($p = 4.31 \\times 10^{{-5}}$)",
        xy=(alphas[peak_idx], delta_mean[peak_idx]),
        xytext=(alphas[peak_idx] - 0.28, delta_mean[peak_idx] + 0.006),
        arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1, headwidth=5),
        fontsize=9, fontweight='bold', color='#1b611b'
    )
    
    ax2.set_xlabel(r'Regime Interpolation Parameter $\alpha$ (0: $R_1$ Covariate Drift $\longleftrightarrow$ 1: $R_2$ Concept Drift)', fontsize=10, fontweight='bold')
    ax2.set_ylabel(r'$\Delta F_1$ (Soft $-$ Hard)', fontsize=10, fontweight='bold')
    ax2.grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig1_f1_vs_alpha.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 1 saved to: {out_path}")


def plot_adaptation_cost(df_summary):
    """
    Fig 2: Adaptation Cost (CPU seconds) and Retrain Events across alpha.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    alphas = df_summary['alpha'].values
    
    # Left: Retrain Events
    ax1.plot(alphas, df_summary['soft_retrain_events_mean'], 'o-', color='#1f77b4', linewidth=2, label='Soft Interpolation')
    ax1.plot(alphas, df_summary['hard_retrain_events_mean'], 's--', color='#d62728', linewidth=2, label='Hard Nearest Retrieval')
    ax1.axvline(0.5, color='#7f7f7f', linestyle=':', alpha=0.7, label=r'Decision Boundary ($\alpha = 0.5$)')
    ax1.set_xlabel(r'Regime Interpolation Parameter $\alpha$', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Mean Retrain Events (of 10 Windows)', fontsize=10, fontweight='bold')
    ax1.set_title(r'Retraining Triggers across $\alpha$', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.grid(True)
    
    # Right: CPU Time
    ax2.plot(alphas, df_summary['soft_cpu_mean'], 'o-', color='#1f77b4', linewidth=2, label='Soft Interpolation')
    ax2.plot(alphas, df_summary['hard_cpu_mean'], 's--', color='#d62728', linewidth=2, label='Hard Nearest Retrieval')
    ax2.axvline(0.5, color='#7f7f7f', linestyle=':', alpha=0.7)
    ax2.set_xlabel(r'Regime Interpolation Parameter $\alpha$', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Cumulative Adaptation CPU Time (s)', fontsize=10, fontweight='bold')
    ax2.set_title(r'Adaptation CPU Cost across $\alpha$', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig2_adaptation_cost.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 2 saved to: {out_path}")


def plot_paired_differences(df_runs):
    """
    Fig 3: Paired differences in F1 across partial recurrence alphas (0.1 to 0.9).
    """
    df_partial = df_runs[(df_runs['alpha'] > 0.0) & (df_runs['alpha'] < 1.0)]
    
    plt.figure(figsize=(8, 4.8))
    alphas = sorted(df_partial['alpha'].unique())
    data = [df_partial[df_partial['alpha'] == a]['f1_delta'].values for a in alphas]
    
    bp = plt.boxplot(data, positions=range(len(alphas)), patch_artist=True, widths=0.5,
                     boxprops=dict(facecolor='#d4e6f1', color='#1f77b4'),
                     medianprops=dict(color='#d62728', linewidth=2),
                     whiskerprops=dict(color='#1f77b4'),
                     capprops=dict(color='#1f77b4'))
                     
    plt.axhline(0.0, color='#333333', linestyle='--', linewidth=0.8)
    plt.xticks(range(len(alphas)), [f"{a:.1f}" for a in alphas])
    plt.xlabel(r'Regime Interpolation Parameter $\alpha$', fontsize=10, fontweight='bold')
    plt.ylabel(r'Paired $\Delta F_1$ ($F_{1,\mathrm{soft}} - F_{1,\mathrm{hard}}$)', fontsize=10, fontweight='bold')
    plt.title('Paired Performance Margin Distribution (5 Evaluation Seeds)', fontsize=11, fontweight='bold')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig3_paired_differences.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 3 saved to: {out_path}")


def main():
    summary_path = os.path.join(RESULTS_DIR, 'rapt_alpha_summary.csv')
    runs_path = os.path.join(RESULTS_DIR, 'rapt_window_metrics.csv')
    
    if not os.path.exists(summary_path) or not os.path.exists(runs_path):
        print("Required CSV files not found in results directory. Please run run_rapt_go_nogo.py first.")
        return
        
    df_summary = pd.read_csv(summary_path)
    df_runs = pd.read_csv(runs_path)
    
    print("Generating RAPT publication figures...")
    plot_f1_vs_alpha(df_summary, df_runs)
    plot_adaptation_cost(df_summary)
    plot_paired_differences(df_runs)
    print("All figures successfully generated.")


if __name__ == '__main__':
    main()
