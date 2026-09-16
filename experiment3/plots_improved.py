"""
================================================================================
EXPERIMENT 3 (IMPROVED) — SEPARATE VISUALIZATION SUITE (9 FIGURES)
================================================================================
Generates 9 dedicated publication figures in figures_improved/:
  1. improved_fig01_f1_comparison.png
  2. improved_fig02_accuracy_comparison.png
  3. improved_fig03_f1_across_windows.png
  4. improved_fig04_ucb1_arm_selection.png
  5. improved_fig05_f1_by_drift_type.png
  6. improved_fig06_cumulative_cpu_time.png
  7. improved_fig07_memory_usage.png
  8. improved_fig08_performance_vs_cpu_cost.png
  9. improved_fig09_retraining_cost.png
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = {
    'Frozen RF': '#7f7f7f',
    'Retrained RF': '#d62728',
    'UCB1 Ensemble': '#1f77b4',
}

ARM_PALETTE = {
    'RF': '#ff7f0e',
    'ET': '#2ca02c',
    'Ensemble': '#1f77b4',
}

DRIFT_COLORS = {
    'none': '#e0e0e0',
    'covariate': '#aec7e8',
    'concept': '#ffbb78',
    'mixed': '#98df8a',
}


def setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2.0,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def safe_savefig(fig, path, dpi=300, max_retries=3, delay=0.5):
    """Robust savefig with retry logic for Windows file systems."""
    for attempt in range(max_retries):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                plt.close(fig)
                raise e


def plot_f1_comparison(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']
    means = [window_df[window_df['approach'] == a]['f1'].mean() for a in approaches]
    stds = [window_df[window_df['approach'] == a]['f1'].std() for a in approaches]
    colors = [PALETTE[a] for a in approaches]

    bars = ax.bar(approaches, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax.set_ylabel('F1 Score')
    ax.set_title('Overall F1 Score Comparison')
    ax.set_ylim(0.5, 0.85)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean + std + 0.01,
                f'{mean:.4f}\n(±{std:.3f})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig01_f1_comparison.png'))


def plot_accuracy_comparison(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']
    means = [window_df[window_df['approach'] == a]['accuracy'].mean() for a in approaches]
    stds = [window_df[window_df['approach'] == a]['accuracy'].std() for a in approaches]
    colors = [PALETTE[a] for a in approaches]

    bars = ax.bar(approaches, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Overall Accuracy Comparison')
    ax.set_ylim(0.75, 0.90)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean + std + 0.005,
                f'{mean:.4f}\n(±{std:.3f})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig02_accuracy_comparison.png'))


def plot_f1_across_windows(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']

    # Background shading for drift regimes
    w_info = window_df[window_df['seed'] == window_df['seed'].iloc[0]].drop_duplicates('window_id')
    for _, row in w_info.iterrows():
        w = row['window_id']
        dtype = row['drift_type']
        color = DRIFT_COLORS.get(dtype, '#ffffff')
        ax.axvspan(w - 0.5, w + 0.5, facecolor=color, alpha=0.4, zorder=0)

    for a in approaches:
        sub = window_df[window_df['approach'] == a]
        grouped = sub.groupby('window_id')['f1'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['window_id'], grouped['mean'], label=a, color=PALETTE[a], marker='o', zorder=3)
        ax.fill_between(grouped['window_id'], grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'],
                        color=PALETTE[a], alpha=0.15, zorder=2)

    ax.set_xlabel('Streaming Window ID (500 samples each)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Streaming F1 Trajectory Across Windows with Drift Regimes')
    ax.set_xticks(range(16))
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig03_f1_across_windows.png'))


def plot_ucb1_arm_selection(bandit_df, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    arms = ['RF', 'ET', 'Ensemble']
    counts = [bandit_df['selected_model'].value_counts().get(a, 0) for a in arms]
    colors = [ARM_PALETTE[a] for a in arms]

    bars = ax1.bar(arms, counts, color=colors, alpha=0.85, edgecolor='black', width=0.45)
    ax1.set_ylabel('Total Selections (Across All Seeds)')
    ax1.set_title('UCB1 Arm Selections Overall')
    ax1.grid(axis='y', alpha=0.3)
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1,
                 f'{int(b.get_height())}', ha='center', va='bottom', fontweight='bold')

    # Breakdown by drift type
    grouped = bandit_df.groupby(['drift_type', 'selected_model']).size().unstack(fill_value=0)
    for arm in arms:
        if arm not in grouped.columns:
            grouped[arm] = 0
    grouped = grouped[arms]
    grouped.plot(kind='bar', stacked=True, ax=ax2, color=[ARM_PALETTE[a] for a in arms], edgecolor='black', alpha=0.85)
    ax2.set_xlabel('Drift Regime')
    ax2.set_ylabel('Selection Count')
    ax2.set_title('Arm Selection Frequency by Drift Type')
    ax2.legend(title='Arm')
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig04_ucb1_arm_selection.png'))


def plot_f1_by_drift_type(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']
    dtypes = ['none', 'covariate', 'concept', 'mixed']
    dlabels = ['No Drift', 'Covariate', 'Concept', 'Mixed']

    x = np.arange(len(dtypes))
    width = 0.25

    for idx, a in enumerate(approaches):
        means = [window_df[(window_df['approach'] == a) & (window_df['drift_type'] == d)]['f1'].mean() for d in dtypes]
        stds = [window_df[(window_df['approach'] == a) & (window_df['drift_type'] == d)]['f1'].std() for d in dtypes]
        ax.bar(x + idx * width, means, width, yerr=stds, capsize=4, label=a, color=PALETTE[a], alpha=0.85, edgecolor='black')

    ax.set_ylabel('F1 Score')
    ax.set_title('Predictive F1 Score Across Heterogeneous Drift Types')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dlabels)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig05_f1_by_drift_type.png'))


def plot_cumulative_cpu_time(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']

    for a in approaches:
        sub = window_df[window_df['approach'] == a]
        seed_cum = []
        for s in sub['seed'].unique():
            s_sub = sub[sub['seed'] == s].sort_values('window_id')
            seed_cum.append(np.cumsum(s_sub['cpu_time'].values))
        cum_arr = np.array(seed_cum)
        mean_cum = np.mean(cum_arr, axis=0)
        std_cum = np.std(cum_arr, axis=0)

        ax.plot(range(16), mean_cum, label=a, color=PALETTE[a], marker='s', zorder=3)
        ax.fill_between(range(16), mean_cum - std_cum, mean_cum + std_cum, color=PALETTE[a], alpha=0.15, zorder=2)

    ax.set_xlabel('Streaming Window ID')
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.set_title('Cumulative Computational Cost Trajectory')
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig06_cumulative_cpu_time.png'))


def plot_memory_usage(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']
    means = [window_df[window_df['approach'] == a]['ram_mb'].mean() for a in approaches]
    stds = [window_df[window_df['approach'] == a]['ram_mb'].std() for a in approaches]
    colors = [PALETTE[a] for a in approaches]

    bars = ax.bar(approaches, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax.set_ylabel('Process Resident RAM (MB)')
    ax.set_title('Memory Consumption Comparison')
    ax.set_ylim(180, 220)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean + std + 0.5,
                f'{mean:.1f} MB', ha='center', va='bottom', fontsize=9, fontweight='bold')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig07_memory_usage.png'))


def plot_performance_vs_cpu_cost(window_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']

    for a in approaches:
        sub = window_df[window_df['approach'] == a]
        f1_vals = sub.groupby('seed')['f1'].mean().values
        cpu_vals = sub.groupby('seed')['cpu_time'].sum().values

        ax.scatter(cpu_vals, f1_vals, label=a, color=PALETTE[a], s=90, alpha=0.8, edgecolor='black', zorder=3)
        ax.scatter(np.mean(cpu_vals), np.mean(f1_vals), color=PALETTE[a], s=250, marker='*', edgecolor='black', zorder=4)

    ax.set_xlabel('Total Streaming CPU Time (seconds)')
    ax.set_ylabel('Mean F1 Score')
    ax.set_title('Performance vs Computational Cost Trade-off (Pareto Frontier)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig08_performance_vs_cpu_cost.png'))


def plot_retraining_cost(retrain_df, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Ensemble']

    events = [retrain_df[retrain_df['approach'] == a].groupby('seed')['retrained'].sum().mean() for a in approaches]
    colors = [PALETTE[a] for a in approaches]

    bars1 = ax1.bar(approaches, events, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax1.set_ylabel('Mean Retraining Events')
    ax1.set_title('Adaptation / Retraining Events Across 16 Windows')
    ax1.grid(axis='y', alpha=0.3)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.3,
                 f'{b.get_height():.1f}', ha='center', va='bottom', fontweight='bold')

    cpu_times = [retrain_df[retrain_df['approach'] == a].groupby('seed')['retraining_cpu_time'].sum().mean() for a in approaches]
    bars2 = ax2.bar(approaches, cpu_times, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    ax2.set_ylabel('Total Retraining CPU Time (s)')
    ax2.set_title('Dedicated Retraining Computational Overhead')
    ax2.grid(axis='y', alpha=0.3)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.3,
                 f'{b.get_height():.2f}s', ha='center', va='bottom', fontweight='bold')

    safe_savefig(fig, os.path.join(out_dir, 'improved_fig09_retraining_cost.png'))


def generate_all_improved_figures(window_df, bandit_df, retrain_df, out_dir):
    """Generate all 9 improved publication figures."""
    os.makedirs(out_dir, exist_ok=True)
    setup_style()
    plot_f1_comparison(window_df, out_dir)
    plot_accuracy_comparison(window_df, out_dir)
    plot_f1_across_windows(window_df, out_dir)
    plot_ucb1_arm_selection(bandit_df, out_dir)
    plot_f1_by_drift_type(window_df, out_dir)
    plot_cumulative_cpu_time(window_df, out_dir)
    plot_memory_usage(window_df, out_dir)
    plot_performance_vs_cpu_cost(window_df, out_dir)
    plot_retraining_cost(retrain_df, out_dir)
