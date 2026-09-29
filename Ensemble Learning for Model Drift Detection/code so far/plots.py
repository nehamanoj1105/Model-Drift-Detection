"""
================================================================================
EXPERIMENT 3 — VISUALIZATION SUITE (9 ESSENTIAL FIGURES ONLY)
================================================================================
Generates exactly the 9 publication figures required for Experiment 3:
  1. fig01_f1_comparison.png         — Overall F1 comparison across 3 approaches
  2. fig02_accuracy_comparison.png   — Overall Accuracy comparison across 3 approaches
  3. fig03_f1_across_windows.png     — F1 trajectory across 16 windows with drift annotations
  4. fig04_ucb1_arm_selection.png    — UCB1 arm selection (RF, ET, Ensemble) overall & by drift type
  5. fig05_f1_by_drift_type.png      — F1 by drift type (No Drift, Covariate, Concept, Mixed)
  6. fig06_cumulative_cpu_time.png   — Cumulative CPU time across streaming windows
  7. fig07_memory_usage.png          — RAM usage across approaches
  8. fig08_performance_vs_cpu_cost.png — F1 vs Cumulative CPU time (Pareto frontier)
  9. fig09_retraining_cost.png       — Retraining time and adaptation/retrain event counts
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Consistent styling & palette
PALETTE = {
    'Frozen RF': '#7f7f7f',               # Grey
    'Retrained RF': '#d62728',            # Red
    'UCB1 Adaptive Ensemble': '#1f77b4',  # Blue
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


def safe_savefig(path, dpi=300):
    """Save figure with retry logic to avoid intermittent Windows file locking."""
    for attempt in range(5):
        try:
            plt.savefig(path, dpi=dpi)
            return
        except OSError:
            time.sleep(0.3)
    plt.savefig(path, dpi=dpi)


# ── Figure 1: F1 Comparison ──────────────────────────────────────────────────
def plot_fig01_f1_comparison(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']
    means = []
    stds = []
    colors = [PALETTE[a] for a in approaches]

    for a in approaches:
        sub = df_window[df_window['approach'] == a]
        means.append(sub['f1'].mean())
        stds.append(sub['f1'].std())

    bars = ax.bar(approaches, means, yerr=stds, capsize=6, color=colors, edgecolor='black', alpha=0.85, width=0.55)
    ax.set_ylabel('F1 Score')
    ax.set_title('Figure 1: F1 Score Comparison Across Approaches')
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y')

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + (s if not np.isnan(s) else 0) + 0.02,
                f'{m:.3f} ± {s:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig01_f1_comparison.png'), dpi=300)
    plt.close()


# ── Figure 2: Accuracy Comparison ────────────────────────────────────────────
def plot_fig02_accuracy_comparison(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']
    means = []
    stds = []
    colors = [PALETTE[a] for a in approaches]

    for a in approaches:
        sub = df_window[df_window['approach'] == a]
        means.append(sub['accuracy'].mean())
        stds.append(sub['accuracy'].std())

    bars = ax.bar(approaches, means, yerr=stds, capsize=6, color=colors, edgecolor='black', alpha=0.85, width=0.55)
    ax.set_ylabel('Accuracy')
    ax.set_title('Figure 2: Accuracy Comparison Across Approaches')
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y')

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + (s if not np.isnan(s) else 0) + 0.02,
                f'{m:.3f} ± {s:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig02_accuracy_comparison.png'), dpi=300)
    plt.close()


# ── Figure 3: F1 Across Streaming Windows ────────────────────────────────────
def plot_fig03_f1_across_windows(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']

    # Annotate drift regimes from typical seed 42
    drift_info = df_window[df_window['seed'] == 42].drop_duplicates('window_id').sort_values('window_id')
    for _, row in drift_info.iterrows():
        w = row['window_id']
        dt = row.get('drift_type', 'none')
        c = DRIFT_COLORS.get(dt, '#ffffff')
        ax.axvspan(w - 0.45, w + 0.45, color=c, alpha=0.35, zorder=0)

    # Plot F1 trajectories
    for a in approaches:
        sub = df_window[df_window['approach'] == a]
        agg = sub.groupby('window_id')['f1'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['window_id'], agg['mean'], label=a, color=PALETTE[a], marker='o', linewidth=2.5, zorder=3)
        ax.fill_between(agg['window_id'], agg['mean'] - agg['std'], agg['mean'] + agg['std'],
                        color=PALETTE[a], alpha=0.15, zorder=2)

    # Legend for drift types
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    drift_patches = [
        Patch(color=DRIFT_COLORS['none'], alpha=0.5, label='No Drift'),
        Patch(color=DRIFT_COLORS['covariate'], alpha=0.5, label='Covariate Drift'),
        Patch(color=DRIFT_COLORS['concept'], alpha=0.5, label='Concept Drift'),
        Patch(color=DRIFT_COLORS['mixed'], alpha=0.5, label='Mixed Drift'),
    ]

    leg1 = ax.legend(handles, labels, loc='lower left', framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=drift_patches, loc='lower right', title='Drift Regimes', framealpha=0.9)

    ax.set_xlabel('Streaming Window (500 samples/window)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Figure 3: F1 Score Trajectory Across Streaming Windows with Drift Annotations')
    ax.set_xticks(range(16))
    ax.set_ylim(0.4, 1.02)
    ax.grid(True)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig03_f1_across_windows.png'), dpi=300)
    plt.close()


# ── Figure 4: UCB1 Arm Selection ─────────────────────────────────────────────
def plot_fig04_ucb1_arm_selection(df_bandit, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    arms = ['RF', 'ET', 'Ensemble']

    # Subplot 1: Total selection counts across all seeds
    counts = df_bandit['selected_model'].value_counts()
    vals = [counts.get(arm, 0) for arm in arms]
    colors = [ARM_PALETTE[arm] for arm in arms]

    bars = ax1.bar(arms, vals, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    ax1.set_ylabel('Total Window Selections (80 Total)')
    ax1.set_title('(a) Overall Arm Selection Frequency')
    ax1.grid(axis='y')
    for bar, v in zip(bars, vals):
        pct = (v / sum(vals)) * 100 if sum(vals) > 0 else 0
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v} ({pct:.1f}%)',
                 ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Selection broken down by drift type
    if 'drift_type' in df_bandit.columns:
        drift_types = ['none', 'covariate', 'concept', 'mixed']
        labels = ['No Drift', 'Covariate', 'Concept', 'Mixed']
        x = np.arange(len(drift_types))
        width = 0.25

        for i, arm in enumerate(arms):
            arm_counts = []
            for dt in drift_types:
                sub = df_bandit[df_bandit['drift_type'] == dt]
                c = (sub['selected_model'] == arm).sum()
                arm_counts.append(c)
            offset = (i - 1) * width
            ax2.bar(x + offset, arm_counts, width, label=arm, color=ARM_PALETTE[arm],
                    edgecolor='black', alpha=0.85)

        ax2.set_xlabel('Drift Type')
        ax2.set_ylabel('Selection Count')
        ax2.set_title('(b) Arm Selection by Drift Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend(title='Bandit Arm')
        ax2.grid(axis='y')
    else:
        # Fallback to selection per window
        window_arm = df_bandit.groupby(['window_id', 'selected_model']).size().unstack(fill_value=0)
        for arm in arms:
            if arm not in window_arm.columns:
                window_arm[arm] = 0
        window_arm[arms].plot(kind='bar', stacked=True, ax=ax2, color=colors, edgecolor='black', alpha=0.85)
        ax2.set_xlabel('Streaming Window')
        ax2.set_ylabel('Selection Count')
        ax2.set_title('(b) Arm Selection Dynamics per Window')
        ax2.legend(title='Bandit Arm')
        ax2.grid(axis='y')

    plt.suptitle('Figure 4: UCB1 Bandit Arm Selection Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig04_ucb1_arm_selection.png'), dpi=300)
    plt.close()


# ── Figure 5: F1 by Drift Type ───────────────────────────────────────────────
def plot_fig05_f1_by_drift_type(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    drift_order = ['none', 'covariate', 'concept', 'mixed']
    drift_labels = ['No Drift', 'Covariate Drift', 'Concept Drift', 'Mixed Drift']
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']

    x = np.arange(len(drift_order))
    width = 0.25

    for i, a in enumerate(approaches):
        means = []
        stds = []
        for dt in drift_order:
            sub = df_window[(df_window['approach'] == a) & (df_window['drift_type'] == dt)]
            means.append(sub['f1'].mean() if len(sub) > 0 else 0)
            stds.append(sub['f1'].std() if len(sub) > 0 else 0)

        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=a,
               color=PALETTE[a], edgecolor='black', alpha=0.85)

    ax.set_ylabel('F1 Score')
    ax.set_title('Figure 5: F1 Performance Breakdown by Drift Type')
    ax.set_xticks(x)
    ax.set_xticklabels(drift_labels)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower left')
    ax.grid(axis='y')

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig05_f1_by_drift_type.png'), dpi=300)
    plt.close()


# ── Figure 6: Cumulative CPU Time ────────────────────────────────────────────
def plot_fig06_cumulative_cpu_time(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']

    for a in approaches:
        sub = df_window[df_window['approach'] == a].sort_values(['seed', 'window_id'])
        sub['cum_cpu'] = sub.groupby('seed')['cpu_time'].cumsum()
        agg = sub.groupby('window_id')['cum_cpu'].agg(['mean', 'std']).reset_index()

        ax.plot(agg['window_id'], agg['mean'], label=a, color=PALETTE[a], marker='s', linewidth=2.5)
        ax.fill_between(agg['window_id'], agg['mean'] - agg['std'], agg['mean'] + agg['std'],
                        color=PALETTE[a], alpha=0.15)

    ax.set_xlabel('Streaming Window')
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.set_title('Figure 6: Cumulative CPU Time Trajectory Across Windows')
    ax.set_xticks(range(16))
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig06_cumulative_cpu_time.png'), dpi=300)
    plt.close()


# ── Figure 7: Memory Usage ───────────────────────────────────────────────────
def plot_fig07_memory_usage(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']
    means = []
    stds = []
    colors = [PALETTE[a] for a in approaches]

    for a in approaches:
        sub = df_window[df_window['approach'] == a]
        means.append(sub['ram_mb'].mean())
        stds.append(sub['ram_mb'].std())

    bars = ax.bar(approaches, means, yerr=stds, capsize=6, color=colors, edgecolor='black', alpha=0.85, width=0.55)
    ax.set_ylabel('RAM Usage (MB)')
    ax.set_title('Figure 7: Memory (RAM) Usage Across Approaches')
    ax.grid(axis='y')

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + (s if not np.isnan(s) else 0) + 2,
                f'{m:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig07_memory_usage.png'), dpi=300)
    plt.close()


# ── Figure 8: Performance vs CPU Cost (Pareto) ───────────────────────────────
def plot_fig08_performance_vs_cpu_cost(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    approaches = ['Frozen RF', 'Retrained RF', 'UCB1 Adaptive Ensemble']

    for a in approaches:
        sub = df_window[df_window['approach'] == a]
        total_cpu_per_seed = sub.groupby('seed')['cpu_time'].sum()
        mean_f1_per_seed = sub.groupby('seed')['f1'].mean()

        cpu_mean = total_cpu_per_seed.mean()
        cpu_std = total_cpu_per_seed.std()
        f1_mean = mean_f1_per_seed.mean()
        f1_std = mean_f1_per_seed.std()

        ax.errorbar(cpu_mean, f1_mean, xerr=cpu_std, yerr=f1_std,
                    fmt='o', markersize=12, label=a, color=PALETTE[a],
                    capsize=6, elinewidth=2, markeredgecolor='black')
        ax.annotate(f'{a}\n(F1: {f1_mean:.3f}, CPU: {cpu_mean:.1f}s)',
                    (cpu_mean, f1_mean), textcoords='offset points',
                    xytext=(15, -5), fontweight='bold', fontsize=10)

    ax.set_xlabel('Total Cumulative CPU Time (seconds)')
    ax.set_ylabel('Mean F1 Score Across Stream')
    ax.set_title('Figure 8: Performance vs Computational Cost Trade-off')
    ax.grid(True)
    ax.legend(loc='lower right')

    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig08_performance_vs_cpu_cost.png'), dpi=300)
    plt.close()


# ── Figure 9: Retraining Cost ────────────────────────────────────────────────
def plot_fig09_retraining_cost(df_retraining, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    target_approaches = ['Retrained RF', 'UCB1 Adaptive Ensemble']
    colors = [PALETTE[a] for a in target_approaches]

    # Subplot 1: Total Retraining CPU Time
    train_times = []
    train_stds = []
    for a in target_approaches:
        sub = df_retraining[df_retraining['approach'] == a]
        per_seed = sub.groupby('seed')['retraining_cpu_time'].sum()
        train_times.append(per_seed.mean())
        train_stds.append(per_seed.std())

    bars1 = ax1.bar(target_approaches, train_times, yerr=train_stds, capsize=6,
                    color=colors, edgecolor='black', alpha=0.85, width=0.5)
    ax1.set_ylabel('Total Retraining CPU Time (s)')
    ax1.set_title('(a) Retraining CPU Time')
    ax1.grid(axis='y')
    for bar, m in zip(bars1, train_times):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 0.2, f'{m:.2f}s',
                 ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Total Retrain Events
    events = []
    event_stds = []
    for a in target_approaches:
        sub = df_retraining[df_retraining['approach'] == a]
        per_seed = sub.groupby('seed')['retrained'].sum()
        events.append(per_seed.mean())
        event_stds.append(per_seed.std())

    bars2 = ax2.bar(target_approaches, events, yerr=event_stds, capsize=6,
                    color=colors, edgecolor='black', alpha=0.85, width=0.5)
    ax2.set_ylabel('Total Retraining Events')
    ax2.set_title('(b) Adaptation / Retraining Events')
    ax2.grid(axis='y')
    for bar, e in zip(bars2, events):
        ax2.text(bar.get_x() + bar.get_width() / 2, e + 0.3, f'{e:.1f}',
                 ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Figure 9: Computational Retraining Cost Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    safe_savefig(os.path.join(output_dir, 'fig09_retraining_cost.png'), dpi=300)
    plt.close()


def generate_all_figures(output_dir, df_window, df_bandit, df_retraining):
    """Generate all 9 essential publication figures."""
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    plot_fig01_f1_comparison(df_window, output_dir)
    plot_fig02_accuracy_comparison(df_window, output_dir)
    plot_fig03_f1_across_windows(df_window, output_dir)
    plot_fig04_ucb1_arm_selection(df_bandit, output_dir)
    plot_fig05_f1_by_drift_type(df_window, output_dir)
    plot_fig06_cumulative_cpu_time(df_window, output_dir)
    plot_fig07_memory_usage(df_window, output_dir)
    plot_fig08_performance_vs_cpu_cost(df_window, output_dir)
    plot_fig09_retraining_cost(df_retraining, output_dir)
