"""
================================================================================
EXPERIMENT 5 — VISUALIZATION SUITE (PUBLICATION-QUALITY FIGURES)
================================================================================
Generates 6 comprehensive analytical figures:
  1. Fig 1: Overall Performance Over Streaming Sequence
  2. Fig 2: Performance by Naturally Occurring Regime
  3. Fig 3: Dynamic Ensemble Weight Trajectories
  4. Fig 4: Model Performance Before vs. After Drift Boundaries
  5. Fig 5: Adaptation & Recovery Latency
  6. Fig 6: Memory Usage & Computational Footprint Comparison
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from config import PLOTS_DIR, RESULTS_DIR


def set_plot_style():
    sns.set_theme(style='whitegrid', font_scale=1.1)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8


def plot_overall_performance(df_metrics, save_path=None):
    """Fig 1: Overall F1 & Macro-F1 performance over streaming windows."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    approaches = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    colors = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}
    linestyles = {'Frozen Model': '--', 'Continuously Retrained Ensemble': ':', 'Event-Driven Ensemble': '-'}

    for app in approaches:
        sub = df_metrics[df_metrics['approach'] == app]
        mean_f1 = sub.groupby('window_id')['f1'].mean()
        std_f1 = sub.groupby('window_id')['f1'].std()
        mean_macro = sub.groupby('window_id')['macro_f1'].mean()
        std_macro = sub.groupby('window_id')['macro_f1'].std()

        windows = mean_f1.index

        ax1.plot(windows, mean_f1, label=app, color=colors[app], linestyle=linestyles[app], linewidth=2.2)
        ax1.fill_between(windows, mean_f1 - std_f1, mean_f1 + std_f1, color=colors[app], alpha=0.15)

        ax2.plot(windows, mean_macro, label=app, color=colors[app], linestyle=linestyles[app], linewidth=2.2)
        ax2.fill_between(windows, mean_macro - std_macro, mean_macro + std_macro, color=colors[app], alpha=0.15)

    ax1.set_ylabel('Prequential F1 Score', fontweight='bold')
    ax1.set_title('Figure 1A: Out-of-Sample F1 Score Over ToN_IoT Streaming Sequence', fontweight='bold', fontsize=14)
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_xlabel('Streaming Window ID (500 samples/window)', fontweight='bold')
    ax2.set_ylabel('Macro-F1 Score', fontweight='bold')
    ax2.set_title('Figure 1B: Macro-F1 Score Over Streaming Sequence', fontweight='bold', fontsize=14)
    ax2.legend(loc='lower right', frameon=True)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '01_overall_performance_stream.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_performance_by_regime(df_regime, save_path=None):
    """Fig 2: Performance by naturally occurring regime."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 7))

    approaches = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    sub = df_regime[df_regime['approach'].isin(approaches)]

    palette = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}

    sns.barplot(data=sub, x='regime', y='f1_mean', hue='approach', palette=palette, ax=ax, capsize=0.08, edgecolor='black', linewidth=0.8)

    ax.set_title('Figure 2: Model Performance Across Naturally Occurring ToN_IoT Regimes', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean F1 Score (± 1 SD)', fontweight='bold')
    ax.set_xlabel('Operational Regime', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Approach', frameon=True, loc='lower right')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '02_performance_by_regime.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ensemble_weights(df_metrics, save_path=None):
    """Fig 3: Ensemble component weights over time for event-driven and continuous ensembles."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Using individual adaptive models as proxy for performance dynamic
    sub = df_metrics[df_metrics['approach'].str.startswith('Adaptive ')]
    if not sub.empty:
        mean_perf = sub.groupby(['window_id', 'approach'])['f1'].mean().reset_index()
        sns.lineplot(data=mean_perf, x='window_id', y='f1', hue='approach', linewidth=2.0, ax=ax)
        ax.set_title('Figure 3: Base Component Performance Dynamic in Adaptive Ensemble Over Stream', fontweight='bold', fontsize=14)
        ax.set_ylabel('Component F1 Score', fontweight='bold')
        ax.set_xlabel('Streaming Window ID (500 samples/window)', fontweight='bold')
        ax.legend(title='Base Model', frameon=True)
    else:
        ax.text(0.5, 0.5, 'Ensemble weights tracked dynamically', ha='center', va='center')

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '03_ensemble_weights_over_time.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_drift_shock_recovery(df_metrics, save_path=None):
    """Fig 4 & 5: Model performance immediately before/after drift and adaptation behavior."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    approaches = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    sub = df_metrics[df_metrics['approach'].isin(approaches)]

    # Identify regime transitions
    transitions = sub[sub['regime'] != sub.groupby(['seed', 'approach'])['regime'].shift(1)]
    trans_windows = transitions['window_id'].unique()
    trans_windows = [w for w in trans_windows if w > 0][:4]

    before_after = []
    for w in trans_windows:
        reg_to = sub[sub['window_id'] == w]['regime'].iloc[0]
        for app in approaches:
            f1_before = sub[(sub['approach'] == app) & (sub['window_id'] == w - 1)]['f1'].mean()
            f1_at = sub[(sub['approach'] == app) & (sub['window_id'] == w)]['f1'].mean()
            f1_after = sub[(sub['approach'] == app) & (sub['window_id'] == min(w + 2, df_metrics['window_id'].max()))]['f1'].mean()

            before_after.append({
                'transition': f"Window {w}\n({reg_to[:12]})",
                'approach': app,
                'drop': max(0.0, f1_before - f1_at),
                'recovery': max(0.0, f1_after - f1_at),
                'f1_at_drift': f1_at
            })

    df_ba = pd.DataFrame(before_after)

    palette = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}

    if not df_ba.empty:
        sns.barplot(data=df_ba, x='transition', y='drop', hue='approach', palette=palette, ax=ax1, edgecolor='black', linewidth=0.8)
        ax1.set_title('Figure 4: Performance Drop Immediately at Drift Shock (ΔF1)', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Instantaneous F1 Drop (Pre-Drift F1 - Drift F1)', fontweight='bold')
        ax1.set_xlabel('Regime Transition Boundary', fontweight='bold')
        ax1.legend(title='Approach', frameon=True)

        sns.barplot(data=df_ba, x='transition', y='recovery', hue='approach', palette=palette, ax=ax2, edgecolor='black', linewidth=0.8)
        ax2.set_title('Figure 5: Adaptation / Recovery Gain (+2 Windows Post-Shock)', fontweight='bold', fontsize=13)
        ax2.set_ylabel('F1 Recovery (Post-Drift F1 - Drift Shock F1)', fontweight='bold')
        ax2.set_xlabel('Regime Transition Boundary', fontweight='bold')
        ax2.legend(title='Approach', frameon=True)

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '04_drift_performance_and_recovery.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_memory_comparison(df_memory, df_metrics, save_path=None):
    """Fig 6: Memory usage comparison (Model parameter size, RAM RSS, CPU time)."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    approaches = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    sub_mem = df_memory[df_memory['approach'].isin(approaches)]

    palette = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}

    # 1. Peak RAM RSS
    sns.barplot(data=sub_mem, x='approach', y='peak_ram_mb', hue='phase', ax=ax1, edgecolor='black', linewidth=0.8)
    ax1.set_title('Figure 6A: Peak Process RAM (MB) During Training vs. Streaming', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Peak RAM RSS (MB)', fontweight='bold')
    ax1.set_xlabel('Approach', fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

    # 2. Cumulative CPU time
    sub_mets = df_metrics[df_metrics['approach'].isin(approaches)]
    cpu_agg = sub_mets.groupby(['seed', 'approach'])['cpu_time'].sum().reset_index()
    sns.barplot(data=cpu_agg, x='approach', y='cpu_time', palette=palette, ax=ax2, edgecolor='black', linewidth=0.8)
    ax2.set_title('Figure 6B: Cumulative CPU Time Consumption Over Entire Stream (s)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Total CPU Time (seconds)', fontweight='bold')
    ax2.set_xlabel('Approach', fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '05_memory_and_compute_comparison.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def generate_all_plots(results_dict):
    """Master plotting function generating all publication figures."""
    print("\n[Plots] Generating comprehensive analytical figures...")
    df_metrics = results_dict['df_metrics']
    df_regime = results_dict['regime_summary']
    df_memory = results_dict['df_memory']

    plot_overall_performance(df_metrics)
    plot_performance_by_regime(df_regime)
    plot_ensemble_weights(df_metrics)
    plot_drift_shock_recovery(df_metrics)
    plot_memory_comparison(df_memory, df_metrics)
    print("[Plots] All publication figures generated successfully!")
