"""
================================================================================
EXPERIMENT 5 — VISUALIZATION SUITE (PUBLICATION-QUALITY FIGURES)
================================================================================
Generates 6 comprehensive analytical figures using pure matplotlib:
  1. Fig 1: Overall Performance Over Streaming Sequence
  2. Fig 2: Performance by Naturally Occurring Regime
  3. Fig 3: Dynamic Base Component Performance Over Stream
  4. Fig 4: Performance Drop at Drift Shock (Delta F1)
  5. Fig 5: Adaptation / Recovery Gain Post-Shock
  6. Fig 6: Memory Usage & Computational Footprint Comparison
================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIR, RESULTS_DIR


def set_plot_style():
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#b0b0b0'
    plt.rcParams['axes.linewidth'] = 0.9
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'


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
        std_f1 = sub.groupby('window_id')['f1'].std().fillna(0)
        mean_macro = sub.groupby('window_id')['macro_f1'].mean()
        std_macro = sub.groupby('window_id')['macro_f1'].std().fillna(0)

        windows = mean_f1.index

        ax1.plot(windows, mean_f1, label=app, color=colors[app], linestyle=linestyles[app], linewidth=2.2)
        ax1.fill_between(windows, mean_f1 - std_f1, mean_f1 + std_f1, color=colors[app], alpha=0.15)

        ax2.plot(windows, mean_macro, label=app, color=colors[app], linestyle=linestyles[app], linewidth=2.2)
        ax2.fill_between(windows, mean_macro - std_macro, mean_macro + std_macro, color=colors[app], alpha=0.15)

    ax1.set_ylabel('Prequential F1 Score', fontweight='bold', fontsize=12)
    ax1.set_title('Figure 1A: Out-of-Sample F1 Score Over ToN_IoT Streaming Sequence', fontweight='bold', fontsize=14)
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_xlabel('Streaming Window ID (500 samples/window)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Macro-F1 Score', fontweight='bold', fontsize=12)
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
    colors = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}

    regimes = df_regime['regime'].unique()
    x = np.arange(len(regimes))
    width = 0.25

    for i, app in enumerate(approaches):
        sub = df_regime[df_regime['approach'] == app]
        reg_means = [sub[sub['regime'] == r]['f1_mean'].values[0] if len(sub[sub['regime'] == r]) > 0 else 0 for r in regimes]
        reg_stds = [sub[sub['regime'] == r]['f1_std'].values[0] if len(sub[sub['regime'] == r]) > 0 else 0 for r in regimes]
        
        ax.bar(x + i * width, reg_means, width, yerr=reg_stds, label=app, color=colors[app], capsize=4, edgecolor='black', linewidth=0.8)

    ax.set_title('Figure 2: Model Performance Across Naturally Occurring ToN_IoT Regimes', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean F1 Score (± 1 SD)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Operational Regime', fontweight='bold', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Approach', frameon=True, loc='lower right')

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '02_performance_by_regime.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ensemble_weights(df_metrics, save_path=None):
    """Fig 3: Base component performance dynamics in adaptive ensemble over stream."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    sub = df_metrics[df_metrics['approach'].str.startswith('Adaptive ')]
    comp_colors = {
        'Adaptive Random Forest': '#1f77b4',
        'Adaptive Extra Trees': '#ff7f0e',
        'Adaptive Gradient Boosting': '#2ca02c'
    }

    for app in sub['approach'].unique():
        app_sub = sub[sub['approach'] == app]
        mean_perf = app_sub.groupby('window_id')['f1'].mean()
        std_perf = app_sub.groupby('window_id')['f1'].std().fillna(0)
        c = comp_colors.get(app, '#333333')
        ax.plot(mean_perf.index, mean_perf, label=app, color=c, linewidth=2.0)
        ax.fill_between(mean_perf.index, mean_perf - std_perf, mean_perf + std_perf, color=c, alpha=0.1)

    ax.set_title('Figure 3: Base Component Performance Dynamic in Adaptive Ensemble Over Stream', fontweight='bold', fontsize=14)
    ax.set_ylabel('Component F1 Score', fontweight='bold', fontsize=12)
    ax.set_xlabel('Streaming Window ID (500 samples/window)', fontweight='bold', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Base Model', frameon=True, loc='lower right')

    plt.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, '03_ensemble_weights_over_time.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_drift_shock_recovery(df_metrics, save_path=None):
    """Fig 4 & 5: Model performance drop immediately at drift shock and adaptation gain."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    approaches = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    colors = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}
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
            })

    df_ba = pd.DataFrame(before_after)

    if not df_ba.empty:
        trans_names = df_ba['transition'].unique()
        x = np.arange(len(trans_names))
        width = 0.25

        for i, app in enumerate(approaches):
            app_df = df_ba[df_ba['approach'] == app]
            drops = [app_df[app_df['transition'] == t]['drop'].values[0] if len(app_df[app_df['transition'] == t]) > 0 else 0 for t in trans_names]
            recs = [app_df[app_df['transition'] == t]['recovery'].values[0] if len(app_df[app_df['transition'] == t]) > 0 else 0 for t in trans_names]

            ax1.bar(x + i * width, drops, width, label=app, color=colors[app], edgecolor='black', linewidth=0.8)
            ax2.bar(x + i * width, recs, width, label=app, color=colors[app], edgecolor='black', linewidth=0.8)

        ax1.set_title('Figure 4: Performance Drop Immediately at Drift Shock (ΔF1)', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Instantaneous F1 Drop (Pre-Drift F1 - Drift F1)', fontweight='bold', fontsize=11)
        ax1.set_xlabel('Regime Transition Boundary', fontweight='bold', fontsize=11)
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(trans_names, fontsize=10)
        ax1.legend(title='Approach', frameon=True)

        ax2.set_title('Figure 5: Adaptation / Recovery Gain (+2 Windows Post-Shock)', fontweight='bold', fontsize=13)
        ax2.set_ylabel('F1 Recovery (Post-Drift F1 - Drift Shock F1)', fontweight='bold', fontsize=11)
        ax2.set_xlabel('Regime Transition Boundary', fontweight='bold', fontsize=11)
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(trans_names, fontsize=10)
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
    colors = {'Frozen Model': '#d62728', 'Continuously Retrained Ensemble': '#1f77b4', 'Event-Driven Ensemble': '#2ca02c'}

    # 1. Peak RAM RSS
    sub_mem = df_memory[df_memory['approach'].isin(approaches)]
    phases = sub_mem['phase'].unique()
    x = np.arange(len(approaches))
    width = 0.35

    phase_colors = {'initial_training': '#4c72b0', 'streaming_end': '#dd8452'}
    for p_idx, ph in enumerate(phases):
        ph_sub = sub_mem[sub_mem['phase'] == ph]
        vals = [ph_sub[ph_sub['approach'] == a]['peak_ram_mb'].mean() if len(ph_sub[ph_sub['approach'] == a]) > 0 else 0 for a in approaches]
        ax1.bar(x + p_idx * width, vals, width, label=ph.replace('_', ' ').title(), color=phase_colors.get(ph, '#888'), edgecolor='black', linewidth=0.8)

    ax1.set_title('Figure 6A: Peak Process RAM (MB) During Training vs. Streaming', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Peak RAM RSS (MB)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Approach', fontweight='bold', fontsize=11)
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(approaches, rotation=15, ha='right', fontsize=10)
    ax1.legend(title='Phase', frameon=True)

    # 2. Cumulative CPU time
    sub_mets = df_metrics[df_metrics['approach'].isin(approaches)]
    cpu_agg = sub_mets.groupby(['seed', 'approach'])['cpu_time'].sum().reset_index()
    mean_cpu = [cpu_agg[cpu_agg['approach'] == a]['cpu_time'].mean() for a in approaches]
    std_cpu = [cpu_agg[cpu_agg['approach'] == a]['cpu_time'].std() for a in approaches]

    ax2.bar(approaches, mean_cpu, yerr=std_cpu, capsize=5, color=[colors[a] for a in approaches], edgecolor='black', linewidth=0.8)
    ax2.set_title('Figure 6B: Cumulative CPU Time Consumption Over Entire Stream (s)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Total CPU Time (seconds)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Approach', fontweight='bold', fontsize=11)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right', fontsize=10)

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
