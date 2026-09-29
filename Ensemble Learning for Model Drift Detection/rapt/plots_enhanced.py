"""
================================================================================
ENHANCED RAPT PUBLICATION PLOT SUITE
================================================================================
Generates 3 simplified, single-idea publication figures at 300 DPI:
  1. fig9_enhanced_f1_accuracy_bar.png: F1 Score and Accuracy comparison across 6 strategies.
  2. fig10_enhanced_f1_vs_cpu_pareto.png: Pareto frontier (F1 Score vs Adaptation CPU time).
  3. fig11_enhanced_streaming_f1_trajectory.png: Prequential F1 score across 38 streaming windows.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_rapt_dir = os.path.dirname(os.path.abspath(__file__))
if _rapt_dir not in sys.path:
    sys.path.insert(0, _rapt_dir)

from config import RESULTS_DIR, FIGURES_DIR

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def generate_enhanced_plots():
    summary_path = os.path.join(RESULTS_DIR, "enhanced_summary_metrics.csv")
    window_path = os.path.join(RESULTS_DIR, "enhanced_window_metrics.csv")

    if not os.path.exists(summary_path) or not os.path.exists(window_path):
        print("Summary or window CSV missing; run run_enhanced_rapt_benchmark.py first.")
        return

    df_summary = pd.read_csv(summary_path)
    df_window = pd.read_csv(window_path)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    brain_fig_dir = r"C:\Users\emhaenn\.gemini\antigravity\brain\df805942-8788-4248-aa21-b6cc1da570ac\figures"
    os.makedirs(brain_fig_dir, exist_ok=True)

    # Order strategies logically
    order = [
        'Frozen Ensemble',
        'RAPT (K=8)',
        'Two-Tier Hybrid RAPT',
        'Enhanced Hybrid RAPT (Ours)',
        'Event-Driven Baseline',
        'Continuous Retraining'
    ]

    df_summary['Strategy_Cat'] = pd.Categorical(df_summary['Strategy'], categories=order, ordered=True)
    df_summary = df_summary.sort_values('Strategy_Cat')

    # --------------------------------------------------------------------------
    # Figure 9: Grouped Bar Chart (F1 & Accuracy)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    x = np.arange(len(df_summary))
    width = 0.35

    rects1 = ax.bar(x - width/2, df_summary['F1_Mean'], width, yerr=df_summary['F1_Std'], capsize=4,
                     label='F1 Score', color='#2b5c8f', edgecolor='black', alpha=0.9, error_kw={'ecolor': '#1a334d', 'linewidth': 1.2})
    rects2 = ax.bar(x + width/2, df_summary['Acc_Mean'], width, yerr=df_summary['Acc_Std'], capsize=4,
                     label='Accuracy', color='#389078', edgecolor='black', alpha=0.9, error_kw={'ecolor': '#1d4d40', 'linewidth': 1.2})

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Performance Across All 6 Evaluated Strategies', fontsize=14, fontweight='bold', pad=18)
    ax.set_xticks(x)
    labels = [s.replace(' (Ours)', '\n(Ours)').replace(' Baseline', '\nBaseline').replace(' Retraining', '\nRetraining').replace(' Ensemble', '\nEnsemble') for s in df_summary['Strategy']]
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.40, 0.95)
    ax.legend(fontsize=11, loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for rect, std in zip(rects1, df_summary['F1_Std']):
        h = rect.get_height()
        ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h + std + 0.012),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    for rect, std in zip(rects2, df_summary['Acc_Std']):
        h = rect.get_height()
        ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h + std + 0.012),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    f9_path1 = os.path.join(FIGURES_DIR, "fig9_enhanced_f1_accuracy_bar.png")
    f9_path2 = os.path.join(brain_fig_dir, "fig9_enhanced_f1_accuracy_bar.png")
    plt.savefig(f9_path1, dpi=300); plt.savefig(f9_path2, dpi=300); plt.close()

    # --------------------------------------------------------------------------
    # Figure 10: Pareto Frontier (F1 vs Adaptation CPU)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    colors = {
        'Frozen Ensemble': '#7f7f7f',
        'RAPT (K=8)': '#d62728',
        'Two-Tier Hybrid RAPT': '#ff7f0e',
        'Enhanced Hybrid RAPT (Ours)': '#9467bd',
        'Event-Driven Baseline': '#1f77b4',
        'Continuous Retraining': '#2ca02c'
    }

    markers = {
        'Frozen Ensemble': 'o',
        'RAPT (K=8)': 's',
        'Two-Tier Hybrid RAPT': '^',
        'Enhanced Hybrid RAPT (Ours)': '*',
        'Event-Driven Baseline': 'D',
        'Continuous Retraining': 'P'
    }

    for _, row in df_summary.iterrows():
        s = row['Strategy']
        ax.errorbar(row['Adapt_CPU_Mean'], row['F1_Mean'], yerr=row['F1_Std'], xerr=row['Adapt_CPU_Std'],
                    fmt=markers.get(s, 'o'), color=colors.get(s, '#333333'), markersize=10 if s != 'Enhanced Hybrid RAPT (Ours)' else 14,
                    capsize=4, label=s, alpha=0.9, markeredgecolor='black')

    ax.set_xlabel('Adaptation CPU Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Whole-Stream Mean F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: F1 Score vs. Adaptation CPU Cost', fontsize=14, fontweight='bold', pad=18)
    ax.legend(fontsize=10, loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
    ax.grid(linestyle='--', alpha=0.6)

    plt.tight_layout()
    f10_path1 = os.path.join(FIGURES_DIR, "fig10_enhanced_f1_vs_cpu_pareto.png")
    f10_path2 = os.path.join(brain_fig_dir, "fig10_enhanced_f1_vs_cpu_pareto.png")
    plt.savefig(f10_path1, dpi=300); plt.savefig(f10_path2, dpi=300); plt.close()

    # --------------------------------------------------------------------------
    # Figure 11: Streaming F1 Trajectory
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    win_grouped = df_window.groupby(['approach', 'window_id'])['f1'].mean().reset_index()

    for app in order:
        sub = win_grouped[win_grouped['approach'] == app]
        if not sub.empty:
            lw = 2.5 if 'Enhanced' in app or 'Event' in app else 1.5
            ls = '-' if 'Ours' in app or 'Continuous' in app or 'Event' in app else '--'
            ax.plot(sub['window_id'], sub['f1'], label=app, color=colors.get(app, '#333333'), linewidth=lw, linestyle=ls, alpha=0.85)

    ax.set_xlabel('Streaming Window ID (500 samples/win)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prequential F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Streaming Prequential F1 Score Trajectory Across 38 Windows', fontsize=14, fontweight='bold', pad=18)
    ax.legend(fontsize=10, loc='lower left', frameon=True, facecolor='white', framealpha=0.95)
    ax.grid(linestyle='--', alpha=0.6)

    plt.tight_layout()
    f11_path1 = os.path.join(FIGURES_DIR, "fig11_enhanced_streaming_f1_trajectory.png")
    f11_path2 = os.path.join(brain_fig_dir, "fig11_enhanced_streaming_f1_trajectory.png")
    plt.savefig(f11_path1, dpi=300); plt.savefig(f11_path2, dpi=300); plt.close()

    print("Successfully generated all 3 enhanced publication figures!")


if __name__ == '__main__':
    generate_enhanced_plots()
