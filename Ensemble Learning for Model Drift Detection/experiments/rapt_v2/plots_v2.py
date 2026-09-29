"""
Publication-Quality Visualization Generator for RAPT-v2
Generates 11 standalone 300-DPI plots in experiments/rapt_v2/plots/ using standard Matplotlib.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = "experiments/rapt_v2/plots"
COMPARISON_DIR = "experiments/rapt_v2/comparison"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

VARIANT_COLORS = {
    'Frozen': '#7f7f7f',
    'Event-Driven': '#d62728',
    'Full Retraining': '#ff7f0e',
    'RAPT-v1': '#1f77b4',
    'RAPT-A': '#aec7e8',
    'RAPT-B': '#98df8a',
    'RAPT-C': '#2ca02c',
    'RAPT-D': '#ffbb78',
    'RAPT-E': '#2b5c8f'  # Dark Cyan/Blue
}

def generate_all_v2_plots(df_summary, df_per_seed, df_per_window, df_hierarchy, dataset_label="9A"):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    prefix = f"{dataset_label}_"
    print(f"Generating RAPT-v2 plots for {dataset_label} in {PLOTS_DIR}...")
    
    methods = df_summary['method'].tolist()
    colors = [VARIANT_COLORS.get(m, '#333333') for m in methods]

    # -------------------------------------------------------------
    # Fig 1: Macro F1 Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(methods, df_summary['macro_f1_mean'], yerr=df_summary['macro_f1_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Macro F1 Score", fontsize=11, fontweight='bold')
    ax.set_title(f"RAPT-v2 ({dataset_label}): Predictive Performance (Macro F1)", fontsize=12, fontweight='bold')
    ax.set_ylim(0.75, 1.02)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width()/2., h),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}v2_1_macro_f1.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 2: Total CPU Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(methods, df_summary['total_cpu_sec_mean'], yerr=df_summary['total_cpu_sec_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Total CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_title(f"RAPT-v2 ({dataset_label}): Total CPU Execution Time", fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s", (bar.get_x() + bar.get_width()/2., h),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}v2_2_total_cpu.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 3: Adaptation CPU Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(methods, df_summary['adaptation_cpu_sec_mean'], yerr=df_summary['adaptation_cpu_sec_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Adaptation CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_title(f"RAPT-v2 ({dataset_label}): Adaptation CPU Overhead", fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}s", (bar.get_x() + bar.get_width()/2., h),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}v2_3_adaptation_cpu.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 4: Retraining Events Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(methods, df_summary['retrain_events_mean'], color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Retraining Events", fontsize=11, fontweight='bold')
    ax.set_title(f"RAPT-v2 ({dataset_label}): Retraining Event Frequency", fontsize=12, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2., h),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}v2_4_retrain_events.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 5: Pareto F1 vs CPU Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for idx, r in df_summary.iterrows():
        m = r['method']
        ax.scatter(r['total_cpu_sec_mean'], r['macro_f1_mean'], s=140,
                   color=VARIANT_COLORS.get(m, '#333'), edgecolors='black', label=m, zorder=5)
        ax.annotate(m, (r['total_cpu_sec_mean'], r['macro_f1_mean']),
                    xytext=(6, -4), textcoords='offset points', fontsize=9, fontweight='bold')
        
    ax.set_xlabel("Total CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Macro F1 Score", fontsize=11, fontweight='bold')
    ax.set_title(f"RAPT-v2 ({dataset_label}): Predictive Performance vs CPU Cost (Pareto)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}v2_5_pareto.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 11: Adaptation Hierarchy Usage (Level 0..3)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    if len(df_hierarchy) > 0:
        h_cols = ['level_0_pure_reuse', 'level_1_weight_adapt', 'level_2_partial_update', 'level_3_full_retrain']
        labels = ['Level 0 (Pure Reuse)', 'Level 1 (Weight Adapt)', 'Level 2 (Partial Update)', 'Level 3 (Full Retrain)']
        h_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        
        df_h_rapt = df_hierarchy[df_hierarchy['method'].str.startswith('RAPT')].copy()
        methods_h = df_h_rapt['method'].tolist()
        x = np.arange(len(methods_h))
        bottom = np.zeros(len(methods_h))
        
        for idx, col in enumerate(h_cols):
            vals = df_h_rapt[col].values
            ax.bar(x, vals, bottom=bottom, label=labels[idx], color=h_colors[idx], edgecolor='black', alpha=0.85)
            bottom += vals
            
        ax.set_xticks(x)
        ax.set_xticklabels(methods_h, rotation=15, ha='right')
        ax.set_ylabel("Adaptation Event Count", fontsize=11, fontweight='bold')
        ax.set_title(f"RAPT-v2 ({dataset_label}): Adaptation Hierarchy Level Usage", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}v2_11_hierarchy_usage.png"))
    plt.close(fig)

    print(f"Completed RAPT-v2 plots for {dataset_label}!")
