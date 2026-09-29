"""
Publication-Quality Visualization Generator for Exp 9B (5G NR Latency Dataset)
Generates 11 standalone 300-DPI plots in experiments/exp9b/plots/ using standard Matplotlib.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = "experiments/exp9b/plots"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
METHOD_COLORS = {
    'Frozen': '#7f7f7f',          # Muted Gray
    'Event-Driven': '#d62728',    # Crimson Red
    'Full Retraining': '#ff7f0e', # Orange
    'RAPT': '#1f77b4'             # Deep Blue
}

def generate_all_9b_plots(df_summary, df_per_seed, df_per_window, df_transition):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Generating 11 publication-quality plots in {PLOTS_DIR}...")
    
    # -------------------------------------------------------------
    # 9B-1: Macro F1 by Method
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    methods = df_summary['method'].tolist()
    f1_means = df_summary['macro_f1_mean'].tolist()
    f1_stds = df_summary['macro_f1_std'].tolist()
    colors = [METHOD_COLORS.get(m, '#333333') for m in methods]
    
    bars = ax.bar(methods, f1_means, yerr=f1_stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Macro F1 Score", fontsize=11, fontweight='bold')
    ax.set_title("9B-1: Predictive Performance (Macro F1)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-1_macro_f1.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-2: Accuracy by Method
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    acc_means = df_summary['accuracy_mean'].tolist()
    acc_stds = df_summary['accuracy_std'].tolist()
    
    bars = ax.bar(methods, acc_means, yerr=acc_stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Accuracy", fontsize=11, fontweight='bold')
    ax.set_title("9B-2: Predictive Performance (Accuracy)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-2_accuracy.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-3: Total CPU Time by Method
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    cpu_means = df_summary['total_cpu_sec_mean'].tolist()
    cpu_stds = df_summary['total_cpu_sec_std'].tolist()
    
    bars = ax.bar(methods, cpu_means, yerr=cpu_stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Total CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_title("9B-3: Total CPU Execution Time", fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-3_total_cpu.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-4: Adaptation CPU Time by Method
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    adapt_cpu_means = df_summary['adaptation_cpu_sec_mean'].tolist()
    adapt_cpu_stds = df_summary['adaptation_cpu_sec_std'].tolist()
    
    bars = ax.bar(methods, adapt_cpu_means, yerr=adapt_cpu_stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Adaptation CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_title("9B-4: Adaptation CPU Overhead", fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-4_adaptation_cpu.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-5: Number of Adaptation/Retraining Events
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    retrain_events = df_summary['retrain_events_mean'].tolist()
    
    bars = ax.bar(methods, retrain_events, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel("Count of Retraining Events", fontsize=11, fontweight='bold')
    ax.set_title("9B-5: Model Retraining Frequency", fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-5_retrain_events.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-6: Streaming F1 Over Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    for m in methods:
        df_m = df_per_window[df_per_window['method'] == m]
        roll_acc = df_m.groupby('window_id')['is_correct'].mean().rolling(15, min_periods=1).mean()
        ax.plot(roll_acc.index, roll_acc.values, label=m, color=METHOD_COLORS.get(m, '#333'), linewidth=2.0)
        
    ax.set_xlabel("Streaming Window Index", fontsize=11, fontweight='bold')
    ax.set_ylabel("Rolling Accuracy (15-window mean)", fontsize=11, fontweight='bold')
    ax.set_title("9B-6: Streaming Accuracy Over Time across Regimes", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-6_streaming_f1_over_time.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-7: Cumulative CPU Over Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    for m in methods:
        df_m = df_per_window[df_per_window['method'] == m]
        cum_cpu = df_m.groupby('window_id')['step_cpu_sec'].mean().cumsum()
        ax.plot(cum_cpu.index, cum_cpu.values, label=m, color=METHOD_COLORS.get(m, '#333'), linewidth=2.0)
        
    ax.set_xlabel("Streaming Window Index", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cumulative CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_title("9B-7: Cumulative CPU Overhead Over Streaming Evaluation", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-7_cumulative_cpu_over_time.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-8: F1 Around Regime Transitions
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    df_tr_agg = df_transition.groupby('method')[['pre_transition_acc', 'post_transition_acc']].mean().reset_index()
    
    x = np.arange(len(df_tr_agg))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_tr_agg['pre_transition_acc'], width, label='Pre-Transition', color='#4c72b0', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_tr_agg['post_transition_acc'], width, label='Post-Transition', color='#c44e52', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_tr_agg['method'])
    ax.set_ylabel("Accuracy", fontsize=11, fontweight='bold')
    ax.set_title("9B-8: Performance Immediately Before vs After Regime Transition", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-8_regime_transitions.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-9: Performance vs CPU Cost (Pareto Plot)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    for idx, r in df_summary.iterrows():
        m = r['method']
        ax.scatter(r['total_cpu_sec_mean'], r['macro_f1_mean'], s=150,
                   color=METHOD_COLORS.get(m, '#333'), edgecolors='black', label=m, zorder=5)
        ax.annotate(m, (r['total_cpu_sec_mean'], r['macro_f1_mean']),
                    xytext=(8, -4), textcoords='offset points', fontsize=10, fontweight='bold')
        
    ax.set_xlabel("Total CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Macro F1 Score", fontsize=11, fontweight='bold')
    ax.set_title("9B-9: Predictive Performance vs CPU Cost (Pareto Efficiency)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-9_pareto_performance_vs_cpu.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-10: Per-Seed F1 Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    seeds = sorted(df_per_seed['seed'].unique())
    x = np.arange(len(seeds))
    width = 0.2
    
    for idx, m in enumerate(methods):
        df_m = df_per_seed[df_per_seed['method'] == m]
        vals = [df_m[df_m['seed'] == s]['macro_f1'].values[0] for s in seeds]
        ax.bar(x + (idx - 1.5) * width, vals, width, label=m, color=METHOD_COLORS.get(m, '#333'), edgecolor='black')
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Macro F1 Score", fontsize=11, fontweight='bold')
    ax.set_title("9B-10: Per-Seed Macro F1 Comparison", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-10_per_seed_f1.png'))
    plt.close(fig)

    # -------------------------------------------------------------
    # 9B-11: Per-Seed CPU Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    for idx, m in enumerate(methods):
        df_m = df_per_seed[df_per_seed['method'] == m]
        vals = [df_m[df_m['seed'] == s]['total_cpu_sec'].values[0] for s in seeds]
        ax.bar(x + (idx - 1.5) * width, vals, width, label=m, color=METHOD_COLORS.get(m, '#333'), edgecolor='black')
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Total CPU Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_title("9B-11: Per-Seed CPU Execution Time Comparison", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '9B-11_per_seed_cpu.png'))
    plt.close(fig)

    print("Successfully generated all 11 Exp 9B individual plots!")
