"""
Plotting Module for Exp 9 — RAPT vs Event-Driven Ensemble
Generates 10 publication-quality charts for performance, CPU savings, Pareto trade-offs, and ablations using Matplotlib.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14, 'axes.labelsize': 12})

def generate_all_plots(summary_df, per_seed_df, per_window_df, transition_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        'Frozen': '#7f7f7f',            # Grey
        'Event-Driven': '#d62728',      # Red
        'RAPT': '#1f77b4',              # Blue
        'RAPT_No_Weights': '#ff7f0e',   # Orange (Ablation D)
        'RAPT_Weights_Only': '#2ca02c', # Green (Ablation E)
        'Full_Retraining': '#9467bd'   # Purple
    }
    
    # 1. Streaming F1 over windows
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, grp in per_window_df.groupby('method'):
        win_avg = grp.groupby('window_id')['f1_macro'].mean().reset_index()
        ax.plot(win_avg['window_id'], win_avg['f1_macro'], label=method, color=colors.get(method, '#333333'), alpha=0.85, linewidth=1.8)
    ax.set_title("Figure 1: Macro F1 Score over Streaming Windows")
    ax.set_xlabel("Streaming Window Index")
    ax.set_ylabel("Macro F1 Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_f1_streaming.png"), dpi=300)
    plt.close()
    
    # 2. Per-window CPU cost
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, grp in per_window_df.groupby('method'):
        win_avg = grp.groupby('window_id')['total_cpu_time'].mean().reset_index()
        ax.plot(win_avg['window_id'], win_avg['total_cpu_time'] * 1000.0, label=method, color=colors.get(method, '#333333'), alpha=0.85, linewidth=1.5)
    ax.set_title("Figure 2: CPU Cost per Streaming Window (ms)")
    ax.set_xlabel("Streaming Window Index")
    ax.set_ylabel("CPU Time (ms)")
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_cpu_cost_streaming.png"), dpi=300)
    plt.close()
    
    # 3. Cumulative CPU Cost
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, grp in per_window_df.groupby('method'):
        win_avg = grp.groupby('window_id')['total_cpu_time'].mean().cumsum().reset_index()
        ax.plot(win_avg['window_id'], win_avg['total_cpu_time'], label=method, color=colors.get(method, '#333333'), linewidth=2.0)
    ax.set_title("Figure 3: Cumulative CPU Compute Time (seconds)")
    ax.set_xlabel("Streaming Window Index")
    ax.set_ylabel("Cumulative CPU Time (s)")
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_cumulative_cpu.png"), dpi=300)
    plt.close()
    
    # 4. F1 around regime transitions
    if not transition_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        trans_ids = sorted(transition_df['transition_id'].unique())
        methods_list = list(transition_df['method'].unique())
        width = 0.8 / len(methods_list)
        x = np.arange(len(trans_ids))
        
        for idx, m in enumerate(methods_list):
            m_df = transition_df[transition_df['method'] == m].groupby('transition_id')['post_event_f1'].mean()
            vals = [m_df.get(tid, 0.0) for tid in trans_ids]
            ax.bar(x + idx * width - 0.4 + width / 2, vals, width, label=m, color=colors.get(m, '#333333'))
            
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in trans_ids])
        ax.set_title("Figure 4: Immediate Post-Event F1 Score across Regime Transitions")
        ax.set_xlabel("Transition Event")
        ax.set_ylabel("Post-Transition F1 Score")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig4_f1_around_transitions.png"), dpi=300)
        plt.close()
        
    # 5. Retraining / Adaptation events count
    fig, ax = plt.subplots(figsize=(8, 5))
    methods_s = summary_df['method'].tolist()
    events_s = summary_df['retrain_events_mean'].tolist()
    bar_colors = [colors.get(m, '#333333') for m in methods_s]
    
    ax.bar(methods_s, events_s, color=bar_colors, edgecolor='black')
    ax.set_title("Figure 5: Total Adaptation / Retraining Events")
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of Retraining Events")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_retraining_events.png"), dpi=300)
    plt.close()
    
    # 6. Performance vs CPU Pareto Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, row in summary_df.iterrows():
        m = row['method']
        f1_m = row['f1_macro_mean']
        cpu_m = row['total_cpu_time_mean']
        c = colors.get(m, '#333333')
        ax.scatter(cpu_m, f1_m, s=160, color=c, label=m, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(m, (cpu_m, f1_m), textcoords="offset points", xytext=(8, 8), ha='left', fontsize=11, weight='bold')
        
    ax.set_title("Figure 6: Performance vs Adaptation CPU Cost (Pareto Efficiency)")
    ax.set_xlabel("Total CPU Time (seconds) [Lower is Better]")
    ax.set_ylabel("Macro F1 Score [Higher is Better]")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_performance_vs_cpu_pareto.png"), dpi=300)
    plt.close()
    
    # 7. Per-seed F1 comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    seeds_list = sorted(per_seed_df['seed'].unique())
    methods_list = list(per_seed_df['method'].unique())
    width = 0.8 / len(methods_list)
    x = np.arange(len(seeds_list))
    
    for idx, m in enumerate(methods_list):
        m_df = per_seed_df[per_seed_df['method'] == m].set_index('seed')['f1_macro']
        vals = [m_df.get(s, 0.0) for s in seeds_list]
        ax.bar(x + idx * width - 0.4 + width / 2, vals, width, label=m, color=colors.get(m, '#333333'))
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds_list])
    ax.set_title("Figure 7: Macro F1 Performance across Random Seeds")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Macro F1 Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_per_seed_f1.png"), dpi=300)
    plt.close()
    
    # 8. Per-seed CPU comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, m in enumerate(methods_list):
        m_df = per_seed_df[per_seed_df['method'] == m].set_index('seed')['total_cpu_time']
        vals = [m_df.get(s, 0.0) for s in seeds_list]
        ax.bar(x + idx * width - 0.4 + width / 2, vals, width, label=m, color=colors.get(m, '#333333'))
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds_list])
    ax.set_title("Figure 8: Total CPU Runtime across Random Seeds")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Total CPU Time (s)")
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_per_seed_cpu.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 9. Figure 9: Macro F1 and Accuracy Bar Chart (Target Models)
    # ---------------------------------------------------------
    target_methods = {
        'Frozen': 'Frozen',
        'Event-Driven': 'Event-Driven',
        'Full_Retraining': 'Full Retraining',
        'RAPT_Weights_Only': 'RAPT (Best)'
    }
    
    selected_rows = []
    for raw_m, disp_m in target_methods.items():
        sub = summary_df[summary_df['method'] == raw_m]
        if not sub.empty:
            r = sub.iloc[0].to_dict()
            r['display_name'] = disp_m
            selected_rows.append(r)
            
    df_sel = pd.DataFrame(selected_rows)
    labels = df_sel['display_name'].tolist()
    x = np.arange(len(labels))
    b_width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_means = df_sel['f1_macro_mean'].values
    f1_stds = df_sel['f1_macro_std'].values
    acc_means = df_sel['accuracy_mean'].values
    acc_stds = df_sel['accuracy_std'].values
    
    rects1 = ax.bar(x - b_width/2, f1_means, b_width, yerr=f1_stds, label='Macro F1 Score', color='#1f77b4', capsize=4, edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + b_width/2, acc_means, b_width, yerr=acc_stds, label='Accuracy', color='#2ca02c', capsize=4, edgecolor='black', alpha=0.85)
    
    ax.set_title("Figure 9: Predictive Performance (Macro F1 & Accuracy) Comparison", fontsize=14, pad=15, weight='bold')
    ax.set_xlabel("Model Architecture", labelpad=10, weight='bold')
    ax.set_ylabel("Score (0.0 to 1.0)", labelpad=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight='bold')
    ax.set_ylim(0.985, 1.002)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig9_f1_accuracy_bar.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 10. Figure 10: Total CPU Time and Adaptation Cost Bar Chart
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    cpu_means = df_sel['total_cpu_time_mean'].values
    cpu_stds = df_sel['total_cpu_time_std'].values
    adapt_means = df_sel['adaptation_cost_mean'].values
    
    rects3 = ax.bar(x - b_width/2, cpu_means, b_width, yerr=cpu_stds, label='Total CPU Time (s)', color='#d62728', capsize=4, edgecolor='black', alpha=0.85)
    rects4 = ax.bar(x + b_width/2, adapt_means, b_width, label='Adaptation CPU Cost (s)', color='#ff7f0e', capsize=4, edgecolor='black', alpha=0.85)
    
    ax.set_title("Figure 10: Computational Cost (Total CPU & Adaptation) Comparison", fontsize=14, pad=15, weight='bold')
    ax.set_xlabel("Model Architecture", labelpad=10, weight='bold')
    ax.set_ylabel("CPU Time (seconds)", labelpad=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight='bold')
    ax.set_ylim(0.0, max(cpu_means) * 1.25)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    for rect in rects3:
        height = rect.get_height()
        ax.annotate(f"{height:.2f} s", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')
    for rect in rects4:
        height = rect.get_height()
        ax.annotate(f"{height:.2f} s", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig10_cpu_cost_bar.png"), dpi=300)
    plt.close()
    
    print(f"All 10 figures successfully generated in {output_dir}")
