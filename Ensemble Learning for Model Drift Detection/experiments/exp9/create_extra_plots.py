"""
Script to generate Figure 9 and Figure 10 bar charts:
Fig 9: Macro F1 and Accuracy for Frozen, Full Retraining, Event-Driven, and Best RAPT.
Fig 10: Total CPU Time and Adaptation Cost for Frozen, Full Retraining, Event-Driven, and Best RAPT.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_fig9_and_fig10(summary_csv='experiments/exp9/results/summary.csv', output_dir='experiments/exp9/plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)
    
    # Map method names to display names
    # Select Best RAPT variant (RAPT_Weights_Only has highest F1 0.9950 among RAPT variants)
    target_methods = {
        'Frozen': 'Frozen',
        'Event-Driven': 'Event-Driven',
        'Full_Retraining': 'Full Retraining',
        'RAPT_Weights_Only': 'RAPT (Best)'
    }
    
    selected_rows = []
    for raw_m, disp_m in target_methods.items():
        sub = df[df['method'] == raw_m]
        if not sub.empty:
            r = sub.iloc[0].to_dict()
            r['display_name'] = disp_m
            selected_rows.append(r)
            
    df_sel = pd.DataFrame(selected_rows)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14, 'axes.labelsize': 12})
    
    colors_f1 = '#1f77b4'  # Deep Blue
    colors_acc = '#2ca02c' # Forest Green
    
    colors_cpu = '#d62728'  # Red
    colors_adapt = '#ff7f0e'# Orange
    
    labels = df_sel['display_name'].tolist()
    x = np.arange(len(labels))
    width = 0.35
    
    # ---------------------------------------------------------
    # FIGURE 9: Macro F1 and Accuracy Bar Chart
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f1_means = df_sel['f1_macro_mean'].values
    f1_stds = df_sel['f1_macro_std'].values
    
    acc_means = df_sel['accuracy_mean'].values
    acc_stds = df_sel['accuracy_std'].values
    
    rects1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, label='Macro F1 Score', color=colors_f1, capsize=4, edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, acc_means, width, yerr=acc_stds, label='Accuracy', color=colors_acc, capsize=4, edgecolor='black', alpha=0.85)
    
    ax.set_title("Figure 9: Predictive Performance (Macro F1 & Accuracy) Comparison", fontsize=14, pad=15, weight='bold')
    ax.set_xlabel("Model Architecture", labelpad=10, weight='bold')
    ax.set_ylabel("Score (0.0 to 1.0)", labelpad=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight='bold')
    ax.set_ylim(0.985, 1.002)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Value annotations on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, weight='bold')
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, weight='bold')
                    
    plt.tight_layout()
    fig9_path = os.path.join(output_dir, "fig9_f1_accuracy_bar.png")
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"Saved Figure 9 to {fig9_path}")
    
    # ---------------------------------------------------------
    # FIGURE 10: Total CPU Time and Adaptation Cost Bar Chart
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cpu_means = df_sel['total_cpu_time_mean'].values
    cpu_stds = df_sel['total_cpu_time_std'].values
    
    adapt_means = df_sel['adaptation_cost_mean'].values
    
    rects3 = ax.bar(x - width/2, cpu_means, width, yerr=cpu_stds, label='Total CPU Time (s)', color=colors_cpu, capsize=4, edgecolor='black', alpha=0.85)
    rects4 = ax.bar(x + width/2, adapt_means, width, label='Adaptation CPU Cost (s)', color=colors_adapt, capsize=4, edgecolor='black', alpha=0.85)
    
    ax.set_title("Figure 10: Computational Cost (Total CPU & Adaptation) Comparison", fontsize=14, pad=15, weight='bold')
    ax.set_xlabel("Model Architecture", labelpad=10, weight='bold')
    ax.set_ylabel("CPU Time (seconds)", labelpad=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight='bold')
    ax.set_ylim(0.0, max(cpu_means) * 1.25)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Value annotations
    for rect in rects3:
        height = rect.get_height()
        ax.annotate(f"{height:.2f} s",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, weight='bold')
                    
    for rect in rects4:
        height = rect.get_height()
        ax.annotate(f"{height:.2f} s",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, weight='bold')
                    
    plt.tight_layout()
    fig10_path = os.path.join(output_dir, "fig10_cpu_cost_bar.png")
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"Saved Figure 10 to {fig10_path}")

if __name__ == '__main__':
    generate_fig9_and_fig10()
