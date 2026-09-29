"""
================================================================================
ALL FIGURES SUBFOLDER PLOT GENERATOR (EXPERIMENT 5 CORE STRATEGIES)
================================================================================
Generates 2 clean, minimal publication plots for the 5 core strategies:
  1. Frozen Ensemble
  2. Event-Driven Baseline
  3. Continuous Retraining
  4. RAPT (K=8)
  5. Two-Tier Hybrid RAPT

Output Files:
  - all_figures/f1_and_accuracy_comparison.png
  - all_figures/adaptation_cpu_comparison.png
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..'))

# Subfolder destinations
EXP5_ALL_FIG_DIR = os.path.join(_this_dir, 'figures', 'all_figures')
ROOT_ALL_FIG_DIR = os.path.join(_project_root, 'figures', 'all_figures')
BRAIN_ALL_FIG_DIR = r"C:\Users\emhaenn\.gemini\antigravity\brain\df805942-8788-4248-aa21-b6cc1da570ac\figures\all_figures"

for d in [EXP5_ALL_FIG_DIR, ROOT_ALL_FIG_DIR, BRAIN_ALL_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Data definition for the 5 core strategies
data = [
    {'strategy': 'Frozen Ensemble', 'f1': 0.6446, 'f1_std': 0.0043, 'accuracy': 0.7571, 'acc_std': 0.0020, 'cpu': 0.00, 'cpu_std': 0.00},
    {'strategy': 'Event-Driven Baseline', 'f1': 0.8234, 'f1_std': 0.0043, 'accuracy': 0.8578, 'acc_std': 0.0029, 'cpu': 10.03, 'cpu_std': 1.82},
    {'strategy': 'Continuous Retraining', 'f1': 0.8486, 'f1_std': 0.0010, 'accuracy': 0.8757, 'acc_std': 0.0010, 'cpu': 19.91, 'cpu_std': 3.10},
    {'strategy': 'RAPT (K=8)', 'f1': 0.7003, 'f1_std': 0.0035, 'accuracy': 0.7824, 'acc_std': 0.0021, 'cpu': 3.15, 'cpu_std': 0.72},
    {'strategy': 'Two-Tier Hybrid RAPT', 'f1': 0.7172, 'f1_std': 0.0143, 'accuracy': 0.7852, 'acc_std': 0.0049, 'cpu': 7.03, 'cpu_std': 1.62},
]

df = pd.DataFrame(data)

# Set clean aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300
})

def save_plot_to_subfolders(fig, filename):
    for d in [EXP5_ALL_FIG_DIR, ROOT_ALL_FIG_DIR, BRAIN_ALL_FIG_DIR]:
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


# ==============================================================================
# PLOT 1: ACCURACY AND F1 SCORE COMPARISON BAR CHART
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

x = np.arange(len(df))
width = 0.35

rects1 = ax1.bar(x - width/2, df['f1'], width, yerr=df['f1_std'], capsize=4,
                 label='F1 Score', color='#2b5c8f', edgecolor='black', alpha=0.9,
                 error_kw={'ecolor': '#1a334d', 'linewidth': 1.2})

rects2 = ax1.bar(x + width/2, df['accuracy'], width, yerr=df['acc_std'], capsize=4,
                 label='Accuracy', color='#389078', edgecolor='black', alpha=0.9,
                 error_kw={'ecolor': '#1d4d40', 'linewidth': 1.2})

ax1.set_ylabel('Score')
ax1.set_title('Classification Performance (F1 Score & Accuracy)')
ax1.set_xticks(x)
labels = [s.replace(' Baseline', '\nBaseline').replace(' Retraining', '\nRetraining').replace(' Ensemble', '\nEnsemble') for s in df['strategy']]
ax1.set_xticklabels(labels, fontweight='bold')
ax1.set_ylim(0.50, 0.95)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Direct numerical labels on top of bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h + 0.01),
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h + 0.01),
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

fig1.tight_layout()
save_plot_to_subfolders(fig1, 'f1_and_accuracy_comparison.png')


# ==============================================================================
# PLOT 2: ADAPTATION CPU TIME COMPARISON BAR CHART
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=300)

colors = ['#7f7f7f', '#ff7f0e', '#d62728', '#1f77b4', '#9467bd']

rects_cpu = ax2.bar(x, df['cpu'], width=0.55, yerr=df['cpu_std'], capsize=5,
                    color=colors, edgecolor='black', alpha=0.88,
                    error_kw={'ecolor': '#333333', 'linewidth': 1.2})

ax2.set_ylabel('Adaptation CPU Time (s)')
ax2.set_title('Adaptation CPU Time Across Strategies')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontweight='bold')
ax2.set_ylim(0.0, max(df['cpu']) * 1.2)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# Direct numerical labels on top of bars
for rect in rects_cpu:
    h = rect.get_height()
    ax2.annotate(f'{h:.2f}s', xy=(rect.get_x() + rect.get_width()/2, h + 0.5),
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')

fig2.tight_layout()
save_plot_to_subfolders(fig2, 'adaptation_cpu_comparison.png')

print("All 2 subfolder figures successfully generated!")
