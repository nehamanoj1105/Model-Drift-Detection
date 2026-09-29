"""
Generate 11 Required Final Validation Publication Plots
Saved to experiments/final_validation/plots/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VAL_DIR = os.path.join(ROOT_DIR, "experiments", "final_validation")
PLOTS_DIR = os.path.join(VAL_DIR, "plots")
RESULTS_DIR = os.path.join(VAL_DIR, "results")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
df_sum = pd.read_csv(os.path.join(RESULTS_DIR, "final_summary.csv"))
df_win = pd.read_csv(os.path.join(RESULTS_DIR, "final_per_window.csv"))
df_trans = pd.read_csv(os.path.join(RESULTS_DIR, "transition_analysis.csv"))

sum_9a = df_sum[df_sum['dataset'] == '9A']
sum_9b = df_sum[df_sum['dataset'] == '9B']
methods = ['Frozen', 'Event-Driven', 'Full Retraining', 'Original RAPT', 'RAPT-E']
colors = ['#7f7f7f', '#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']

# Fig 1: Macro F1 9A
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.bar(sum_9a['method'], sum_9a['macro_f1_mean'], yerr=sum_9a['macro_f1_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_ylim(0.985, 1.002)
ax.set_title("Figure 1: Macro F1 Score Comparison — Experiment 9A", fontsize=11, fontweight='bold')
ax.set_ylabel("Macro F1")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig1_macro_f1_9a.png"))
plt.close(fig)

# Fig 2: Macro F1 9B
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.bar(sum_9b['method'], sum_9b['macro_f1_mean'], yerr=sum_9b['macro_f1_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_ylim(0.83, 0.93)
ax.set_title("Figure 2: Macro F1 Score Comparison — Experiment 9B", fontsize=11, fontweight='bold')
ax.set_ylabel("Macro F1")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig2_macro_f1_9b.png"))
plt.close(fig)

# Fig 3: Adaptation CPU 9A
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.bar(sum_9a['method'], sum_9a['adaptation_cpu_mean'], yerr=sum_9a['adaptation_cpu_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.3f}s", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_title("Figure 3: Adaptation CPU Overhead — Experiment 9A", fontsize=11, fontweight='bold')
ax.set_ylabel("Adaptation CPU Time (seconds)")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig3_adapt_cpu_9a.png"))
plt.close(fig)

# Fig 4: Adaptation CPU 9B
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.bar(sum_9b['method'], sum_9b['adaptation_cpu_mean'], yerr=sum_9b['adaptation_cpu_std'], capsize=4, color=colors, edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.3f}s", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_title("Figure 4: Adaptation CPU Overhead — Experiment 9B", fontsize=11, fontweight='bold')
ax.set_ylabel("Adaptation CPU Time (seconds)")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig4_adapt_cpu_9b.png"))
plt.close(fig)

# Fig 5: Total Runtime Side-by-Side
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
x = np.arange(len(methods))
w = 0.35
bars1 = ax.bar(x - w/2, sum_9a['total_runtime_mean'], w, label='9A Campus QoS', color='#1f77b4', edgecolor='black', alpha=0.85)
bars2 = ax.bar(x + w/2, sum_9b['total_runtime_mean'], w, label='9B 5G NR Latency', color='#2ca02c', edgecolor='black', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_title("Figure 5: Total Streaming Runtime Comparison", fontsize=11, fontweight='bold')
ax.set_ylabel("Total Runtime (seconds)")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig5_total_runtime.png"))
plt.close(fig)

# Fig 6: Streaming F1 (Seed 42 9A)
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
win_seed42 = df_win[(df_win['dataset'] == '9A') & (df_win['seed'] == 42)]
for m, c in zip(['Event-Driven', 'RAPT-E'], ['#d62728', '#2ca02c']):
    df_m = win_seed42[win_seed42['method'] == m].sort_values('window_id')
    roll_acc = df_m['is_correct'].rolling(50, min_periods=1).mean()
    ax.plot(df_m['window_id'], roll_acc, label=m, color=c, linewidth=1.5)
ax.set_title("Figure 6: RAPT-E vs Event-Driven Streaming Accuracy (9A Seed 42)", fontsize=11, fontweight='bold')
ax.set_xlabel("Streaming Window ID")
ax.set_ylabel("50-Window Moving Accuracy")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig6_streaming_f1.png"))
plt.close(fig)

# Fig 7: Cumulative Adaptation CPU
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
for m, c in zip(['Event-Driven', 'Original RAPT', 'RAPT-E'], ['#d62728', '#1f77b4', '#2ca02c']):
    df_m = win_seed42[win_seed42['method'] == m].sort_values('window_id')
    cum_cpu = np.cumsum(df_m['adaptation_time'].values)
    ax.plot(df_m['window_id'], cum_cpu, label=m, color=c, linewidth=1.8)
ax.set_title("Figure 7: Cumulative Adaptation CPU Overhead over Telemetry Stream", fontsize=11, fontweight='bold')
ax.set_xlabel("Streaming Window ID")
ax.set_ylabel("Cumulative Adaptation CPU Time (s)")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig7_cum_adapt_cpu.png"))
plt.close(fig)

# Fig 8: Retraining Events Comparison
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
x = np.arange(len(methods))
bars1 = ax.bar(x - w/2, sum_9a['retrain_events_mean'], w, label='9A Campus QoS', color='#1f77b4', edgecolor='black', alpha=0.85)
bars2 = ax.bar(x + w/2, sum_9b['retrain_events_mean'], w, label='9B 5G NR Latency', color='#2ca02c', edgecolor='black', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_title("Figure 8: Total Retraining Events Across Methods", fontsize=11, fontweight='bold')
ax.set_ylabel("Retraining Events Count")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig8_retrain_events.png"))
plt.close(fig)

# Fig 9: F1 around Regime Recurrence
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
df_tr_summary = df_trans.groupby('method')[['pre_event_f1', 'post_event_f1', 'min_post_f1']].mean().reset_index()
x_m = np.arange(len(df_tr_summary))
ax.bar(x_m - w, df_tr_summary['pre_event_f1'], w, label='Pre-Event F1', color='#1f77b4', edgecolor='black')
ax.bar(x_m, df_tr_summary['post_event_f1'], w, label='Post-Event F1', color='#2ca02c', edgecolor='black')
ax.bar(x_m + w, df_tr_summary['min_post_f1'], w, label='Min Post F1', color='#d62728', edgecolor='black')
ax.set_xticks(x_m)
ax.set_xticklabels(df_tr_summary['method'])
ax.set_ylim(0.85, 1.02)
ax.set_title("Figure 9: Performance Stability Around Regime Transitions", fontsize=11, fontweight='bold')
ax.set_ylabel("Accuracy / F1")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig9_regime_recurrence.png"))
plt.close(fig)

# Fig 10: Performance vs Adaptation CPU Pareto
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
for idx, r in sum_9a.iterrows():
    ax.scatter(r['adaptation_cpu_mean'], r['macro_f1_mean'], s=120, label=f"9A-{r['method']}", marker='o')
for idx, r in sum_9b.iterrows():
    ax.scatter(r['adaptation_cpu_mean'], r['macro_f1_mean'], s=120, label=f"9B-{r['method']}", marker='^')
ax.set_title("Figure 10: Macro F1 vs Adaptation CPU Pareto Frontier", fontsize=11, fontweight='bold')
ax.set_xlabel("Adaptation CPU Overhead (seconds)")
ax.set_ylabel("Macro F1 Score")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig10_pareto.png"))
plt.close(fig)

# Fig 11: Cross-Dataset RAPT-E Summary
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
categories = ['Campus QoS (9A)', '5G NR Latency (9B)']
f1_vals = [sum_9a[sum_9a['method']=='RAPT-E']['macro_f1_mean'].values[0], sum_9b[sum_9b['method']=='RAPT-E']['macro_f1_mean'].values[0]]
cpu_red = [
    (1 - sum_9a[sum_9a['method']=='RAPT-E']['adaptation_cpu_mean'].values[0] / sum_9a[sum_9a['method']=='Event-Driven']['adaptation_cpu_mean'].values[0]) * 100,
    (1 - sum_9b[sum_9b['method']=='RAPT-E']['adaptation_cpu_mean'].values[0] / sum_9b[sum_9b['method']=='Event-Driven']['adaptation_cpu_mean'].values[0]) * 100
]

ax2 = ax.twinx()
b1 = ax.bar(np.arange(len(categories)) - 0.2, f1_vals, 0.4, label='Macro F1', color='#1f77b4', edgecolor='black')
b2 = ax2.bar(np.arange(len(categories)) + 0.2, cpu_red, 0.4, label='Adaptation CPU Reduction (%)', color='#2ca02c', edgecolor='black')

ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(categories)
ax.set_ylabel("Macro F1", color='#1f77b4', fontweight='bold')
ax2.set_ylabel("Adaptation CPU Reduction (%)", color='#2ca02c', fontweight='bold')
ax.set_ylim(0.8, 1.05)
ax2.set_ylim(0, 100)
ax.set_title("Figure 11: Cross-Dataset RAPT-E Performance & Compute Savings", fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "fig11_cross_summary.png"))
plt.close(fig)

print("All 11 publication plots generated in experiments/final_validation/plots/")
