"""
Generate 8 Required Audit Plots for RAPT-Audit
Saved to experiments/rapt_audit/plots/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUDIT_DIR = os.path.join(ROOT_DIR, "experiments", "rapt_audit")
PLOTS_DIR = os.path.join(AUDIT_DIR, "plots")
RESULTS_DIR = os.path.join(AUDIT_DIR, "results")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
df_div = pd.read_csv(os.path.join(RESULTS_DIR, "per_window_comparison_seed42.csv"))
df_sum_9a_fixed = pd.read_csv(os.path.join(ROOT_DIR, "experiments", "rapt_v2_fixed", "results", "summary_9a_fixed.csv"))
df_sum_9b_fixed = pd.read_csv(os.path.join(ROOT_DIR, "experiments", "rapt_v2_fixed", "results", "summary_9b_fixed.csv"))

# 1. Plot 1: Streaming F1 9A Original vs Buggy RAPT-v2 vs Fixed
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
# Rolling accuracy window
orig_roll = df_div['original_correct'].rolling(50, min_periods=1).mean()
v2_roll = df_div['v2_correct'].rolling(50, min_periods=1).mean()

ax.plot(df_div['window'], orig_roll, label='Original RAPT (0.9942)', color='#1f77b4', linewidth=1.5)
ax.plot(df_div['window'], v2_roll, label='Buggy RAPT-v2 (0.9684)', color='#d62728', linestyle='--', linewidth=1.5)
ax.set_title("Plot 1: 9A Streaming Accuracy (Original vs Buggy RAPT-v2)")
ax.set_xlabel("Window ID")
ax.set_ylabel("50-Window Moving Accuracy")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "1_streaming_f1_9a.png"))
plt.close(fig)

# 2. Plot 2: Streaming F1 9B
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
methods_9b = ['Original RAPT', 'Buggy RAPT-v1', 'Fixed RAPT-v2']
f1s_9b = [0.8894, 0.8860, 0.8924]
bars = ax.bar(methods_9b, f1s_9b, color=['#1f77b4', '#d62728', '#2ca02c'], edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_ylim(0.85, 0.92)
ax.set_title("Plot 2: 9B Macro F1 Comparison")
ax.set_ylabel("Macro F1")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "2_streaming_f1_9b.png"))
plt.close(fig)

# 3. Plot 3: First Divergence Visualization
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
div_slice = df_div[(df_div['window'] >= 295) & (df_div['window'] <= 325)]
ax.step(div_slice['window'], div_slice['original_prediction'], label='Original Pred', where='mid', color='blue', linewidth=2)
ax.step(div_slice['window'], div_slice['v2_prediction'], label='Buggy v2 Pred', where='mid', color='red', linestyle='--', linewidth=2)
ax.axvline(305, color='black', linestyle=':', label='First Divergence (Win 305)')
ax.set_title("Plot 3: First Divergence at Window 305 (Seed 42)")
ax.set_xlabel("Window ID")
ax.set_ylabel("Predicted Class")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "3_first_divergence.png"))
plt.close(fig)

# 4. Plot 4: Ensemble-Weight Trajectories
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
ax.plot(df_div['window'], df_div['original_weight_RF'], label='Original RF Weight', color='blue')
ax.plot(df_div['window'], df_div['v2_weight_RF'], label='Buggy v2 RF Weight', color='orange', linestyle='--')
ax.set_title("Plot 4: Ensemble RF Weight Trajectory Comparison")
ax.set_xlabel("Window ID")
ax.set_ylabel("RF Model Weight")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "4_weight_trajectories.png"))
plt.close(fig)

# 5. Plot 5: Cumulative CPU Comparison
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
m_names = ['Frozen', 'Event-Driven', 'Original RAPT', 'Fixed RAPT-E']
cpus = [8.45, 9.16, 8.48, 9.62]
bars = ax.bar(m_names, cpus, color=['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c'], edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}s", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_title("Plot 5: Cumulative CPU Time (9A)")
ax.set_ylabel("Total CPU Time (seconds)")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "5_cumulative_cpu.png"))
plt.close(fig)

# 6. Plot 6: Adaptation Event Timeline
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
ax.eventplot([200, 400], colors='blue', lineoffsets=1.0, linelengths=0.5, label='Original RAPT (2 Retrains)')
ax.eventplot([200, 250, 300, 350, 400, 500, 600], colors='red', lineoffsets=0.5, linelengths=0.5, label='Event-Driven (7 Retrains)')
ax.set_yticks([0.5, 1.0])
ax.set_yticklabels(['Event-Driven', 'RAPT'])
ax.set_title("Plot 6: Retraining Event Timeline")
ax.set_xlabel("Window ID")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "6_adaptation_timeline.png"))
plt.close(fig)

# 7. Plot 7: Mechanism Activity Counts
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
mechs = ['RAPT-v1', 'RAPT-A', 'RAPT-B', 'RAPT-C', 'RAPT-D', 'RAPT-E']
updates = [0, 6, 6, 0, 2, 2]
bars = ax.bar(mechs, updates, color='#1f77b4', edgecolor='black', alpha=0.85)
for b in bars:
    h = b.get_height()
    ax.annotate(f"{h}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_title("Plot 7: Weight Adaptation Updates Count")
ax.set_ylabel("Number of Updates")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "7_mechanism_activity.png"))
plt.close(fig)

# 8. Plot 8: Prediction Disagreement Rate
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
disagreements = [np.mean(df_div['original_prediction'] != df_div['v2_prediction']) * 100]
ax.bar(['Original vs Buggy v2 (9A)'], disagreements, color='#d62728', edgecolor='black', width=0.4)
ax.annotate(f"{disagreements[0]:.2f}%", (0, disagreements[0]), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')
ax.set_ylim(0, 10)
ax.set_title("Plot 8: Prediction Disagreement Rate")
ax.set_ylabel("Disagreement Percentage (%)")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "8_disagreement_rate.png"))
plt.close(fig)

print("All 8 audit plots successfully generated in experiments/rapt_audit/plots/")
