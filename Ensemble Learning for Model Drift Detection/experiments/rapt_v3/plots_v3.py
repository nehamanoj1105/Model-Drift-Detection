"""
Publication Plot Generator for RAPT-v3 Benchmark Suite
Generates 10 required publication-ready figures saved to experiments/rapt_v3/plots/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAPT_V3_DIR = os.path.join(ROOT_DIR, "experiments", "rapt_v3")
RESULTS_DIR = os.path.join(RAPT_V3_DIR, "results")
PLOTS_DIR = os.path.join(RAPT_V3_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_all_plots():
    print("Generating 10 publication-ready plots for RAPT-v3...", flush=True)
    
    # Load summary and per-window results
    df_sum = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    df_win = pd.read_csv(os.path.join(RESULTS_DIR, "per_window_results.csv"))
    df_trace = pd.read_csv(os.path.join(RESULTS_DIR, "champion_challenger_trace.csv"))
    df_trans = pd.read_csv(os.path.join(RESULTS_DIR, "transition_analysis.csv"))
    
    sum_9a = df_sum[df_sum['dataset'] == '9A']
    sum_9b = df_sum[df_sum['dataset'] == '9B']
    
    methods = ['Frozen', 'Event-Driven', 'Full Retraining', 'Original RAPT', 'RAPT-E', 'RAPT-v3-50', 'RAPT-v3-100', 'RAPT-v3-250']
    palette = ['#7f7f7f', '#d62728', '#ff7f0e', '#1f77b4', '#17becf', '#2ca02c', '#9467bd', '#8c564b']
    
    # Fig 1: Macro F1 by Method — 9A
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(sum_9a['method'], sum_9a['macro_f1_mean'], yerr=sum_9a['macro_f1_std'], capsize=4, color=palette, edgecolor='black', alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    ax.set_ylim(0.985, 1.002)
    ax.set_title("Figure 1: Macro F1 Score Comparison — Experiment 9A", fontsize=11, fontweight='bold')
    ax.set_ylabel("Macro F1")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig1_macro_f1_9a.png"))
    plt.close(fig)

    # Fig 2: Macro F1 by Method — 9B
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(sum_9b['method'], sum_9b['macro_f1_mean'], yerr=sum_9b['macro_f1_std'], capsize=4, color=palette, edgecolor='black', alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    ax.set_ylim(0.85, 0.93)
    ax.set_title("Figure 2: Macro F1 Score Comparison — Experiment 9B", fontsize=11, fontweight='bold')
    ax.set_ylabel("Macro F1")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig2_macro_f1_9b.png"))
    plt.close(fig)

    # Fig 3: Adaptation CPU — 9A
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(sum_9a['method'], sum_9a['adaptation_cpu_mean'], yerr=sum_9a['adaptation_cpu_std'], capsize=4, color=palette, edgecolor='black', alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}s", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    ax.set_title("Figure 3: Adaptation CPU Overhead — Experiment 9A", fontsize=11, fontweight='bold')
    ax.set_ylabel("Adaptation CPU Time (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig3_adapt_cpu_9a.png"))
    plt.close(fig)

    # Fig 4: Adaptation CPU — 9B
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(sum_9b['method'], sum_9b['adaptation_cpu_mean'], yerr=sum_9b['adaptation_cpu_std'], capsize=4, color=palette, edgecolor='black', alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}s", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    ax.set_title("Figure 4: Adaptation CPU Overhead — Experiment 9B", fontsize=11, fontweight='bold')
    ax.set_ylabel("Adaptation CPU Time (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig4_adapt_cpu_9b.png"))
    plt.close(fig)

    # Fig 5: RAPT-v1 (RAPT-E) vs RAPT-v3 Variants F1
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    v_methods = ['RAPT-E', 'RAPT-v3-50', 'RAPT-v3-100', 'RAPT-v3-250']
    v_colors = ['#17becf', '#2ca02c', '#9467bd', '#8c564b']
    
    x = np.arange(len(v_methods))
    w = 0.35
    f1_9a_v = [sum_9a[sum_9a['method']==m]['macro_f1_mean'].values[0] for m in v_methods]
    f1_9b_v = [sum_9b[sum_9b['method']==m]['macro_f1_mean'].values[0] for m in v_methods]
    
    ax.bar(x - w/2, f1_9a_v, w, label='9A Campus QoS', color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.bar(x + w/2, f1_9b_v, w, label='9B 5G NR Latency', color='#2ca02c', edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(v_methods)
    ax.set_ylim(0.85, 1.02)
    ax.set_title("Figure 5: RAPT-v1 (RAPT-E) vs RAPT-v3 Variant Comparison", fontsize=11, fontweight='bold')
    ax.set_ylabel("Macro F1")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig5_v1_vs_v3_f1.png"))
    plt.close(fig)

    # Fig 6: Streaming F1 around Regime Recurrence
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    win_seed42 = df_win[(df_win['dataset'] == '9A') & (df_win['seed'] == 42)]
    for m, c in zip(['Event-Driven', 'RAPT-E', 'RAPT-v3-50'], ['#d62728', '#17becf', '#2ca02c']):
        df_m = win_seed42[win_seed42['method'] == m].sort_values('window_id')
        roll_acc = df_m['is_correct'].rolling(40, min_periods=1).mean()
        ax.plot(df_m['window_id'], roll_acc, label=m, color=c, linewidth=1.5)
    ax.set_title("Figure 6: Streaming Moving Accuracy around Regime Transitions (9A Seed 42)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Streaming Window ID")
    ax.set_ylabel("40-Window Moving Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig6_streaming_recurrence.png"))
    plt.close(fig)

    # Fig 7: Champion vs Challenger Performance at Recurring Regimes (PRIMARY DIAGNOSTIC)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    if len(df_trace) > 0:
        trace_sample = df_trace.head(30)
        x_t = np.arange(len(trace_sample))
        ax.plot(x_t, trace_sample['champion_score'], 'o-', label='Champion Score', color='#1f77b4', linewidth=1.8)
        ax.plot(x_t, trace_sample['challenger_score'], 's--', label='Challenger Score', color='#2ca02c', linewidth=1.8)
        
        # Highlight selections
        for idx_t, r_t in trace_sample.iterrows():
            pos = idx_t - trace_sample.index[0]
            if r_t['selected_policy'] == 'challenger':
                ax.axvline(x=pos, color='#2ca02c', alpha=0.3, linestyle=':')
        ax.set_xticks(x_t[::3])
        ax.set_xticklabels([f"W{int(w)}" for w in trace_sample['window_id'].values[::3]], rotation=30)
    ax.set_title("Figure 7: Champion vs Challenger Validation Scores at Recurring Regimes", fontsize=11, fontweight='bold')
    ax.set_ylabel("Validation Score (F1 / Accuracy)")
    ax.set_xlabel("Transition Event Window")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig7_champion_vs_challenger_scores.png"))
    plt.close(fig)

    # Fig 8: Ensemble Weight Trajectories
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    if len(df_trace) > 0:
        trace_v50 = df_trace[df_trace['variant'] == 'RAPT-v3-50'].head(25)
        x_tr = np.arange(len(trace_v50))
        ax.plot(x_tr, trace_v50['champion_w_rf'], 'o-', label='Champion RF Weight', color='#1f77b4')
        ax.plot(x_tr, trace_v50['challenger_w_rf'], 's--', label='Challenger RF Weight', color='#ff7f0e')
        ax.set_xticks(x_tr[::2])
        ax.set_xticklabels([f"W{int(w)}" for w in trace_v50['window_id'].values[::2]], rotation=30)
    ax.set_title("Figure 8: Ensemble Weight Movement (Champion vs Challenger)", fontsize=11, fontweight='bold')
    ax.set_ylabel("RandomForest Weight ($w_{RF}$)")
    ax.set_xlabel("Transition Event Window")
    ax.set_ylim(0.2, 0.8)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig8_weight_trajectories.png"))
    plt.close(fig)

    # Fig 9: Total CPU vs Macro F1 Pareto Frontier
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    for idx, r in sum_9a.iterrows():
        ax.scatter(r['adaptation_cpu_mean'], r['macro_f1_mean'], s=120, label=f"9A-{r['method']}", marker='o')
    for idx, r in sum_9b.iterrows():
        ax.scatter(r['adaptation_cpu_mean'], r['macro_f1_mean'], s=120, label=f"9B-{r['method']}", marker='^')
    ax.set_title("Figure 9: Macro F1 vs Adaptation CPU Pareto Frontier", fontsize=11, fontweight='bold')
    ax.set_xlabel("Adaptation CPU Overhead (seconds)")
    ax.set_ylabel("Macro F1 Score")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig9_cpu_vs_f1_pareto.png"))
    plt.close(fig)

    # Fig 10: Champion vs Challenger Selection Counts
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    v3_sum = df_sum[df_sum['method'].str.startswith('RAPT-v3')]
    x_sel = np.arange(len(v3_sum))
    w_sel = 0.35
    ax.bar(x_sel - w_sel/2, v3_sum['champion_selections_mean'], w_sel, label='Champion Selected', color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.bar(x_sel + w_sel/2, v3_sum['challenger_selections_mean'], w_sel, label='Challenger Selected', color='#2ca02c', edgecolor='black', alpha=0.85)
    ax.set_xticks(x_sel)
    ax.set_xticklabels(v3_sum['method'])
    ax.set_title("Figure 10: Champion vs Challenger Selection Counts", fontsize=11, fontweight='bold')
    ax.set_ylabel("Selection Frequency (Events)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig10_selection_counts.png"))
    plt.close(fig)

    print("All 10 publication plots successfully saved to experiments/rapt_v3/plots/", flush=True)

if __name__ == '__main__':
    generate_all_plots()
