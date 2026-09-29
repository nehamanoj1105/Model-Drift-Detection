"""
================================================================================
EXPERIMENT 5 — ORACLE HISTORICAL MODEL RETRIEVAL FIGURE GENERATOR
================================================================================
Generates all 6 required publication-quality figures:
  - Figure 1: Method Comparison (F1 across all 6 methods)
  - Figure 2: Historical Retrieval Gain Over Time (Oracle F1 - Current F1 vs Window)
  - Figure 3: Checkpoint Age vs Transfer Gain Scatter Plot
  - Figure 4: Precision / Recall / F1 Multi-Metric Comparison
  - Figure 5: Confusion Matrix & FP/FN Statistics
  - Figure 6: Weight-Only Headroom vs Historical Model Retrieval Headroom
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

METHOD_COLORS = {
    'Fixed Ensemble': '#4C72B0',
    'Global Adaptive Ensemble': '#DD8452',
    'Regime-Aware Ensemble': '#55A868',
    'Current Event-Driven': '#8172B3',
    'Oracle Weighting': '#CCB974',
    'Oracle Model Retrieval (In-Sample)': '#C44E52',
    'Oracle Model Retrieval (Held-Out)': '#9370DB',
}


def plot_fig1_method_comparison(df_summary, save_dir):
    """Figure 1 — Method Comparison (Macro F1 across all methods)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    methods_order = [
        'Fixed Ensemble',
        'Global Adaptive Ensemble',
        'Regime-Aware Ensemble',
        'Current Event-Driven',
        'Oracle Weighting',
        'Oracle Model Retrieval (In-Sample)',
        'Oracle Model Retrieval (Held-Out)'
    ]

    agg = df_summary.groupby('method')['f1'].agg(['mean', 'std']).reset_index()
    agg = agg[agg['method'].isin(methods_order)]
    agg['method'] = pd.Categorical(agg['method'], categories=methods_order, ordered=True)
    agg = agg.sort_values('method')

    colors = [METHOD_COLORS.get(m, '#333333') for m in agg['method']]
    bars = ax.bar(agg['method'], agg['mean'], yerr=agg['std'], capsize=5, color=colors, edgecolor='black', alpha=0.85, width=0.55)

    for bar, mean_val in zip(bars, agg['mean']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean_val + 0.005, f"{mean_val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_ylabel("Macro F1 Score", fontweight='bold')
    ax.set_title("Figure 1: Streaming Macro F1 Performance Across Methods (Mean ± Std)", fontweight='bold', pad=12)
    ax.set_ylim(0.90, 1.02)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_method_comparison.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'fig1_method_comparison.svg'), dpi=300)
    plt.close()


def plot_fig2_retrieval_gain_over_time(df_window, save_dir):
    """Figure 2 — Historical Retrieval Gain Over Time."""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    # Calculate window-level delta: Oracle Retrieval F1 - Current Event-Driven F1
    w_curr = df_window[df_window['method'] == 'Current Event-Driven'].groupby('window_id')['f1'].mean()
    w_insample = df_window[df_window['method'] == 'Oracle Model Retrieval (In-Sample)'].groupby('window_id')['f1'].mean()
    w_heldout = df_window[df_window['method'] == 'Oracle Model Retrieval (Held-Out)'].groupby('window_id')['f1'].mean()

    delta_insample = (w_insample - w_curr).dropna()
    delta_heldout = (w_heldout - w_curr).dropna()

    ax.plot(delta_insample.index, delta_insample.values, label='In-Sample Oracle Gain', color='#C44E52', lw=2.2, marker='o', ms=4)
    ax.plot(delta_heldout.index, delta_heldout.values, label='Held-Out Oracle Gain', color='#9370DB', lw=2.2, marker='s', ms=4)
    ax.axhline(0, color='black', linestyle='--', alpha=0.7, lw=1.2)

    ax.set_xlabel("Streaming Window Index (500 samples/window)", fontweight='bold')
    ax.set_ylabel("F1 Gain (Oracle - Baseline)", fontweight='bold')
    ax.set_title("Figure 2: Historical Model Retrieval Gain Over Time", fontweight='bold', pad=12)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_retrieval_gain_over_time.png'), dpi=300)
    plt.close()


def plot_fig3_checkpoint_age_vs_gain(df_age, save_dir):
    """Figure 3 — Checkpoint Age vs Transfer Gain Scatter Plot."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    if df_age.empty or 'checkpoint_age' not in df_age.columns:
        ax.text(0.5, 0.5, "No Checkpoint Retrieval Events", ha='center', va='center')
    else:
        x_vals = df_age['checkpoint_age'].values
        y_vals = df_age['f1_gain'].values

        ax.scatter(x_vals, y_vals, color='#8172B3', alpha=0.7, s=45, edgecolor='black', linewidth=0.5, label='Retrieved Checkpoint')

        if len(x_vals) > 1 and np.std(x_vals) > 1e-5:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            ax.plot(x_line, slope * x_line + intercept, color='#4B0082', lw=2.2, label=f'Linear Trend (slope={slope:.4f})')

        ax.set_xlabel("Retrieved Checkpoint Age (Target Window - Training Window)", fontweight='bold')
        ax.set_ylabel("Macro F1 Transfer Gain", fontweight='bold')
        ax.set_title("Figure 3: Historical Checkpoint Age vs. Transfer Gain", fontweight='bold', pad=12)
        ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_checkpoint_age_vs_gain.png'), dpi=300)
    plt.close()


def plot_fig4_precision_recall_f1(df_summary, save_dir):
    """Figure 4 — Multi-metric Comparison (Precision, Recall, F1)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    methods = [
        'Fixed Ensemble',
        'Global Adaptive Ensemble',
        'Regime-Aware Ensemble',
        'Current Event-Driven',
        'Oracle Weighting',
        'Oracle Model Retrieval (Held-Out)'
    ]

    agg = df_summary[df_summary['method'].isin(methods)].groupby('method')[['precision', 'recall', 'f1']].mean().reindex(methods).reset_index()

    x = np.arange(len(methods))
    width = 0.25

    ax.bar(x - width, agg['precision'], width, label='Precision', color='#4C72B0', edgecolor='black', alpha=0.85)
    ax.bar(x, agg['recall'], width, label='Recall', color='#55A868', edgecolor='black', alpha=0.85)
    ax.bar(x + width, agg['f1'], width, label='Macro F1', color='#C44E52', edgecolor='black', alpha=0.85)

    ax.set_ylabel("Score", fontweight='bold')
    ax.set_title("Figure 4: Precision, Recall, and Macro F1 Across Methods", fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylim(0.90, 1.05)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_precision_recall_f1.png'), dpi=300)
    plt.close()


def plot_fig5_confusion_matrices(cm_dict, save_dir):
    """Figure 5 — Confusion Matrix & FP/FN Statistics."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Format confusion matrix values as strings
    tp = cm_dict.get('tp', 0)
    tn = cm_dict.get('tn', 0)
    fp = cm_dict.get('fp', 0)
    fn = cm_dict.get('fn', 0)

    cm_data = np.array([[tn, fp], [fn, tp]])

    im = ax.imshow(cm_data, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    classes = ['Normal (0)', 'Attack (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontweight='bold')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontweight='bold')

    for i in range(2):
        for j in range(2):
            val = cm_data[i, j]
            color = 'white' if val > cm_data.max() / 2.0 else 'black'
            ax.text(j, i, f"{val:,}\n({val/cm_data.sum():.1%})", ha='center', va='center', color=color, fontweight='bold', fontsize=12)

    ax.set_xlabel("Predicted Label", fontweight='bold')
    ax.set_ylabel("True Label", fontweight='bold')
    ax.set_title(f"Figure 5: Confusion Matrix (FP = {fp} | Precision = {tp/(tp+fp) if (tp+fp)>0 else 1.0:.4f})", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_confusion_matrix.png'), dpi=300)
    plt.close()


def plot_fig6_headroom_comparison(headroom_dict, save_dir):
    """Figure 6 — Weight Oracle vs Historical Model Retrieval Headroom."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    categories = [
        'Weight-Only Oracle Headroom\n(H_weight)',
        'In-Sample Model Retrieval Headroom\n(H_retrieval_insample)',
        'Held-Out Model Retrieval Headroom\n(H_retrieval_heldout)'
    ]

    values = [
        headroom_dict.get('h_weight', 0.0),
        headroom_dict.get('h_insample', 0.0),
        headroom_dict.get('h_heldout', 0.0)
    ]

    colors = ['#CCB974', '#C44E52', '#9370DB']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.85, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.0005, f"+{val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel("Macro F1 Headroom over Current Baseline", fontweight='bold')
    ax.set_title("Figure 6: Weight-Only vs. Historical Model Retrieval Headroom", fontweight='bold', pad=12)
    ax.set_ylim(min(0.0, min(values) - 0.002), max(values) + 0.005)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_headroom_comparison.png'), dpi=300)
    plt.close()


def generate_all_retrieval_figures(df_summary, df_window, df_age, cm_dict, headroom_dict, save_dir):
    """Generates all 6 publication figures for the retrieval experiment."""
    os.makedirs(save_dir, exist_ok=True)
    plot_fig1_method_comparison(df_summary, save_dir)
    plot_fig2_retrieval_gain_over_time(df_window, save_dir)
    plot_fig3_checkpoint_age_vs_gain(df_age, save_dir)
    plot_fig4_precision_recall_f1(df_summary, save_dir)
    plot_fig5_confusion_matrices(cm_dict, save_dir)
    plot_fig6_headroom_comparison(headroom_dict, save_dir)
