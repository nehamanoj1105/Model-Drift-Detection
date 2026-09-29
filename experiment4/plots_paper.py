"""
================================================================================
EXPERIMENT 4 v4 — PAPER-READY FIGURE CONSOLIDATION (7 CORE FIGURES)
================================================================================
Generates the consolidated 7 figures for the paper manuscript in figures_paper/:
  1. accuracy_vs_cost_tradeoff.png: Headline Pareto scatter (direct point labels)
  2. cpu_cost_by_strategy.png: Sorted CPU bar chart with Frozen reference line
  3. f1_score_by_strategy.png: Mean F1 bar chart aligned with CPU ordering
  4. buffer_cap_ablation.png: Two-panel ablation of Bounded vs. Unbounded buffer
  5. drift_detector_quality.png: Precision / Recall / F1 grouped bar chart for 5 tuned detectors
  6. f1_by_drift_type.png: F1 across Stationary, Covariate, Concept, Mixed (Frozen vs. Cont vs. Comp-Selective)
  7. component_retraining_frequency.png: Retraining event breakdown (RF vs. ET vs. GB)
================================================================================
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER_PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures_paper')

# Manuscript-grade plotting aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.30,
    'grid.linestyle': '--',
})

PALETTE = {
    'Frozen Model': '#555555',
    'Continuously Retrained Ensemble': '#d62728',
    'Component-Selective Ensemble': '#2ca02c',
    'Warm-Start Ensemble': '#17becf',
    'Event-Driven Ensemble (Custom Dual-Trigger)': '#8c564b',
    'Event-Driven Ensemble (Page-Hinkley)': '#bcbd22',
    'Two-Tier Hybrid Ensemble': '#ff7f0e',
    'Event-Driven Ensemble (ADWIN)': '#1f77b4',
    'Event-Driven Ensemble (DDM)': '#e377c2',
    'Event-Driven Ensemble (EDDM)': '#9467bd',
    'Adaptive Random Forest (river)': '#7f7f7f',
    'Streaming Random Patches (river)': '#aec7e8',
}


def generate_paper_figures(df_window, df_retrain, df_detector_quality, df_ablation=None, output_dir=PAPER_PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating 7 paper-ready figures in: {output_dir}")

    fig1_pareto_scatter(df_window, output_dir)
    fig2_cpu_cost(df_window, output_dir)
    fig3_f1_score(df_window, output_dir)
    if df_ablation is not None and len(df_ablation) > 0:
        fig4_buffer_ablation(df_ablation, output_dir)
    fig5_detector_quality(df_detector_quality, output_dir)
    fig6_f1_by_drift_type(df_window, output_dir)
    fig7_component_frequency(df_retrain, output_dir)
    print("All 7 paper-ready figures successfully generated!")


# -----------------------------------------------------------------------------
# FIGURE 1: Headline Accuracy vs. Cost Pareto Trade-Off
# -----------------------------------------------------------------------------
def fig1_pareto_scatter(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    approaches = list(df_window['approach'].unique())

    # Direct labels positioning offsets to prevent collision
    offsets = {
        'Frozen Model': (10, -5),
        'Warm-Start Ensemble': (-60, -8),
        'Component-Selective Ensemble': (-65, 8),
        'Event-Driven Ensemble (Page-Hinkley)': (12, -18),
        'Event-Driven Ensemble (Custom Dual-Trigger)': (12, 16),
        'Two-Tier Hybrid Ensemble': (12, -18),
        'Event-Driven Ensemble (DDM)': (12, 16),
        'Event-Driven Ensemble (ADWIN)': (12, -18),
        'Event-Driven Ensemble (EDDM)': (-45, 16),
        'Continuously Retrained Ensemble': (12, 16),
        'Adaptive Random Forest (river)': (10, 5),
        'Streaming Random Patches (river)': (-55, -18),
    }

    short_names = {
        'Frozen Model': 'Frozen Model',
        'Continuously Retrained Ensemble': 'Continuous Retraining',
        'Component-Selective Ensemble': 'Component-Selective (Ours)',
        'Warm-Start Ensemble': 'Warm-Start',
        'Event-Driven Ensemble (Page-Hinkley)': 'ED (Page-Hinkley)',
        'Event-Driven Ensemble (Custom Dual-Trigger)': 'ED (Custom Dual-Trigger)',
        'Two-Tier Hybrid Ensemble': 'Two-Tier Hybrid',
        'Event-Driven Ensemble (DDM)': 'ED (DDM)',
        'Event-Driven Ensemble (ADWIN)': 'ED (ADWIN)',
        'Event-Driven Ensemble (EDDM)': 'ED (EDDM)',
        'Adaptive Random Forest (river)': 'ARF (river)',
        'Streaming Random Patches (river)': 'SRP (river)',
    }

    for app in approaches:
        sub = df_window[df_window['approach'] == app]
        mean_cpu = sub.groupby('seed')['cpu_time'].sum().mean()
        std_cpu = sub.groupby('seed')['cpu_time'].sum().std()
        mean_f1 = sub.groupby('seed')['f1'].mean().mean()
        std_f1 = sub.groupby('seed')['f1'].mean().std()

        c = PALETTE.get(app, '#333333')
        is_highlight = 'Component-Selective' in app or 'Warm-Start' in app
        sz = 140 if is_highlight else 90
        marker = '*' if 'Component-Selective' in app else ('^' if 'Warm-Start' in app else 'o')

        ax.errorbar(mean_cpu, mean_f1, xerr=std_cpu, yerr=std_f1,
                    fmt=marker, color=c, ecolor=c, elinewidth=1.6, capsize=4,
                    markersize=11 if is_highlight else 8, zorder=5 if is_highlight else 3)

        label_txt = short_names.get(app, app)
        ox, oy = offsets.get(app, (8, 5))
        ax.annotate(f"{label_txt}\n({mean_f1:.3f}, {mean_cpu:.1f}s)",
                    xy=(mean_cpu, mean_f1), xytext=(ox, oy), textcoords='offset points',
                    fontsize=8.5, fontweight='bold' if is_highlight else 'normal',
                    color='#000000',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.82, edgecolor='#cccccc', linewidth=0.5))

    ax.set_title('Figure 1: Predictive Performance vs. Computational Cost Trade-off')
    ax.set_xlabel('Cumulative CPU Execution Time (s) [Lower is Better]')
    all_cpus = [df_window[df_window['approach'] == a].groupby('seed')['cpu_time'].sum().mean() for a in approaches]
    max_c = max(all_cpus) if all_cpus else 90.0
    ax.set_xlim(-2, max_c * 1.15)
    ax.set_ylim(0.44, 0.80)

    # Annotate Pareto Frontier region cleanly straight above Component-Selective
    cs_sub = df_window[df_window['approach'] == 'Component-Selective Ensemble']
    if not cs_sub.empty:
        cs_cpu = float(cs_sub.groupby('seed')['cpu_time'].sum().mean())
        cs_f1 = float(cs_sub.groupby('seed')['f1'].mean().mean())
        ax.annotate('Optimal Frontier\n(High F1, Low CPU)', xy=(cs_cpu, cs_f1), xytext=(cs_cpu, cs_f1 + 0.025),
                    ha='center',
                    arrowprops=dict(facecolor='#2ca02c', edgecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='#2ca02c',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#2ca02c', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_cost_tradeoff.png'), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# FIGURE 2: CPU Cost by Strategy (Sorted Low to High with Frozen Reference Line)
# -----------------------------------------------------------------------------
def fig2_cpu_cost(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    approaches = list(df_window['approach'].unique())

    data = []
    for app in approaches:
        sub = df_window[df_window['approach'] == app]
        mean_cpu = sub.groupby('seed')['cpu_time'].sum().mean()
        std_cpu = sub.groupby('seed')['cpu_time'].sum().std()
        data.append({'app': app, 'mean_cpu': mean_cpu, 'std_cpu': std_cpu})

    df_sort = pd.DataFrame(data).sort_values('mean_cpu').reset_index(drop=True)

    short_map = {
        'Frozen Model': 'Frozen Model',
        'Continuously Retrained Ensemble': 'Continuous',
        'Component-Selective Ensemble': 'Component-Selective (Ours)',
        'Warm-Start Ensemble': 'Warm-Start',
        'Event-Driven Ensemble (Page-Hinkley)': 'ED (Page-Hinkley)',
        'Event-Driven Ensemble (Custom Dual-Trigger)': 'ED (Custom)',
        'Two-Tier Hybrid Ensemble': 'Two-Tier Hybrid',
        'Event-Driven Ensemble (DDM)': 'ED (DDM)',
        'Event-Driven Ensemble (ADWIN)': 'ED (ADWIN)',
        'Event-Driven Ensemble (EDDM)': 'ED (EDDM)',
        'Adaptive Random Forest (river)': 'ARF (river)',
        'Streaming Random Patches (river)': 'SRP (river)',
    }

    labels = [short_map.get(a, a) for a in df_sort['app']]
    colors = [PALETTE.get(a, '#333') for a in df_sort['app']]
    frozen_val = df_sort[df_sort['app'] == 'Frozen Model']['mean_cpu'].values[0]

    bars = ax.bar(range(len(df_sort)), df_sort['mean_cpu'], yerr=df_sort['std_cpu'],
                  color=colors, width=0.6, capsize=3.5, alpha=0.88, edgecolor='black', linewidth=0.7)

    # Reference line for Frozen
    ax.axhline(frozen_val, color='#555555', linestyle='--', linewidth=1.5,
               label=f'Frozen Baseline CPU ({frozen_val:.2f}s)')

    for bar, val, app_name in zip(bars, df_sort['mean_cpu'], df_sort['app']):
        ratio = val / frozen_val if frozen_val > 0 else 0
        ratio_str = f"{ratio:.1f}x" if app_name != 'Frozen Model' else "1.0x"
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 2.0,
                f"{val:.1f}s\n({ratio_str})", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Figure 2: Cumulative CPU Time by Deployment Strategy (Sorted Low-to-High)')
    ax.set_xlabel('Deployment Strategy')
    ax.set_ylabel('Cumulative CPU Time (seconds) [Lower is Better]')
    ax.set_xticks(range(len(df_sort)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9.5)
    ax.set_ylim(0, max(df_sort['mean_cpu']) * 1.25)
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_cost_by_strategy.png'), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# FIGURE 3: F1-Score by Strategy (Matching Figure 2 Ordering)
# -----------------------------------------------------------------------------
def fig3_f1_score(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    approaches = list(df_window['approach'].unique())

    # Sort in identical order as Figure 2
    cpu_data = []
    for app in approaches:
        sub = df_window[df_window['approach'] == app]
        cpu_data.append({'app': app, 'cpu': sub.groupby('seed')['cpu_time'].sum().mean()})
    sorted_order = [x['app'] for x in sorted(cpu_data, key=lambda x: x['cpu'])]

    f1_data = []
    for app in sorted_order:
        sub = df_window[df_window['approach'] == app]
        m = sub.groupby('seed')['f1'].mean().mean()
        s = sub.groupby('seed')['f1'].mean().std()
        f1_data.append({'app': app, 'mean_f1': m, 'std_f1': s})
    df_f1 = pd.DataFrame(f1_data)

    short_map = {
        'Frozen Model': 'Frozen Model',
        'Continuously Retrained Ensemble': 'Continuous',
        'Component-Selective Ensemble': 'Component-Selective (Ours)',
        'Warm-Start Ensemble': 'Warm-Start',
        'Event-Driven Ensemble (Page-Hinkley)': 'ED (Page-Hinkley)',
        'Event-Driven Ensemble (Custom Dual-Trigger)': 'ED (Custom)',
        'Two-Tier Hybrid Ensemble': 'Two-Tier Hybrid',
        'Event-Driven Ensemble (DDM)': 'ED (DDM)',
        'Event-Driven Ensemble (ADWIN)': 'ED (ADWIN)',
        'Event-Driven Ensemble (EDDM)': 'ED (EDDM)',
        'Adaptive Random Forest (river)': 'ARF (river)',
        'Streaming Random Patches (river)': 'SRP (river)',
    }

    labels = [short_map.get(a, a) for a in df_f1['app']]
    colors = [PALETTE.get(a, '#333') for a in df_f1['app']]
    cont_f1 = df_f1[df_f1['app'] == 'Continuously Retrained Ensemble']['mean_f1'].values[0]

    bars = ax.bar(range(len(df_f1)), df_f1['mean_f1'], yerr=df_f1['std_f1'],
                  color=colors, width=0.6, capsize=3.5, alpha=0.88, edgecolor='black', linewidth=0.7)

    # Reference line for Continuous Retraining parity
    ax.axhline(cont_f1, color='#d62728', linestyle=':', linewidth=1.5,
               label=f'Continuous Retraining Parity ({cont_f1:.4f})')

    for bar, val in zip(bars, df_f1['mean_f1']):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.015,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_title('Figure 3: Mean Prequential F1-Score by Deployment Strategy (Ordered by CPU Cost)')
    ax.set_xlabel('Deployment Strategy (Identical Ordering to CPU Cost Chart)')
    ax.set_ylabel('Mean F1-Score (± 1 SD) [Higher is Better]')
    ax.set_xticks(range(len(df_f1)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9.5)
    ax.set_ylim(0.40, 0.82)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_by_strategy.png'), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# FIGURE 4: Buffer Cap Ablation (Two-Panel: CPU and F1)
# -----------------------------------------------------------------------------
def fig4_buffer_ablation(df_ablation, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    approaches = ['Continuous Retraining', 'Custom Dual-Trigger', 'Component-Selective', 'Warm-Start Ensemble']
    short_labels = ['Continuous', 'Event-Driven Ensemble', 'Component-Selective', 'Warm-Start']

    x = np.arange(len(approaches))
    width = 0.35

    # Filter by condition
    sub_b = df_ablation[df_ablation['condition'].str.contains('Bounded')]
    sub_u = df_ablation[df_ablation['condition'].str.contains('Unbounded')]

    cpu_b = [sub_b[sub_b['approach'] == a]['cpu_time'].mean() for a in approaches]
    cpu_b_std = [sub_b[sub_b['approach'] == a]['cpu_time'].std() for a in approaches]
    cpu_u = [sub_u[sub_u['approach'] == a]['cpu_time'].mean() for a in approaches]
    cpu_u_std = [sub_u[sub_u['approach'] == a]['cpu_time'].std() for a in approaches]

    f1_b = [sub_b[sub_b['approach'] == a]['f1'].mean() for a in approaches]
    f1_b_std = [sub_b[sub_b['approach'] == a]['f1'].std() for a in approaches]
    f1_u = [sub_u[sub_u['approach'] == a]['f1'].mean() for a in approaches]
    f1_u_std = [sub_u[sub_u['approach'] == a]['f1'].std() for a in approaches]

    # Panel 1: CPU Time
    b1 = ax1.bar(x - width/2, cpu_b, width, yerr=cpu_b_std, label='Bounded (5,000 samples)', color='#1f77b4', capsize=3)
    b2 = ax1.bar(x + width/2, cpu_u, width, yerr=cpu_u_std, label='Unbounded (up to 50k)', color='#ff7f0e', capsize=3)
    ax1.set_title('(A) Cumulative CPU Time (s)')
    ax1.set_ylabel('CPU Time (s) [Lower is Better]')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=20, ha='right', fontsize=9.5)
    ax1.legend(loc='upper right', framealpha=0.9)

    for bar, val in zip(b1, cpu_b):
        ax1.text(bar.get_x() + bar.get_width()/2.0, val + 1.0, f"{val:.1f}s", ha='center', va='bottom', fontsize=8)
    for bar, val in zip(b2, cpu_u):
        ax1.text(bar.get_x() + bar.get_width()/2.0, val + 1.0, f"{val:.1f}s", ha='center', va='bottom', fontsize=8)

    # Panel 2: F1 Score
    b3 = ax2.bar(x - width/2, f1_b, width, yerr=f1_b_std, label='Bounded (5,000 samples)', color='#1f77b4', capsize=3)
    b4 = ax2.bar(x + width/2, f1_u, width, yerr=f1_u_std, label='Unbounded (up to 50k)', color='#ff7f0e', capsize=3)
    ax2.set_title('(B) Prequential F1-Score')
    ax2.set_ylabel('Mean F1-Score [Higher is Better]')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=20, ha='right', fontsize=9.5)
    ax2.set_ylim(0.60, 0.80)
    ax2.legend(loc='lower left', framealpha=0.9)

    for bar, val in zip(b3, f1_b):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val + 0.005, f"{val:.3f}", ha='center', va='bottom', fontsize=8)
    for bar, val in zip(b4, f1_u):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val + 0.005, f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    fig.suptitle('Figure 4: Controlled Buffer Cap Ablation (Bounded 5,000 vs. Unbounded)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'buffer_cap_ablation.png'), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# FIGURE 5: Drift Detector Quality (Precision, Recall, F1 under Fair Tuning)
# -----------------------------------------------------------------------------
def fig5_detector_quality(df_detector_quality, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    detectors = ['Custom Dual-Trigger', 'ADWIN', 'DDM', 'EDDM', 'Page-Hinkley']
    detector_labels = ['Event-Driven Ensemble', 'ADWIN', 'DDM', 'EDDM', 'Page-Hinkley']

    sub = df_detector_quality.groupby('detector')[['precision', 'recall', 'f1']].mean().loc[detectors]

    x = np.arange(len(detectors))
    width = 0.25

    p_bars = ax.bar(x - width, sub['precision'], width, label='Precision', color='#1f77b4', alpha=0.9)
    r_bars = ax.bar(x, sub['recall'], width, label='Recall', color='#2ca02c', alpha=0.9)
    f_bars = ax.bar(x + width, sub['f1'], width, label='F1-Score', color='#d62728', alpha=0.9)

    for bars in [p_bars, r_bars, f_bars]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_title('Figure 5: Drift Detector Quality Metrics Under Equalized Validation Tuning')
    ax.set_xlabel('Drift Detection Algorithm')
    ax.set_ylabel('Score [0.0 – 1.0]')
    ax.set_xticks(x)
    ax.set_xticklabels(detector_labels, fontsize=9.5)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drift_detector_quality.png'), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# FIGURE 6: F1 by Drift Type (Frozen vs. Continuous vs. Component-Selective)
# -----------------------------------------------------------------------------
def fig6_f1_by_drift_type(df_window, output_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    drift_types = ['none', 'covariate', 'concept', 'mixed']
    type_labels = ['Stationary', 'Covariate Shift', 'Concept Drift', 'Mixed Transitions']

    triad = ['Frozen Model', 'Continuously Retrained Ensemble', 'Component-Selective Ensemble']
    triad_labels = ['Frozen Model', 'Continuous Retraining', 'Component-Selective (Ours)']
    colors = ['#555555', '#d62728', '#2ca02c']

    x = np.arange(len(drift_types))
    width = 0.26

    for i, (app, label, c) in enumerate(zip(triad, triad_labels, colors)):
        means = []
        stds = []
        for dt in drift_types:
            sub = df_window[(df_window['approach'] == app) & (df_window['drift_type'] == dt)]
            m = sub.groupby('seed')['f1'].mean().mean()
            s = sub.groupby('seed')['f1'].mean().std()
            means.append(m)
            stds.append(s)

        bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=label,
                      color=c, capsize=3.5, alpha=0.9, edgecolor='black', linewidth=0.7)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.015, f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    ax.set_title('Figure 6: Regime-Specific Resilience Across Drift Regimes')
    ax.set_xlabel('Environmental Regime')
    ax.set_ylabel('Mean F1-Score (± 1 SD)')
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=10)
    ax.set_ylim(0.50, 0.92)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_drift_type.png'), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# FIGURE 7: Component Retraining Frequency Breakdown
# -----------------------------------------------------------------------------
def fig7_component_frequency(df_retrain, output_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sub = df_retrain[df_retrain['approach'] == 'Component-Selective Ensemble']

    comp_counts = {'RF': 0, 'ET': 0, 'GB': 0}
    for comp_list_str in sub['retrained_components']:
        if isinstance(comp_list_str, str) and comp_list_str:
            for c in comp_list_str.split(','):
                c = c.strip()
                if c in comp_counts:
                    comp_counts[c] += 1

    components = ['Random Forest (RF)', 'Extra Trees (ET)', 'Gradient Boosting (GB)']
    keys = ['RF', 'ET', 'GB']
    counts = [comp_counts[k] for k in keys]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(components, counts, color=colors, width=0.5, alpha=0.9, edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, counts):
        pct = (val / sum(counts)) * 100.0 if sum(counts) > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 2, f"{val} events\n({pct:.1f}%)",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Figure 7: Component-Selective Retraining Frequency by Sub-Model')
    ax.set_xlabel('Base Model Component (Multi-Temporal Horizon)')
    ax.set_ylabel('Total Adaptation Events Across 5 Seeds')
    ax.set_ylim(0, max(counts) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_retraining_frequency.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    from config import RESULTS_DIR
    df_win = pd.read_csv(os.path.join(RESULTS_DIR, 'window_results.csv'))
    df_ret = pd.read_csv(os.path.join(RESULTS_DIR, 'retraining_results.csv'))
    df_det = pd.read_csv(os.path.join(RESULTS_DIR, 'detector_quality.csv'))
    abl_path = os.path.join(RESULTS_DIR, 'buffer_cap_ablation.csv')
    df_abl = pd.read_csv(abl_path) if os.path.exists(abl_path) else None

    generate_paper_figures(df_win, df_ret, df_det, df_abl)

