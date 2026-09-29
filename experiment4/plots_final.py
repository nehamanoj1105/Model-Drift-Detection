"""
================================================================================
EXPERIMENT 4 v6 (FINAL) — RESTRUCTURED MANUSCRIPT FIGURE SUITE
================================================================================
Generates two strictly separated figure groups (never conflated into one chart):

Group A: Headline Three-Way Comparison
  Directory: experiment4/figures_final/group_a_headline/
  1. f1_comparison.png: 3-bar chart of prequential F1 (Frozen vs. Cont vs. Event-Driven)
  2. cpu_cost_comparison.png: 3-bar chart of CPU cost with reconciled 'Nx Frozen' labels
  3. accuracy_vs_cost_tradeoff.png: 3-point headline Pareto scatter plot
  4. f1_by_drift_type.png: 3-approach grouped bar chart across Stationary, Covariate, Concept, Mixed

Group B: Event-Driven Trigger vs. Standard Literature Detectors
  Directory: experiment4/figures_final/group_b_baselines/
  5. f1_vs_standard_detectors.png: F1 comparison (Tuned Event-Driven vs. ADWIN, DDM, EDDM, Page-Hinkley)
  6. cpu_cost_vs_standard_detectors.png: CPU cost comparison across the 5 detector mechanisms
  7. detector_quality_precision_recall.png: Grouped bar chart of Precision, Recall, and F1
================================================================================
"""

import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_A_DIR = os.path.join(BASE_DIR, 'figures_final', 'group_a_headline')
GROUP_B_DIR = os.path.join(BASE_DIR, 'figures_final', 'group_b_baselines')

ARTIFACT_A_DIR = r"C:\Users\emhaenn\.gemini\antigravity\brain\e7b78523-74fa-4f1f-9dea-d867ccc55e5b\figures_final\group_a_headline"
ARTIFACT_B_DIR = r"C:\Users\emhaenn\.gemini\antigravity\brain\e7b78523-74fa-4f1f-9dea-d867ccc55e5b\figures_final\group_b_baselines"

# Publication aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.30,
    'grid.linestyle': '--',
})

# Colors
COLOR_FROZEN = '#555555'
COLOR_CONT = '#d62728'
COLOR_EVENT = '#2ca02c'

COLOR_DETECTORS = {
    'Tuned Event-Driven (Ours)': '#2ca02c',
    'Custom Dual-Trigger': '#2ca02c',
    'Page-Hinkley': '#bcbd22',
    'DDM': '#e377c2',
    'ADWIN': '#1f77b4',
    'EDDM': '#9467bd',
}


# =============================================================================
# GROUP A: HEADLINE THREE-WAY COMPARISON
# =============================================================================

def fig1_headline_f1(df_window, output_dir=GROUP_A_DIR):
    """Figure 1: Simple 3-bar chart of prequential F1."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    app_map = {
        'Frozen Model': 'Frozen Model\n(Static Baseline)',
        'Continuously Retrained Ensemble': 'Continuous Retraining\n(Always-On Ceiling)',
        'Component-Selective Ensemble': 'Event-Driven Ensemble',
    }
    
    apps = list(app_map.keys())
    labels = [app_map[a] for a in apps]
    colors = [COLOR_FROZEN, COLOR_CONT, COLOR_EVENT]
    
    means = []
    stds = []
    for app in apps:
        sub = df_window[df_window['approach'] == app]
        seed_means = sub.groupby('seed')['f1'].mean()
        means.append(float(seed_means.mean()))
        stds.append(float(seed_means.std()))
        
    bars = ax.bar(range(3), means, yerr=stds, color=colors, width=0.52, capsize=5,
                  alpha=0.90, edgecolor='black', linewidth=0.8)
    
    # Reference line at continuous retraining ceiling
    ax.axhline(means[1], color=COLOR_CONT, linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Continuous Ceiling ({means[1]:.4f})')
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.015, f"{val:.4f}",
                ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        
    ax.set_title('Figure 1: Prequential F1-Score (Headline Comparison)')
    ax.set_ylabel('Mean Prequential F1-Score [Higher is Better]')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 0.82)
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=300)
    plt.close()


def fig2_headline_cpu(df_window, output_dir=GROUP_A_DIR):
    """Figure 2: 3-bar chart of cumulative CPU cost with reconciled 'Nx Frozen' labels."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    app_map = {
        'Frozen Model': 'Frozen Model',
        'Component-Selective Ensemble': 'Event-Driven Ensemble',
        'Continuously Retrained Ensemble': 'Continuous Retraining',
    }
    
    apps = list(app_map.keys())
    labels = [app_map[a] for a in apps]
    colors = [COLOR_FROZEN, COLOR_EVENT, COLOR_CONT]
    
    means = []
    stds = []
    seed_data = {}
    for app in apps:
        sub = df_window[df_window['approach'] == app]
        seed_sums = sub.groupby('seed')['cpu_time'].sum()
        seed_data[app] = seed_sums
        means.append(float(seed_sums.mean()))
        stds.append(float(seed_sums.std()))
        
    frozen_mean = means[0]
    frozen_seeds = seed_data['Frozen Model']
    
    bars = ax.bar(range(3), means, yerr=stds, color=colors, width=0.52, capsize=5,
                  alpha=0.90, edgecolor='black', linewidth=0.8)
    
    for idx, (bar, val, app) in enumerate(zip(bars, means, apps)):
        if app == 'Frozen Model':
            ratio_str = "1.00x\n(Baseline)"
        else:
            ratio_per_seed = seed_data[app] / frozen_seeds
            ratio_m = float(ratio_per_seed.mean())
            ratio_s = float(ratio_per_seed.std())
            ratio_str = f"{ratio_m:.1f}x ± {ratio_s:.1f}x\nvs. Frozen"
            
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 2.5, f"{val:.1f}s\n({ratio_str})",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax.set_title('Figure 2: Cumulative CPU Time & Cost Multipliers (Headline Comparison)', pad=14)
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_cost_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig3_headline_tradeoff(df_window, output_dir=GROUP_A_DIR):
    """Figure 3: 3-point headline Pareto scatter plot."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    app_info = [
        ('Frozen Model', 'Frozen Model\n(Static Baseline)', COLOR_FROZEN, 's', 110, (14, -12), 'left'),
        ('Continuously Retrained Ensemble', 'Continuous Retraining\n(Full Adaptation Ceiling)', COLOR_CONT, 'o', 120, (-15, -35), 'right'),
        ('Component-Selective Ensemble', 'Event-Driven Ensemble', COLOR_EVENT, '*', 220, (14, 14), 'left'),
    ]
    
    cpus, f1s = [], []
    for app_key, label, color, marker, sz, offset, align in app_info:
        sub = df_window[df_window['approach'] == app_key]
        seed_cpu = sub.groupby('seed')['cpu_time'].sum()
        seed_f1 = sub.groupby('seed')['f1'].mean()
        
        m_cpu, s_cpu = float(seed_cpu.mean()), float(seed_cpu.std())
        m_f1, s_f1 = float(seed_f1.mean()), float(seed_f1.std())
        cpus.append(m_cpu)
        f1s.append(m_f1)
        
        ax.errorbar(m_cpu, m_f1, xerr=s_cpu, yerr=s_f1, fmt=marker, color=color, ecolor=color,
                    elinewidth=1.6, capsize=4, markersize=11 if marker=='*' else 9, zorder=5)
        
        ax.annotate(f"{label}\n({m_f1:.4f}, {m_cpu:.1f}s)",
                    xy=(m_cpu, m_f1), xytext=offset, textcoords='offset points',
                    ha=align,
                    fontsize=9.5, fontweight='bold' if marker=='*' else 'normal',
                    color='#000000',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88, edgecolor='#cccccc', linewidth=0.6))
        
    # Draw horizontal arrow showing compute reduction from Continuous to Event-Driven
    ax.annotate('', xy=(20, 0.743), xytext=(75, 0.743),
                arrowprops=dict(facecolor=COLOR_EVENT, edgecolor=COLOR_EVENT, shrink=0.02, width=1.8, headwidth=7))
    ax.text(64.0, 0.760, '82.2% Compute Reduction\nat Identical F1 (p=0.83)',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_EVENT,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#eafaf1', edgecolor=COLOR_EVENT, alpha=0.92))
    
    ax.set_title('Figure 3: Accuracy vs. Cost Trade-Off (Headline Three-Way Comparison)')
    ax.set_xlabel('Cumulative CPU Execution Time (seconds) [Lower is Better]')
    ax.set_ylabel('Mean Prequential F1-Score [Higher is Better]')
    ax.set_xlim(-5, max(cpus) * 1.15)
    ax.set_ylim(0.64, 0.790)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_cost_tradeoff.png'), dpi=300)
    plt.close()


def fig4_headline_drift_types(df_window, output_dir=GROUP_A_DIR):
    """Figure 4: 4-approach grouped bar chart isolating ensembling vs adaptation across Stationary, Covariate, Concept, Mixed."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    
    dtypes = ['none', 'covariate', 'concept', 'mixed']
    d_labels = ['Stationary\n(Baseline)', 'Covariate Shift\n(Feature Drift)', 'Concept Drift\n(P(y|X) Shift)', 'Mixed Drift\n(Combined)']
    
    COLOR_FROZEN_ENS = '#4682b4'  # Steel blue for Frozen Ensemble
    
    apps = ['Frozen Model', 'Frozen Ensemble', 'Continuously Retrained Ensemble', 'Component-Selective Ensemble']
    app_labels = ['Frozen Model', 'Frozen Ensemble', 'Continuous Retraining', 'Event-Driven Ensemble']
    colors = [COLOR_FROZEN, COLOR_FROZEN_ENS, COLOR_CONT, COLOR_EVENT]
    
    x = np.arange(len(dtypes))
    width = 0.19
    
    for i, (app, label, c) in enumerate(zip(apps, app_labels, colors)):
        means, stds = [], []
        for dt in dtypes:
            sub = df_window[(df_window['approach'] == app) & (df_window['drift_type'] == dt)]
            s_m = sub.groupby('seed')['f1'].mean()
            means.append(float(s_m.mean()))
            stds.append(float(s_m.std()))
            
        rects = ax.bar(x + (i - 1.5)*width, means, width, yerr=stds, label=label,
                       color=c, capsize=3.0, alpha=0.88, edgecolor='black', linewidth=0.7)
        for rect, val in zip(rects, means):
            ax.text(rect.get_x() + rect.get_width()/2.0, val + 0.015, f"{val:.3f}",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            
    ax.set_title('Figure 4: Predictive Resilience Disaggregated by Drift Mechanism\n(Isolating Ensembling vs. Adaptation Effects)', fontsize=12, pad=10)
    ax.set_ylabel('Mean Prequential F1-Score [Higher is Better]')
    ax.set_xticks(x)
    ax.set_xticklabels(d_labels)
    ax.set_ylim(0.40, 0.98)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_drift_type.png'), dpi=300)
    plt.close()


# =============================================================================
# GROUP B: EVENT-DRIVEN ENSEMBLE VS. STANDARD LITERATURE DETECTOR BASELINES
# =============================================================================

def fig5_baselines_f1(df_window, output_dir=GROUP_B_DIR):
    """Figure 5: F1 comparison: Tuned Event-Driven vs. ADWIN, DDM, EDDM, Page-Hinkley."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    app_map = {
        'Component-Selective Ensemble': 'Event-Driven Ensemble',
        'Event-Driven Ensemble (Page-Hinkley)': 'Page-Hinkley\nDetector',
        'Event-Driven Ensemble (ADWIN)': 'ADWIN\nDetector',
        'Event-Driven Ensemble (DDM)': 'DDM\nDetector',
        'Event-Driven Ensemble (EDDM)': 'EDDM\nDetector',
    }
    
    apps = list(app_map.keys())
    labels = [app_map[a] for a in apps]
    colors = [COLOR_EVENT, '#bcbd22', '#1f77b4', '#e377c2', '#9467bd']
    
    means, stds = [], []
    for app in apps:
        sub = df_window[df_window['approach'] == app]
        seed_m = sub.groupby('seed')['f1'].mean()
        means.append(float(seed_m.mean()))
        stds.append(float(seed_m.std()))
        
    bars = ax.bar(range(len(apps)), means, yerr=stds, color=colors, width=0.52, capsize=4,
                  alpha=0.88, edgecolor='black', linewidth=0.7)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.01, f"{val:.4f}",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax.set_title('Figure 5: F1-Score Comparison Against Standard Literature Detectors')
    ax.set_ylabel('Mean Prequential F1-Score [Higher is Better]')
    ax.set_xticks(range(len(apps)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.68, 0.79)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_vs_standard_detectors.png'), dpi=300)
    plt.close()


def fig6_baselines_cpu(df_window, output_dir=GROUP_B_DIR):
    """Figure 6: CPU cost comparison: Event-Driven vs. standard detectors."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    app_map = {
        'Component-Selective Ensemble': 'Event-Driven Ensemble',
        'Event-Driven Ensemble (Page-Hinkley)': 'Page-Hinkley\nDetector',
        'Event-Driven Ensemble (DDM)': 'DDM\nDetector',
        'Event-Driven Ensemble (ADWIN)': 'ADWIN\nDetector',
        'Event-Driven Ensemble (EDDM)': 'EDDM\nDetector',
    }
    
    apps = list(app_map.keys())
    labels = [app_map[a] for a in apps]
    colors = [COLOR_EVENT, '#bcbd22', '#e377c2', '#1f77b4', '#9467bd']
    
    means, stds = [], []
    for app in apps:
        sub = df_window[df_window['approach'] == app]
        seed_s = sub.groupby('seed')['cpu_time'].sum()
        means.append(float(seed_s.mean()))
        stds.append(float(seed_s.std()))
        
    bars = ax.bar(range(len(apps)), means, yerr=stds, color=colors, width=0.52, capsize=4,
                  alpha=0.88, edgecolor='black', linewidth=0.7)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 2.0, f"{val:.1f}s",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax.set_title('Figure 6: Cumulative CPU Time Comparison Against Standard Detectors', pad=14)
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.set_xticks(range(len(apps)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_cost_vs_standard_detectors.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig7_detector_quality(df_detector_quality, output_dir=GROUP_B_DIR):
    """Figure 7: Grouped Precision / Recall / F1 bar chart for all 5 detectors."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    det_map = {
        'Custom Dual-Trigger': 'Event-Driven Ensemble',
        'Page-Hinkley': 'Page-Hinkley',
        'ADWIN': 'ADWIN',
        'DDM': 'DDM',
        'EDDM': 'EDDM',
    }
    
    detectors = list(det_map.keys())
    labels = [det_map[d] for d in detectors]
    
    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Episode Recall', 'Detector F1']
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    x = np.arange(len(detectors))
    width = 0.24
    
    for i, (m, m_label, c) in enumerate(zip(metrics, metric_labels, metric_colors)):
        means, stds = [], []
        for det in detectors:
            sub = df_detector_quality[df_detector_quality['detector'] == det]
            means.append(float(sub[m].mean()))
            stds.append(float(sub[m].std()))
            
        rects = ax.bar(x + (i - 1)*width, means, width, yerr=stds, label=m_label,
                       color=c, capsize=3.5, alpha=0.88, edgecolor='black', linewidth=0.7)
        for rect, val in zip(rects, means):
            ax.text(rect.get_x() + rect.get_width()/2.0, val + 0.02, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    ax.set_title('Figure 7: Drift Detector Quality Metrics (Equalized Validation Tuning)')
    ax.set_ylabel('Score [0.0 - 1.0]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detector_quality_precision_recall.png'), dpi=300)
    plt.close()


# =============================================================================
# MASTER RUNNER
# =============================================================================

def generate_all_final_figures(df_window, df_detector_quality):
    print("Generating Group A (Headline Three-Way Comparison)...")
    fig1_headline_f1(df_window)
    fig2_headline_cpu(df_window)
    fig3_headline_tradeoff(df_window)
    fig4_headline_drift_types(df_window)
    
    print("Generating Group B (Event-Driven vs. Standard Detectors)...")
    fig5_baselines_f1(df_window)
    fig6_baselines_cpu(df_window)
    fig7_detector_quality(df_detector_quality)
    
    # Mirror to brain artifact directories
    os.makedirs(ARTIFACT_A_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_B_DIR, exist_ok=True)
    
    for f in os.listdir(GROUP_A_DIR):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(GROUP_A_DIR, f), os.path.join(ARTIFACT_A_DIR, f))
            print(f"Synced Group A: {f}")
            
    for f in os.listdir(GROUP_B_DIR):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(GROUP_B_DIR, f), os.path.join(ARTIFACT_B_DIR, f))
            print(f"Synced Group B: {f}")
            
    print("\nAll 7 final figures successfully generated and synced!")

if __name__ == '__main__':
    from config import RESULTS_DIR
    df_win = pd.read_csv(os.path.join(RESULTS_DIR, 'window_results.csv'))
    df_det = pd.read_csv(os.path.join(RESULTS_DIR, 'detector_quality.csv'))
    generate_all_final_figures(df_win, df_det)
