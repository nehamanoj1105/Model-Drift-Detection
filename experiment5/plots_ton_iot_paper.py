"""
================================================================================
EXPERIMENT 5 — ToN_IoT DATASET PAPER-READY FIGURE SUITE
================================================================================
Generates publication-quality figures for the ToN_IoT dataset (Experiment 5),
matching the exact set of figures in the "figures for the paper" folder:

  1. f1_comparison.png: Headline F1 comparison (Frozen vs. Continuous vs. Event-Driven)
  2. cpu_cost_comparison.png: Cumulative CPU time comparison
  3. accuracy_vs_cost_tradeoff.png: F1 vs CPU Time Pareto efficiency plot
  4. f1_by_drift_type.png: F1 disaggregated by ToN_IoT attack regime
  5. f1_vs_standard_detectors.png: F1 comparison vs individual adaptive models
  6. cpu_cost_vs_standard_detectors.png: CPU cost comparison vs individual adaptive models
  7. detector_quality_precision_recall.png: Precision, Recall, F1 metrics on ToN_IoT
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
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ton_iot')
PAPER_DIR = os.path.join(BASE_DIR, 'figures_paper')
ALL_PAPER_DIR = os.path.join(BASE_DIR, 'all_figures', 'figures for the paper')
BRAIN_PAPER_DIR = r"C:\Users\emhaenn\.gemini\antigravity\brain\e7b78523-74fa-4f1f-9dea-d867ccc55e5b\ton_iot_figures_paper"

# Publication style
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
COLOR_RF = '#1f77b4'
COLOR_ET = '#9467bd'
COLOR_GB = '#ff7f0e'


def fig1_f1_comparison(df_metrics, output_dir):
    """Figure 1: Headline 3-way F1 comparison on ToN_IoT."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    apps = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    labels = ['Frozen Model\n(Static Baseline)', 'Continuous Retraining\n(Always-On Ceiling)', 'Event-Driven Ensemble\n(Selective Adaptation)']
    colors = [COLOR_FROZEN, COLOR_CONT, COLOR_EVENT]
    
    means, stds = [], []
    for app in apps:
        sub = df_metrics[df_metrics['approach'] == app]
        seed_m = sub.groupby('seed')['f1'].mean()
        means.append(float(seed_m.mean()))
        stds.append(float(seed_m.std()))
        
    bars = ax.bar(range(3), means, yerr=stds, color=colors, width=0.52, capsize=5,
                  alpha=0.90, edgecolor='black', linewidth=0.8)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.02, f"{val:.4f}",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax.set_title('Figure 1: Prequential F1-Score Comparison (ToN_IoT Dataset)', pad=14)
    ax.set_ylabel('Mean Prequential F1-Score [Higher is Better]')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=300)
    plt.close()


def fig2_cpu_cost_comparison(df_metrics, output_dir):
    """Figure 2: Cumulative CPU time comparison on ToN_IoT."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    apps = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    labels = ['Frozen Model\n(Static)', 'Continuous Retraining\n(Always-On)', 'Event-Driven Ensemble\n(Selective)']
    colors = [COLOR_FROZEN, COLOR_CONT, COLOR_EVENT]
    
    means, stds = [], []
    for app in apps:
        sub = df_metrics[df_metrics['approach'] == app]
        seed_s = sub.groupby('seed')['cpu_time'].sum()
        means.append(float(seed_s.mean()))
        stds.append(float(seed_s.std()))
        
    bars = ax.bar(range(3), means, yerr=stds, color=colors, width=0.52, capsize=5,
                  alpha=0.90, edgecolor='black', linewidth=0.8)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 1.5, f"{val:.2f}s",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax.set_title('Figure 2: Cumulative CPU Execution Time (ToN_IoT Dataset)', pad=14)
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.22)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_cost_comparison.png'), dpi=300)
    plt.close()


def fig3_accuracy_vs_cost_tradeoff(df_metrics, output_dir):
    """Figure 3: F1 vs CPU cost Pareto tradeoff on ToN_IoT."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    apps = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble',
            'Adaptive Random Forest', 'Adaptive Extra Trees', 'Adaptive Gradient Boosting']
    labels = {
        'Frozen Model': 'Frozen Model',
        'Continuously Retrained Ensemble': 'Continuous Retraining',
        'Event-Driven Ensemble': 'Event-Driven Ensemble',
        'Adaptive Random Forest': 'Adaptive RF',
        'Adaptive Extra Trees': 'Adaptive ET',
        'Adaptive Gradient Boosting': 'Adaptive GB',
    }
    colors = {
        'Frozen Model': COLOR_FROZEN,
        'Continuously Retrained Ensemble': COLOR_CONT,
        'Event-Driven Ensemble': COLOR_EVENT,
        'Adaptive Random Forest': COLOR_RF,
        'Adaptive Extra Trees': COLOR_ET,
        'Adaptive Gradient Boosting': COLOR_GB,
    }
    markers = {
        'Frozen Model': 's',
        'Continuously Retrained Ensemble': '^',
        'Event-Driven Ensemble': 'o',
        'Adaptive Random Forest': 'd',
        'Adaptive Extra Trees': 'p',
        'Adaptive Gradient Boosting': 'v',
    }
    
    for app in apps:
        sub = df_metrics[df_metrics['approach'] == app]
        f1_m = float(sub.groupby('seed')['f1'].mean().mean())
        f1_s = float(sub.groupby('seed')['f1'].mean().std())
        cpu_m = float(sub.groupby('seed')['cpu_time'].sum().mean())
        cpu_s = float(sub.groupby('seed')['cpu_time'].sum().std())
        
        ax.errorbar(cpu_m, f1_m, xerr=cpu_s, yerr=f1_s, fmt=markers[app], color=colors[app],
                    markersize=10, capsize=4, label=labels[app], alpha=0.9, elinewidth=1.2)
        ax.annotate(labels[app], (cpu_m, f1_m), textcoords="offset points", xytext=(8, 5),
                    ha='left', fontsize=9, fontweight='bold', color=colors[app])
        
    ax.set_title('Figure 3: F1-Score vs. Computational Cost Trade-off (ToN_IoT)', pad=14)
    ax.set_xlabel('Cumulative CPU Execution Time (seconds)')
    ax.set_ylabel('Mean Prequential F1-Score')
    ax.set_ylim(0.30, 1.05)
    ax.set_xlim(-2, max([df_metrics[df_metrics['approach']==a].groupby('seed')['cpu_time'].sum().mean() for a in apps]) * 1.15)
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_cost_tradeoff.png'), dpi=300)
    plt.close()


def fig4_f1_by_drift_type(df_metrics, output_dir):
    """Figure 4: F1 disaggregated by ToN_IoT attack regimes."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    regimes = ['Regime 1: DDoS', 'Regime 2: Password', 'Regime 3: XSS/Ransomware', 'Regime 4: Backdoor']
    regime_short = ['Regime 1\n(DDoS)', 'Regime 2\n(Password)', 'Regime 3\n(XSS/Ransomware)', 'Regime 4\n(Backdoor)']
    
    apps = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble']
    labels = ['Frozen Model', 'Continuous Retraining', 'Event-Driven Ensemble']
    colors = [COLOR_FROZEN, COLOR_CONT, COLOR_EVENT]
    
    x = np.arange(len(regimes))
    width = 0.25
    
    for i, app in enumerate(apps):
        means, stds = [], []
        for reg in regimes:
            sub = df_metrics[(df_metrics['approach'] == app) & (df_metrics['regime'] == reg)]
            s_m = sub.groupby('seed')['f1'].mean()
            means.append(float(s_m.mean()))
            stds.append(float(s_m.std()))
            
        rects = ax.bar(x + (i - 1)*width, means, width, yerr=stds, label=labels[i],
                       color=colors[i], capsize=3.5, alpha=0.88, edgecolor='black', linewidth=0.7)
        for rect, val in zip(rects, means):
            ax.text(rect.get_x() + rect.get_width()/2.0, val + 0.02, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    ax.set_title('Figure 4: Predictive Resilience Across ToN_IoT Operational Regimes', pad=14)
    ax.set_ylabel('Mean Prequential F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_short)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_by_drift_type.png'), dpi=300)
    plt.close()


def fig5_f1_vs_standard_detectors(df_metrics, output_dir):
    """Figure 5: F1 comparison of Event-Driven Ensemble vs Individual Adaptive Baselines."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    
    apps = ['Event-Driven Ensemble', 'Continuously Retrained Ensemble', 'Adaptive Extra Trees', 'Adaptive Random Forest', 'Adaptive Gradient Boosting']
    labels = ['Event-Driven\nEnsemble', 'Continuous\nEnsemble', 'Adaptive\nExtra Trees', 'Adaptive\nRandom Forest', 'Adaptive\nGradient Boosting']
    colors = [COLOR_EVENT, COLOR_CONT, COLOR_ET, COLOR_RF, COLOR_GB]
    
    means, stds = [], []
    for app in apps:
        sub = df_metrics[df_metrics['approach'] == app]
        seed_m = sub.groupby('seed')['f1'].mean()
        means.append(float(seed_m.mean()))
        stds.append(float(seed_m.std()))
        
    bars = ax.bar(range(len(apps)), means, yerr=stds, color=colors, width=0.52, capsize=4,
                  alpha=0.88, edgecolor='black', linewidth=0.7)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.01, f"{val:.4f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax.set_title('Figure 5: F1-Score Comparison Against Adaptive Baselines (ToN_IoT)', pad=14)
    ax.set_ylabel('Mean Prequential F1-Score')
    ax.set_xticks(range(len(apps)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.90, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_vs_standard_detectors.png'), dpi=300)
    plt.close()


def fig6_cpu_cost_vs_standard_detectors(df_metrics, output_dir):
    """Figure 6: CPU cost comparison against adaptive baseline models."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    
    apps = ['Event-Driven Ensemble', 'Continuously Retrained Ensemble', 'Adaptive Extra Trees', 'Adaptive Random Forest', 'Adaptive Gradient Boosting']
    labels = ['Event-Driven\nEnsemble', 'Continuous\nEnsemble', 'Adaptive\nExtra Trees', 'Adaptive\nRandom Forest', 'Adaptive\nGradient Boosting']
    colors = [COLOR_EVENT, COLOR_CONT, COLOR_ET, COLOR_RF, COLOR_GB]
    
    means, stds = [], []
    for app in apps:
        sub = df_metrics[df_metrics['approach'] == app]
        seed_s = sub.groupby('seed')['cpu_time'].sum()
        means.append(float(seed_s.mean()))
        stds.append(float(seed_s.std()))
        
    bars = ax.bar(range(len(apps)), means, yerr=stds, color=colors, width=0.52, capsize=4,
                  alpha=0.88, edgecolor='black', linewidth=0.7)
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 1.5, f"{val:.1f}s",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax.set_title('Figure 6: Cumulative CPU Time Comparison Across Approaches (ToN_IoT)', pad=14)
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.set_xticks(range(len(apps)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.22)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_cost_vs_standard_detectors.png'), dpi=300)
    plt.close()


def fig7_detector_quality(df_metrics, output_dir):
    """Figure 7: Precision, Recall, and F1 across models on ToN_IoT."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    apps = ['Frozen Model', 'Event-Driven Ensemble', 'Continuously Retrained Ensemble', 'Adaptive Extra Trees', 'Adaptive Random Forest']
    labels = ['Frozen Model', 'Event-Driven', 'Continuous', 'Adaptive ET', 'Adaptive RF']
    
    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    x = np.arange(len(apps))
    width = 0.24
    
    for i, (m, m_label, c) in enumerate(zip(metrics, metric_labels, metric_colors)):
        means, stds = [], []
        for app in apps:
            sub = df_metrics[df_metrics['approach'] == app]
            s_m = sub.groupby('seed')[m].mean()
            means.append(float(s_m.mean()))
            stds.append(float(s_m.std()))
            
        rects = ax.bar(x + (i - 1)*width, means, width, yerr=stds, label=m_label,
                       color=c, capsize=3.5, alpha=0.88, edgecolor='black', linewidth=0.7)
        for rect, val in zip(rects, means):
            ax.text(rect.get_x() + rect.get_width()/2.0, val + 0.02, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    ax.set_title('Figure 7: Classification Quality Metrics Across Deployment Paradigms (ToN_IoT)', pad=14)
    ax.set_ylabel('Metric Score [0.0 - 1.0]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detector_quality_precision_recall.png'), dpi=300)
    plt.close()


def generate_all_ton_iot_paper_figures():
    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'metrics.csv'))
    
    for d in [PAPER_DIR, ALL_PAPER_DIR, BRAIN_PAPER_DIR]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Generating ToN_IoT paper figures in {PAPER_DIR}...")
    fig1_f1_comparison(df_metrics, PAPER_DIR)
    fig2_cpu_cost_comparison(df_metrics, PAPER_DIR)
    fig3_accuracy_vs_cost_tradeoff(df_metrics, PAPER_DIR)
    fig4_f1_by_drift_type(df_metrics, PAPER_DIR)
    fig5_f1_vs_standard_detectors(df_metrics, PAPER_DIR)
    fig6_cpu_cost_vs_standard_detectors(df_metrics, PAPER_DIR)
    fig7_detector_quality(df_metrics, PAPER_DIR)
    
    print(f"Syncing to {ALL_PAPER_DIR} and {BRAIN_PAPER_DIR}...")
    for f in os.listdir(PAPER_DIR):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(PAPER_DIR, f), os.path.join(ALL_PAPER_DIR, f))
            shutil.copy2(os.path.join(PAPER_DIR, f), os.path.join(BRAIN_PAPER_DIR, f))
            
    print("All ToN_IoT paper figures generated and synced successfully!")


if __name__ == '__main__':
    generate_all_ton_iot_paper_figures()
