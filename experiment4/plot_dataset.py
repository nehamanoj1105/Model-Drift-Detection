"""
EXPERIMENT 4 — TELEMETRY DATASET & DRIFT TAXONOMY VISUALIZATION
Generates a publication-grade multi-panel figure illustrating:
  (a) Telemetry feature streams (Speed, Distance, Delay, Throughput) across 50k samples with drift regime shading
  (b) Window-level QoS violation base rate dynamics (showing stationary troughs 10-28% vs drift surges 60-80%)
  (c) Feature distribution shifts under Covariate Drift (P(X) shift)
  (d) Concept Drift mechanics: Ground-truth relationship shifts (P(y|X) shift)
"""
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import sys
BASE_DIR = r"C:\Users\emhaenn\Downloads\Model-Drift-Detection\experiment4"
sys.path.insert(0, BASE_DIR)
from data_generation import generate_experiment_dataset
from config import COVARIATE_DRIFT_FULL, COVARIATE_MAGNITUDES, CONCEPT_DRIFT_SHIFTS

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

# Colors for drift regimes
COLOR_MAP = {
    'none': '#4a7bb0',       # Blue (Stationary)
    'covariate': '#f28e2b',  # Orange (Covariate Shift)
    'concept': '#b07aa1',    # Purple (Concept Drift)
    'mixed': '#e15759',      # Coral/Red (Mixed Drift)
}

def generate_dataset_figure(seed=42):
    df, schedule = generate_experiment_dataset(seed)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.9, 0.9], hspace=0.32, wspace=0.22)
    
    # -------------------------------------------------------------------------
    # Panel 1: Telemetry Feature Streams over Time (Full Width)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    
    # Downsample for crisp plotting of 50k samples
    step = 10
    t_down = df['timestamp'].iloc[::step]
    dist_down = df['distance'].iloc[::step]
    delay_down = df['delay'].iloc[::step]
    tp_down = df['throughput'].iloc[::step]
    speed_down = df['speed'].iloc[::step]
    
    # Plot Distance and Delay on ax1
    l1, = ax1.plot(t_down, dist_down, color='#2b5c8f', lw=1.2, alpha=0.85, label='Distance (m) [Left]')
    l2, = ax1.plot(t_down, delay_down, color='#d95f02', lw=1.1, alpha=0.85, label='Delay (ms) [Left]')
    ax1.set_ylabel('Distance (m) / Delay (ms)', fontweight='bold')
    ax1.set_ylim(0, 130)
    
    # Secondary y-axis for Throughput and Speed
    ax1_twin = ax1.twinx()
    l3, = ax1_twin.plot(t_down, tp_down, color='#2ca02c', lw=1.1, alpha=0.75, linestyle='-', label='Throughput (Mbps) [Right]')
    l4, = ax1_twin.plot(t_down, speed_down, color='#7570b3', lw=1.1, alpha=0.75, linestyle='--', label='Speed (m/s) [Right]')
    ax1_twin.set_ylabel('Throughput (Mbps) / Speed (m/s)', fontweight='bold')
    ax1_twin.set_ylim(0, 160)
    ax1_twin.grid(False)
    
    # Shade Initial Training Period (0 - 20k)
    ax1.axvspan(0, 20000, color='#e0e0e0', alpha=0.35, zorder=0)
    ax1.text(10000, 120, 'Initial Training Baseline (Stationary, 20,000 samples)', 
             ha='center', va='top', fontsize=10, fontweight='bold', color='#444444',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#999999', alpha=0.85))
    
    # Vertical line separating initial training from deployment
    ax1.axvline(20000, color='#333333', linestyle='-', lw=1.8, zorder=3)
    ax1.text(20200, 10, 'Deployment Start (t = 20,000)', fontsize=8.5, fontweight='bold', rotation=90, color='#222222')
    
    # Shade deployment windows by drift type
    for w_info in schedule:
        w_id = w_info['window_id']
        start_t = 20000 + w_id * 500
        end_t = start_t + 500
        dtype = w_info['drift_type']
        if dtype != 'none':
            ax1.axvspan(start_t, end_t, color=COLOR_MAP[dtype], alpha=0.18, zorder=0)
            
    # Combine legends
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right', framealpha=0.92, ncol=4, fontsize=9)
    ax1.set_title('(a) Synthetic Telemetry Feature Streams & Physical Drift Injection (50,000 Total Samples)', pad=8)
    ax1.set_xlabel('Sample Index (Time Step t)')
    ax1.set_xlim(0, 50000)
    
    # -------------------------------------------------------------------------
    # Panel 2: Window-Level QoS Violation Rate Dynamics (Full Width)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, :])
    
    # Compute violation rate for 40 training windows (500 samples each) + 60 deployment windows
    win_indices = []
    win_rates = []
    win_types = []
    
    for i in range(100):
        start = i * 500
        end = start + 500
        w_df = df.iloc[start:end]
        rate = w_df['qos_violation'].mean()
        win_indices.append(start + 250)
        win_rates.append(rate)
        if i < 40:
            win_types.append('training')
        else:
            w_id = i - 40
            win_types.append(schedule[w_id]['drift_type'])
            
    # Plot baseline rate curve
    ax2.plot(win_indices[:40], win_rates[:40], color='#666666', lw=1.5, marker='o', markersize=4, label='Stationary Training Windows')
    ax2.plot(win_indices[39:], win_rates[39:], color='#333333', lw=1.2, linestyle=':', alpha=0.6)
    
    # Scatter points for deployment windows colored by drift type
    for dtype, label, color in [
        ('none', 'Stationary Deployment (Base Rate 10-28%)', COLOR_MAP['none']),
        ('covariate', 'Covariate Shift (Surge to 60-80%)', COLOR_MAP['covariate']),
        ('concept', 'Concept Drift (P(y|X) Shift)', COLOR_MAP['concept']),
        ('mixed', 'Mixed Drift (Simultaneous Shift)', COLOR_MAP['mixed']),
    ]:
        xs = [win_indices[i] for i in range(40, 100) if win_types[i] == dtype]
        ys = [win_rates[i] for i in range(40, 100) if win_types[i] == dtype]
        ax2.scatter(xs, ys, color=color, s=40, edgecolors='black', linewidths=0.7, label=label, zorder=5)
        
    ax2.axvspan(0, 20000, color='#e0e0e0', alpha=0.30, zorder=0)
    ax2.axvline(20000, color='#333333', linestyle='-', lw=1.8, zorder=3)
    ax2.axhline(0.26, color='#4a7bb0', linestyle='--', lw=1.2, alpha=0.7, label='Stationary Mean Violation Rate (~26%)')
    
    # Annotate the stationary troughs and drift surges
    ax2.annotate('Periodic Distance Troughs\n(Base rate drops to 10-12%)', xy=(6000, 0.12), xytext=(3000, 0.45),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color='#4a7bb0'),
                 fontsize=8.5, fontweight='bold', color='#2b5c8f',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#4a7bb0', alpha=0.9))
    
    ax2.annotate('Drift Regimes\n(Violation surges to 60-80%)', xy=(26000, 0.76), xytext=(22000, 0.90),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color='#e15759'),
                 fontsize=8.5, fontweight='bold', color='#c0392b',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#e15759', alpha=0.9))
    
    ax2.set_ylabel('QoS Violation Rate P(y=1)', fontweight='bold')
    ax2.set_xlabel('Sample Index (Time Step t)')
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlim(0, 50000)
    ax2.set_title('(b) Dynamic QoS Violation Rate across Telemetry Windows (Explaining the Base-Rate & Buffer Carry-Over Effects)', pad=8)
    ax2.legend(loc='lower left', framealpha=0.92, ncol=3, fontsize=8.5)
    
    # -------------------------------------------------------------------------
    # Panel 3: Covariate Shift Demonstration (Density P(X))
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Extract baseline distance and severe covariate distance
    train_dist = df.iloc[:20000]['distance'].values
    train_delay = df.iloc[:20000]['delay'].values
    
    # Find severe covariate windows
    cov_windows = [w_info['window_id'] for w_info in schedule if w_info['drift_type'] == 'covariate' and w_info['severity'] == 'severe']
    cov_samples = []
    for w in cov_windows:
        cov_samples.append(df.iloc[20000 + w*500 : 20000 + (w+1)*500])
    df_cov = pd.concat(cov_samples) if cov_samples else df.iloc[20000:25000]
    
    kde_train_dist = gaussian_kde(train_dist)
    kde_cov_dist = gaussian_kde(df_cov['distance'])
    x_grid = np.linspace(10, 120, 300)
    
    ax3.plot(x_grid, kde_train_dist(x_grid), color=COLOR_MAP['none'], lw=2.0, label='Stationary Baseline: P(Distance)')
    ax3.fill_between(x_grid, kde_train_dist(x_grid), color=COLOR_MAP['none'], alpha=0.25)
    
    ax3.plot(x_grid, kde_cov_dist(x_grid), color=COLOR_MAP['covariate'], lw=2.0, linestyle='--', label='Severe Covariate Shift: P*(Distance)')
    ax3.fill_between(x_grid, kde_cov_dist(x_grid), color=COLOR_MAP['covariate'], alpha=0.25)
    
    ax3.set_title('(c) Covariate Shift: Feature Distribution P(X) Shifts', pad=8)
    ax3.set_xlabel('Distance (meters)')
    ax3.set_ylabel('Probability Density')
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    # -------------------------------------------------------------------------
    # Panel 4: Concept Drift Demonstration (Shifts in P(y|X) Coefficients)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    
    kpis = ['Distance\n($w_d$)', 'Delay\n($w_{del}$)', 'Throughput\n($w_{tp}$)', 'Speed\n($w_{sp}$)', 'Speed×Delay\n($w_{i1}$)', 'Delay×TP\n($w_{i2}$)']
    base_weights = [0.7, 1.1, -0.9, 0.3, 0.6, 0.5]
    severe_shifts = [CONCEPT_DRIFT_SHIFTS['severe'][k] for k in ['dist_w', 'delay_w', 'tp_w', 'speed_w', 'i1_w', 'i2_w']]
    drifted_weights = [bw + sw for bw, sw in zip(base_weights, severe_shifts)]
    
    x = np.arange(len(kpis))
    width = 0.36
    
    r1 = ax4.bar(x - width/2, base_weights, width, label='Stationary Baseline P(y|X)', color=COLOR_MAP['none'], alpha=0.85, edgecolor='black', linewidth=0.7)
    r2 = ax4.bar(x + width/2, drifted_weights, width, label='Severe Concept Drift P*(y|X)', color=COLOR_MAP['concept'], alpha=0.85, edgecolor='black', linewidth=0.7)
    
    for rect, val in zip(r1, base_weights):
        ax4.text(rect.get_x() + rect.get_width()/2.0, val + (0.08 if val >= 0 else -0.25), f"{val:+.1f}",
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for rect, val in zip(r2, drifted_weights):
        ax4.text(rect.get_x() + rect.get_width()/2.0, val + (0.08 if val >= 0 else -0.25), f"{val:+.1f}",
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        
    ax4.set_title('(d) Concept Drift: Ground-Truth Relationship P(y|X) Shifts', pad=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(kpis, fontsize=8.5)
    ax4.set_ylabel('Logistic Model Weight Coefficient')
    ax4.axhline(0, color='black', lw=0.8, linestyle='-')
    ax4.set_ylim(-2.0, 3.2)
    ax4.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    # Save figure
    out_path = os.path.join(BASE_DIR, 'figures_final', 'dataset_overview.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    # Also save to headline folder
    out_path_a = os.path.join(BASE_DIR, 'figures_final', 'group_a_headline', 'dataset_overview.png')
    plt.savefig(out_path_a, dpi=300, bbox_inches='tight')
    
    # Save to brain artifact directory
    brain_dir = r"C:\Users\emhaenn\.gemini\antigravity\brain\e7b78523-74fa-4f1f-9dea-d867ccc55e5b\figures_final"
    os.makedirs(brain_dir, exist_ok=True)
    brain_path = os.path.join(brain_dir, 'dataset_overview.png')
    shutil.copy2(out_path, brain_path)
    
    plt.close()
    print(f"Dataset overview plot successfully generated at:\n - {out_path}\n - {brain_path}")

if __name__ == '__main__':
    generate_dataset_figure(seed=42)
