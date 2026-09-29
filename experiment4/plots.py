"""
================================================================================
EXPERIMENT 4 v2 — PUBLICATION-QUALITY DIAGNOSTIC VISUALIZATIONS (14 FIGURES)
================================================================================
Generates all 14 publication-grade figures for Experiment 4 v2:
  1. F1 vs. stream position (all 7 approaches across 60 windows)
  2. F1 vs. drift intensity (none, mild, moderate, severe)
  3. F1 by drift type (Stationary, Covariate, Concept, Mixed)
  4. CPU usage vs. stream position (instantaneous per-window)
  5. Memory usage (Incremental RSS Delta MB) vs. stream position
  6. Cumulative CPU time trajectories
  7. Retraining events over time (Continuous vs. 5 Event-Driven detectors)
  8. Performance vs. computational cost (2D Pareto trade-off)
  9. Drift event timeline overlay (Drift -> Detector Triggers -> Recovery)
  10. Detector Precision / Recall / F1 bar chart (Custom vs. Standard)
  11. Detection Latency box plot across all drift episodes and seeds
  12. Detector performance split by drift type
  13. Model footprint bar chart (model_footprint_bytes: Frozen vs. Ensembles)
  14. CPU vs. F1 Pareto frontier across all seven approaches
================================================================================
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIR

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.35,
})

COLORS = {
    'Frozen Model': '#7f7f7f',
    'Continuously Retrained Ensemble': '#d62728',
    'Event-Driven Ensemble (Custom Dual-Trigger)': '#2ca02c',
    'Event-Driven Ensemble': '#2ca02c',
    'Event-Driven Ensemble (ADWIN)': '#1f77b4',
    'Event-Driven Ensemble (DDM)': '#ff7f0e',
    'Event-Driven Ensemble (EDDM)': '#9467bd',
    'Event-Driven Ensemble (Page-Hinkley)': '#8c564b',
    'Warm-Start Ensemble': '#17becf',
    'Component-Selective Ensemble': '#bcbd22',
    'Adaptive Random Forest (river)': '#e377c2',
    'Streaming Random Patches (river)': '#17becf',
    'Two-Tier Hybrid Ensemble': '#ff9896',
    # Short names / detectors
    'Custom Dual-Trigger': '#2ca02c',
    'ADWIN': '#1f77b4',
    'DDM': '#ff7f0e',
    'EDDM': '#9467bd',
    'Page-Hinkley': '#8c564b',
    # Drift types
    'none': '#7f7f7f',
    'covariate': '#1f77b4',
    'concept': '#ff7f0e',
    'mixed': '#d62728',
}

LINESTYLES = {
    'Frozen Model': '--',
    'Continuously Retrained Ensemble': '-',
    'Event-Driven Ensemble (Custom Dual-Trigger)': '-',
    'Event-Driven Ensemble': '-',
    'Event-Driven Ensemble (ADWIN)': '-.',
    'Event-Driven Ensemble (DDM)': ':',
    'Event-Driven Ensemble (EDDM)': '--',
    'Event-Driven Ensemble (Page-Hinkley)': '-',
    'Warm-Start Ensemble': '-.',
    'Component-Selective Ensemble': ':',
    'Adaptive Random Forest (river)': '--',
    'Streaming Random Patches (river)': '-.',
    'Two-Tier Hybrid Ensemble': '-',
}


def generate_all_plots(df_window, df_retrain, df_drift, df_detector_quality=None, df_episodes=None, output_dir=PLOTS_DIR):
    """
    Generate all 14 diagnostic figures for Experiment 4 v2.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. F1 vs Stream Position
    _plot_01_f1_vs_stream(df_window, output_dir)

    # 2. F1 vs Drift Intensity
    _plot_02_f1_vs_severity(df_window, output_dir)

    # 3. F1 by Drift Type
    _plot_03_f1_by_drift_type(df_window, output_dir)

    # 4. CPU Usage vs Stream Position
    _plot_04_cpu_vs_stream(df_window, output_dir)

    # 5. Incremental Memory Usage vs Stream Position
    _plot_05_memory_vs_stream(df_window, output_dir)

    # 6. Cumulative CPU Time
    _plot_06_cumulative_cpu(df_window, output_dir)

    # 7. Retraining Events Over Time
    _plot_07_retraining_timeline(df_retrain, output_dir)

    # 8. Performance vs Computational Cost (Pareto)
    _plot_08_pareto_frontier(df_window, output_dir)

    # 9. Drift Event Timeline Overlay
    _plot_09_drift_event_overlay(df_window, df_retrain, df_drift, output_dir)

    # 10. Detector Precision / Recall / F1
    if df_detector_quality is not None and len(df_detector_quality) > 0:
        _plot_10_detector_precision_recall_f1(df_detector_quality, output_dir)

    # 11. Detection Latency Box Plot
    if df_episodes is not None and len(df_episodes) > 0:
        _plot_11_detection_latency_distribution(df_episodes, output_dir)
    elif df_detector_quality is not None and 'mean_latency' in df_detector_quality.columns:
        _plot_11_detection_latency_bar(df_detector_quality, output_dir)

    # 12. Detector Performance Split by Drift Type
    _plot_12_detector_performance_by_drift_type(df_window, output_dir)

    # 13. Model Footprint Bar Chart
    _plot_13_model_footprint(df_window, output_dir)

    # 14. CPU vs F1 Pareto Frontier (All Approaches)
    _plot_14_pareto_all_approaches(df_window, output_dir)

    # 15. Component-Selective Retraining Breakdown
    _plot_15_component_selective_breakdown(df_retrain, output_dir)

    # 16. Two-Tier Weight Dynamics
    _plot_16_two_tier_weight_dynamics(df_window, output_dir)

    # 17. Cost-Normalized Efficiency Comparison
    _plot_17_efficiency_frontier(df_window, output_dir)


def _plot_01_f1_vs_stream(df, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    approaches = list(df['approach'].unique())

    for app in approaches:
        sub = df[df['approach'] == app].groupby('window_id')['f1'].agg(['mean', 'std']).reset_index()
        ls = LINESTYLES.get(app, '-')
        c = COLORS.get(app, '#333')
        lw = 2.2 if ('Custom' in app or 'Continuously' in app or 'Frozen' in app) else 1.6
        ax.plot(sub['window_id'], sub['mean'], label=app, color=c, lw=lw, linestyle=ls)
        if app in ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble (Custom Dual-Trigger)', 'Event-Driven Ensemble']:
            ax.fill_between(sub['window_id'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                            color=c, alpha=0.10)

    ax.set_title('Figure 1: Prequential F1-Score Progression Across 60 Streaming Windows')
    ax.set_xlabel('Streaming Window ID (500 samples/window, 30,000 deployment samples)')
    ax.set_ylabel('Prequential F1-Score')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_f1_vs_stream_position.png'), dpi=300)
    plt.close()


def _plot_02_f1_vs_severity(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    approaches = list(df['approach'].unique())
    severities = ['none', 'mild', 'moderate', 'severe']

    n_app = len(approaches)
    bar_width = 0.8 / n_app
    x = np.arange(len(severities))

    for i, app in enumerate(approaches):
        sub = df[df['approach'] == app]
        means = []
        stds = []
        for sev in severities:
            s_data = sub[sub['severity'] == sev]['f1']
            means.append(s_data.mean() if len(s_data) > 0 else 0.0)
            stds.append(s_data.std() if len(s_data) > 1 else 0.0)

        offset = (i - (n_app - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, label=app,
               color=COLORS.get(app, '#333'), capsize=3, alpha=0.9)

    ax.set_title('Figure 2: Predictive Performance (F1) by Drift Intensity')
    ax.set_xlabel('Drift Severity')
    ax.set_ylabel('Mean F1-Score (± 1 SD)')
    ax.set_xticks(x)
    ax.set_xticklabels(['None (Stationary)', 'Mild (20%)', 'Moderate (45%)', 'Severe (75%)'])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_f1_vs_drift_intensity.png'), dpi=300)
    plt.close()


def _plot_03_f1_by_drift_type(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    approaches = list(df['approach'].unique())
    types = ['none', 'covariate', 'concept', 'mixed']

    n_app = len(approaches)
    bar_width = 0.8 / n_app
    x = np.arange(len(types))

    for i, app in enumerate(approaches):
        sub = df[df['approach'] == app]
        means = []
        stds = []
        for dt in types:
            d_data = sub[sub['drift_type'] == dt]['f1']
            means.append(d_data.mean() if len(d_data) > 0 else 0.0)
            stds.append(d_data.std() if len(d_data) > 1 else 0.0)

        offset = (i - (n_app - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, label=app,
               color=COLORS.get(app, '#333'), capsize=3, alpha=0.9)

    ax.set_title('Figure 3: Predictive Performance (F1) Across Drift Categories')
    ax.set_xlabel('Drift Type')
    ax.set_ylabel('Mean F1-Score (± 1 SD)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Stationary (None)', 'Covariate P(X)', 'Concept P(Y|X)', 'Mixed Shift'])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_f1_by_drift_type.png'), dpi=300)
    plt.close()


def _plot_04_cpu_vs_stream(df, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    approaches = list(df['approach'].unique())

    for app in approaches:
        sub = df[df['approach'] == app].groupby('window_id')['cpu_time'].mean().reset_index()
        ls = LINESTYLES.get(app, '-')
        lw = 2.2 if ('Custom' in app or 'Continuously' in app or 'Frozen' in app) else 1.5
        ax.plot(sub['window_id'], sub['cpu_time'], label=app, color=COLORS.get(app, '#333'), lw=lw, linestyle=ls)

    ax.set_title('Figure 4: Instantaneous CPU Time per Window Over Streaming Timeline')
    ax.set_xlabel('Streaming Window ID')
    ax.set_ylabel('Total Window CPU Time (Inference + Adaptation, s)')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_cpu_vs_stream_position.png'), dpi=300)
    plt.close()


def _plot_05_memory_vs_stream(df, output_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    approaches = list(df['approach'].unique())

    col = 'incremental_rss_delta_mb' if 'incremental_rss_delta_mb' in df.columns else 'peak_ram_mb'
    col_label = 'Incremental Memory Allocated per Window (tracemalloc MB)' if col == 'incremental_rss_delta_mb' else 'Process Peak RSS (MB)'

    for app in approaches:
        sub = df[df['approach'] == app].groupby('window_id')[col].mean().reset_index()
        ls = LINESTYLES.get(app, '-')
        lw = 2.0 if ('Custom' in app or 'Continuously' in app or 'Frozen' in app) else 1.5
        ax.plot(sub['window_id'], sub[col], label=app, color=COLORS.get(app, '#333'), lw=lw, linestyle=ls)

    ax.set_title(f'Figure 5: Memory Profile Across Streaming Deployment ({col_label})')
    ax.set_xlabel('Streaming Window ID')
    ax.set_ylabel(col_label)
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_memory_vs_stream_position.png'), dpi=300)
    plt.close()


def _plot_06_cumulative_cpu(df, output_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    approaches = list(df['approach'].unique())

    for app in approaches:
        sub = df[df['approach'] == app].groupby(['seed', 'window_id'])['cpu_time'].sum().unstack('seed')
        cum = sub.cumsum()
        mean_cum = cum.mean(axis=1)
        ls = LINESTYLES.get(app, '-')
        lw = 2.5 if ('Custom' in app or 'Continuously' in app or 'Frozen' in app) else 1.8
        ax.plot(mean_cum.index, mean_cum, label=app, color=COLORS.get(app, '#333'), lw=lw, linestyle=ls)

    ax.set_title('Figure 6: Cumulative CPU Time Consumption (Mean Across Seeds)')
    ax.set_xlabel('Streaming Window ID')
    ax.set_ylabel('Cumulative CPU Time (seconds)')
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_cumulative_cpu_time.png'), dpi=300)
    plt.close()


def _plot_07_retraining_timeline(df_retrain, output_dir):
    approaches = [a for a in df_retrain['approach'].unique() if a != 'Frozen Model']
    n_app = len(approaches)
    fig, axes = plt.subplots(n_app, 1, figsize=(13, 2.2 * n_app), sharex=True)
    if n_app == 1:
        axes = [axes]

    for ax, app in zip(axes, approaches):
        sub = df_retrain[df_retrain['approach'] == app].groupby('window_id')['retrained'].mean()
        ax.bar(sub.index, sub.values * 100, color=COLORS.get(app, '#333'), alpha=0.85)
        ax.set_ylabel('Retrain (%)', fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_title(f"{app} (Avg Freq: {sub.mean()*100:.1f}%)", fontsize=10, loc='left')

    axes[-1].set_xlabel('Streaming Window ID')
    plt.suptitle('Figure 7: Retraining Events Over Time Across Deployment Strategies', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_retraining_events_over_time.png'), dpi=300)
    plt.close()


def _plot_08_pareto_frontier(df, output_dir):
    _plot_14_pareto_all_approaches(df, output_dir, filename='08_performance_vs_computational_cost.png', title='Figure 8: Performance-Cost Pareto Frontier (F1 vs. Cumulative CPU Time)')


def _plot_09_drift_event_overlay(df_window, df_retrain, df_drift, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1.2, 1.8]})

    seed_focus = 42
    drift_seed = df_drift[df_drift['seed'] == seed_focus].sort_values('window')
    win_seed = df_window[(df_window['seed'] == seed_focus)].sort_values('window_id')
    retrain_seed = df_retrain[(df_retrain['seed'] == seed_focus)].sort_values('window_id')

    windows = np.arange(len(drift_seed))

    # Top Panel: Drift Regimes
    sev_map = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    sev_vals = [sev_map.get(s, 0) for s in drift_seed['severity']]
    type_colors = [COLORS.get(t.lower(), '#555') for t in drift_seed['drift_type']]

    ax1.bar(windows, sev_vals, color=type_colors, alpha=0.7, width=0.8)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['None', 'Mild', 'Mod', 'Sev'])
    ax1.set_ylabel('Drift Severity')
    ax1.set_title('Figure 9: Detailed Streaming Timeline: Drift Schedule -> Detector Triggers -> Recovery (Seed 42)')

    # Middle Panel: Detector Activations
    ed_approaches = [a for a in retrain_seed['approach'].unique() if 'Event-Driven' in a]
    y_pos = 1
    for ed_app in ed_approaches:
        short_name = ed_app.replace('Event-Driven Ensemble (', '').replace(')', '').replace('Event-Driven Ensemble', 'Event-Driven Ensemble')
        sub_r = retrain_seed[(retrain_seed['approach'] == ed_app) & (retrain_seed['retrained'] == True)]
        if len(sub_r) > 0:
            ax2.scatter(sub_r['window_id'], [y_pos] * len(sub_r), marker='|', s=120, lw=2.5,
                        color=COLORS.get(ed_app, COLORS.get(short_name, '#333')), label=short_name)
        y_pos += 1

    ax2.set_yticks(range(1, len(ed_approaches) + 1))
    ax2.set_yticklabels([a.replace('Event-Driven Ensemble (', '').replace(')', '').replace('Event-Driven Ensemble', 'Event-Driven Ensemble') for a in ed_approaches], fontsize=9)
    ax2.set_ylabel('Detector Retrain')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

    # Bottom Panel: F1 Trajectories (Key Approaches)
    key_approaches = ['Frozen Model', 'Continuously Retrained Ensemble', 'Event-Driven Ensemble', 'Event-Driven Ensemble (ADWIN)']
    available_keys = [k for k in key_approaches if k in win_seed['approach'].unique()]
    if not available_keys:
        available_keys = list(win_seed['approach'].unique())[:4]

    for app in available_keys:
        app_data = win_seed[win_seed['approach'] == app]
        ax3.plot(app_data['window_id'], app_data['f1'], label=app, color=COLORS.get(app, '#333'), lw=2, linestyle=LINESTYLES.get(app, '-'))

    ax3.set_xlabel('Streaming Window ID (500 samples per window)')
    ax3.set_ylabel('Prequential F1-Score')
    ax3.set_ylim(0.4, 1.0)
    ax3.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_drift_event_timeline_overlay.png'), dpi=300)
    plt.close()


def _plot_10_detector_precision_recall_f1(df_quality, output_dir):
    """
    Figure 10: Grouped bar chart comparing Precision, Recall, and F1 across detectors.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    detectors = list(df_quality['detector'].unique())

    x = np.arange(len(detectors))
    bar_width = 0.25

    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (m, label, c) in enumerate(zip(metrics, metric_labels, metric_colors)):
        means = []
        stds = []
        for det in detectors:
            sub = df_quality[df_quality['detector'] == det][m]
            means.append(sub.mean() if len(sub) > 0 else 0.0)
            stds.append(sub.std() if len(sub) > 1 else 0.0)

        offset = (i - 1) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, label=label, color=c, capsize=4, alpha=0.85)

    ax.set_title('Figure 10: Drift Detector Quality Comparison (Precision, Recall, F1 Relative to Injected Drift)')
    ax.set_xlabel('Drift Detector')
    ax.set_ylabel('Metric Score (Mean ± 1 SD Across Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_detector_precision_recall_f1.png'), dpi=300)
    plt.close()


def _plot_11_detection_latency_distribution(df_episodes, output_dir):
    """
    Figure 11: Box plot of windows to detect across all drift episodes and seeds.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    detectors = list(df_episodes['detector'].unique())

    data_to_plot = []
    clean_detectors = []
    for det in detectors:
        sub = df_episodes[(df_episodes['detector'] == det) & (df_episodes['detected'] == True)]['latency']
        if len(sub) > 0:
            data_to_plot.append(sub.values)
            clean_detectors.append(det)

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=clean_detectors, patch_artist=True)
        for patch, det in zip(bp['boxes'], clean_detectors):
            patch.set_facecolor(COLORS.get(det, '#1f77b4'))
            patch.set_alpha(0.7)

    ax.set_title('Figure 11: Detection Latency Distribution (Windows Elapsed Until First Trigger)')
    ax.set_xlabel('Drift Detector')
    ax.set_ylabel('Detection Latency (Windows, 0 = Immediate Onset Detection)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_detection_latency_distribution.png'), dpi=300)
    plt.close()


def _plot_11_detection_latency_bar(df_quality, output_dir):
    """Fallback bar chart for detection latency if episode records are aggregate only."""
    fig, ax = plt.subplots(figsize=(9, 5))
    detectors = list(df_quality['detector'].unique())

    means = [df_quality[df_quality['detector'] == d]['mean_latency'].mean() for d in detectors]
    stds = [df_quality[df_quality['detector'] == d]['mean_latency'].std() for d in detectors]

    colors = [COLORS.get(d, '#1f77b4') for d in detectors]
    ax.bar(detectors, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_title('Figure 11: Mean Detection Latency Across Drift Episodes (Windows Elapsed)')
    ax.set_xlabel('Drift Detector')
    ax.set_ylabel('Mean Windows to Detect (0 = Immediate Onset)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_detection_latency_distribution.png'), dpi=300)
    plt.close()


def _plot_12_detector_performance_by_drift_type(df_window, output_dir):
    """
    Figure 12: Grouped bar chart of F1 score broken out by drift_type across detectors.
    Tests whether ADWIN/DDM/EDDM underperform on concept drift relative to Custom Dual-Trigger.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ed_approaches = [a for a in df_window['approach'].unique() if 'Event-Driven' in a]
    types = ['none', 'covariate', 'concept', 'mixed']

    n_app = len(ed_approaches)
    bar_width = 0.8 / n_app
    x = np.arange(len(types))

    for i, app in enumerate(ed_approaches):
        short_name = app.replace('Event-Driven Ensemble (', '').replace(')', '').replace('Event-Driven Ensemble', 'Event-Driven Ensemble')
        sub = df_window[df_window['approach'] == app]
        means = []
        stds = []
        for dt in types:
            d_data = sub[sub['drift_type'] == dt]['f1']
            means.append(d_data.mean() if len(d_data) > 0 else 0.0)
            stds.append(d_data.std() if len(d_data) > 1 else 0.0)

        offset = (i - (n_app - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, label=short_name,
               color=COLORS.get(app, COLORS.get(short_name, '#333')), capsize=3, alpha=0.85)

    ax.set_title('Figure 12: Event-Driven Detector Performance (F1) Split by Drift Type')
    ax.set_xlabel('Drift Type')
    ax.set_ylabel('Mean F1-Score (± 1 SD)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Stationary (None)', 'Covariate P(X)', 'Concept P(Y|X)', 'Mixed Shift'])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_detector_performance_by_drift_type.png'), dpi=300)
    plt.close()


def _plot_13_model_footprint(df_window, output_dir):
    """
    Figure 13: Bar chart of model_footprint_bytes (KB) showing honest difference between
    the single Frozen RF and 3-model Ensembles.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    approaches = list(df_window['approach'].unique())

    col = 'model_footprint_bytes' if 'model_footprint_bytes' in df_window.columns else None
    if col is None:
        return

    means_kb = []
    stds_kb = []
    short_names = []

    for app in approaches:
        sub = df_window[df_window['approach'] == app][col] / 1024.0
        means_kb.append(sub.mean() if len(sub) > 0 else 0.0)
        stds_kb.append(sub.std() if len(sub) > 1 else 0.0)
        sname = app.replace('Continuously Retrained Ensemble', 'Continuous Ens').replace('Event-Driven Ensemble (', 'ED (').replace(')', '')
        short_names.append(sname)

    colors = [COLORS.get(app, '#333') for app in approaches]
    bars = ax.bar(range(len(approaches)), means_kb, yerr=stds_kb, color=colors, capsize=4, alpha=0.85)

    for bar, val in zip(bars, means_kb):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 15, f"{val:.0f} KB", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Figure 13: Model Footprint Comparison (Serialized In-Memory Byte Size)')
    ax.set_xlabel('Deployment Approach')
    ax.set_ylabel('Model Footprint (KB)')
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(short_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, max(means_kb) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_model_footprint_comparison.png'), dpi=300)
    plt.close()


def _plot_14_pareto_all_approaches(df, output_dir, filename='14_pareto_frontier_all_approaches.png', title='Figure 14: Accuracy-Cost Pareto Frontier Across All Seven Deployment Approaches'):
    """
    Figure 14: Scatter plot of Cumulative CPU Time vs. Mean F1 across all seven approaches.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    approaches = list(df['approach'].unique())

    markers = ['o', 's', '^', 'v', 'D', 'P', 'X']

    for i, app in enumerate(approaches):
        sub = df[df['approach'] == app]
        seed_cpu = sub.groupby('seed')['cpu_time'].sum()
        seed_f1 = sub.groupby('seed')['f1'].mean()

        mean_cpu = seed_cpu.mean()
        std_cpu = seed_cpu.std() if len(seed_cpu) > 1 else 0.0
        mean_f1 = seed_f1.mean()
        std_f1 = seed_f1.std() if len(seed_f1) > 1 else 0.0

        marker = markers[i % len(markers)]
        c = COLORS.get(app, '#333')

        short_label = app.replace('Continuously Retrained Ensemble', 'Continuous').replace('Event-Driven Ensemble (', 'ED (').replace(')', '')

        ax.errorbar(mean_cpu, mean_f1, xerr=std_cpu, yerr=std_f1,
                    fmt=marker, color=c, ecolor=c, elinewidth=1.5, capsize=4,
                    markersize=10, label=f"{short_label} ({mean_f1:.4f}, {mean_cpu:.1f}s)")

    ax.set_title(title)
    ax.set_xlabel('Cumulative CPU Time (seconds) [Lower is Better]')
    ax.set_ylabel('Mean F1-Score [Higher is Better]')
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def _plot_15_component_selective_breakdown(df_retrain, output_dir):
    """
    Figure 15: Component retraining frequency breakdown for Component-Selective Ensemble.
    """
    sub = df_retrain[df_retrain['approach'] == 'Component-Selective Ensemble']
    if len(sub) == 0 or 'retrained_components' not in sub.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    comp_counts = {'RF': 0, 'ET': 0, 'GB': 0}
    for comp_list_str in sub['retrained_components']:
        if isinstance(comp_list_str, str) and comp_list_str:
            for c in comp_list_str.split(','):
                c = c.strip()
                if c in comp_counts:
                    comp_counts[c] += 1

    components = list(comp_counts.keys())
    counts = [comp_counts[c] for c in components]
    bars = ax.bar(components, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.5, alpha=0.85)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')

    ax.set_title('Figure 15: Component-Selective Retraining Events Breakdown')
    ax.set_xlabel('Base Model Component')
    ax.set_ylabel('Total Retraining Events Across Seeds')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '15_component_selective_breakdown.png'), dpi=300)
    plt.close()


def _plot_16_two_tier_weight_dynamics(df_window, output_dir):
    """
    Figure 16: Two-Tier Hybrid Ensemble dynamic online weight trajectory over streaming windows.
    """
    sub = df_window[df_window['approach'] == 'Two-Tier Hybrid Ensemble']
    if len(sub) == 0 or 'online_weight' not in sub.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 4.5))
    seed_0 = sub['seed'].iloc[0]
    sub_seed = sub[sub['seed'] == seed_0]

    ax.plot(sub_seed['window_id'], sub_seed['online_weight'], color='#ff7f0e', lw=2.2, label='Online Learner (HAT) Weight')
    ax.axhline(0.10, color='gray', linestyle='--', alpha=0.7, label='Baseline Weight (0.10)')
    ax.axhline(0.40, color='red', linestyle=':', alpha=0.7, label='Post-Drift Boost Peak (0.40)')

    ax.set_title(f'Figure 16: Dynamic Online Learner Weight Trajectory (Seed {seed_0})')
    ax.set_xlabel('Streaming Window ID')
    ax.set_ylabel('Online Learner Soft-Vote Weight')
    ax.set_ylim(0.0, 0.50)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '16_two_tier_weight_dynamics.png'), dpi=300)
    plt.close()


def _plot_17_efficiency_frontier(df_window, output_dir):
    """
    Figure 17: Performance-to-Cost Efficiency (F1 / CPU-second) Bar Chart across all approaches.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    approaches = list(df_window['approach'].unique())

    effs = []
    short_names = []
    colors = []

    for app in approaches:
        sub = df_window[df_window['approach'] == app]
        mean_f1 = sub.groupby('seed')['f1'].mean().mean()
        mean_cpu = sub.groupby('seed')['cpu_time'].sum().mean()
        eff = mean_f1 / mean_cpu if mean_cpu > 0 else 0.0
        effs.append(eff)
        colors.append(COLORS.get(app, '#333'))

        short_label = app.replace('Continuously Retrained Ensemble', 'Continuous') \
                         .replace('Event-Driven Ensemble (', 'ED (') \
                         .replace('Adaptive Random Forest (river)', 'ARF (river)') \
                         .replace('Streaming Random Patches (river)', 'SRP (river)') \
                         .replace(')', '')
        short_names.append(short_label)

    bars = ax.bar(range(len(approaches)), effs, color=colors, alpha=0.85, width=0.6)

    for bar, val in zip(bars, effs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Figure 17: Performance-to-Cost Efficiency Comparison (F1 / Cumulative CPU-Second)')
    ax.set_xlabel('Deployment Strategy')
    ax.set_ylabel('Efficiency Ratio (F1 / CPU-second) [Higher is Better]')
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '17_performance_cost_efficiency.png'), dpi=300)
    plt.close()

