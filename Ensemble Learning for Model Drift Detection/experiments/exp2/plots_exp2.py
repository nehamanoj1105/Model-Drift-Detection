"""
Visualization Generator for Experiment 2 — Probabilistic Regime Transfer
Generates 13 publication-quality figures plus the primary diagnostic overlay plot
(Similarity vs Delta F1 with outcome markers and probability decision boundary).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Styling setup
plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'

def generate_all_plots():
    results_dir = os.path.join("experiments", "exp2", "results")
    plots_dir = os.path.join("experiments", "exp2", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_file = os.path.join(results_dir, "summary.csv")
    pairs_file = os.path.join(results_dir, "transfer_pairs.csv")
    windows_file = os.path.join(results_dir, "per_window_results.csv")
    seeds_file = os.path.join(results_dir, "per_seed_results.csv")

    if not os.path.exists(summary_file):
        print("Error: summary.csv not found in results directory.")
        return

    df_summary = pd.read_csv(summary_file)
    df_pairs = pd.read_csv(pairs_file) if os.path.exists(pairs_file) else pd.DataFrame()
    df_windows = pd.read_csv(windows_file) if os.path.exists(windows_file) else pd.DataFrame()
    df_seeds = pd.read_csv(seeds_file) if os.path.exists(seeds_file) else pd.DataFrame()

    # ---------------------------------------------------------
    # PRIMARY DIAGNOSTIC OVERLAY PLOT (Req 31)
    # x = Similarity, y = Delta F1, Color = Transfer Label
    # ---------------------------------------------------------
    if not df_pairs.empty:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        ax.grid(True)
        
        # Mapping labels
        label_map = {1: 'Positive Transfer (> +0.005)', 0: 'Neutral Transfer', -1: 'Negative Transfer (< -0.005)'}
        color_map = {1: '#2ca02c', 0: '#7f7f7f', -1: '#d62728'}
        marker_map = {1: '^', 0: 'o', -1: 'v'}

        for lbl in [1, 0, -1]:
            sub = df_pairs[df_pairs['transfer_label'] == lbl]
            if not sub.empty:
                ax.scatter(sub['combined_similarity'], sub['delta_F1'],
                           c=color_map[lbl], label=label_map[lbl],
                           marker=marker_map[lbl], alpha=0.75, s=60, edgecolors='k', linewidth=0.5)

        # Decision threshold line at delta_F1 = 0
        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.2, label='Zero Gain Line')
        ax.axhline(0.005, color='#2ca02c', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.axhline(-0.005, color='#d62728', linestyle=':', linewidth=1.0, alpha=0.7)
        
        ax.set_xlabel('Combined Regime Similarity S(Rs, Rt)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Transfer F1 Gain ΔF1', fontsize=12, fontweight='bold')
        ax.set_title('Primary Diagnostic: Similarity vs Actual Transfer Outcome', fontsize=14, fontweight='bold', pad=12)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "primary_overlay_similarity_vs_delta_f1.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 1: Regime Representation Visualization (PCA)
    # ---------------------------------------------------------
    if not df_pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        feat_cols = [c for c in df_pairs.columns if 'distance' in c or 'diff' in c]
        if len(feat_cols) >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_pairs[feat_cols].fillna(0.0))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df_pairs['combined_similarity'], cmap='viridis', s=50, alpha=0.8)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Combined Similarity', fontsize=11)
            ax.set_xlabel('PCA Component 1', fontsize=12)
            ax.set_ylabel('PCA Component 2', fontsize=12)
            ax.set_title('Figure 1: Regime Representation Space (PCA)', fontsize=13, fontweight='bold')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fig1_regime_representation_pca.png"))
            plt.close()

    # ---------------------------------------------------------
    # FIGURE 2: Similarity vs Actual Transfer Gain
    # ---------------------------------------------------------
    if not df_pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.scatter(df_pairs['combined_similarity'], df_pairs['delta_F1'], alpha=0.6, color='#1f77b4', edgecolors='k', linewidth=0.3)
        # Linear regression trend line
        if len(df_pairs) > 1:
            z = np.polyfit(df_pairs['combined_similarity'], df_pairs['delta_F1'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_pairs['combined_similarity'].min(), df_pairs['combined_similarity'].max(), 100)
            ax.plot(x_line, p(x_line), color='#d62728', linewidth=2, label='Linear Trend')
        ax.set_xlabel('Regime Similarity S(Rs, Rt)', fontsize=12)
        ax.set_ylabel('Transfer F1 Gain ΔF1', fontsize=12)
        ax.set_title('Figure 2: Similarity vs Actual Transfer Gain', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig2_similarity_vs_transfer_gain.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 3 & 4: Reliability Diagram / Calibration Curve
    # ---------------------------------------------------------
    if not df_pairs.empty and 'predicted_probability' in df_pairs.columns:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        probs = df_pairs['predicted_probability'].values
        labels = (df_pairs['transfer_label'] > 0).astype(int).values

        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = []

        for i in range(10):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(labels[mask]))
            else:
                bin_accs.append(np.nan)

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(bin_centers, bin_accs, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Probability Estimator')
        ax.set_xlabel('Predicted Transfer Probability P(positive)', fontsize=12)
        ax.set_ylabel('Observed Positive Transfer Fraction', fontsize=12)
        ax.set_title('Figure 3 & 4: Probability Calibration Curve (Reliability Diagram)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig3_fig4_calibration_reliability_diagram.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 5: Negative Transfer Rate by Method
    # ---------------------------------------------------------
    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        df_sub = df_summary[df_summary['Total Transfers'] > 0]
        if not df_sub.empty:
            methods = df_sub['Method'].unique()
            streams = df_sub['Stream'].unique()
            x = np.arange(len(methods))
            width = 0.8 / max(1, len(streams))
            
            for i, st in enumerate(streams):
                st_data = df_sub[df_sub['Stream'] == st]
                val_dict = dict(zip(st_data['Method'], st_data['Negative Transfer Rate']))
                vals = [val_dict.get(m, 0.0) for m in methods]
                ax.bar(x + i * width, vals, width, label=st)

            ax.set_xticks(x + width * (len(streams) - 1) / 2)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            ax.set_ylabel('Negative Transfer Rate (NTR)', fontsize=12)
            ax.set_title('Figure 5: Negative Transfer Rate by Method', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fig5_negative_transfer_rate_by_method.png"))
            plt.close()

    # ---------------------------------------------------------
    # FIGURE 6: Transfer vs Abstention Decisions
    # ---------------------------------------------------------
    if not df_seeds.empty and 'total_transfers' in df_seeds.columns:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        df_prob = df_seeds[df_seeds['method'].str.contains('Probability')]
        if not df_prob.empty:
            methods = df_prob['method'].unique()
            vals = [df_prob[df_prob['method'] == m]['total_transfers'].values for m in methods]
            ax.boxplot(vals, tick_labels=methods)
            ax.set_ylabel('Number of Approved Transfers', fontsize=12)
            ax.set_title('Figure 6: Transfer vs Abstention Decisions under tau=0.60', fontsize=13, fontweight='bold')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fig6_transfer_vs_abstention.png"))
            plt.close()

    # ---------------------------------------------------------
    # FIGURE 7: Macro F1 by Method (Bar Chart)
    # ---------------------------------------------------------
    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        methods = df_summary['Method'].unique()
        streams = df_summary['Stream'].unique()
        x = np.arange(len(methods))
        width = 0.8 / max(1, len(streams))

        for i, st in enumerate(streams):
            st_data = df_summary[df_summary['Stream'] == st]
            val_dict = dict(zip(st_data['Method'], st_data['Macro F1 Mean']))
            vals = [val_dict.get(m, 0.0) for m in methods]
            ax.bar(x + i * width, vals, width, label=st)

        ax.set_xticks(x + width * (len(streams) - 1) / 2)
        ax.set_xticklabels(methods, rotation=35, ha='right')
        ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Figure 7: Macro F1 Performance Across Benchmark Methods', fontsize=14, fontweight='bold')
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig7_macro_f1_by_method.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 8: Adaptation CPU by Method
    # ---------------------------------------------------------
    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        methods = df_summary['Method'].unique()
        streams = df_summary['Stream'].unique()
        x = np.arange(len(methods))
        width = 0.8 / max(1, len(streams))

        for i, st in enumerate(streams):
            st_data = df_summary[df_summary['Stream'] == st]
            val_dict = dict(zip(st_data['Method'], st_data['Adapt CPU Mean']))
            vals = [val_dict.get(m, 0.0) for m in methods]
            ax.bar(x + i * width, vals, width, label=st)

        ax.set_xticks(x + width * (len(streams) - 1) / 2)
        ax.set_xticklabels(methods, rotation=35, ha='right')
        ax.set_ylabel('Adaptation CPU (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Figure 8: Total Adaptation CPU Cost by Method', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig8_adaptation_cpu_by_method.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 9: Cumulative CPU Over Time
    # ---------------------------------------------------------
    if not df_windows.empty:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        top_methods = ['Frozen', 'Event-Driven', 'RAPT-E', 'Similarity-Only', 'Probability-Guided Top-1']
        df_top = df_windows[df_windows['method'].isin(top_methods)].copy()
        
        for m in top_methods:
            sub = df_top[df_top['method'] == m]
            if not sub.empty:
                avg_sub = sub.groupby('window_id')['cpu_cost'].mean().reset_index()
                avg_sub['cum_cpu'] = avg_sub['cpu_cost'].cumsum()
                ax.plot(avg_sub['window_id'], avg_sub['cum_cpu'], label=m, linewidth=2)

        ax.set_xlabel('Streaming Window ID', fontsize=12)
        ax.set_ylabel('Cumulative Adaptation CPU (s)', fontsize=12)
        ax.set_title('Figure 9: Cumulative CPU Cost Across Streaming Timeline', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig9_cumulative_cpu.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 10: F1 Around Recurrent Transitions
    # ---------------------------------------------------------
    if not df_windows.empty:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        target_m = ['Event-Driven', 'RAPT-E', 'Probability-Guided Top-1']
        df_trans = df_windows[df_windows['method'].isin(target_m)]
        
        if not df_trans.empty:
            stable_f1 = [df_trans[(df_trans['method'] == m) & (df_trans['is_transition'] == False)]['f1_score'].mean() for m in target_m]
            trans_f1 = [df_trans[(df_trans['method'] == m) & (df_trans['is_transition'] == True)]['f1_score'].mean() for m in target_m]
            
            x = np.arange(len(target_m))
            width = 0.35
            ax.bar(x - width/2, stable_f1, width, label='Stable Window')
            ax.bar(x + width/2, trans_f1, width, label='Regime Transition Window')
            
            ax.set_xticks(x)
            ax.set_xticklabels(target_m)
            ax.set_ylabel('Macro F1 Score', fontsize=12)
            ax.set_title('Figure 10: Performance Stability Around Recurrent Transitions', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fig10_f1_around_transitions.png"))
            plt.close()

    # ---------------------------------------------------------
    # FIGURE 11: Oracle Headroom Analysis
    # ---------------------------------------------------------
    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        df_head = df_summary[df_summary['Method'].isin(['Similarity-Only', 'RAPT-E', 'Probability-Guided Top-1', 'Oracle Transfer'])]
        if not df_head.empty:
            methods = df_head['Method'].unique()
            streams = df_head['Stream'].unique()
            x = np.arange(len(methods))
            width = 0.8 / max(1, len(streams))

            for i, st in enumerate(streams):
                st_data = df_head[df_head['Stream'] == st]
                val_dict = dict(zip(st_data['Method'], st_data['Macro F1 Mean']))
                vals = [val_dict.get(m, 0.0) for m in methods]
                ax.bar(x + i * width, vals, width, label=st)

            ax.set_xticks(x + width * (len(streams) - 1) / 2)
            ax.set_xticklabels(methods, rotation=20, ha='right')
            ax.set_ylabel('Macro F1 Score', fontsize=12)
            ax.set_title('Figure 11: Oracle Headroom Analysis (Gap to Upper Bound)', fontsize=13, fontweight='bold')
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fig11_oracle_headroom.png"))
            plt.close()

    # ---------------------------------------------------------
    # FIGURE 12: Performance-CPU Pareto Plot
    # ---------------------------------------------------------
    if not df_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        for stream_name, group in df_summary.groupby('Stream'):
            ax.scatter(group['Adapt CPU Mean'], group['Macro F1 Mean'], s=100, label=stream_name, alpha=0.8)
            for _, row in group.iterrows():
                ax.annotate(row['Method'], (row['Adapt CPU Mean'], row['Macro F1 Mean']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Adaptation CPU Cost (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Figure 12: Performance–CPU Pareto Tradeoff', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig12_pareto_f1_vs_cpu.png"))
        plt.close()

    # ---------------------------------------------------------
    # FIGURE 13: Probability Distribution for Transfer Outcomes
    # ---------------------------------------------------------
    if not df_pairs.empty and 'predicted_probability' in df_pairs.columns:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        for lbl, name, col in [(1, 'Positive Transfer', '#2ca02c'), (0, 'Neutral Transfer', '#7f7f7f'), (-1, 'Negative Transfer', '#d62728')]:
            sub = df_pairs[df_pairs['transfer_label'] == lbl]
            if not sub.empty:
                ax.hist(sub['predicted_probability'], bins=10, alpha=0.5, label=name, color=col, density=True)
        
        ax.set_xlabel('Predicted Probability P(positive transfer)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Figure 13: Predicted Probability Distributions for Transfer Outcomes', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fig13_probability_distribution_outcomes.png"))
        plt.close()

    print(f"All 13 publication figures + primary overlay plot saved in {plots_dir}")

if __name__ == "__main__":
    generate_all_plots()
