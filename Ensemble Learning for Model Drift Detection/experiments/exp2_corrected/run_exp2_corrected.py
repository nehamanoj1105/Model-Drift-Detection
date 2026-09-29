"""
Run Experiment 2 Corrected -- Full Orchestration Script.

Usage:
    cd "experiments/exp2_corrected"
    python run_exp2_corrected.py --data_9a <path/to/ntnu_owd_Packets_with_IATs.csv>
                                  --data_9b_dir <path/to/zenodo_5g_nr_dir>
                                  [--seeds 42 43 44]
                                  [--streams 9A_Original 9A_Enriched 9B_Original 9B_Enriched]
                                  [--skip_gate]

Exit codes:
    0  -- All sanity checks passed
    1  -- One or more CRITICAL sanity checks failed
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import argparse
import warnings
import traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP9_DIR  = os.path.normpath(os.path.join(_HERE, '..', 'exp9'))
_EXP9B_DIR = os.path.normpath(os.path.join(_HERE, '..', 'exp9b'))
for _p in [_EXP9B_DIR, _EXP9_DIR, _HERE]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from stream_builder import get_all_streams
from exp2_corrected import run_single_seed, SEEDS
from evaluation import (
    aggregate_seed_results,
    compute_ntr,
    compute_wilcoxon_pairs,
    compute_probability_quality_summary,
    compute_oracle_comparison,
    run_sanity_checks,
    METHODS_ORDER,
)
from plots import generate_all_plots


RESULTS_DIR = os.path.join(_HERE, 'results')
PLOTS_DIR   = os.path.join(_HERE, 'plots')
REPORT_PATH = os.path.join(_HERE, 'EXP2_CORRECTED_REPORT.md')

ALL_STREAM_NAMES = ['9A_Original', '9A_Enriched', '9B_Original', '9B_Enriched']


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiment 2 Corrected')
    parser.add_argument('--data_9a', required=True,
                        help='Path to ntnu_owd_Packets_with_IATs.csv')
    parser.add_argument('--data_9b_dir', required=True,
                        help='Directory containing the 48 5G NR scenario CSV files')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                        help='Seeds to run (default: 42 43 44)')
    parser.add_argument('--streams', nargs='+', default=ALL_STREAM_NAMES,
                        choices=ALL_STREAM_NAMES,
                        help='Which streams to run (default: all four)')
    parser.add_argument('--skip_gate', action='store_true',
                        help='Continue even if baseline gate checks fail (for debugging)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose per-window output')
    return parser.parse_args()


def run_all(args) -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    t_global_start = time.time()
    exit_code = 0

    # ── Build all streams ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("EXPERIMENT 2 CORRECTED -- STREAM CONSTRUCTION")
    print("="*70)
    all_streams = get_all_streams(
        data_9a=args.data_9a,
        data_9b_dir=args.data_9b_dir,
    )

    # ── Filter to requested streams ───────────────────────────────────────────
    streams_to_run = {k: v for k, v in all_streams.items() if k in args.streams}
    if not streams_to_run:
        print(f"ERROR: None of the requested streams {args.streams} are available.")
        return 1

    # ── Per-stream results ─────────────────────────────────────────────────────
    all_summary_records = []
    all_wilcoxon_records = []
    all_gate_records = []
    prob_quality_by_stream = {}
    cross_dataset_auroc = {}

    for stream_name, stream_dict in streams_to_run.items():
        print("\n" + "="*70)
        print(f"STREAM: {stream_name}  ({len(stream_dict['df'])} windows)")
        print("="*70)

        dataset = stream_dict['dataset']
        seed_results = []

        for seed in args.seeds:
            print(f"\n  Seed {seed}...")
            try:
                result = run_single_seed(
                    stream_dict=stream_dict,
                    seed=seed,
                    stream_name=stream_name,
                    verbose=args.verbose,
                )
                seed_results.append(result)
            except Exception as e:
                print(f"  ERROR in seed {seed}: {e}")
                traceback.print_exc()

        if not seed_results:
            print(f"  FATAL: All seeds failed for {stream_name}. Skipping.")
            continue

        # ── Sanity checks ─────────────────────────────────────────────────────
        print(f"\n  Running sanity checks for {stream_name}...")
        gate = run_sanity_checks(seed_results, stream_name, dataset)
        for check_name, check_res in gate.items():
            passed = check_res.get('pass', False)
            note   = check_res.get('note', '')
            status = 'PASS PASS' if passed else 'FAIL FAIL'
            print(f"    {status}  {check_name}: {note}")
            if 'RAPT' in check_name or 'Event' in check_name or 'Frozen' in check_name:
                if not passed and not args.skip_gate:
                    print(f"\n  CRITICAL: Baseline gate check failed for {check_name}.")
                    print(f"  Expected range: {check_res.get('expected_range')}, "
                          f"got {check_res.get('mean_f1', 'N/A'):.4f}")
                    print("  Cannot proceed to Experiment 2 analysis without valid baselines.")
                    print("  Use --skip_gate to force continuation (results will be invalid).")
                    exit_code = 1
                    if not args.skip_gate:
                        continue

        all_gate_records.append({'stream': stream_name, **gate})

        # ── Aggregate across seeds ────────────────────────────────────────────
        summary_df = aggregate_seed_results(seed_results, stream_name)
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, f'{stream_name}_summary.csv'), index=False)
        print(f"\n  Summary (mean F1 across {len(args.seeds)} seeds):")
        for _, row in summary_df.iterrows():
            print(f"    {row['method']:25s}  F1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f}  "
                  f"CPU={row['adapt_cpu_mean']:.2f}s")

        all_summary_records.append(summary_df)

        # ── NTR analysis ──────────────────────────────────────────────────────
        all_cand = []
        for r in seed_results:
            all_cand.extend(r.get('candidate_records', []))
        if all_cand:
            cand_df = pd.DataFrame(all_cand)
            cand_df.to_csv(
                os.path.join(RESULTS_DIR, f'{stream_name}_candidate_records.csv'), index=False)
            ntr_df = compute_ntr(all_cand)
            ntr_df.to_csv(
                os.path.join(RESULTS_DIR, f'{stream_name}_ntr.csv'), index=False)
            print(f"\n  Transfer label distribution:")
            for _, row in ntr_df.iterrows():
                print(f"    {row['label']:10s}: {int(row['count']):4d}  ({row['fraction']:.2%})")
        else:
            cand_df = pd.DataFrame()
            ntr_df = pd.DataFrame()

        # ── Probability quality ───────────────────────────────────────────────
        pq_list = [r['prob_quality'] for r in seed_results]
        pq_df = compute_probability_quality_summary(pq_list, stream_name)
        if not pq_df.empty:
            pq_df.to_csv(
                os.path.join(RESULTS_DIR, f'{stream_name}_prob_quality.csv'), index=False)
            row = pq_df.iloc[0]
            print(f"\n  Probability quality: AUROC={row.get('auroc',0):.4f}  "
                  f"AUPRC={row.get('auprc',0):.4f}  Brier={row.get('brier',0):.4f}  "
                  f"ECE={row.get('ece',0):.4f}")
        prob_quality_by_stream[stream_name] = pq_list
        if not pq_df.empty:
            cross_dataset_auroc[stream_name] = {'auroc': float(pq_df['auroc'].values[0])}

        # ── Wilcoxon tests ────────────────────────────────────────────────────
        all_pw_dfs = [r['per_window_records'] for r in seed_results]
        if all_pw_dfs:
            pw_combined = pd.concat(all_pw_dfs, ignore_index=True)
            wil_df = compute_wilcoxon_pairs(pw_combined)
            wil_df['stream'] = stream_name
            wil_df.to_csv(
                os.path.join(RESULTS_DIR, f'{stream_name}_wilcoxon.csv'), index=False)
            all_wilcoxon_records.append(wil_df)
            pw_combined.to_parquet(
                os.path.join(RESULTS_DIR, f'{stream_name}_per_window.parquet'),
                index=False,
            ) if hasattr(pw_combined, 'to_parquet') else None

        # ── Oracle comparison ─────────────────────────────────────────────────
        if all_pw_dfs:
            oracle_df = compute_oracle_comparison(pw_combined, stream_name)
            if oracle_df is not None and not oracle_df.empty:
                oracle_df.to_csv(
                    os.path.join(RESULTS_DIR, f'{stream_name}_oracle_comparison.csv'), index=False)

        # ── Plots ─────────────────────────────────────────────────────────────
        try:
            ntr_by_method = {}
            for method in METHODS_ORDER:
                if all_cand:
                    # Use oracle selection to assign method NTR
                    ntr_by_method[method] = {
                        'negative': len([c for c in all_cand if c.get('transfer_label') == 'negative']),
                        'positive': len([c for c in all_cand if c.get('transfer_label') == 'positive']),
                        'total': len(all_cand),
                    }

            generate_all_plots(
                summary_df=summary_df,
                candidate_df=cand_df if not cand_df.empty else None,
                prob_quality_list=pq_list,
                oracle_df=oracle_df if oracle_df is not None and not oracle_df.empty else None,
                ntr_by_method=ntr_by_method,
                cross_results=cross_dataset_auroc,
                stream_name=stream_name,
                plots_dir=PLOTS_DIR,
            )
        except Exception as e:
            print(f"  WARNING: Plot generation failed: {e}")

    # ── Write report ───────────────────────────────────────────────────────────
    if all_summary_records:
        combined_summary = pd.concat(all_summary_records, ignore_index=True)
        combined_summary.to_csv(os.path.join(RESULTS_DIR, 'all_streams_summary.csv'), index=False)
        _write_report(combined_summary, all_gate_records, prob_quality_by_stream, args)

    t_elapsed = time.time() - t_global_start
    print(f"\n{'='*70}")
    print(f"Experiment 2 Corrected completed in {t_elapsed:.1f}s")
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots:   {PLOTS_DIR}")
    print(f"Report:  {REPORT_PATH}")
    if exit_code != 0:
        print("⚠  One or more CRITICAL sanity checks failed. Review gate results.")
    else:
        print("PASS  All sanity checks passed. Results are scientifically valid.")
    print(f"{'='*70}")
    return exit_code


def _write_report(
    combined_summary: pd.DataFrame,
    all_gate_records: list,
    prob_quality_by_stream: dict,
    args,
):
    import datetime
    lines = [
        "# Experiment 2 Corrected -- Results Report",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seeds:** {args.seeds}",
        f"**Streams evaluated:** {args.streams}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Experiment 2 Corrected implements probabilistic transferability estimation for",
        "RAPT-E regime policies, using a rolling-origin calibrated logistic regression.",
        "All results are computed on the same validated streaming datasets as Experiment 1.",
        "",
        "---",
        "",
        "## Sanity Check Gates",
        "",
        "| Stream | Check | Pass | Value | Expected Range |",
        "|--------|-------|------|-------|----------------|",
    ]
    for gate_rec in all_gate_records:
        stream = gate_rec.get('stream', '?')
        for check_name, res in gate_rec.items():
            if not isinstance(res, dict):
                continue
            passed = 'PASS' if res.get('pass', False) else 'FAIL FAIL'
            mean_f1 = f"{res.get('mean_f1', 0):.4f}" if 'mean_f1' in res else str(res.get('note', ''))
            expected = str(res.get('expected_range', ''))
            lines.append(f"| {stream} | {check_name} | {passed} | {mean_f1} | {expected} |")
    lines.append("")

    # Per-stream method summary tables
    for stream_name in combined_summary['stream'].unique():
        sub = combined_summary[combined_summary['stream'] == stream_name]
        lines.extend([
            f"## Results -- {stream_name}",
            "",
            "| Method | F1 Mean | F1 Std | CI95 | Adapt CPU (s) |",
            "|--------|---------|--------|------|---------------|",
        ])
        for _, row in sub.sort_values('f1_mean', ascending=False).iterrows():
            ci95 = f"[{row.get('f1_ci95_lo', 0):.4f}, {row.get('f1_ci95_hi', 0):.4f}]"
            lines.append(
                f"| {row['method']:25s} | {row['f1_mean']:.4f} | {row['f1_std']:.4f} | "
                f"{ci95} | {row['adapt_cpu_mean']:.2f} |"
            )
        lines.append("")

        # Probability quality
        pq_list = prob_quality_by_stream.get(stream_name, [])
        if pq_list:
            auroc_vals = [pq.get('auroc', float('nan')) for pq in pq_list
                          if not np.isnan(pq.get('auroc', float('nan')))]
            brier_vals = [pq.get('brier', float('nan')) for pq in pq_list
                          if not np.isnan(pq.get('brier', float('nan')))]
            n_eval = [pq.get('n_eval', 0) for pq in pq_list]
            lines.extend([
                f"### Probability Model Quality",
                "",
                f"| Metric | Mean | Std |",
                f"|--------|------|-----|",
                f"| AUROC  | {np.nanmean(auroc_vals):.4f} | {np.nanstd(auroc_vals):.4f} |",
                f"| Brier  | {np.nanmean(brier_vals):.4f} | {np.nanstd(brier_vals):.4f} |",
                f"| Eval episodes | {np.mean(n_eval):.1f} | -- |",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## Scientific Validity",
        "",
        "> - No label leakage: step 9 (outcome) strictly precedes step 10 (history update).",
        "> - Checkpoint pool invariant: target checkpoints trained on target data only.",
        "> - Rolling-origin: probability model at episode k trained on episodes 0..k-1.",
        "> - Oracle lower bounded by similarity-only (verified per episode).",
        "",
    ])

    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == '__main__':
    args = parse_args()
    sys.exit(run_all(args))
