"""
================================================================================
EXPERIMENT 4 v4 — PART 1a: CONTROLLED BUFFER CAP ABLATION
================================================================================
Compares two conditions across 5 seeds and 60 streaming windows:
  Condition 1: Bounded (5,000 samples)
  Condition 2: Unbounded (None / up to 50,000 samples)

Evaluated across four primary approaches:
  1. Continuously Retrained Ensemble
  2. Event-Driven Ensemble (Custom Dual-Trigger)
  3. Component-Selective Ensemble
  4. Warm-Start Ensemble

Measures:
  - F1 Score (mean +- std across windows)
  - Classification Accuracy
  - Cumulative CPU Time (inference + retraining)
  - Pure Model Footprint (KB, strictly serializing fitted estimators)
  - Post-Drift Recovery Latency (windows to recover to F1 >= 0.85)
================================================================================
"""

import os

# Pin single-threaded execution for numpy / scikit-learn / BLAS / OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_INITIAL_TRAINING, N_WINDOWS, WINDOW_SIZE, SEEDS,
    KPI_COLS, TARGET_COL, DECISION_THRESHOLD, RESULTS_DIR
)
from data_generation import generate_experiment_dataset
from ensemble import (
    HeterogeneousAdaptiveEnsemble,
    WarmStartEnsemble,
    ComponentSelectiveEnsemble,
)
from drift import DualTriggerDriftDetector, tune_detector_hyperparameters
from resource_monitor import measure_execution, measure_model_footprint
from metrics import compute_metrics, extract_drift_episodes


def compute_recovery_latency(f1_series, episodes, threshold=0.85, max_penalty=10):
    """
    Compute the mean number of windows required after drift onset
    for the model's F1 to recover to >= threshold.
    """
    latencies = []
    for ep in episodes:
        start_w = ep['start_window']
        recovered = False
        for w in range(start_w, min(len(f1_series), start_w + max_penalty)):
            if f1_series[w] >= threshold:
                latencies.append(w - start_w)
                recovered = True
                break
        if not recovered:
            latencies.append(max_penalty)
    return float(np.mean(latencies)) if latencies else 0.0


def compute_multi_threshold_recovery(f1_series, episodes, thresholds=[0.70, 0.75, 0.80, 0.85], max_penalty=10):
    """
    Compute recovery latencies across multiple operational thresholds (Part 4).
    """
    return {
        f"rec_lat_{int(t*100)}": compute_recovery_latency(f1_series, episodes, threshold=t, max_penalty=max_penalty)
        for t in thresholds
    }


def run_ablation(seeds=SEEDS, n_windows=N_WINDOWS):
    print("=" * 80, flush=True)
    print("RUNNING CONTROLLED BUFFER CAP ABLATION (Part 1a)", flush=True)
    print("Conditions: Bounded (5,000) vs. Unbounded (No Cap)", flush=True)
    print("Approaches: Continuous, Custom Dual-Trigger, Component-Selective, Warm-Start", flush=True)
    print("Seeds:", seeds, flush=True)
    print("=" * 80, flush=True)

    conditions = [
        ('Bounded (5,000)', 5000),
        ('Unbounded (None)', None),
    ]

    all_rows = []

    for cond_name, cap_val in conditions:
        print(f"\n{'='*70}", flush=True)
        print(f"EVALUATING CONDITION: {cond_name} (cap = {cap_val})", flush=True)
        print(f"{'='*70}", flush=True)

        for seed in seeds:
            t0 = time.time()
            df, schedule = generate_experiment_dataset(seed)
            episodes = extract_drift_episodes(schedule[:n_windows])

            X_all = df[KPI_COLS].values
            y_all = df[TARGET_COL].values

            X_init = X_all[:N_INITIAL_TRAINING]
            y_init = y_all[:N_INITIAL_TRAINING]
            X_ref_val = X_init[16000:]
            y_ref_val = y_init[16000:]

            # 1. Continuous Retraining Ensemble
            cont_ens = HeterogeneousAdaptiveEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=cap_val)
            cont_ens.fit_initial(X_init, y_init)

            tuned_configs = tune_detector_hyperparameters(X_ref_val, y_ref_val, cont_ens, seed=seed)
            custom_p = tuned_configs['Custom Dual-Trigger']

            # 2. Event-Driven (Custom Dual-Trigger)
            ed_ens = HeterogeneousAdaptiveEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=cap_val)
            ed_ens.fit_initial(X_init, y_init)
            ed_det = DualTriggerDriftDetector(
                wasserstein_threshold=custom_p['wasserstein_threshold'],
                perf_drop_threshold=custom_p['perf_drop_threshold'],
                initial_baseline_f1=0.85
            )

            # 3. Component-Selective Ensemble
            cs_ens = ComponentSelectiveEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=cap_val)
            cs_ens.fit_initial(X_init, y_init)
            cs_det = DualTriggerDriftDetector(
                wasserstein_threshold=custom_p['wasserstein_threshold'],
                perf_drop_threshold=custom_p['perf_drop_threshold'],
                initial_baseline_f1=0.85
            )

            # 4. Warm-Start Ensemble
            ws_ens = WarmStartEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=cap_val)
            ws_ens.fit_initial(X_init, y_init)
            ws_det = DualTriggerDriftDetector(
                wasserstein_threshold=custom_p['wasserstein_threshold'],
                perf_drop_threshold=custom_p['perf_drop_threshold'],
                initial_baseline_f1=0.85
            )

            models_dict = {
                'Continuous Retraining': (cont_ens, None),
                'Custom Dual-Trigger': (ed_ens, ed_det),
                'Component-Selective': (cs_ens, cs_det),
                'Warm-Start Ensemble': (ws_ens, ws_det),
            }

            cpu_totals = {k: 0.0 for k in models_dict}
            f1_scores = {k: [] for k in models_dict}
            acc_scores = {k: [] for k in models_dict}

            for w in range(n_windows):
                idx_s = N_INITIAL_TRAINING + w * WINDOW_SIZE
                idx_e = idx_s + WINDOW_SIZE
                X_win = X_all[idx_s:idx_e]
                y_win = y_all[idx_s:idx_e]

                # Reference slice for Wasserstein distance
                X_ref_slice = X_init[:cap_val] if cap_val is not None else X_init

                # 1. Continuous Retraining
                (pred, prob, cp, cpr), inf_res = measure_execution(cont_ens.predict, X_win)
                m = compute_metrics(y_win, pred, prob)
                cont_ens.update_weights(y_win, cp, cpr)
                _, ret_res, _ = cont_ens.adapt(X_win, y_win)
                r_cpu = sum(r['total_cpu_time'] for r in ret_res.values())
                cpu_totals['Continuous Retraining'] += (inf_res['total_cpu_time'] + r_cpu)
                f1_scores['Continuous Retraining'].append(m['f1'])
                acc_scores['Continuous Retraining'].append(m['accuracy'])

                # 2. Custom Dual-Trigger
                (pred, prob, cp, cpr), inf_res = measure_execution(ed_ens.predict, X_win)
                m = compute_metrics(y_win, pred, prob)
                ed_ens.update_weights(y_win, cp, cpr)
                dec = ed_det.check_drift(X_ref_slice, X_win, m['f1'])
                if dec['drift_detected']:
                    _, ret_res, _ = ed_ens.adapt(X_win, y_win)
                    r_cpu = sum(r['total_cpu_time'] for r in ret_res.values())
                else:
                    ed_ens.buffer_X.extend(X_win)
                    ed_ens.buffer_y.extend(y_win)
                    if cap_val is not None and len(ed_ens.buffer_X) > cap_val:
                        ed_ens.buffer_X = ed_ens.buffer_X[-cap_val:]
                        ed_ens.buffer_y = ed_ens.buffer_y[-cap_val:]
                    r_cpu = 0.0
                cpu_totals['Custom Dual-Trigger'] += (inf_res['total_cpu_time'] + r_cpu)
                f1_scores['Custom Dual-Trigger'].append(m['f1'])
                acc_scores['Custom Dual-Trigger'].append(m['accuracy'])

                # 3. Component-Selective
                (pred, prob, cp, cpr), inf_res = measure_execution(cs_ens.predict, X_win)
                m = compute_metrics(y_win, pred, prob)
                comp_m, _, _, _ = cs_ens.update_weights(y_win, cp, cpr)
                dec = cs_det.check_drift(X_ref_slice, X_win, m['f1'])
                if dec['drift_detected']:
                    _, ret_res, _, _ = cs_ens.adapt_selective(X_win, y_win, comp_m)
                    r_cpu = sum(r['total_cpu_time'] for r in ret_res.values())
                else:
                    cs_ens.buffer_X.extend(X_win)
                    cs_ens.buffer_y.extend(y_win)
                    if cap_val is not None and len(cs_ens.buffer_X) > cap_val:
                        cs_ens.buffer_X = cs_ens.buffer_X[-cap_val:]
                        cs_ens.buffer_y = cs_ens.buffer_y[-cap_val:]
                    r_cpu = 0.0
                cpu_totals['Component-Selective'] += (inf_res['total_cpu_time'] + r_cpu)
                f1_scores['Component-Selective'].append(m['f1'])
                acc_scores['Component-Selective'].append(m['accuracy'])

                # 4. Warm-Start Ensemble
                (pred, prob, cp, cpr), inf_res = measure_execution(ws_ens.predict, X_win)
                m = compute_metrics(y_win, pred, prob)
                ws_ens.update_weights(y_win, cp, cpr)
                dec = ws_det.check_drift(X_ref_slice, X_win, m['f1'])
                if dec['drift_detected']:
                    _, ret_res, _ = ws_ens.adapt(X_win, y_win)
                    r_cpu = sum(r['total_cpu_time'] for r in ret_res.values())
                else:
                    ws_ens.buffer_X.extend(X_win)
                    ws_ens.buffer_y.extend(y_win)
                    if cap_val is not None and len(ws_ens.buffer_X) > cap_val:
                        ws_ens.buffer_X = ws_ens.buffer_X[-cap_val:]
                        ws_ens.buffer_y = ws_ens.buffer_y[-cap_val:]
                    r_cpu = 0.0
                cpu_totals['Warm-Start Ensemble'] += (inf_res['total_cpu_time'] + r_cpu)
                f1_scores['Warm-Start Ensemble'].append(m['f1'])
                acc_scores['Warm-Start Ensemble'].append(m['accuracy'])

            elapsed = time.time() - t0
            print(f"   [{cond_name}] Seed {seed} completed in {elapsed:.1f}s", flush=True)

            for k in models_dict:
                fp = measure_model_footprint(models_dict[k][0])
                rec_lat_dict = compute_multi_threshold_recovery(f1_scores[k], episodes)
                rec_lat = rec_lat_dict['rec_lat_85']
                all_rows.append({
                    'condition': cond_name,
                    'buffer_cap': str(cap_val) if cap_val is not None else 'Unbounded',
                    'approach': k,
                    'seed': seed,
                    'f1': float(np.mean(f1_scores[k])),
                    'accuracy': float(np.mean(acc_scores[k])),
                    'cpu_time': float(cpu_totals[k]),
                    'model_footprint_kb': float(fp) / 1024.0,
                    'recovery_latency': rec_lat,
                    'rec_lat_70': rec_lat_dict['rec_lat_70'],
                    'rec_lat_75': rec_lat_dict['rec_lat_75'],
                    'rec_lat_80': rec_lat_dict['rec_lat_80'],
                    'rec_lat_85': rec_lat_dict['rec_lat_85'],
                })

    df_res = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULTS_DIR, 'buffer_cap_ablation.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"\nAblation raw data saved to: {out_csv}", flush=True)

    # Summary table
    summary = []
    for (cond, app), grp in df_res.groupby(['condition', 'approach'], sort=False):
        summary.append({
            'Condition': cond,
            'Approach': app,
            'Mean F1': f"{grp['f1'].mean():.4f} ± {grp['f1'].std():.4f}",
            'Mean Accuracy': f"{grp['accuracy'].mean():.4f} ± {grp['accuracy'].std():.4f}",
            'Cumulative CPU (s)': f"{grp['cpu_time'].mean():.2f} ± {grp['cpu_time'].std():.2f}",
            'Pure Footprint (KB)': f"{grp['model_footprint_kb'].mean():.1f} ± {grp['model_footprint_kb'].std():.1f}",
            'Recovery Latency (win)': f"{grp['recovery_latency'].mean():.2f} ± {grp['recovery_latency'].std():.2f}",
            'Rec Lat (>=0.70)': f"{grp['rec_lat_70'].mean():.2f} ± {grp['rec_lat_70'].std():.2f}",
            'Rec Lat (>=0.75)': f"{grp['rec_lat_75'].mean():.2f} ± {grp['rec_lat_75'].std():.2f}",
            'Rec Lat (>=0.80)': f"{grp['rec_lat_80'].mean():.2f} ± {grp['rec_lat_80'].std():.2f}",
            'Rec Lat (>=0.85)': f"{grp['rec_lat_85'].mean():.2f} ± {grp['rec_lat_85'].std():.2f}",
            'raw_f1': grp['f1'].mean(),
            'raw_cpu': grp['cpu_time'].mean(),
            'raw_footprint': grp['model_footprint_kb'].mean(),
            'raw_rec_lat': grp['recovery_latency'].mean(),
            'raw_rec_lat_70': grp['rec_lat_70'].mean(),
            'raw_rec_lat_75': grp['rec_lat_75'].mean(),
            'raw_rec_lat_80': grp['rec_lat_80'].mean(),
            'raw_rec_lat_85': grp['rec_lat_85'].mean(),
        })

    df_sum = pd.DataFrame(summary)
    sum_csv = os.path.join(RESULTS_DIR, 'buffer_cap_ablation_summary.csv')
    df_sum.to_csv(sum_csv, index=False)

    print("\n" + "=" * 90, flush=True)
    print("BUFFER CAP ABLATION SUMMARY TABLE (Bounded vs. Unbounded)", flush=True)
    print("=" * 90, flush=True)
    print(df_sum[['Condition', 'Approach', 'Mean F1', 'Cumulative CPU (s)', 'Pure Footprint (KB)', 'Recovery Latency (win)']].to_string(index=False), flush=True)
    print("=" * 90, flush=True)

    return df_sum


if __name__ == '__main__':
    run_ablation()
