"""
================================================================================
EXPERIMENT 4 v3 — MAIN EXECUTION RUNNER
================================================================================
Compares twelve strategies across 5 seeds and 60 streaming windows (50,000 samples):
  1. Frozen Model (Static baseline, zero adaptation)
  2. Continuously Retrained Ensemble (RF + ET + GB, 100% retraining)
  3. Event-Driven Ensemble (Custom Dual-Trigger: Wasserstein + F1 drop)
  4. Event-Driven Ensemble (ADWIN)
  5. Event-Driven Ensemble (DDM)
  6. Event-Driven Ensemble (EDDM)
  7. Event-Driven Ensemble (Page-Hinkley)
  8. Warm-Start Ensemble (Incremental tree addition & oldest tree pruning)
  9. Component-Selective Ensemble (Retrains only degraded components)
  10. Adaptive Random Forest (river) (Continuous streaming ensemble)
  11. Streaming Random Patches (river) (Continuous streaming ensemble)
  12. Two-Tier Hybrid Ensemble (Batch ensemble + always-on online streaming learner)

Measurement Fixes in v3:
  - Clean TimingMonitor: zero tracemalloc contamination on CPU timing hot path
  - Disambiguated metrics: window_trigger_rate vs. episode_detection_rate
  - Fair validation tuning: equal tuning budget on 4,000 reference validation split
  - Bounded buffer cap: 5,000 samples preventing memory leaks

Outputs:
  - Results saved to experiment4/results/
  - Figures saved to experiment4/figures/ (17 publication figures)
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
    N_TOTAL_SAMPLES, N_INITIAL_TRAINING, N_DEPLOYMENT_SAMPLES,
    N_WINDOWS, WINDOW_SIZE, SEEDS, KPI_COLS, TARGET_COL,
    RESULTS_DIR, FIGURES_DIR, PLOTS_DIR,
    DECISION_THRESHOLD, APPROACH_NAMES, DETECTOR_NAMES,
    BUFFER_CAP
)
from data_generation import generate_experiment_dataset, export_drift_configuration
from models import create_frozen_rf
from ensemble import (
    HeterogeneousAdaptiveEnsemble,
    WarmStartEnsemble,
    ComponentSelectiveEnsemble,
    TwoTierHybridEnsemble,
)
from streaming_models import RiverStreamingEnsembleWrapper
from drift import (
    DualTriggerDriftDetector,
    create_drift_detector,
    tune_detector_hyperparameters
)
from resource_monitor import measure_execution, measure_model_footprint
from metrics import compute_metrics, compute_detector_quality
from statistics import compute_summary_stats, run_paired_tests
from plots import generate_all_plots


def run_experiment4(seeds=SEEDS, n_windows=N_WINDOWS, output_dir=RESULTS_DIR, plots_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 85)
    print("STARTING EXPERIMENT 4 v3: MULTI-STRATEGY DRIFT BENCHMARKING & MEASUREMENT AUDIT")
    print(f"Approaches ({len(APPROACH_NAMES)}): {APPROACH_NAMES}")
    print(f"Scale: 50,000 samples ({N_INITIAL_TRAINING} initial training, {n_windows} windows x {WINDOW_SIZE} samples)")
    print(f"Seeds: {seeds}")
    print(f"Output Directory: {output_dir}")
    print(f"Figures Directory: {plots_dir}")
    print("=" * 85)

    # Export drift schedule across seeds
    drift_cfg_path = os.path.join(output_dir, 'drift_configuration.csv')
    export_drift_configuration(seeds, drift_cfg_path)

    window_rows = []
    retrain_rows = []
    drift_rows = []
    detector_quality_rows = []
    episode_records_all = []

    for seed in seeds:
        print(f"\n---> Running Seed {seed}...")
        t0_seed = time.time()

        df, schedule = generate_experiment_dataset(seed)
        X_all = df[KPI_COLS].values
        y_all = df[TARGET_COL].values

        X_init = X_all[:N_INITIAL_TRAINING]
        y_init = y_all[:N_INITIAL_TRAINING]

        # Initial validation split for threshold tuning and detector warm-up
        X_ref_train = X_init[:16000]
        y_ref_train = y_init[:16000]
        X_ref_val = X_init[16000:]
        y_ref_val = y_init[16000:]

        # =====================================================================
        # MODEL INITIALIZATION
        # =====================================================================
        # 1. Model 1: Frozen Model (RF)
        frozen_rf = create_frozen_rf(seed)
        _, frozen_init_res = measure_execution(frozen_rf.fit, X_init, y_init)
        frozen_footprint = measure_model_footprint(frozen_rf)

        # 1b. Model 1b: Frozen Ensemble (RF+ET+GB)
        frozen_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=True, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = frozen_ens.fit_initial(X_init, y_init)
        frozen_ens_footprint = measure_model_footprint(frozen_ens)

        # 2. Model 2: Continuously Retrained Ensemble
        cont_ensemble = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = cont_ensemble.fit_initial(X_init, y_init)

        # Tune all 5 detectors on validation split (zero stream leakage)
        print(f"   [Seed {seed}] Tuning detector hyperparameters on validation split...")
        tuned_configs = tune_detector_hyperparameters(X_ref_val, y_ref_val, cont_ensemble, seed=seed)

        # 3. Model 3: Event-Driven Ensemble (Custom Dual-Trigger)
        ed_custom_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = ed_custom_ens.fit_initial(X_init, y_init)
        custom_params = tuned_configs['Custom Dual-Trigger']
        custom_detector = DualTriggerDriftDetector(
            wasserstein_threshold=custom_params['wasserstein_threshold'],
            perf_drop_threshold=custom_params['perf_drop_threshold'],
            initial_baseline_f1=0.85
        )

        # 4. Model 4: Event-Driven Ensemble (ADWIN)
        ed_adwin_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = ed_adwin_ens.fit_initial(X_init, y_init)
        adwin_detector = create_drift_detector('ADWIN', **tuned_configs['ADWIN'])
        (v_pred, _, _, _) = ed_adwin_ens.predict(X_ref_val)
        adwin_detector.warm_start(y_ref_val, v_pred)

        # 5. Model 5: Event-Driven Ensemble (DDM)
        ed_ddm_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = ed_ddm_ens.fit_initial(X_init, y_init)
        ddm_detector = create_drift_detector('DDM', **tuned_configs['DDM'])
        (v_pred, _, _, _) = ed_ddm_ens.predict(X_ref_val)
        ddm_detector.warm_start(y_ref_val, v_pred)

        # 6. Model 6: Event-Driven Ensemble (EDDM)
        ed_eddm_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = ed_eddm_ens.fit_initial(X_init, y_init)
        eddm_detector = create_drift_detector('EDDM', **tuned_configs['EDDM'])
        (v_pred, _, _, _) = ed_eddm_ens.predict(X_ref_val)
        eddm_detector.warm_start(y_ref_val, v_pred)

        # 7. Model 7: Event-Driven Ensemble (Page-Hinkley)
        ed_ph_ens = HeterogeneousAdaptiveEnsemble(seed, is_frozen=False, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = ed_ph_ens.fit_initial(X_init, y_init)
        ph_detector = create_drift_detector('Page-Hinkley', **tuned_configs['Page-Hinkley'])
        (v_pred, _, _, _) = ed_ph_ens.predict(X_ref_val)
        ph_detector.warm_start(y_ref_val, v_pred)

        # 8. Model 8: Warm-Start Ensemble
        warm_start_ens = WarmStartEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = warm_start_ens.fit_initial(X_init, y_init)
        ws_detector = DualTriggerDriftDetector(
            wasserstein_threshold=custom_params['wasserstein_threshold'],
            perf_drop_threshold=custom_params['perf_drop_threshold'],
            initial_baseline_f1=0.85
        )

        # 9. Model 9: Component-Selective Ensemble
        comp_sel_ens = ComponentSelectiveEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = comp_sel_ens.fit_initial(X_init, y_init)
        cs_detector = DualTriggerDriftDetector(
            wasserstein_threshold=custom_params['wasserstein_threshold'],
            perf_drop_threshold=custom_params['perf_drop_threshold'],
            initial_baseline_f1=0.85
        )

        # 10. Model 10: Adaptive Random Forest (river)
        arf_model = RiverStreamingEnsembleWrapper('ARF', seed=seed, threshold=DECISION_THRESHOLD)
        arf_model.fit_initial(X_init, y_init)

        # 11. Model 11: Streaming Random Patches (river)
        srp_model = RiverStreamingEnsembleWrapper('SRP', seed=seed, threshold=DECISION_THRESHOLD)
        srp_model.fit_initial(X_init, y_init)

        # 12. Model 12: Two-Tier Hybrid Ensemble
        two_tier_ens = TwoTierHybridEnsemble(seed, threshold=DECISION_THRESHOLD, buffer_cap=BUFFER_CAP)
        _ = two_tier_ens.fit_initial(X_init, y_init)
        tt_detector = DualTriggerDriftDetector(
            wasserstein_threshold=custom_params['wasserstein_threshold'],
            perf_drop_threshold=custom_params['perf_drop_threshold'],
            initial_baseline_f1=0.85
        )

        # Tracking detector fires for quality evaluation
        detector_fires = {
            'Custom Dual-Trigger': [],
            'ADWIN': [],
            'DDM': [],
            'EDDM': [],
            'Page-Hinkley': [],
        }

        # Streaming evaluation loop across 60 windows
        for w in range(n_windows):
            idx_s = N_INITIAL_TRAINING + w * WINDOW_SIZE
            idx_e = idx_s + WINDOW_SIZE
            X_win = X_all[idx_s:idx_e]
            y_win = y_all[idx_s:idx_e]

            win_cfg = schedule[w]
            dtype = win_cfg['drift_type']
            sev = win_cfg['severity']
            trans = win_cfg['transition']

            # Helper for appending standard window record
            def _log_window(app_name, m, tot_cpu, tot_wall, footprint, inc_mem, rss, fired, w_mean=0.0, online_w=None):
                row = {
                    'seed': seed,
                    'window_id': w,
                    'approach': app_name,
                    'f1': round(m['f1'], 4),
                    'accuracy': round(m['accuracy'], 4),
                    'precision': round(m['precision'], 4),
                    'recall': round(m['recall'], 4),
                    'auc': round(m['auc'], 4),
                    'pr_auc': round(m['pr_auc'], 4),
                    'balanced_accuracy': round(m['balanced_accuracy'], 4),
                    'cpu_time': round(tot_cpu, 4),
                    'wall_time': round(tot_wall, 4),
                    'model_footprint_bytes': footprint,
                    'incremental_rss_delta_mb': round(inc_mem, 4),
                    'process_rss_mb': round(rss, 2),
                    'drift_type': dtype,
                    'severity': sev,
                    'transition': trans,
                    'drift_detected': bool(fired),
                    'wasserstein_mean': round(w_mean, 4),
                }
                if online_w is not None:
                    row['online_weight'] = round(online_w, 4)
                window_rows.append(row)

            def _log_retrain(app_name, did_retrain, retrain_cpu, retrain_wall, samples, footprint, inc_mem, comps=None):
                row = {
                    'seed': seed,
                    'window_id': w,
                    'approach': app_name,
                    'retrained': bool(did_retrain),
                    'retraining_cpu_time': round(retrain_cpu, 4),
                    'retraining_wall_time': round(retrain_wall, 4),
                    'samples_adapted': samples,
                    'model_footprint_bytes': footprint,
                    'incremental_rss_delta_mb': round(inc_mem, 4),
                }
                if comps is not None:
                    row['retrained_components'] = ','.join(comps) if isinstance(comps, list) else str(comps)
                retrain_rows.append(row)

            # =========================================================
            # 1. FROZEN MODEL
            # =========================================================
            def _frozen_infer():
                prob = frozen_rf.predict_proba(X_win)[:, 1]
                pred = (prob >= DECISION_THRESHOLD).astype(int)
                return pred, prob

            (frozen_pred, frozen_prob), frozen_inf_res = measure_execution(_frozen_infer)
            frozen_m = compute_metrics(y_win, frozen_pred, frozen_prob)
            frozen_cpu = frozen_inf_res['total_cpu_time']
            frozen_wall = frozen_inf_res['wall_clock_time']
            frozen_inc_mem = frozen_inf_res.get('incremental_rss_delta_mb', 0.0)
            frozen_rss = frozen_inf_res.get('process_rss_mb', 0.0)

            _log_window('Frozen Model', frozen_m, frozen_cpu, frozen_wall, frozen_footprint, frozen_inc_mem, frozen_rss, False)
            _log_retrain('Frozen Model', False, 0.0, 0.0, 0, frozen_footprint, frozen_inc_mem)

            # =========================================================
            # 1b. FROZEN ENSEMBLE
            # =========================================================
            (f_ens_pred, f_ens_prob, _, _), f_ens_inf_res = measure_execution(
                lambda: frozen_ens.predict(X_win)
            )
            f_ens_m = compute_metrics(y_win, f_ens_pred, f_ens_prob)
            f_ens_cpu = f_ens_inf_res['total_cpu_time']
            f_ens_wall = f_ens_inf_res['wall_clock_time']
            f_ens_inc_mem = f_ens_inf_res.get('incremental_rss_delta_mb', 0.0)
            f_ens_rss = f_ens_inf_res.get('process_rss_mb', 0.0)

            _log_window('Frozen Ensemble', f_ens_m, f_ens_cpu, f_ens_wall, frozen_ens_footprint, f_ens_inc_mem, f_ens_rss, False)
            _log_retrain('Frozen Ensemble', False, 0.0, 0.0, 0, frozen_ens_footprint, f_ens_inc_mem)

            # =========================================================
            # 2. CONTINUOUSLY RETRAINED ENSEMBLE
            # =========================================================
            (cont_pred, cont_prob, cont_cp, cont_cpr), cont_inf_res = measure_execution(
                lambda: cont_ensemble.predict(X_win)
            )
            cont_m = compute_metrics(y_win, cont_pred, cont_prob)
            cont_ensemble.update_weights(y_win, cont_cp, cont_cpr)

            _, cont_retrain_res, cont_samples = cont_ensemble.adapt(X_win, y_win)
            cont_retrain_cpu = sum(r['total_cpu_time'] for r in cont_retrain_res.values())
            cont_retrain_wall = sum(r['wall_clock_time'] for r in cont_retrain_res.values())
            cont_retrain_inc_mem = max([r.get('incremental_rss_delta_mb', 0.0) for r in cont_retrain_res.values()] or [0.0])
            cont_footprint = measure_model_footprint(cont_ensemble)

            cont_tot_cpu = cont_inf_res['total_cpu_time'] + cont_retrain_cpu
            cont_tot_wall = cont_inf_res['wall_clock_time'] + cont_retrain_wall
            cont_tot_inc_mem = cont_inf_res.get('incremental_rss_delta_mb', 0.0) + cont_retrain_inc_mem
            cont_rss = cont_inf_res.get('process_rss_mb', 0.0)

            _log_window('Continuously Retrained Ensemble', cont_m, cont_tot_cpu, cont_tot_wall, cont_footprint, cont_tot_inc_mem, cont_rss, True)
            _log_retrain('Continuously Retrained Ensemble', True, cont_retrain_cpu, cont_retrain_wall, cont_samples, cont_footprint, cont_retrain_inc_mem)

            # =========================================================
            # 3. EVENT-DRIVEN ENSEMBLE (CUSTOM DUAL-TRIGGER)
            # =========================================================
            (ed_c_pred, ed_c_prob, ed_c_cp, ed_c_cpr), ed_c_inf_res = measure_execution(
                lambda: ed_custom_ens.predict(X_win)
            )
            ed_c_m = compute_metrics(y_win, ed_c_pred, ed_c_prob)
            ed_custom_ens.update_weights(y_win, ed_c_cp, ed_c_cpr)

            drift_dec = custom_detector.check_drift(X_init[:BUFFER_CAP], X_win, ed_c_m['f1'])
            custom_fired = drift_dec['drift_detected']
            w_score = drift_dec['wasserstein_score']
            detector_fires['Custom Dual-Trigger'].append(custom_fired)

            drift_rows.append({
                'seed': seed,
                'window': w,
                'drift_type': dtype,
                'severity': sev,
                'transition': trans,
                'custom_drift_detected': custom_fired,
                'covariate_drift': drift_dec['covariate_drift'],
                'concept_drift': drift_dec['concept_drift'],
                'wasserstein_score': round(w_score, 4),
                'f1_drop': round(drift_dec['f1_drop'], 4),
            })

            if custom_fired:
                _, ed_c_retrain_res, ed_c_samples = ed_custom_ens.adapt(X_win, y_win)
                ed_c_retrain_cpu = sum(r['total_cpu_time'] for r in ed_c_retrain_res.values())
                ed_c_retrain_wall = sum(r['wall_clock_time'] for r in ed_c_retrain_res.values())
                ed_c_retrain_inc_mem = max([r.get('incremental_rss_delta_mb', 0.0) for r in ed_c_retrain_res.values()] or [0.0])
                ed_c_did_retrain = True
            else:
                ed_custom_ens.buffer_X.extend(X_win)
                ed_custom_ens.buffer_y.extend(y_win)
                if len(ed_custom_ens.buffer_X) > ed_custom_ens.buffer_cap:
                    ed_custom_ens.buffer_X = ed_custom_ens.buffer_X[-ed_custom_ens.buffer_cap:]
                    ed_custom_ens.buffer_y = ed_custom_ens.buffer_y[-ed_custom_ens.buffer_cap:]
                ed_c_retrain_cpu = 0.0
                ed_c_retrain_wall = 0.0
                ed_c_retrain_inc_mem = 0.0
                ed_c_samples = 0
                ed_c_did_retrain = False

            ed_c_footprint = measure_model_footprint(ed_custom_ens)
            ed_c_tot_cpu = ed_c_inf_res['total_cpu_time'] + ed_c_retrain_cpu
            ed_c_tot_wall = ed_c_inf_res['wall_clock_time'] + ed_c_retrain_wall
            ed_c_tot_inc_mem = ed_c_inf_res.get('incremental_rss_delta_mb', 0.0) + ed_c_retrain_inc_mem
            ed_c_rss = ed_c_inf_res.get('process_rss_mb', 0.0)

            _log_window('Event-Driven Ensemble (Custom Dual-Trigger)', ed_c_m, ed_c_tot_cpu, ed_c_tot_wall, ed_c_footprint, ed_c_tot_inc_mem, ed_c_rss, custom_fired, w_score)
            _log_retrain('Event-Driven Ensemble (Custom Dual-Trigger)', ed_c_did_retrain, ed_c_retrain_cpu, ed_c_retrain_wall, ed_c_samples, ed_c_footprint, ed_c_retrain_inc_mem)

            # =========================================================
            # 4–7. STANDARD RIVER DETECTORS (ADWIN, DDM, EDDM, Page-Hinkley)
            # =========================================================
            river_approaches = [
                ('Event-Driven Ensemble (ADWIN)', ed_adwin_ens, adwin_detector, 'ADWIN'),
                ('Event-Driven Ensemble (DDM)', ed_ddm_ens, ddm_detector, 'DDM'),
                ('Event-Driven Ensemble (EDDM)', ed_eddm_ens, eddm_detector, 'EDDM'),
                ('Event-Driven Ensemble (Page-Hinkley)', ed_ph_ens, ph_detector, 'Page-Hinkley'),
            ]

            for app_name, ensemble_obj, detector_obj, det_key in river_approaches:
                (pred, prob, cp, cpr), inf_res = measure_execution(
                    lambda ens=ensemble_obj: ens.predict(X_win)
                )
                m = compute_metrics(y_win, pred, prob)
                ensemble_obj.update_weights(y_win, cp, cpr)

                det_res = detector_obj.check_drift_window(y_win, pred)
                fired = det_res['drift_detected']
                detector_fires[det_key].append(fired)

                if fired:
                    _, retrain_res, samples = ensemble_obj.adapt(X_win, y_win)
                    retrain_cpu = sum(r['total_cpu_time'] for r in retrain_res.values())
                    retrain_wall = sum(r['wall_clock_time'] for r in retrain_res.values())
                    retrain_inc_mem = max([r.get('incremental_rss_delta_mb', 0.0) for r in retrain_res.values()] or [0.0])
                    did_retrain = True
                else:
                    ensemble_obj.buffer_X.extend(X_win)
                    ensemble_obj.buffer_y.extend(y_win)
                    if len(ensemble_obj.buffer_X) > ensemble_obj.buffer_cap:
                        ensemble_obj.buffer_X = ensemble_obj.buffer_X[-ensemble_obj.buffer_cap:]
                        ensemble_obj.buffer_y = ensemble_obj.buffer_y[-ensemble_obj.buffer_cap:]
                    retrain_cpu = 0.0
                    retrain_wall = 0.0
                    retrain_inc_mem = 0.0
                    samples = 0
                    did_retrain = False

                footprint = measure_model_footprint(ensemble_obj)
                tot_cpu = inf_res['total_cpu_time'] + retrain_cpu
                tot_wall = inf_res['wall_clock_time'] + retrain_wall
                tot_inc_mem = inf_res.get('incremental_rss_delta_mb', 0.0) + retrain_inc_mem
                rss = inf_res.get('process_rss_mb', 0.0)

                _log_window(app_name, m, tot_cpu, tot_wall, footprint, tot_inc_mem, rss, fired)
                _log_retrain(app_name, did_retrain, retrain_cpu, retrain_wall, samples, footprint, retrain_inc_mem)

            # =========================================================
            # 8. WARM-START ENSEMBLE
            # =========================================================
            (ws_pred, ws_prob, ws_cp, ws_cpr), ws_inf_res = measure_execution(
                lambda: warm_start_ens.predict(X_win)
            )
            ws_m = compute_metrics(y_win, ws_pred, ws_prob)
            warm_start_ens.update_weights(y_win, ws_cp, ws_cpr)

            ws_drift_dec = ws_detector.check_drift(X_init[:BUFFER_CAP], X_win, ws_m['f1'])
            ws_fired = ws_drift_dec['drift_detected']

            if ws_fired:
                _, ws_retrain_res, ws_samples = warm_start_ens.adapt(X_win, y_win)
                ws_retrain_cpu = sum(r['total_cpu_time'] for r in ws_retrain_res.values())
                ws_retrain_wall = sum(r['wall_clock_time'] for r in ws_retrain_res.values())
                ws_retrain_inc_mem = max([r.get('incremental_rss_delta_mb', 0.0) for r in ws_retrain_res.values()] or [0.0])
                ws_did_retrain = True
            else:
                warm_start_ens.buffer_X.extend(X_win)
                warm_start_ens.buffer_y.extend(y_win)
                if len(warm_start_ens.buffer_X) > warm_start_ens.buffer_cap:
                    warm_start_ens.buffer_X = warm_start_ens.buffer_X[-warm_start_ens.buffer_cap:]
                    warm_start_ens.buffer_y = warm_start_ens.buffer_y[-warm_start_ens.buffer_cap:]
                ws_retrain_cpu = 0.0
                ws_retrain_wall = 0.0
                ws_retrain_inc_mem = 0.0
                ws_samples = 0
                ws_did_retrain = False

            ws_footprint = measure_model_footprint(warm_start_ens)
            ws_tot_cpu = ws_inf_res['total_cpu_time'] + ws_retrain_cpu
            ws_tot_wall = ws_inf_res['wall_clock_time'] + ws_retrain_wall
            ws_tot_inc_mem = ws_inf_res.get('incremental_rss_delta_mb', 0.0) + ws_retrain_inc_mem
            ws_rss = ws_inf_res.get('process_rss_mb', 0.0)

            _log_window('Warm-Start Ensemble', ws_m, ws_tot_cpu, ws_tot_wall, ws_footprint, ws_tot_inc_mem, ws_rss, ws_fired)
            _log_retrain('Warm-Start Ensemble', ws_did_retrain, ws_retrain_cpu, ws_retrain_wall, ws_samples, ws_footprint, ws_retrain_inc_mem)

            # =========================================================
            # 9. COMPONENT-SELECTIVE ENSEMBLE
            # =========================================================
            (cs_pred, cs_prob, cs_cp, cs_cpr), cs_inf_res = measure_execution(
                lambda: comp_sel_ens.predict(X_win)
            )
            cs_m = compute_metrics(y_win, cs_pred, cs_prob)
            cs_comp_mets, _, _, _ = comp_sel_ens.update_weights(y_win, cs_cp, cs_cpr)

            cs_drift_dec = cs_detector.check_drift(X_init[:BUFFER_CAP], X_win, cs_m['f1'])
            cs_fired = cs_drift_dec['drift_detected']

            if cs_fired:
                _, cs_retrain_res, cs_samples, cs_sel_comps = comp_sel_ens.adapt_selective(X_win, y_win, cs_comp_mets)
                cs_retrain_cpu = sum(r['total_cpu_time'] for r in cs_retrain_res.values())
                cs_retrain_wall = sum(r['wall_clock_time'] for r in cs_retrain_res.values())
                cs_retrain_inc_mem = max([r.get('incremental_rss_delta_mb', 0.0) for r in cs_retrain_res.values()] or [0.0])
                cs_did_retrain = True
            else:
                comp_sel_ens.buffer_X.extend(X_win)
                comp_sel_ens.buffer_y.extend(y_win)
                if len(comp_sel_ens.buffer_X) > comp_sel_ens.buffer_cap:
                    comp_sel_ens.buffer_X = comp_sel_ens.buffer_X[-comp_sel_ens.buffer_cap:]
                    comp_sel_ens.buffer_y = comp_sel_ens.buffer_y[-comp_sel_ens.buffer_cap:]
                cs_retrain_cpu = 0.0
                cs_retrain_wall = 0.0
                cs_retrain_inc_mem = 0.0
                cs_samples = 0
                cs_did_retrain = False
                cs_sel_comps = []

            cs_footprint = measure_model_footprint(comp_sel_ens)
            cs_tot_cpu = cs_inf_res['total_cpu_time'] + cs_retrain_cpu
            cs_tot_wall = cs_inf_res['wall_clock_time'] + cs_retrain_wall
            cs_tot_inc_mem = cs_inf_res.get('incremental_rss_delta_mb', 0.0) + cs_retrain_inc_mem
            cs_rss = cs_inf_res.get('process_rss_mb', 0.0)

            _log_window('Component-Selective Ensemble', cs_m, cs_tot_cpu, cs_tot_wall, cs_footprint, cs_tot_inc_mem, cs_rss, cs_fired)
            _log_retrain('Component-Selective Ensemble', cs_did_retrain, cs_retrain_cpu, cs_retrain_wall, cs_samples, cs_footprint, cs_retrain_inc_mem, cs_sel_comps)

            # =========================================================
            # 10. ADAPTIVE RANDOM FOREST (river)
            # =========================================================
            (arf_pred, arf_prob), arf_inf_res = measure_execution(
                lambda: arf_model.predict(X_win)
            )
            arf_m = compute_metrics(y_win, arf_pred, arf_prob)

            _, arf_learn_res = measure_execution(arf_model.learn, X_win, y_win)
            arf_tot_cpu = arf_inf_res['total_cpu_time'] + arf_learn_res['total_cpu_time']
            arf_tot_wall = arf_inf_res['wall_clock_time'] + arf_learn_res['wall_clock_time']
            arf_inc_mem = arf_inf_res.get('incremental_rss_delta_mb', 0.0) + arf_learn_res.get('incremental_rss_delta_mb', 0.0)
            arf_rss = arf_inf_res.get('process_rss_mb', 0.0)
            arf_footprint = measure_model_footprint(arf_model.classifier)

            # Note: Streaming models adapt continuously without discrete retrain events
            _log_window('Adaptive Random Forest (river)', arf_m, arf_tot_cpu, arf_tot_wall, arf_footprint, arf_inc_mem, arf_rss, False)
            _log_retrain('Adaptive Random Forest (river)', False, arf_learn_res['total_cpu_time'], arf_learn_res['wall_clock_time'], len(X_win), arf_footprint, arf_inc_mem)

            # =========================================================
            # 11. STREAMING RANDOM PATCHES (river)
            # =========================================================
            (srp_pred, srp_prob), srp_inf_res = measure_execution(
                lambda: srp_model.predict(X_win)
            )
            srp_m = compute_metrics(y_win, srp_pred, srp_prob)

            _, srp_learn_res = measure_execution(srp_model.learn, X_win, y_win)
            srp_tot_cpu = srp_inf_res['total_cpu_time'] + srp_learn_res['total_cpu_time']
            srp_tot_wall = srp_inf_res['wall_clock_time'] + srp_learn_res['wall_clock_time']
            srp_inc_mem = srp_inf_res.get('incremental_rss_delta_mb', 0.0) + srp_learn_res.get('incremental_rss_delta_mb', 0.0)
            srp_rss = srp_inf_res.get('process_rss_mb', 0.0)
            srp_footprint = measure_model_footprint(srp_model.classifier)

            _log_window('Streaming Random Patches (river)', srp_m, srp_tot_cpu, srp_tot_wall, srp_footprint, srp_inc_mem, srp_rss, False)
            _log_retrain('Streaming Random Patches (river)', False, srp_learn_res['total_cpu_time'], srp_learn_res['wall_clock_time'], len(X_win), srp_footprint, srp_inc_mem)

            # =========================================================
            # 12. TWO-TIER HYBRID ENSEMBLE
            # =========================================================
            (tt_pred, tt_prob, tt_cp, tt_cpr), tt_inf_res = measure_execution(
                lambda: two_tier_ens.predict(X_win)
            )
            tt_m = compute_metrics(y_win, tt_pred, tt_prob)
            two_tier_ens.update_weights(y_win, tt_cp, tt_cpr)

            tt_drift_dec = tt_detector.check_drift(X_init[:BUFFER_CAP], X_win, tt_m['f1'])
            tt_fired = tt_drift_dec['drift_detected']

            _, _, tt_samples, tt_retrain_cpu, tt_retrain_wall = two_tier_ens.adapt(X_win, y_win, drift_fired=tt_fired)
            tt_did_retrain = tt_fired

            tt_footprint = measure_model_footprint(two_tier_ens.batch_ensemble) + measure_model_footprint(two_tier_ens.online_learner.learner)
            tt_tot_cpu = tt_inf_res['total_cpu_time'] + tt_retrain_cpu
            tt_tot_wall = tt_inf_res['wall_clock_time'] + tt_retrain_wall
            tt_tot_inc_mem = tt_inf_res.get('incremental_rss_delta_mb', 0.0)
            tt_rss = tt_inf_res.get('process_rss_mb', 0.0)

            _log_window('Two-Tier Hybrid Ensemble', tt_m, tt_tot_cpu, tt_tot_wall, tt_footprint, tt_tot_inc_mem, tt_rss, tt_fired, online_w=two_tier_ens.online_weight)
            _log_retrain('Two-Tier Hybrid Ensemble', tt_did_retrain, tt_retrain_cpu, tt_retrain_wall, tt_samples, tt_footprint, tt_tot_inc_mem)

        # Evaluate detector quality metrics for this seed
        for det_name in DETECTOR_NAMES:
            det_summary, ep_recs = compute_detector_quality(
                schedule[:n_windows], detector_fires[det_name], det_name, seed
            )
            det_summary['tuned'] = True
            det_summary['hyperparameters'] = str(tuned_configs.get(det_name, {}))
            detector_quality_rows.append(det_summary)
            episode_records_all.extend(ep_recs)

        # Seed Summary Print
        sub_df = pd.DataFrame(window_rows)
        print(f"Seed {seed} completed in {time.time() - t0_seed:.1f}s")
        for app in APPROACH_NAMES:
            s_val = sub_df[(sub_df['seed'] == seed) & (sub_df['approach'] == app)]['f1'].mean()
            s_cpu = sub_df[(sub_df['seed'] == seed) & (sub_df['approach'] == app)]['cpu_time'].sum()
            print(f"   {app:40s}: F1 = {s_val:.4f} | CPU = {s_cpu:.2f}s")

    # =========================================================================
    # POST-PROCESSING & ARTIFACT GENERATION
    # =========================================================================
    df_window = pd.DataFrame(window_rows)
    df_retrain = pd.DataFrame(retrain_rows)
    df_drift = pd.DataFrame(drift_rows)
    df_detector_quality = pd.DataFrame(detector_quality_rows)
    df_episodes = pd.DataFrame(episode_records_all)

    df_window.to_csv(os.path.join(output_dir, 'window_results.csv'), index=False)
    df_retrain.to_csv(os.path.join(output_dir, 'retraining_results.csv'), index=False)
    df_drift.to_csv(os.path.join(output_dir, 'drift_results.csv'), index=False)
    df_detector_quality.to_csv(os.path.join(output_dir, 'detector_quality.csv'), index=False)
    df_episodes.to_csv(os.path.join(output_dir, 'detector_episodes.csv'), index=False)

    # 1. Summary Results Table
    summary_list = []
    for app in APPROACH_NAMES:
        sub_w = df_window[df_window['approach'] == app]
        sub_r = df_retrain[df_retrain['approach'] == app]

        seed_f1 = sub_w.groupby('seed')['f1'].mean()
        seed_acc = sub_w.groupby('seed')['accuracy'].mean()
        seed_prec = sub_w.groupby('seed')['precision'].mean()
        seed_rec = sub_w.groupby('seed')['recall'].mean()
        seed_cpu = sub_w.groupby('seed')['cpu_time'].sum()
        seed_footprint_kb = sub_w.groupby('seed')['model_footprint_bytes'].mean() / 1024.0
        seed_inc_ram = sub_w.groupby('seed')['incremental_rss_delta_mb'].mean()
        seed_rss = sub_w.groupby('seed')['process_rss_mb'].mean()
        seed_events = sub_r.groupby('seed')['retrained'].sum()
        seed_samples = sub_r.groupby('seed')['samples_adapted'].sum()

        f1_st = compute_summary_stats(seed_f1)
        acc_st = compute_summary_stats(seed_acc)
        prec_st = compute_summary_stats(seed_prec)
        rec_st = compute_summary_stats(seed_rec)
        cpu_st = compute_summary_stats(seed_cpu)
        fp_st = compute_summary_stats(seed_footprint_kb)
        inc_st = compute_summary_stats(seed_inc_ram)
        rss_st = compute_summary_stats(seed_rss)
        ev_st = compute_summary_stats(seed_events)
        sm_st = compute_summary_stats(seed_samples)

        retrain_freq_pct = (ev_st['mean'] / n_windows) * 100.0

        summary_list.append({
            'Approach': app,
            'Mean F1': f"{f1_st['mean']:.4f} ± {f1_st['std']:.4f}",
            'F1 95% CI': f"[{f1_st['ci_lower']:.4f}, {f1_st['ci_upper']:.4f}]",
            'Mean Accuracy': f"{acc_st['mean']:.4f} ± {acc_st['std']:.4f}",
            'Acc 95% CI': f"[{acc_st['ci_lower']:.4f}, {acc_st['ci_upper']:.4f}]",
            'Mean Precision': f"{prec_st['mean']:.4f} ± {prec_st['std']:.4f}",
            'Mean Recall': f"{rec_st['mean']:.4f} ± {rec_st['std']:.4f}",
            'Cumulative CPU Time (s)': f"{cpu_st['mean']:.2f} ± {cpu_st['std']:.2f}",
            'Model Footprint (KB)': f"{fp_st['mean']:.1f} ± {fp_st['std']:.1f}",
            'Incremental RAM Delta (MB)': f"{inc_st['mean']:.3f} ± {inc_st['std']:.3f}",
            'Process RSS (MB)': f"{rss_st['mean']:.1f} ± {rss_st['std']:.1f}",
            'Retraining Events': f"{ev_st['mean']:.1f} ± {ev_st['std']:.1f}",
            'Retraining Freq (%)': f"{retrain_freq_pct:.1f}%",
            'Total Adapt Samples': f"{int(sm_st['mean']):,}",
            'raw_f1': f1_st['mean'],
            'raw_acc': acc_st['mean'],
            'raw_cpu': cpu_st['mean'],
            'raw_events': ev_st['mean'],
            'raw_footprint_kb': fp_st['mean'],
            'raw_incremental_ram': inc_st['mean'],
            'raw_rss': rss_st['mean'],
        })

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)

    # 2. Performance by Drift Type Table
    drift_type_rows = []
    type_display_map = {'none': 'Stationary (None)', 'covariate': 'Covariate', 'concept': 'Concept', 'mixed': 'Mixed'}
    for dtype_raw, dtype_disp in type_display_map.items():
        row = {'Drift Type': dtype_disp}
        for app in APPROACH_NAMES:
            vals = df_window[(df_window['drift_type'] == dtype_raw) & (df_window['approach'] == app)].groupby('seed')['f1'].mean()
            st = compute_summary_stats(vals)
            row[f'{app} F1'] = f"{st['mean']:.4f} ± {st['std']:.4f}"
        drift_type_rows.append(row)

    df_drift_type = pd.DataFrame(drift_type_rows)
    df_drift_type.to_csv(os.path.join(output_dir, 'drift_type_summary.csv'), index=False)

    # 3. Performance by Severity Table
    sev_rows = []
    sev_map = {'none': 'None (Stationary)', 'mild': 'Mild', 'moderate': 'Moderate', 'severe': 'Severe'}
    for sev_raw, sev_disp in sev_map.items():
        row = {'Severity': sev_disp}
        for app in APPROACH_NAMES:
            vals = df_window[(df_window['severity'] == sev_raw) & (df_window['approach'] == app)].groupby('seed')['f1'].mean()
            st = compute_summary_stats(vals)
            row[f'{app} F1'] = f"{st['mean']:.4f} ± {st['std']:.4f}"
        sev_rows.append(row)

    df_sev = pd.DataFrame(sev_rows)
    df_sev.to_csv(os.path.join(output_dir, 'severity_summary.csv'), index=False)

    # 4. Computational Savings Table across all variants
    c_cpu = df_summary[df_summary['Approach'] == 'Continuously Retrained Ensemble']['raw_cpu'].values[0]
    c_ev = df_summary[df_summary['Approach'] == 'Continuously Retrained Ensemble']['raw_events'].values[0]
    c_f1 = df_summary[df_summary['Approach'] == 'Continuously Retrained Ensemble']['raw_f1'].values[0]

    savings_rows = []
    for app in APPROACH_NAMES:
        row_data = df_summary[df_summary['Approach'] == app]
        app_cpu = row_data['raw_cpu'].values[0]
        app_ev = row_data['raw_events'].values[0]
        app_f1 = row_data['raw_f1'].values[0]

        cpu_sav_pct = 100.0 * (c_cpu - app_cpu) / c_cpu if c_cpu > 0 else 0.0
        ev_red_pct = 100.0 * (c_ev - app_ev) / c_ev if c_ev > 0 else 0.0
        eff = app_f1 / app_cpu if app_cpu > 0 else 0.0

        savings_rows.append({
            'Approach': app,
            'Cumulative CPU Time (s)': f"{app_cpu:.2f}s",
            'CPU Savings (%) vs. Continuous': f"{cpu_sav_pct:.2f}%",
            'Retraining Events': f"{app_ev:.1f}",
            'Retraining Event Reduction (%)': f"{ev_red_pct:.2f}%",
            'Performance-Cost Efficiency (F1 / CPU s)': f"{eff:.4f}",
        })

    df_savings = pd.DataFrame(savings_rows)
    df_savings.to_csv(os.path.join(output_dir, 'computational_savings.csv'), index=False)

    # 5. Statistical Hypothesis Tests (Wilcoxon Signed-Rank + Holm-Bonferroni)
    stat_results = run_paired_tests(df_window, metrics=['f1', 'accuracy', 'cpu_time', 'incremental_rss_delta_mb'])
    df_stats = pd.DataFrame(stat_results)
    df_stats.to_csv(os.path.join(output_dir, 'statistical_results.csv'), index=False)

    # 6. Generate All 17 Diagnostic Figures
    print("\nGenerating all 17 publication figures...")
    generate_all_plots(
        df_window, df_retrain, df_drift,
        df_detector_quality=df_detector_quality,
        df_episodes=df_episodes,
        output_dir=plots_dir
    )
    print("All 17 figures successfully generated!")

    print("\n" + "=" * 85)
    print("EXPERIMENT 4 v3 COMPLETE!")
    print("=" * 85)
    print("\n--- SUMMARY RESULTS TABLE ---")
    print(df_summary[['Approach', 'Mean F1', 'Mean Accuracy', 'Cumulative CPU Time (s)', 'Model Footprint (KB)', 'Retraining Events', 'Retraining Freq (%)']].to_string(index=False))
    print("\n--- DETECTOR QUALITY TABLE ---")
    quality_disp = df_detector_quality.groupby('detector')[['precision', 'recall', 'f1', 'mean_latency', 'window_trigger_rate', 'episode_detection_rate']].mean().reset_index()
    print(quality_disp.to_string(index=False))
    print("\n--- COMPUTATIONAL SAVINGS ---")
    print(df_savings.to_string(index=False))


if __name__ == '__main__':
    run_experiment4()
