# Experiment 9B — Final Results (v3)

Generated 2026-10-01T14:41:10Z

Branch `final-results-20260930`; frozen config `config_frozen_v3.json` sha256 `aa37746a23417cc9dfef10d6cae52fcecf4c5f3a9d732e0dd7081d27a1daeb08`.

Primary metric: **pooled macro-F1** (concatenate predictions over all evaluation windows of a run). Per-window macro-F1 is secondary and always labelled. The two are never mixed in one table.

## 1. Detector-harness provenance (paper Fig. 8 / Table II)

The paper's detector results were produced by `experiments/exp9a`. That harness has two defects, both established from committed files and a rerun:

1. **river return-value bug.** `river` 0.26.1 `ADWIN.update()` and `PageHinkley.update()` return `None`; the paper harness stores the return value as the detection flag, so those two detectors are recorded as never firing regardless of the data.

2. **inflated EDD counts.** The harness records `detected_events` per detector *fire* (114/115/117) while the retrain loop throttles retrains to one per 3 windows (38/39/39). The two columns measure different things.

| dataset | detector | paper_detected_events | paper_retrain_events | paper_macro_f1 | river_return_value_bug | edd_detected_equals_reval_original | edd_retrain_equals_reval_original |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | ADWIN | 0.0000 | 0 | 0.9374 | True | False | False |
| 5g_campus | EDD | 114.0000 | 38 | 0.9867 | True | True | False |
| 5g_campus | Page-Hinkley | 0.0000 | 0 | 0.9374 | True | False | False |
| 5g_campus | EDMA | 0.0000 | 0 | 0.9374 | True | False | False |
| ugr16 | ADWIN | 0.0000 | 0 | 0.9667 | True | False | False |
| ugr16 | EDD | 115.0000 | 39 | 0.9675 | True | True | False |
| ugr16 | Page-Hinkley | 0.0000 | 0 | 0.9667 | True | False | False |
| ugr16 | EDMA | 0.0000 | 0 | 0.9667 | True | False | False |
| nordicdat | ADWIN | 0.0000 | 0 | 0.2567 | True | False | False |
| nordicdat | EDD | 117.0000 | 39 | 0.4063 | True | True | False |
| nordicdat | Page-Hinkley | 0.0000 | 0 | 0.2567 | True | False | False |
| nordicdat | EDMA | 0.0000 | 0 | 0.2567 | True | False | False |

2. **EDMA zero is a wiring artefact, not a detector property.** With the river bug removed and the EWMA read *before* it is updated (the correct pre-update rule, as used by the revalidation harness), EDMA fires 10/1/4/14 times on the window signal (5G Campus / UGR'16 / NordicDat / 5G NR). The published "EDMA never fires" is therefore also a harness artefact.

| dataset | detector | fires_every_fire_rule | retrains_throttle3_rule |
| --- | --- | --- | --- |
| 5g_campus | EDMA | 10 | 10 |
| ugr16 | EDMA | 1 | 1 |
| nordicdat | EDMA | 4 | 4 |
| 5g_nr | EDMA | 14 | 13 |


**ADWIN / Page-Hinkley stay at zero even after the bug is removed.** The window error signal is a single scalar in [0,1] per window over only 143–146 windows; ADWIN's default `delta=0.002` and Page-Hinkley's `threshold=50` are not reached. On the per-sample error signal the same detectors do fire (ADWIN 12 on UGR'16 and 187 on NordicDat; Page-Hinkley 4 and 115). This is a signal-scale artefact, not evidence that no drift occurs.

| dataset | n_windows | err_mean | err_std | err_min | err_max | n_unique_err | adwin_delta | adwin_fires | ph_threshold | ph_min_instances | ph_fires | adwin_sanity_0to1_120pts_fires |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | 143 | 0.0175 | 0.0447 | 0.0000 | 0.2000 | 3 | 0.0020 | 0 | 50 | 30 | 0 | 1 |
| ugr16 | 144 | 0.0097 | 0.0410 | 0.0000 | 0.2833 | 11 | 0.0020 | 0 | 50 | 30 | 0 | 1 |
| nordicdat | 146 | 0.5319 | 0.3838 | 0.0000 | 1.0000 | 99 | 0.0020 | 0 | 50 | 30 | 0 | 1 |
| 5g_nr | 400 | 0.0925 | 0.2897 | 0.0000 | 1.0000 | 2 | 0.0020 | 0 | 50 | 30 | 0 | 1 |


Corrected detector wiring (river bug removed, per-sample signal):

| dataset | detector | fires_window_signal | fires_per_sample_signal |
| --- | --- | --- | --- |
| 5g_campus | ADWIN | 0 | 0 |
| 5g_campus | Page-Hinkley | 0 | 0 |
| 5g_campus | EDD | 114 | 1220 |
| 5g_campus | EDMA | 10 | 17 |
| ugr16 | ADWIN | 0 | 12 |
| ugr16 | Page-Hinkley | 0 | 4 |
| ugr16 | EDD | 115 | 33841 |
| ugr16 | EDMA | 1 | 84 |
| nordicdat | ADWIN | 0 | 187 |
| nordicdat | Page-Hinkley | 0 | 115 |
| nordicdat | EDD | 111 | 72517 |
| nordicdat | EDMA | 4 | 740 |
| 5g_nr | ADWIN | 0 | 0 |
| 5g_nr | Page-Hinkley | 0 | 0 |
| 5g_nr | EDD | 344 | 344 |
| 5g_nr | EDMA | 14 | 14 |

## 2. 5G NR stream structure

| dataset | row_kind | n_rows | packets_per_row | n_packets | window_size_rows | total_windows | prefix_rows | prefix_packets | eval_windows | eval_rows | eval_packets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | sample (1 packet/row) | 1790 | 1 | 1790 | 10 | 179 | 360 | 360 | 143 | 1430 | 1430 |
| ugr16 | sample (1 packet/row) | 43200 | 1 | 43200 | 240 | 180 | 8640 | 8640 | 144 | 34560 | 34560 |
| nordicdat | sample (1 packet/row) | 91000 | 1 | 91000 | 500 | 182 | 18000 | 18000 | 146 | 73000 | 73000 |
| 5g_nr | window (500 packets/row) | 499 | 500 | 249500 | 1 | 499 | 99 | 49500 | 400 | 400 | 200000 |


**One 5G NR row is one 500-packet telemetry window, not one packet.** The stream has 499 rows (249,500 packets); the prefix is 99 rows (49,500 packets). The paper's "143 windows / 1,430 samples" figures are 5G Campus; they do not describe 5G NR. All window-count settings in rows and packets are in `T2_window_settings.csv`.

## 3. Provenance-fixed RAPT variants

The published RAPT stores a checkpoint under the *new* regime key although the buffer it trains on still ends in the *previous* regime. `RAPT_v2`/`RAPT-Enhanced_v2` store only after the new regime's first window labels are known; `*_pure` additionally train the stored policy on the new regime's own rows. Pooled macro-F1, mean over seeds 42–46:

| dataset | method | pooled_macro_f1 | pooled_macro_f1_sd | per_window_macro_f1 | accuracy | precision | recall | adapt_cpu | retrain_events | reuse_events | trees_trained | trees_reused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | Event-Driven | 0.9908 | 0.0003 | 0.9636 | 0.9908 | 0.9637 | 0.9646 | 0.1089 | 1.0000 | 0.0000 | 200.0000 | 0.0000 |
| 5g_campus | Frozen | 0.9822 | 0.0006 | 0.9364 | 0.9822 | 0.9379 | 0.9377 | 0.0841 | 0.0000 | 0.0000 | 100.0000 | 0.0000 |
| 5g_campus | Full Retraining | 0.9951 | 0.0000 | 0.9836 | 0.9951 | 0.9838 | 0.9843 | 1.5038 | 14.0000 | 0.0000 | 1500.0000 | 0.0000 |
| 5g_campus | RAPT | 0.9838 | 0.0003 | 0.9381 | 0.9838 | 0.9393 | 0.9392 | 0.2381 | 2.0000 | 12.0000 | 300.0000 | 1200.0000 |
| 5g_campus | RAPT-Enhanced | 0.9838 | 0.0003 | 0.9381 | 0.9838 | 0.9393 | 0.9392 | 0.2407 | 2.0000 | 12.0000 | 300.0000 | 1200.0000 |
| 5g_campus | RAPT-Enhanced_v2 | 0.9838 | 0.0003 | 0.9381 | 0.9838 | 0.9393 | 0.9392 | 0.3181 | 2.0000 | 12.0000 | 300.0000 | 1200.0000 |
| 5g_campus | RAPT-Enhanced_v2_pure | 0.9639 | 0.0076 | 0.9087 | 0.9641 | 0.9150 | 0.9177 | 0.3008 | 2.0000 | 12.0000 | 300.0000 | 1200.0000 |
| 5g_campus | RAPT_v2 | 0.9838 | 0.0003 | 0.9381 | 0.9838 | 0.9393 | 0.9392 | 0.3172 | 2.0000 | 12.0000 | 300.0000 | 1200.0000 |
| 5g_campus | RAPT_v2_pure | 0.9639 | 0.0076 | 0.9087 | 0.9641 | 0.9150 | 0.9177 | 0.2997 | 2.0000 | 12.0000 | 300.0000 | 1200.0000 |
| 5g_nr | Event-Driven | 0.8923 | 0.0065 | 0.9060 | 0.9060 | 0.9060 | 0.9060 | 0.8887 | 10.0000 | 0.0000 | 1100.0000 | 0.0000 |
| 5g_nr | Frozen | 0.8961 | 0.0099 | 0.9075 | 0.9075 | 0.9075 | 0.9075 | 0.0790 | 0.0000 | 0.0000 | 100.0000 | 0.0000 |
| 5g_nr | Full Retraining | 0.9018 | 0.0054 | 0.9110 | 0.9110 | 0.9110 | 0.9110 | 0.7790 | 8.0000 | 0.0000 | 900.0000 | 0.0000 |
| 5g_nr | RAPT | 0.8930 | 0.0067 | 0.9040 | 0.9040 | 0.9040 | 0.9040 | 0.2672 | 3.0000 | 5.0000 | 400.0000 | 500.0000 |
| 5g_nr | RAPT-Enhanced | 0.8965 | 0.0082 | 0.9085 | 0.9085 | 0.9085 | 0.9085 | 0.7786 | 3.0000 | 5.0000 | 940.0000 | 500.0000 |
| 5g_nr | RAPT-Enhanced_v2 | 0.8925 | 0.0086 | 0.9065 | 0.9065 | 0.9065 | 0.9065 | 0.8663 | 3.0000 | 5.0000 | 940.0000 | 500.0000 |
| 5g_nr | RAPT-Enhanced_v2_pure | 0.8329 | 0.0116 | 0.8720 | 0.8720 | 0.8720 | 0.8720 | 0.9462 | 3.0000 | 5.0000 | 1040.0000 | 500.0000 |
| 5g_nr | RAPT_v2 | 0.8866 | 0.0114 | 0.9010 | 0.9010 | 0.9010 | 0.9010 | 0.3524 | 3.0000 | 5.0000 | 400.0000 | 500.0000 |
| 5g_nr | RAPT_v2_pure | 0.8001 | 0.0259 | 0.8510 | 0.8510 | 0.8510 | 0.8510 | 0.3312 | 3.0000 | 5.0000 | 400.0000 | 500.0000 |
| nordicdat | Event-Driven | 0.4399 | 0.0288 | 0.2547 | 0.4583 | 0.3909 | 0.2274 | 0.2592 | 2.2000 | 0.0000 | 320.0000 | 0.0000 |
| nordicdat | Frozen | 0.3250 | 0.0079 | 0.2779 | 0.4770 | 0.4215 | 0.2487 | 1.1594 | 0.0000 | 0.0000 | 100.0000 | 0.0000 |
| nordicdat | Full Retraining | 0.4806 | 0.0099 | 0.4227 | 0.5694 | 0.4938 | 0.4085 | 3.3780 | 18.0000 | 0.0000 | 1900.0000 | 0.0000 |
| nordicdat | RAPT | 0.4618 | 0.0112 | 0.3818 | 0.6172 | 0.4703 | 0.3582 | 0.2717 | 2.0000 | 16.0000 | 300.0000 | 1600.0000 |
| nordicdat | RAPT-Enhanced | 0.4662 | 0.0172 | 0.3783 | 0.5677 | 0.4600 | 0.3614 | 2.6814 | 2.0000 | 16.0000 | 2120.0000 | 1600.0000 |
| nordicdat | RAPT-Enhanced_v2 | 0.4615 | 0.0167 | 0.3850 | 0.5617 | 0.4692 | 0.3683 | 3.8506 | 2.0000 | 16.0000 | 2120.0000 | 1600.0000 |
| nordicdat | RAPT-Enhanced_v2_pure | 0.4522 | 0.0194 | 0.3528 | 0.5502 | 0.4413 | 0.3341 | 3.7865 | 2.0000 | 16.0000 | 2120.0000 | 1600.0000 |
| nordicdat | RAPT_v2 | 0.4572 | 0.0104 | 0.4000 | 0.6101 | 0.4913 | 0.3766 | 1.4738 | 2.0000 | 16.0000 | 300.0000 | 1600.0000 |
| nordicdat | RAPT_v2_pure | 0.4483 | 0.0070 | 0.3641 | 0.5977 | 0.4600 | 0.3386 | 1.4269 | 2.0000 | 16.0000 | 300.0000 | 1600.0000 |
| ugr16 | Event-Driven | 0.8413 | 0.0170 | 0.8972 | 0.9898 | 0.9042 | 0.8955 | 0.5757 | 3.0000 | 0.0000 | 400.0000 | 0.0000 |
| ugr16 | Frozen | 0.8216 | 0.0177 | 0.9690 | 0.9899 | 0.9795 | 0.9662 | 1.2294 | 0.0000 | 0.0000 | 100.0000 | 0.0000 |
| ugr16 | Full Retraining | 0.7603 | 0.0194 | 0.9595 | 0.9873 | 0.9726 | 0.9580 | 25.0037 | 144.0000 | 0.0000 | 14500.0000 | 0.0000 |
| ugr16 | RAPT | 0.6629 | 0.0750 | 0.8360 | 0.9635 | 0.8527 | 0.8336 | 1.9968 | 11.0000 | 133.0000 | 1200.0000 | 13300.0000 |
| ugr16 | RAPT-Enhanced | 0.8008 | 0.0305 | 0.9276 | 0.9880 | 0.9375 | 0.9251 | 2.7602 | 11.0000 | 133.0000 | 1200.0000 | 13300.0000 |
| ugr16 | RAPT-Enhanced_v2 | 0.7805 | 0.0165 | 0.9157 | 0.9861 | 0.9261 | 0.9128 | 3.9911 | 11.0000 | 133.0000 | 1200.0000 | 13300.0000 |
| ugr16 | RAPT-Enhanced_v2_pure | 0.6292 | 0.0316 | 0.8640 | 0.9706 | 0.8834 | 0.8610 | 2.9029 | 11.0000 | 133.0000 | 1200.0000 | 13300.0000 |
| ugr16 | RAPT_v2 | 0.7805 | 0.0165 | 0.9157 | 0.9861 | 0.9261 | 0.9128 | 3.9867 | 11.0000 | 133.0000 | 1200.0000 | 13300.0000 |
| ugr16 | RAPT_v2_pure | 0.6292 | 0.0316 | 0.8640 | 0.9706 | 0.8834 | 0.8610 | 2.8933 | 11.0000 | 133.0000 | 1200.0000 | 13300.0000 |


The provenance fix changes the 5G NR number materially (RAPT 0.8930 → RAPT_v2 0.8866 → RAPT_v2_pure 0.8001) and the UGR'16 number strongly (RAPT 0.6629 → RAPT_v2 0.7805). The direction is reported, not the better-looking variant.

## 4. Cost-matched Pareto analysis

| dataset | method | pooled_macro_f1 | pooled_macro_f1_sd | adaptation_cpu_sec | per_window_macro_f1 | retrain_events | refresh_events | pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | Event-Driven | 0.9908 | 0.0003 | 0.1075 | 0.9636 | 1.0000 | 0.0000 | True |
| 5g_campus | Event-Driven-Cheap | 0.9906 | 0.0006 | 0.0761 | 0.9624 | 1.0000 | 0.0000 | True |
| 5g_campus | FR-Cheap | 0.9913 | 0.0009 | 0.5388 | 0.9780 | 14.0000 | 0.0000 | True |
| 5g_campus | Frozen | 0.9822 | 0.0006 | 0.0865 | 0.9364 | 0.0000 | 0.0000 | False |
| 5g_campus | Full Retraining | 0.9951 | 0.0000 | 1.5761 | 0.9836 | 14.0000 | 0.0000 | True |
| 5g_campus | Periodic-Cheap-10 | 0.9923 | 0.0013 | 0.5576 | 0.9784 | 14.0000 | 0.0000 | True |
| 5g_campus | Periodic-Cheap-5 | 0.9924 | 0.0013 | 1.1217 | 0.9804 | 28.0000 | 0.0000 | False |
| 5g_campus | Periodic-Cheap-Matched | 0.9924 | 0.0013 | 1.0931 | 0.9804 | 28.0000 | 0.0000 | False |
| 5g_campus | RAPT-Cheap | 0.9944 | 0.0009 | 1.1100 | 0.9817 | 2.0000 | 24.0000 | False |
| 5g_campus | RAPT-Cheap-Floor | 0.9938 | 0.0006 | 0.6409 | 0.9788 | 2.0000 | 10.0000 | True |
| 5g_campus | RAPT-Cheap-Original | 0.9944 | 0.0009 | 1.1023 | 0.9817 | 2.0000 | 22.0000 | True |
| 5g_campus | RAPT-Cheap-WinEq | 0.9945 | 0.0009 | 1.3588 | 0.9813 | 2.0000 | 30.0000 | True |
| 5g_nr | Event-Driven | 0.8923 | 0.0065 | 0.9032 | 0.9060 | 10.0000 | 0.0000 | False |
| 5g_nr | Event-Driven-Cheap | 0.8699 | 0.0092 | 0.4961 | 0.8885 | 11.6000 | 0.0000 | False |
| 5g_nr | FR-Cheap | 0.8764 | 0.0105 | 0.3267 | 0.8935 | 8.0000 | 0.0000 | False |
| 5g_nr | Frozen | 0.8961 | 0.0099 | 0.0872 | 0.9075 | 0.0000 | 0.0000 | True |
| 5g_nr | Full Retraining | 0.9018 | 0.0054 | 0.7957 | 0.9110 | 8.0000 | 0.0000 | True |
| 5g_nr | Periodic-Cheap-10 | 0.8639 | 0.0084 | 1.5130 | 0.8855 | 40.0000 | 0.0000 | False |
| 5g_nr | Periodic-Cheap-5 | 0.8669 | 0.0153 | 3.0323 | 0.8870 | 80.0000 | 0.0000 | False |
| 5g_nr | Periodic-Cheap-Matched | 0.8669 | 0.0153 | 3.0084 | 0.8870 | 80.0000 | 0.0000 | False |
| 5g_nr | RAPT-Cheap | 0.8583 | 0.0153 | 3.2153 | 0.8805 | 3.0000 | 81.0000 | False |
| 5g_nr | RAPT-Cheap-Floor | 0.8818 | 0.0147 | 2.0389 | 0.8950 | 3.0000 | 45.0000 | False |
| 5g_nr | RAPT-Cheap-Original | 0.8614 | 0.0159 | 3.2551 | 0.8830 | 3.0000 | 79.0000 | False |
| 5g_nr | RAPT-Cheap-WinEq | 0.8691 | 0.0073 | 3.3109 | 0.8880 | 3.0000 | 82.0000 | False |
| nordicdat | Event-Driven | 0.4399 | 0.0288 | 0.2628 | 0.2547 | 2.2000 | 0.0000 | True |
| nordicdat | Event-Driven-Cheap | 0.4514 | 0.0278 | 0.5486 | 0.2458 | 2.4000 | 0.0000 | True |
| nordicdat | FR-Cheap | 0.4764 | 0.0095 | 1.1343 | 0.3916 | 18.0000 | 0.0000 | False |
| nordicdat | Frozen | 0.3250 | 0.0079 | 1.1694 | 0.2779 | 0.0000 | 0.0000 | False |
| nordicdat | Full Retraining | 0.4806 | 0.0099 | 3.4640 | 0.4227 | 18.0000 | 0.0000 | False |
| nordicdat | Periodic-Cheap-10 | 0.5037 | 0.0227 | 1.0364 | 0.4577 | 14.0000 | 0.0000 | True |
| nordicdat | Periodic-Cheap-5 | 0.4983 | 0.0138 | 1.5348 | 0.4662 | 29.0000 | 0.0000 | False |
| nordicdat | Periodic-Cheap-Matched | 0.4983 | 0.0138 | 1.6082 | 0.4662 | 29.0000 | 0.0000 | False |
| nordicdat | RAPT-Cheap | 0.4410 | 0.0070 | 2.3176 | 0.4300 | 2.0000 | 24.0000 | False |
| nordicdat | RAPT-Cheap-Floor | 0.5349 | 0.0072 | 4.1490 | 0.5471 | 2.0000 | 66.4000 | True |
| nordicdat | RAPT-Cheap-Original | 0.4283 | 0.0127 | 2.2868 | 0.4100 | 2.0000 | 21.0000 | False |
| nordicdat | RAPT-Cheap-WinEq | 0.4248 | 0.0113 | 2.5116 | 0.4329 | 2.0000 | 30.0000 | False |
| ugr16 | Event-Driven | 0.8413 | 0.0170 | 0.6034 | 0.8972 | 3.0000 | 0.0000 | True |
| ugr16 | Event-Driven-Cheap | 0.6399 | 0.0286 | 0.6603 | 0.8259 | 3.6000 | 0.0000 | False |
| ugr16 | FR-Cheap | 0.5984 | 0.0456 | 6.4333 | 0.8641 | 144.0000 | 0.0000 | False |
| ugr16 | Frozen | 0.8216 | 0.0177 | 1.2408 | 0.9690 | 0.0000 | 0.0000 | False |
| ugr16 | Full Retraining | 0.7603 | 0.0194 | 25.5187 | 0.9595 | 144.0000 | 0.0000 | False |
| ugr16 | Periodic-Cheap-10 | 0.6462 | 0.0823 | 1.0798 | 0.8910 | 14.0000 | 0.0000 | False |
| ugr16 | Periodic-Cheap-5 | 0.6451 | 0.0892 | 1.6268 | 0.8566 | 28.0000 | 0.0000 | False |
| ugr16 | Periodic-Cheap-Matched | 0.6451 | 0.0892 | 1.6328 | 0.8566 | 28.0000 | 0.0000 | False |
| ugr16 | RAPT-Cheap | 0.7805 | 0.0165 | 4.0589 | 0.9157 | 11.0000 | 11.0000 | False |
| ugr16 | RAPT-Cheap-Floor | 0.7571 | 0.0143 | 4.3860 | 0.9125 | 11.0000 | 22.4000 | False |
| ugr16 | RAPT-Cheap-Original | 0.8008 | 0.0305 | 4.0811 | 0.9276 | 11.0000 | 0.0000 | False |
| ugr16 | RAPT-Cheap-WinEq | 0.6878 | 0.0196 | 4.8772 | 0.9025 | 11.0000 | 37.0000 | False |


**Per-stream verdict (RAPT-Cheap vs the reference):**

| dataset | reference | rapt_cheap_f1 | reference_f1 | rapt_cheap_cpu | reference_cpu | beats_on_f1 | beats_on_cpu | beats_on_both | beats_on_neither |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | Periodic-Cheap-5 | 0.9944 | 0.9924 | 1.1100 | 1.1217 | True | True | True | False |
| 5g_campus | FR-Cheap | 0.9944 | 0.9913 | 1.1100 | 0.5388 | True | False | False | False |
| 5g_nr | Periodic-Cheap-5 | 0.8583 | 0.8669 | 3.2153 | 3.0323 | False | False | False | True |
| 5g_nr | FR-Cheap | 0.8583 | 0.8764 | 3.2153 | 0.3267 | False | False | False | True |
| nordicdat | Periodic-Cheap-5 | 0.4410 | 0.4983 | 2.3176 | 1.5348 | False | False | False | True |
| nordicdat | FR-Cheap | 0.4410 | 0.4764 | 2.3176 | 1.1343 | False | False | False | True |
| ugr16 | Periodic-Cheap-5 | 0.7805 | 0.6451 | 4.0589 | 1.6268 | True | False | False | False |
| ugr16 | FR-Cheap | 0.7805 | 0.5984 | 4.0589 | 6.4333 | True | True | True | False |



| claim | streams | beats_periodic_cheap_on_f1_or_cpu | beats_on_both | verdict |
| --- | --- | --- | --- | --- |
| RAPT-Cheap matches Full Retraining accuracy at much lower cost | 4 | 2 | 1 | repository claim not supported |



## 5. Statistics

Block bootstrap (24 day-blocks for UGR'16, regime-visit blocks otherwise):

| dataset | method_a | method_b | n_blocks | mean_diff_per_window | ci_low | ci_high | excludes_zero | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | Frozen | Event-Driven | 14 | -0.0272 | -0.0488 | -0.0075 | True | a<b |
| 5g_campus | Frozen | Full Retraining | 14 | -0.0472 | -0.0743 | -0.0231 | True | a<b |
| 5g_campus | Frozen | RAPT | 14 | -0.0017 | -0.0048 | 0.0000 | False | a<b |
| 5g_campus | Frozen | RAPT-Enhanced | 14 | -0.0017 | -0.0048 | 0.0000 | False | a<b |
| 5g_campus | Event-Driven | Full Retraining | 14 | -0.0200 | -0.0449 | -0.0013 | True | a<b |
| 5g_campus | Event-Driven | RAPT | 14 | 0.0255 | 0.0051 | 0.0481 | True | a>b |
| 5g_campus | Event-Driven | RAPT-Enhanced | 14 | 0.0255 | 0.0051 | 0.0481 | True | a>b |
| 5g_campus | Full Retraining | RAPT | 14 | 0.0455 | 0.0206 | 0.0734 | True | a>b |
| 5g_campus | Full Retraining | RAPT-Enhanced | 14 | 0.0455 | 0.0206 | 0.0734 | True | a>b |
| 5g_campus | RAPT | RAPT-Enhanced | 14 | 0.0000 | 0.0000 | 0.0000 | False | a<b |
| ugr16 | Frozen | Event-Driven | 20 | 0.0718 | 0.0392 | 0.1044 | True | a>b |
| ugr16 | Frozen | Full Retraining | 20 | 0.0095 | -0.0027 | 0.0237 | False | a>b |
| ugr16 | Frozen | RAPT | 20 | 0.1330 | 0.1014 | 0.1670 | True | a>b |
| ugr16 | Frozen | RAPT-Enhanced | 20 | 0.0413 | 0.0178 | 0.0695 | True | a>b |
| ugr16 | Event-Driven | Full Retraining | 20 | -0.0624 | -0.0987 | -0.0254 | True | a<b |
| ugr16 | Event-Driven | RAPT | 20 | 0.0612 | 0.0248 | 0.0990 | True | a>b |
| ugr16 | Event-Driven | RAPT-Enhanced | 20 | -0.0305 | -0.0670 | 0.0070 | False | a<b |
| ugr16 | Full Retraining | RAPT | 20 | 0.1236 | 0.0855 | 0.1624 | True | a>b |
| ugr16 | Full Retraining | RAPT-Enhanced | 20 | 0.0319 | 0.0050 | 0.0612 | True | a>b |
| ugr16 | RAPT | RAPT-Enhanced | 20 | -0.0917 | -0.1219 | -0.0634 | True | a<b |
| nordicdat | Frozen | Event-Driven | 19 | 0.0232 | -0.2106 | 0.1459 | False | a>b |
| nordicdat | Frozen | Full Retraining | 19 | -0.1449 | -0.4316 | 0.0020 | False | a<b |
| nordicdat | Frozen | RAPT | 19 | -0.1039 | -0.3254 | -0.0146 | True | a<b |
| nordicdat | Frozen | RAPT-Enhanced | 19 | -0.1005 | -0.3744 | 0.0113 | False | a<b |
| nordicdat | Event-Driven | Full Retraining | 19 | -0.1680 | -0.3395 | -0.0309 | True | a<b |
| nordicdat | Event-Driven | RAPT | 19 | -0.1271 | -0.2731 | -0.0017 | True | a<b |
| nordicdat | Event-Driven | RAPT-Enhanced | 19 | -0.1236 | -0.2429 | -0.0354 | True | a<b |
| nordicdat | Full Retraining | RAPT | 19 | 0.0409 | -0.0670 | 0.1646 | False | a>b |
| nordicdat | Full Retraining | RAPT-Enhanced | 19 | 0.0444 | -0.0549 | 0.1421 | False | a>b |
| nordicdat | RAPT | RAPT-Enhanced | 19 | 0.0035 | -0.0937 | 0.0676 | False | a>b |
| 5g_nr | Frozen | Event-Driven | 9 | 0.0015 | -0.0413 | 0.0381 | False | a>b |
| 5g_nr | Frozen | Full Retraining | 9 | -0.0035 | -0.0519 | 0.0360 | False | a<b |
| 5g_nr | Frozen | RAPT | 9 | 0.0035 | -0.0020 | 0.0108 | False | a>b |
| 5g_nr | Frozen | RAPT-Enhanced | 9 | -0.0010 | -0.0430 | 0.0316 | False | a<b |
| 5g_nr | Event-Driven | Full Retraining | 9 | -0.0050 | -0.0211 | 0.0127 | False | a<b |
| 5g_nr | Event-Driven | RAPT | 9 | 0.0020 | -0.0369 | 0.0478 | False | a>b |
| 5g_nr | Event-Driven | RAPT-Enhanced | 9 | -0.0025 | -0.0189 | 0.0136 | False | a<b |
| 5g_nr | Full Retraining | RAPT | 9 | 0.0070 | -0.0353 | 0.0589 | False | a>b |
| 5g_nr | Full Retraining | RAPT-Enhanced | 9 | 0.0025 | -0.0188 | 0.0210 | False | a>b |
| 5g_nr | RAPT | RAPT-Enhanced | 9 | -0.0045 | -0.0481 | 0.0292 | False | a<b |


Seed-level exact Wilcoxon (min two-sided p at n=5 is 0.0625):

| dataset | method_a | method_b | seed_mean_diff | wilcoxon_p | cohen_dz | note |
| --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | Frozen | Event-Driven | -0.0085 | 0.0625 | -27.3098 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Frozen | Full Retraining | -0.0129 | 0.0625 | -20.6350 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Frozen | RAPT | -0.0015 | 0.0625 | -4.9346 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Frozen | RAPT-Enhanced | -0.0015 | 0.0625 | -4.9346 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Event-Driven | Full Retraining | -0.0043 | 0.0625 | -13.9264 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Event-Driven | RAPT | 0.0070 | 0.0625 | 50233.2429 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Event-Driven | RAPT-Enhanced | 0.0070 | 0.0625 | 50233.2429 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Full Retraining | RAPT | 0.0113 | 0.0625 | 36.4010 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | Full Retraining | RAPT-Enhanced | 0.0113 | 0.0625 | 36.4010 | min two-sided p at n=5 is 0.0625 |
| 5g_campus | RAPT | RAPT-Enhanced | 0.0000 | 1.0000 |  | min two-sided p at n=5 is 0.0625 |
| ugr16 | Frozen | Event-Driven | -0.0196 | 0.0625 | -1.6724 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Frozen | Full Retraining | 0.0613 | 0.0625 | 2.5100 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Frozen | RAPT | 0.1588 | 0.0625 | 1.7616 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Frozen | RAPT-Enhanced | 0.0208 | 0.4375 | 0.5097 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Event-Driven | Full Retraining | 0.0809 | 0.0625 | 4.7254 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Event-Driven | RAPT | 0.1784 | 0.0625 | 2.0273 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Event-Driven | RAPT-Enhanced | 0.0405 | 0.1250 | 1.1823 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Full Retraining | RAPT | 0.0975 | 0.1250 | 1.1193 | min two-sided p at n=5 is 0.0625 |
| ugr16 | Full Retraining | RAPT-Enhanced | -0.0405 | 0.0625 | -1.0906 | min two-sided p at n=5 is 0.0625 |
| ugr16 | RAPT | RAPT-Enhanced | -0.1379 | 0.0625 | -2.0519 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Frozen | Event-Driven | -0.1148 | 0.0625 | -3.1588 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Frozen | Full Retraining | -0.1556 | 0.0625 | -35.6799 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Frozen | RAPT | -0.1368 | 0.0625 | -21.5578 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Frozen | RAPT-Enhanced | -0.1412 | 0.0625 | -6.4182 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Event-Driven | Full Retraining | -0.0407 | 0.1250 | -1.0756 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Event-Driven | RAPT | -0.0219 | 0.4375 | -0.5610 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Event-Driven | RAPT-Enhanced | -0.0263 | 0.1250 | -1.0764 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Full Retraining | RAPT | 0.0188 | 0.0625 | 7.2125 | min two-sided p at n=5 is 0.0625 |
| nordicdat | Full Retraining | RAPT-Enhanced | 0.0144 | 0.1875 | 0.6947 | min two-sided p at n=5 is 0.0625 |
| nordicdat | RAPT | RAPT-Enhanced | -0.0044 | 0.4375 | -0.2079 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Frozen | Event-Driven | 0.0038 | 0.8125 | 0.3026 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Frozen | Full Retraining | -0.0056 | 0.3125 | -0.5588 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Frozen | RAPT | 0.0031 | 0.6250 | 0.4339 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Frozen | RAPT-Enhanced | -0.0004 | 1.0000 | -0.0397 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Event-Driven | Full Retraining | -0.0094 | 0.0625 | -1.5070 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Event-Driven | RAPT | -0.0007 | 1.0000 | -0.0919 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Event-Driven | RAPT-Enhanced | -0.0042 | 0.6250 | -0.3385 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Full Retraining | RAPT | 0.0088 | 0.0625 | 1.2601 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | Full Retraining | RAPT-Enhanced | 0.0052 | 0.3125 | 0.6200 | min two-sided p at n=5 is 0.0625 |
| 5g_nr | RAPT | RAPT-Enhanced | -0.0035 | 0.4375 | -0.4609 | min two-sided p at n=5 is 0.0625 |


20-seed cheap-config Wilcoxon with Holm correction (raises power above the n=5 floor):

| dataset | comparison | n_seeds | mean_diff | cohen_dz | wilcoxon_p | holm_p | significant_holm_05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | RAPT-Cheap vs FR-Cheap | 20 | 0.0024 | 1.8463 | 0.0001 | 0.0001 | True |
| 5g_campus | RAPT-Cheap vs Periodic-Cheap-5 | 20 | 0.0021 | 1.2416 | 0.0000 | 0.0000 | True |
| 5g_nr | RAPT-Cheap vs FR-Cheap | 20 | -0.0195 | -1.0804 | 0.0002 | 0.0004 | True |
| 5g_nr | RAPT-Cheap vs Periodic-Cheap-5 | 20 | -0.0116 | -0.6346 | 0.0083 | 0.0083 | True |
| nordicdat | RAPT-Cheap vs FR-Cheap | 20 | -0.0401 | -3.0221 | 0.0000 | 0.0000 | True |
| nordicdat | RAPT-Cheap vs Periodic-Cheap-5 | 20 | -0.0588 | -5.0868 | 0.0000 | 0.0000 | True |
| ugr16 | RAPT-Cheap vs FR-Cheap | 20 | 0.1572 | 3.3682 | 0.0000 | 0.0000 | True |
| ugr16 | RAPT-Cheap vs Periodic-Cheap-5 | 20 | 0.0921 | 1.1680 | 0.0000 | 0.0000 | True |


TOST on 5G Campus (primary margin 0.01):

| dataset | method_a | method_b | margin | tost_p | equivalent | primary |
| --- | --- | --- | --- | --- | --- | --- |
| 5g_campus | RAPT | RAPT-Enhanced | 0.0050 | 0.0312 | True | False |
| 5g_campus | RAPT | RAPT-Enhanced | 0.0100 | 0.0312 | True | True |
| 5g_campus | RAPT | RAPT-Enhanced | 0.0200 | 0.0312 | True | False |



## 6. Component ablation

| dataset | variant | delta_f1_vs_base | wilcoxon_p | base_f1 | variant_f1 |
| --- | --- | --- | --- | --- | --- |
| 5g_campus | cheap_refit | 0.0000 | 1.0000 | 0.9838 | 0.9838 |
| 5g_campus | floor_trigger | 0.0101 | 0.0625 | 0.9838 | 0.9938 |
| 5g_campus | gate_on | 0.0110 | 0.0625 | 0.9838 | 0.9948 |
| 5g_campus | novelty_refit_small | 0.0000 | 1.0000 | 0.9838 | 0.9838 |
| 5g_campus | parity_on | 0.0000 | 1.0000 | 0.9838 | 0.9838 |
| 5g_campus | periodic_refresh | 0.0106 | 0.0625 | 0.9838 | 0.9944 |
| 5g_campus | relative_trigger | 0.0052 | 0.0625 | 0.9838 | 0.9890 |
| 5g_campus | tier1_off | 0.0000 | 1.0000 | 0.9838 | 0.9838 |
| ugr16 | cheap_refit | 0.0000 | 1.0000 | 0.7805 | 0.7805 |
| ugr16 | floor_trigger | -0.0234 | 0.0625 | 0.7805 | 0.7571 |
| ugr16 | gate_on | -0.0046 | 0.6250 | 0.7805 | 0.7759 |
| ugr16 | novelty_refit_small | -0.0278 | 0.0625 | 0.7805 | 0.7527 |
| ugr16 | parity_on | 0.0000 | 1.0000 | 0.7805 | 0.7805 |
| ugr16 | periodic_refresh | 0.0000 | 1.0000 | 0.7805 | 0.7805 |
| ugr16 | relative_trigger | -0.0352 | 0.0625 | 0.7805 | 0.7454 |
| ugr16 | tier1_off | -0.0013 | 0.3750 | 0.7805 | 0.7792 |



## 7. Timing

| dataset | method | adapt_cpu_median | fr_adapt_cpu_median | saving_vs_fr_pct |
| --- | --- | --- | --- | --- |
| 5g_campus | Frozen | 0.0870 | 1.5538 | 94.3978 |
| 5g_campus | Event-Driven | 0.1088 | 1.5538 | 92.9990 |
| 5g_campus | Full Retraining | 1.5538 | 1.5538 | 0.0000 |
| 5g_campus | RAPT | 0.2377 | 1.5538 | 84.7018 |
| 5g_campus | RAPT-Enhanced | 0.2400 | 1.5538 | 84.5567 |
| 5g_campus | RAPT_v2 | 0.3278 | 1.5538 | 78.9057 |
| 5g_campus | FR-Cheap | 0.5451 | 1.5538 | 64.9222 |
| 5g_campus | Periodic-Cheap-5 | 1.0480 | 1.5538 | 32.5532 |
| 5g_campus | RAPT-Cheap | 1.1313 | 1.5538 | 27.1923 |
| 5g_nr | Frozen | 0.0780 | 0.7790 | 89.9856 |
| 5g_nr | Event-Driven | 0.9240 | 0.7790 | -18.6164 |
| 5g_nr | Full Retraining | 0.7790 | 0.7790 | 0.0000 |
| 5g_nr | RAPT | 0.2705 | 0.7790 | 65.2744 |
| 5g_nr | RAPT-Enhanced | 0.7464 | 0.7790 | 4.1799 |
| 5g_nr | RAPT_v2 | 0.3523 | 0.7790 | 54.7726 |
| 5g_nr | FR-Cheap | 0.3195 | 0.7790 | 58.9892 |
| 5g_nr | Periodic-Cheap-5 | 2.9292 | 0.7790 | -276.0208 |
| 5g_nr | RAPT-Cheap | 3.1985 | 0.7790 | -310.6002 |



## 7b. Label-delay sensitivity

Delay 0 vs delay 1 (the next-window target is only known one window later). 5G Campus is unchanged at every method; NordicDat degrades sharply for every adapting method; UGR'16 is not applicable.

| dataset | method | pooled_f1_delay0 | pooled_f1_delay1 | delta_f1 | wilcoxon_p |
| --- | --- | --- | --- | --- | --- |
| 5g_campus | Event-Driven | 0.9908 | 0.9908 | 0.0000 | 1.0000 |
| 5g_campus | Frozen | 0.9822 | 0.9822 | 0.0000 | 1.0000 |
| 5g_campus | Full Retraining | 0.9951 | 0.9951 | 0.0000 | 1.0000 |
| 5g_campus | RAPT | 0.9838 | 0.9838 | 0.0000 | 1.0000 |
| 5g_campus | RAPT-Enhanced | 0.9838 | 0.9838 | 0.0000 | 1.0000 |
| 5g_campus | RAPT_v2 | 0.9838 | 0.9838 | 0.0000 | 1.0000 |
| 5g_nr | Event-Driven | 0.8923 | 0.8923 | 0.0000 | 1.0000 |
| 5g_nr | Frozen | 0.8961 | 0.8961 | 0.0000 | 1.0000 |
| 5g_nr | Full Retraining | 0.9018 | 0.9007 | -0.0011 | 0.8750 |
| 5g_nr | RAPT | 0.8930 | 0.8922 | -0.0008 | 0.5000 |
| 5g_nr | RAPT-Enhanced | 0.8965 | 0.8984 | 0.0019 | 1.0000 |
| 5g_nr | RAPT_v2 | 0.8866 | 0.8930 | 0.0064 | 0.3125 |
| nordicdat | Event-Driven | 0.4399 | 0.4399 | 0.0000 | 1.0000 |
| nordicdat | Frozen | 0.3250 | 0.3250 | 0.0000 | 1.0000 |
| nordicdat | Full Retraining | 0.4806 | 0.4205 | -0.0601 | 0.0625 |
| nordicdat | RAPT | 0.4618 | 0.3362 | -0.1256 | 0.0625 |
| nordicdat | RAPT-Enhanced | 0.4662 | 0.4126 | -0.0536 | 0.0625 |
| nordicdat | RAPT_v2 | 0.4572 | 0.4436 | -0.0136 | 0.1250 |



## 8. Paper number audit

Every F1 cell of the manuscript's primary table (Table II) re-derived from a committed CSV:

| stream | model | paper_f1 | harness_scale | harness_f1 | abs_diff | source | matches_within_0.001 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5G Campus | Frozen | 0.9364 | f1_per_window | 0.9364 | 0.0000 | C_metric_reconciliation | True |
| 5G Campus | Event-Driven | 0.9636 | f1_per_window | 0.9636 | 0.0000 | C_metric_reconciliation | True |
| 5G Campus | Full Retraining | 0.9836 | f1_per_window | 0.9836 | 0.0000 | C_metric_reconciliation | True |
| 5G Campus | RAPT | 0.9381 | f1_per_window | 0.9381 | 0.0000 | C_metric_reconciliation | True |
| 5G Campus | RAPT-Enhanced | 0.9381 | f1_per_window | 0.9381 | 0.0000 | C_metric_reconciliation | True |
| UGR'16 | Frozen | 0.9690 | f1_per_window | 0.9690 | 0.0000 | C_metric_reconciliation | True |
| UGR'16 | Event-Driven | 0.8972 | f1_per_window | 0.8972 | 0.0000 | C_metric_reconciliation | True |
| UGR'16 | Full Retraining | 0.9595 | f1_per_window | 0.9595 | 0.0000 | C_metric_reconciliation | True |
| UGR'16 | RAPT | 0.8360 | f1_per_window | 0.8360 | 0.0000 | C_metric_reconciliation | True |
| UGR'16 | RAPT-Enhanced | 0.9276 | f1_per_window | 0.9276 | 0.0000 | C_metric_reconciliation | True |
| NordicDat | Frozen | 0.2779 | f1_per_window | 0.2779 | 0.0000 | C_metric_reconciliation | True |
| NordicDat | Event-Driven | 0.2547 | f1_per_window | 0.2547 | 0.0000 | C_metric_reconciliation | True |
| NordicDat | Full Retraining | 0.4227 | f1_per_window | 0.4227 | 0.0000 | C_metric_reconciliation | True |
| NordicDat | RAPT | 0.3818 | f1_per_window | 0.3818 | 0.0000 | C_metric_reconciliation | True |
| NordicDat | RAPT-Enhanced | 0.3783 | f1_per_window | 0.3783 | 0.0000 | C_metric_reconciliation | True |
| 5G NR Lat. | Frozen | 0.8961 | f1_per_window | 0.8961 | 0.0000 | exp9b per_seed_results | True |
| 5G NR Lat. | Event-Driven | 0.8903 | f1_per_window | 0.8903 | 0.0000 | exp9b per_seed_results | True |
| 5G NR Lat. | Full Retraining | 0.9027 | f1_per_window | 0.9069 | 0.0042 | exp9b per_seed_results | False |
| 5G NR Lat. | RAPT | 0.8894 | f1_per_window | 0.8894 | 0.0000 | exp9b per_seed_results | True |
| 5G NR Lat. | RAPT-Enhanced | 0.8915 | nan |  |  | nan | False |



## 9. Figures

- `fig9b_detector_fixed.png` — generated
- `fig9b_cost_pareto.png` — generated

## 10. Reproducibility

Every number above is read from a CSV written by a committed script in `results/final/v3/` (`t1_*.py` … `t9_report.py`). The configuration is frozen and hashed before evaluation. Raw per-window results are in `results/final/v3/raw/`.
