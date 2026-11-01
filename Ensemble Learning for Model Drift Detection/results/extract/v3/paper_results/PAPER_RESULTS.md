# RAPT paper — results and figures

Paper: *When Is Reuse Safe? Regime-Aware Policy Transfer for Recurring Drift in Network Telemetry* (`paper/main.tex`, identical to `Paper_Final/manuscript.tex`).

Every value below is copied from a committed CSV; the source path is given for each table. `audit/verify_paper_numbers.py` re-derives 205/205 of the manuscript's quantitative claims from these artefacts (0 failures).

Contents of this archive:
- `figures/` — the four figures the manuscript includes, plus the supplementary figures produced by the same committed scripts.
- `tables/` — the committed source tables behind Table II, Table III and the statistics.
- `notes/` — the experiment reports and the figure/verification scripts.

## Experiment 1 — primary comparison (manuscript Table II)

Source: `experiments/exp9a/tables/table9a_main.csv` (Campus / UGR'16 / NordicDat) and `results/experiment_9b/natural_drift/summary.csv` (5G NR). Mean +/- sd over seeds 42-46.

| Dataset | Model | Macro-F1 | Accuracy | Precision | Recall | Adapt CPU | Runtime | Retrains | Reuses |
|---|---|---|---|---|---|---|---|---|---|
| 5G Campus QoS | Frozen | 0.9364 +/- 0.0023 | 0.9822 +/- 0.0006 | 0.9379 +/- 0.0025 | 0.9377 +/- 0.0017 | 0.0000 +/- 0.0000 | 1.2482 +/- 0.0126 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 5G Campus QoS | Event-Driven | 0.9636 +/- 0.0006 | 0.9908 +/- 0.0003 | 0.9637 +/- 0.0008 | 0.9646 +/- 0.0002 | 0.1080 +/- 0.0062 | 1.3926 +/- 0.0187 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 5G Campus QoS | Full Retraining | 0.9836 +/- 0.0013 | 0.9951 +/- 0.0000 | 0.9838 +/- 0.0015 | 0.9843 +/- 0.0012 | 1.4008 +/- 0.0103 | 2.6677 +/- 0.0158 | 14.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 5G Campus QoS | RAPT | 0.9381 +/- 0.0016 | 0.9838 +/- 0.0003 | 0.9393 +/- 0.0017 | 0.9392 +/- 0.0016 | 0.2292 +/- 0.0057 | 1.4931 +/- 0.0181 | 2.0000 +/- 0.0000 | 12.0000 +/- 0.0000 |
| 5G Campus QoS | RAPT-Enhanced | 0.9381 +/- 0.0016 | 0.9838 +/- 0.0003 | 0.9393 +/- 0.0017 | 0.9392 +/- 0.0016 | 0.2337 +/- 0.0147 | 2.1046 +/- 0.0468 | 2.0000 +/- 0.0000 | 12.0000 +/- 0.0000 |
| UGR'16 | Frozen | 0.9690 +/- 0.0029 | 0.9899 +/- 0.0007 | 0.9795 +/- 0.0030 | 0.9662 +/- 0.0028 | 0.0000 +/- 0.0000 | 2.4914 +/- 0.0201 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| UGR'16 | Event-Driven | 0.8972 +/- 0.0277 | 0.9898 +/- 0.0013 | 0.9042 +/- 0.0269 | 0.8955 +/- 0.0280 | 0.5754 +/- 0.0066 | 3.1303 +/- 0.0293 | 3.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| UGR'16 | Full Retraining | 0.9595 +/- 0.0012 | 0.9873 +/- 0.0007 | 0.9726 +/- 0.0014 | 0.9580 +/- 0.0015 | 23.6318 +/- 0.3942 | 26.2627 +/- 0.4072 | 144.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| UGR'16 | RAPT | 0.8360 +/- 0.0262 | 0.9635 +/- 0.0182 | 0.8527 +/- 0.0205 | 0.8336 +/- 0.0294 | 1.9711 +/- 0.0198 | 4.4990 +/- 0.0165 | 11.0000 +/- 0.0000 | 133.0000 +/- 0.0000 |
| UGR'16 | RAPT-Enhanced | 0.9276 +/- 0.0123 | 0.9880 +/- 0.0017 | 0.9375 +/- 0.0117 | 0.9251 +/- 0.0123 | 2.7182 +/- 0.0243 | 5.9329 +/- 0.0465 | 11.0000 +/- 0.0000 | 133.0000 +/- 0.0000 |
| NordicDat | Frozen | 0.2779 +/- 0.0224 | 0.4770 +/- 0.0123 | 0.4215 +/- 0.0226 | 0.2487 +/- 0.0229 | 0.0000 +/- 0.0000 | 2.5488 +/- 0.0128 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| NordicDat | Event-Driven | 0.2547 +/- 0.0177 | 0.4583 +/- 0.0367 | 0.3909 +/- 0.0208 | 0.2274 +/- 0.0151 | 0.2533 +/- 0.0474 | 2.8435 +/- 0.0490 | 2.2000 +/- 0.4472 | 0.0000 +/- 0.0000 |
| NordicDat | Full Retraining | 0.4227 +/- 0.0299 | 0.5694 +/- 0.0075 | 0.4938 +/- 0.0233 | 0.4085 +/- 0.0293 | 2.2034 +/- 0.0100 | 4.7593 +/- 0.0193 | 18.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| NordicDat | RAPT | 0.3818 +/- 0.0184 | 0.6172 +/- 0.0127 | 0.4703 +/- 0.0185 | 0.3582 +/- 0.0192 | 0.2594 +/- 0.0021 | 2.8257 +/- 0.0331 | 2.0000 +/- 0.0000 | 16.0000 +/- 0.0000 |
| NordicDat | RAPT-Enhanced | 0.3783 +/- 0.0321 | 0.5677 +/- 0.0222 | 0.4600 +/- 0.0307 | 0.3614 +/- 0.0318 | 2.6492 +/- 0.2561 | 5.8812 +/- 0.2327 | 2.0000 +/- 0.0000 | 16.0000 +/- 0.0000 |

5G NR latency (natural drift):

| method | Macro-F1 | Accuracy | Adapt CPU | Retrains | Reuses |
|---|---|---|---|---|---|
| Frozen | 0.8961 +/- 0.0099 | 0.9075 +/- 0.0061 | 0.0 | 0.0 | 0.0 |
| Event-Driven | 0.8903 +/- 0.0107 | 0.9055 +/- 0.006 | 1.0406 | 11.0 | 0.0 |
| Full Retraining | 0.9027 +/- 0.0059 | 0.9145 +/- 0.0045 | 0.7636 | 9.0 | 0.0 |
| RAPT | 0.8894 +/- 0.011 | 0.9025 +/- 0.0068 | 0.3955 | 3.0 | 6.0 |
| RAPT-Enhanced | 0.8915 +/- 0.0117 | 0.9055 +/- 0.0076 | 0.8609 | 3.0 | 6.0 |

Figure: `figures/primary_performance.png` (supplementary; the manuscript keeps the numbers in Table II).

## Experiment 2 — computational cost and reuse (Fig. 3)

Adaptation-CPU reduction of RAPT vs Full Retraining, computed from Table II:

| dataset | FR CPU (s) | RAPT CPU (s) | reduction (%) |
|---|---|---|---|
| 5G Campus QoS | 1.4008454945999964 | 0.2292361316000046 | 83.63587330054114 |
| UGR'16 | 23.63180814759998 | 1.9711001212000496 | 91.65912270068834 |
| NordicDat | 2.203364262999992 | 0.2593979052000463 | 88.22718923257551 |
| 5G NR | 0.763613117200007 | 0.3955139520000049 | 48.20492955251172 |

Figure: `figures/computational_cost.png` (adaptation CPU, UGR'16 + NordicDat).

## Experiment 3 — UGR'16 rolling accuracy (Fig. 2)

Source: `results/experiment_9a_three/raw/per_window_ugr16_full.csv`; generator `notes/make_ugr_rolling_figure.py`. Vertical ticks are novelty refits (first appearance of each regime).

Figure: `figures/ugr_rolling_accuracy.png`.

## Experiment 4 — accuracy-cost trade-off (Fig. 4, inline TikZ)

Coordinates are (CPU saved vs Full Retraining %, delta macro-F1) from Table II:

| model | Campus | UGR16 | Nordic | NR |
|---|---|---|---|---|
| RAPT | (83.6, -0.0455) | (91.7, -0.1235) | (88.2, -0.0409) | (48.2, -0.0133) |
| RAPT-Enhanced | (83.3, -0.0455) | (88.5, -0.0319) | (-20.2, -0.0444) | (-12.7, -0.0112) |

## Experiment 5 — efficiency ablation on 5G Campus (Table III, Fig. 5)

Source: `Final_Experiments/results/tables/table_final_main.csv`. The manuscript quotes Full Retrain, RAPT-Full, RAPT-Evidence, RAPT-Floor, RAPT-Cheap and RAPT-Incremental; the full ladder is in the file.

| Model | Macro-F1 | Adapt CPU (s) | Runtime (s) | Retrains | Reuse events | Trees trained | Trees reused |
|---|---|---|---|---|---|---|---|
| Frozen | 0.9367 +/- 0.0016 | 0.0000 +/- 0.0000 | 1.282 +/- 0.027 | 0.0 | 0.0 | 100.0 | 0.0 |
| Event-Driven | 0.9639 +/- 0.0000 | 0.1066 +/- 0.0034 | 1.395 +/- 0.007 | 1.0 | 0.0 | 200.0 | 0.0 |
| Full Retraining | 0.9829 +/- 0.0023 | 1.4063 +/- 0.0221 | 2.696 +/- 0.038 | 14.0 | 0.0 | 1500.0 | 0.0 |
| RAPT_T2 | 0.9381 +/- 0.0016 | 0.2292 +/- 0.0031 | 1.509 +/- 0.028 | 2.0 | 12.0 | 300.0 | 1200.0 |
| RAPT_T1 | 0.9404 +/- 0.0020 | 0.2297 +/- 0.0028 | 1.786 +/- 0.010 | 2.0 | 12.0 | 300.0 | 1200.0 |
| RAPT_T1_REFIT | 0.9404 +/- 0.0020 | 0.2303 +/- 0.0038 | 2.481 +/- 0.024 | 2.0 | 12.0 | 300.0 | 1200.0 |
| RAPT_FULL | 0.9587 +/- 0.0000 | 0.4283 +/- 0.0019 | 2.731 +/- 0.028 | 4.0 | 10.0 | 500.0 | 1000.0 |
| RAPT_REL_REFIT | 0.9587 +/- 0.0000 | 0.4270 +/- 0.0026 | 2.699 +/- 0.012 | 4.0 | 10.0 | 500.0 | 1000.0 |
| RAPT_REFRESH_W5 | 0.9847 +/- 0.0028 | 2.7168 +/- 0.0216 | 4.898 +/- 0.056 | 4.0 | 10.0 | 2700.0 | 1000.0 |
| RAPT_REFRESH_W10 | 0.9774 +/- 0.0004 | 0.9618 +/- 0.0035 | 3.229 +/- 0.022 | 4.0 | 10.0 | 1000.0 | 1000.0 |
| RAPT_EVIDENCE | 0.9587 +/- 0.0000 | 0.4260 +/- 0.0019 | 3.414 +/- 0.013 | 4.0 | 10.0 | 500.0 | 1000.0 |
| RAPT_CHEAP | 0.9851 +/- 0.0029 | 0.8477 +/- 0.0032 | 2.330 +/- 0.015 | 4.0 | 10.0 | 940.0 | 280.0 |
| RAPT_COMBO | 0.9587 +/- 0.0000 | 0.4329 +/- 0.0038 | 3.429 +/- 0.022 | 4.0 | 10.0 | 500.0 | 1000.0 |
| RAPT_FLOOR | 0.9797 +/- 0.0050 | 0.5865 +/- 0.0414 | 2.604 +/- 0.029 | 4.0 | 10.0 | 652.0 | 440.0 |
| RAPT_INCR | 0.9549 +/- 0.0020 | 0.5813 +/- 0.0073 | 2.921 +/- 0.014 | 4.0 | 10.0 | 500.0 | 1000.0 |

Figure: `figures/rapt_ablation.png`.

## Experiment 6 — historical detector comparison (Fig. 6)

Source: `experiments/exp9a/tables/table9a_drift_detectors.csv`.

| Dataset | Method | Macro-F1 | Adapt CPU | Runtime | Adaptation Events |
|---|---|---|---|---|---|
| 5G Campus QoS | ADWIN | 0.9364 +/- 0.0023 | 0.0000 +/- 0.0000 | 1.2604 +/- 0.0218 | 0.0000 +/- 0.0000 |
| 5G Campus QoS | EDD | 0.9847 +/- 0.0019 | 3.9824 +/- 0.0416 | 5.2656 +/- 0.0440 | 38.0000 +/- 0.0000 |
| 5G Campus QoS | Page-Hinkley | 0.9364 +/- 0.0023 | 0.0000 +/- 0.0000 | 1.2603 +/- 0.0138 | 0.0000 +/- 0.0000 |
| 5G Campus QoS | EDMA | 0.9364 +/- 0.0023 | 0.0000 +/- 0.0000 | 1.2665 +/- 0.0148 | 0.0000 +/- 0.0000 |
| 5G Campus QoS | RAPT | 0.9381 +/- 0.0016 | 0.2292 +/- 0.0057 | 1.4931 +/- 0.0181 | 2.0000 +/- 0.0000 |
| UGR'16 | ADWIN | 0.9690 +/- 0.0029 | 0.0000 +/- 0.0000 | 2.5015 +/- 0.0310 | 0.0000 +/- 0.0000 |
| UGR'16 | EDD | 0.9655 +/- 0.0016 | 6.1012 +/- 0.1002 | 8.6511 +/- 0.1096 | 39.0000 +/- 0.0000 |
| UGR'16 | Page-Hinkley | 0.9690 +/- 0.0029 | 0.0000 +/- 0.0000 | 2.5107 +/- 0.0266 | 0.0000 +/- 0.0000 |
| UGR'16 | EDMA | 0.9690 +/- 0.0029 | 0.0000 +/- 0.0000 | 2.5355 +/- 0.0331 | 0.0000 +/- 0.0000 |
| UGR'16 | RAPT | 0.8360 +/- 0.0262 | 1.9711 +/- 0.0198 | 4.4990 +/- 0.0165 | 11.0000 +/- 0.0000 |
| NordicDat | ADWIN | 0.2779 +/- 0.0224 | 0.0000 +/- 0.0000 | 2.5807 +/- 0.0294 | 0.0000 +/- 0.0000 |
| NordicDat | EDD | 0.4184 +/- 0.0165 | 4.6875 +/- 0.0309 | 7.2563 +/- 0.0363 | 39.0000 +/- 0.0000 |
| NordicDat | Page-Hinkley | 0.2779 +/- 0.0224 | 0.0000 +/- 0.0000 | 2.5862 +/- 0.0128 | 0.0000 +/- 0.0000 |
| NordicDat | EDMA | 0.2779 +/- 0.0224 | 0.0000 +/- 0.0000 | 2.5756 +/- 0.0151 | 0.0000 +/- 0.0000 |
| NordicDat | RAPT | 0.3818 +/- 0.0184 | 0.2594 +/- 0.0021 | 2.8257 +/- 0.0331 | 2.0000 +/- 0.0000 |

ADWIN, Page-Hinkley and EDMA never fire on any stream; EDD fires 38 (Campus), 39 (UGR'16), 39 (NordicDat) times. Figure: `figures/detector_comparison.png`.

## Experiment 7 — statistical tests

### 9A window-level Wilcoxon + Cohen's d (Campus / UGR'16 / NordicDat)

| dataset | method_a | method_b | n_windows | mean_diff_f1 | cohen_d | wilcoxon_stat | p_value | ci_low | ci_high | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5G Campus QoS | RAPT | Frozen | 143 | 0.0017311120252296 | 0.0977755440655506 | 0.0 | 0.1797124948789997 | 0.0 | 0.0046448649389825 | False |
| 5G Campus QoS | RAPT | Event-Driven | 143 | -0.0254588033225804 | -0.2348451331691349 | 1.0 | 0.0056648121852525 | -0.0449783543340044 | -0.0096439357855766 | True |
| 5G Campus QoS | RAPT | Full Retraining | 143 | -0.0454865647173339 | -0.3200694003505945 | 0.0 | 0.0002584648985523 | -0.0694848294418132 | -0.0254543280606033 | True |
| 5G Campus QoS | RAPT-Enhanced | Frozen | 143 | 0.0017311120252296 | 0.0977755440655506 | 0.0 | 0.1797124948789997 | 0.0 | 0.0046448649389825 | False |
| 5G Campus QoS | RAPT-Enhanced | Full Retraining | 143 | -0.0454865647173339 | -0.3200694003505945 | 0.0 | 0.0002584648985523 | -0.0694848294418132 | -0.0254543280606033 | True |
| 5G Campus QoS | RAPT-Enhanced | RAPT | 143 | 0.0 | 0.0 | 0.0 |  | 0.0 | 0.0 | False |
| UGR'16 | RAPT | Frozen | 144 | -0.1330259172848156 | -0.7334613922418588 | 34.0 | 3.4042674926036448e-12 | -0.1625556616744827 | -0.1020885620715884 | True |
| UGR'16 | RAPT | Event-Driven | 144 | -0.0611943870555189 | -0.304620435199173 | 1022.0 | 0.0002584863356693 | -0.0916531061298326 | -0.0280659664553718 | True |
| UGR'16 | RAPT | Full Retraining | 144 | -0.1235548405216922 | -0.6735377915001062 | 126.0 | 2.609097143243066e-11 | -0.1531369575146523 | -0.0927539965801903 | True |
| UGR'16 | RAPT-Enhanced | Frozen | 144 | -0.0413479067912171 | -0.3474544356476848 | 75.0 | 4.988928751179327e-05 | -0.0611900446244151 | -0.0230885701795742 | True |
| UGR'16 | RAPT-Enhanced | Full Retraining | 144 | -0.0318768300280937 | -0.2541190372973838 | 178.0 | 0.0030680228526091 | -0.0506499559093041 | -0.01083329680673 | True |
| UGR'16 | RAPT-Enhanced | RAPT | 144 | 0.0916780104935984 | 0.5624572288878424 | 289.5 | 5.414696529584801e-09 | 0.0651176740217888 | 0.1170994597640597 | True |
| NordicDat | RAPT | Frozen | 146 | 0.1039269572981871 | 0.3804326372404317 | 564.0 | 0.0008905016855279 | 0.066819179964304 | 0.1482479727136376 | True |
| NordicDat | RAPT | Event-Driven | 146 | 0.1270892912649248 | 0.357631759280515 | 1875.0 | 0.0001824659765944 | 0.0705086455846682 | 0.1804659565148387 | True |
| NordicDat | RAPT | Full Retraining | 146 | -0.0409241503960447 | -0.1249478660722081 | 2149.0 | 0.2551881622082571 | -0.0953704634570038 | 0.0072447293779824 | False |
| NordicDat | RAPT-Enhanced | Frozen | 146 | 0.1004594741152063 | 0.2700565940572405 | 1888.0 | 0.0093486029930615 | 0.0427054222957064 | 0.153850560150886 | True |
| NordicDat | RAPT-Enhanced | Full Retraining | 146 | -0.0443916335790256 | -0.1288647700400844 | 2320.0 | 0.2388796615398436 | -0.1061375387346065 | 0.0084292339719042 | False |
| NordicDat | RAPT-Enhanced | RAPT | 146 | -0.0034674831829808 | -0.012707492320353 | 2390.0 | 0.6425231921087418 | -0.0475150627186415 | 0.0365152472129202 | False |

### 5G NR window-level Wilcoxon

| comparison | method1 | method2 | mean_accuracy_diff | std_diff | wilcoxon_stat | p_value | cohen_d | significant_p05 |
|---|---|---|---|---|---|---|---|---|
| RAPT vs Event-Driven | RAPT | Event-Driven | -0.003 | 0.2323596360487751 | 99271.5 | 0.5637028616507731 | -0.0129110203949977 | False |
| RAPT vs Frozen | RAPT | Frozen | -0.005 | 0.0835164664424503 | 3987.0 | 0.0075263151664578 | -0.059868433292079 | True |
| RAPT vs Full Retraining | RAPT | Full Retraining | -0.014 | 0.2186412596864611 | 66385.0 | 0.0042667248221761 | -0.0640318301315884 | True |
| Event-Driven vs Frozen | Event-Driven | Frozen | -0.002 | 0.2345122608074565 | 103111.5 | 0.7029175632453667 | -0.0085283387449071 | False |

### Ablation seed-level comparison vs Full Retraining (5G Campus)

| Model | Comparison | mean_delta | ci95_low | ci95_high | cohens_d | wilcoxon_p | n_seeds |
|---|---|---|---|---|---|---|---|
| Frozen | Frozen vs Full Retraining | -0.0462073351749466 | -0.0508246150424237 | -0.0415900553074696 | -12.425927426187409 | 0.0625 | 5 |
| Event-Driven | Event-Driven vs Full Retraining | -0.0190174198271363 | -0.0219233070429292 | -0.0161115326113434 | -8.126002072533453 | 0.0625 | 5 |
| RAPT_T2 | RAPT_T2 vs Full Retraining | -0.04475045871807 | -0.0493677385855469 | -0.040133178850593 | -12.034148911942356 | 0.0625 | 5 |
| RAPT_T1 | RAPT_T1 vs Full Retraining | -0.0425421407202783 | -0.0462620184542601 | -0.0388222629862965 | -14.20020987688372 | 0.0625 | 5 |
| RAPT_T1_REFIT | RAPT_T1_REFIT vs Full Retraining | -0.0425421407202783 | -0.0462620184542601 | -0.0388222629862965 | -14.20020987688372 | 0.0625 | 5 |
| RAPT_FULL | RAPT_FULL vs Full Retraining | -0.0241548262803323 | -0.0270607134961252 | -0.0212489390645394 | -10.321177646590495 | 0.0625 | 5 |
| RAPT_REL_REFIT | RAPT_REL_REFIT vs Full Retraining | -0.0241548262803323 | -0.0270607134961252 | -0.0212489390645394 | -10.321177646590495 | 0.0625 | 5 |
| RAPT_REFRESH_W5 | RAPT_REFRESH_W5 vs Full Retraining | 0.0017632502450316 | -0.0002357686900373 | 0.0037622691801005 | 1.095219415219884 | 0.25 | 5 |
| RAPT_REFRESH_W10 | RAPT_REFRESH_W10 vs Full Retraining | -0.0054519714530756 | -0.0084472103451166 | -0.0024567325610347 | -2.2600924054862657 | 0.0625 | 5 |
| RAPT_EVIDENCE | RAPT_EVIDENCE vs Full Retraining | -0.0241548262803323 | -0.0270607134961252 | -0.0212489390645394 | -10.321177646590495 | 0.0625 | 5 |
| RAPT_CHEAP | RAPT_CHEAP vs Full Retraining | 0.0022087280390121 | -0.0036076726074494 | 0.0080251286854737 | 0.4715112067689102 | 0.3125 | 5 |
| RAPT_COMBO | RAPT_COMBO vs Full Retraining | -0.0241548262803323 | -0.0270607134961252 | -0.0212489390645394 | -10.321177646590495 | 0.0625 | 5 |
| RAPT_FLOOR | RAPT_FLOOR vs Full Retraining | -0.0032393893269859 | -0.0108091790039456 | 0.0043304003499736 | -0.5313533499783475 | 0.375 | 5 |
| RAPT_INCR | RAPT_INCR vs Full Retraining | -0.0279492110555783 | -0.0305171373137788 | -0.0253812847973778 | -13.514223406956118 | 0.0625 | 5 |

## Experiment 8 — refit cost drivers (micro-benchmarks)

Source: `notes/FINAL_EXPERIMENTS_REPORT.md` section 4.

| Factor | Setting | Cost per refit |
|---|---|---|
| n_estimators | 20 trees (10 RF + 10 ET) | 22 ms |
| | 50 trees | 54 ms |
| | 100 trees (full) | 104 ms |
| Buffer size | 200 samples | 80 ms |
| | 500 | 92 ms |
| | 1000 (full) | 104 ms |

## Experiment 9 — independent re-validation

Source: `results/revalidation/` (A1 reproduction, A7 tests, A8 label delay). `tables/revalidation_A1_summary.csv` reproduces the Table II predictive metrics; the adaptation CPU re-run shifts by up to 0.14 s but the ordering and savings are stable.

`notes/verify_paper_numbers.py` re-derives every claim from the committed artefacts: 205 checks, 0 failures.

