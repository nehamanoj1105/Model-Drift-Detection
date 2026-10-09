# RAPT Final-2 Report — fast job (2026-10-09)

**One-file deliverable.** Every number below is read from `results/final2/` files produced by code in this repo; nothing is hand-typed. The paper (`paper/main.tex`) was **not** modified.

## Contents

1. Protocol & frozen config
2. Main results (12 methods x 4 streams, 30 seeds)
3. Cost tables
4. Statistics (Wilcoxon+Holm, Cohen's dz, TOST)
5. RAPT-ED-Guard gate accuracy
6. Long vs short window gate
7. Detector dev sweep
8. Concept-drift stress panel
9. Class-weight / leakage diagnosis
10. What changed / got worse / claims softened
11. How to reproduce


## 1. Protocol & frozen config

- Tuning: 5 seeds (42-46), first 60% of post-prefix windows, dev only.

- Final: all 30 seeds (42-71), full streams, run once with frozen configs.


`configs/final.yaml`:

```yaml
adwin_delta: 0.2
anchor_lookback: 75
anchor_share_prefix: 0.5
cheap_buffer: 300
cheap_n_trees: 20
detector_min_retrain_interval: 3
ed_refit_interval: 3
edd_drift: 3
edd_warning: 2
edma_alpha: 0.3
edma_k: 1.5
enable_floor: false
floor_acc_threshold: 0.6
floor_min_interval: 2
floor_window: 3
guard_lookback: 100
guard_margin: 0
guard_min_rows: 30
periodic_buffer: 300
periodic_interval: 5
periodic_n_trees: 20
ph_alpha: 0.9999
ph_delta: 0.005
ph_min_instances: 10
ph_threshold: 0.002

```


`configs/final2.yaml`:

```yaml
adwin_delta: 0.2
anchor_lookback: 75
anchor_share_prefix: 0.5
cheap_buffer: 300
cheap_n_trees: 20
detector_min_retrain_interval: 3
ed_refit_interval: 3
edd_drift: 3
edd_warning: 2
edguard_lookback: 50
edguard_margin: 0.05
edguard_min_rows: 30
edma_alpha: 0.3
edma_k: 1.5
enable_floor: false
error_threshold_k: 2.0
error_window_size: 20
floor_acc_threshold: 0.6
floor_min_interval: 2
floor_window: 3
guard_lookback: 100
guard_margin: 0
guard_min_rows: 30
periodic_buffer: 300
periodic_interval: 5
periodic_n_trees: 20
ph_alpha: 0.9999
ph_delta: 0.005
ph_min_instances: 10
ph_threshold: 0.002

```


## 2. Main results (30 seeds, full stream)


### Macro-F1

| method | 5g_campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 0.9371 | 0.9693 | 0.9110 | 0.6797 |
| Frozen+Periodic | 0.9778 | 0.8367 | 0.8839 | 0.8928 |
| Event-Driven | 0.9638 | 0.8978 | 0.9081 | 0.7147 |
| Event-Driven+Reuse | 0.9638 | 0.9639 | 0.9001 | 0.7160 |
| ADWIN | 0.9371 | 0.9498 | 0.9114 | 0.8924 |
| Page-Hinkley | 0.9843 | 0.9598 | 0.9045 | 0.8861 |
| Full Retraining | 0.9836 | 0.9511 | 0.9121 | 0.9060 |
| RAPT | 0.9385 | 0.9137 | 0.8969 | 0.6797 |
| RAPT-Guard | 0.9391 | 0.9357 | 0.9148 | 0.6797 |
| RAPT-Enhanced | 0.9385 | 0.9137 | 0.9045 | 0.6797 |
| RAPT-Anchor | 0.9385 | 0.9247 | 0.9042 | 0.6797 |
| RAPT-ED-Guard | 0.9838 | 0.9522 | 0.9048 | 0.7819 |


## 3. Cost tables (mean CPU seconds / counts per seed)


### adaptation_cpu_sec

| method | 5g_campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Frozen+Periodic | 1.6305 | 1.6981 | 4.7298 | 1.8740 |
| Event-Driven | 0.1678 | 0.9313 | 1.5886 | 0.8118 |
| Event-Driven+Reuse | 0.1678 | 0.6446 | 0.2275 | 0.6684 |
| ADWIN | 0.0000 | 2.2674 | 0.0091 | 1.7646 |
| Page-Hinkley | 0.9513 | 3.4248 | 2.8498 | 2.6829 |
| Full Retraining | 2.2295 | 33.1000 | 1.1511 | 26.6097 |
| RAPT | 0.2896 | 2.8954 | 0.4149 | 0.6294 |
| RAPT-Guard | 0.5226 | 6.6283 | 0.6857 | 3.5499 |
| RAPT-Enhanced | 0.2886 | 2.9067 | 1.1564 | 0.6352 |
| RAPT-Anchor | 0.2881 | 2.8995 | 0.4186 | 0.6355 |
| RAPT-ED-Guard | 2.8059 | 37.3975 | 2.5617 | 26.3742 |


### prediction_cpu_sec

| method | 5g_campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 1.2658 | 1.2570 | 3.5063 | 1.5017 |
| Frozen+Periodic | 0.6041 | 0.6172 | 1.6347 | 0.7080 |
| Event-Driven | 1.2623 | 1.2879 | 3.5120 | 1.5263 |
| Event-Driven+Reuse | 1.2575 | 1.2824 | 3.5255 | 1.5237 |
| ADWIN | 1.2735 | 1.2631 | 3.5329 | 1.5065 |
| Page-Hinkley | 1.2475 | 1.2631 | 3.5317 | 1.4842 |
| Full Retraining | 1.2407 | 1.2777 | 3.5302 | 1.5191 |
| RAPT | 1.2561 | 1.2700 | 3.5245 | 1.4866 |
| RAPT-Guard | 1.2654 | 1.2799 | 3.5437 | 1.5035 |
| RAPT-Enhanced | 1.2569 | 1.3045 | 3.5460 | 1.5010 |
| RAPT-Anchor | 3.8365 | 3.6648 | 10.9957 | 4.4933 |
| RAPT-ED-Guard | 1.2352 | 1.2659 | 3.5108 | 1.5108 |


### total_cpu_sec

| method | 5g_campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 1.2658 | 1.2570 | 3.5063 | 1.5017 |
| Frozen+Periodic | 2.2346 | 2.3153 | 6.3645 | 2.5820 |
| Event-Driven | 1.4302 | 2.2192 | 5.1006 | 2.3380 |
| Event-Driven+Reuse | 1.4253 | 1.9269 | 3.7530 | 2.1922 |
| ADWIN | 1.2735 | 3.5305 | 3.5420 | 3.2711 |
| Page-Hinkley | 2.1988 | 4.6879 | 6.3815 | 4.1670 |
| Full Retraining | 3.4702 | 34.3777 | 4.6813 | 28.1288 |
| RAPT | 1.5457 | 4.1654 | 3.9394 | 2.1160 |
| RAPT-Guard | 1.7880 | 7.9081 | 4.2293 | 5.0534 |
| RAPT-Enhanced | 1.5455 | 4.2112 | 4.7024 | 2.1362 |
| RAPT-Anchor | 4.1246 | 6.5643 | 11.4143 | 5.1288 |
| RAPT-ED-Guard | 4.0411 | 38.6634 | 6.0725 | 27.8851 |


### retrains

| method | 5g_campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Frozen+Periodic | 28.0000 | 28.0000 | 80.0000 | 36.0000 |
| Event-Driven | 1.0000 | 3.5667 | 10.8667 | 4.8667 |
| Event-Driven+Reuse | 1.0000 | 2.4667 | 1.6333 | 4.0000 |
| ADWIN | 0.0000 | 8.7000 | 0.0667 | 10.8667 |
| Page-Hinkley | 6.0000 | 13.2667 | 19.6000 | 16.7000 |
| Full Retraining | 14.0000 | 144.0000 | 8.0000 | 178.0000 |
| RAPT | 2.0000 | 11.0000 | 3.0000 | 5.0000 |
| RAPT-Guard | 2.0667 | 17.3667 | 4.2000 | 5.0333 |
| RAPT-Enhanced | 2.0000 | 11.0000 | 8.0333 | 5.0000 |
| RAPT-Anchor | 2.0000 | 11.0000 | 3.0000 | 5.0000 |
| RAPT-ED-Guard | 14.8333 | 142.5667 | 14.1667 | 137.6000 |


### reuses

| method | 5g_campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Frozen+Periodic | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Event-Driven | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Event-Driven+Reuse | 0.0000 | 0.9333 | 6.3333 | 1.0000 |
| ADWIN | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Page-Hinkley | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Full Retraining | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RAPT | 12.0000 | 133.0000 | 5.0000 | 173.0000 |
| RAPT-Guard | 11.9333 | 126.6333 | 3.8000 | 172.9667 |
| RAPT-Enhanced | 12.0000 | 133.0000 | 5.0000 | 173.0000 |
| RAPT-Anchor | 12.0000 | 133.0000 | 5.0000 | 173.0000 |
| RAPT-ED-Guard | 0.0000 | 1.4333 | 3.5000 | 40.4000 |


## 4. Statistics (window Wilcoxon + Holm; seed Cohen's dz; TOST +/-0.02)

| stream | comparison | window_mean_diff | window_p_holm | seed_dz | seed_t_p | tost_p | tost_equiv | ci95_low | ci95_high |
|---|---|---|---|---|---|---|---|---|---|
| 5g_campus | RAPT-Guard vs Frozen | 0.0020 | 0.0000 | 1.0275 | 0.0000 | 0.0000 | True | 0.0013 | 0.0027 |
| 5g_campus | RAPT-Guard vs Event-Driven | -0.0247 | 0.0000 | -8.1491 | 0.0000 | 1.0000 | False | -0.0259 | -0.0236 |
| 5g_campus | RAPT vs Frozen | 0.0014 | 0.0000 | 3.6106 | 0.0000 | 0.0000 | True | 0.0013 | 0.0016 |
| 5g_campus | RAPT vs Event-Driven | -0.0253 | 0.0000 | -12.9658 | 0.0000 | 1.0000 | False | -0.0260 | -0.0246 |
| 5g_campus | RAPT-ED-Guard vs Frozen | 0.0467 | 0.0000 | 13.9775 | 0.0000 | 1.0000 | False | 0.0454 | 0.0479 |
| 5g_campus | RAPT-ED-Guard vs Event-Driven | 0.0199 | 0.0000 | 8.9255 | 0.0000 | 0.4412 | False | 0.0191 | 0.0208 |
| 5g_campus | Event-Driven+Reuse vs Event-Driven | 0.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| 5g_campus | Frozen+Periodic vs Frozen | 0.0407 | 0.0000 | 8.8063 | 0.0000 | 1.0000 | False | 0.0390 | 0.0425 |
| 5g_campus | Full Retraining vs Frozen | 0.0466 | 0.0000 | 12.6422 | 0.0000 | 1.0000 | False | 0.0452 | 0.0479 |
| 5g_campus | RAPT vs Full Retraining | -0.0451 | 0.0000 | -12.4805 | 0.0000 | 1.0000 | False | -0.0465 | -0.0438 |
| 5g_campus | RAPT-Guard vs Full Retraining | -0.0445 | 0.0000 | -10.1527 | 0.0000 | 1.0000 | False | -0.0462 | -0.0429 |
| 5g_campus | ADWIN vs Frozen | 0.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| 5g_campus | Page-Hinkley vs Frozen | 0.0472 | 0.0000 | 18.5605 | 0.0000 | 1.0000 | False | 0.0463 | 0.0482 |
| ugr16 | RAPT-Guard vs Frozen | -0.0336 | 0.0000 | -2.4291 | 0.0000 | 1.0000 | False | -0.0388 | -0.0284 |
| ugr16 | RAPT-Guard vs Event-Driven | 0.0380 | 0.0000 | 1.2536 | 0.0000 | 0.9985 | False | 0.0266 | 0.0493 |
| ugr16 | RAPT vs Frozen | -0.0556 | 0.0000 | -3.6063 | 0.0000 | 1.0000 | False | -0.0614 | -0.0499 |
| ugr16 | RAPT vs Event-Driven | 0.0159 | 0.7238 | 0.4980 | 0.0107 | 0.2436 | False | 0.0040 | 0.0278 |
| ugr16 | RAPT-ED-Guard vs Frozen | -0.0171 | 0.0000 | -1.5317 | 0.0000 | 0.0862 | False | -0.0213 | -0.0130 |
| ugr16 | RAPT-ED-Guard vs Event-Driven | 0.0544 | 0.0000 | 1.7680 | 0.0000 | 1.0000 | False | 0.0429 | 0.0659 |
| ugr16 | Event-Driven+Reuse vs Event-Driven | 0.0661 | 0.0000 | 1.9862 | 0.0000 | 1.0000 | False | 0.0537 | 0.0785 |
| ugr16 | Frozen+Periodic vs Frozen | -0.1326 | 0.0000 | -3.2465 | 0.0000 | 1.0000 | False | -0.1479 | -0.1174 |
| ugr16 | Full Retraining vs Frozen | -0.0182 | 0.0000 | -2.2847 | 0.0000 | 0.1120 | False | -0.0212 | -0.0152 |
| ugr16 | RAPT vs Full Retraining | -0.0375 | 0.0000 | -3.0236 | 0.0000 | 1.0000 | False | -0.0421 | -0.0328 |
| ugr16 | RAPT-Guard vs Full Retraining | -0.0154 | 0.0010 | -1.3676 | 0.0000 | 0.0166 | True | -0.0196 | -0.0112 |
| ugr16 | ADWIN vs Frozen | -0.0195 | 0.7238 | -1.1410 | 0.0000 | 0.4408 | False | -0.0259 | -0.0131 |
| ugr16 | Page-Hinkley vs Frozen | -0.0096 | 0.6960 | -1.7584 | 0.0000 | 0.0000 | True | -0.0116 | -0.0075 |
| 5g_nr | RAPT-Guard vs Frozen | 0.0038 | 0.1102 | 0.1340 | 0.4687 | 0.0018 | True | -0.0067 | 0.0142 |
| 5g_nr | RAPT-Guard vs Event-Driven | 0.0067 | 0.0046 | 0.2323 | 0.2133 | 0.0084 | True | -0.0041 | 0.0174 |
| 5g_nr | RAPT vs Frozen | -0.0141 | 0.0000 | -0.5017 | 0.0102 | 0.1309 | False | -0.0246 | -0.0036 |
| 5g_nr | RAPT vs Event-Driven | -0.0112 | 0.0000 | -0.4032 | 0.0353 | 0.0465 | True | -0.0216 | -0.0008 |
| 5g_nr | RAPT-ED-Guard vs Frozen | -0.0062 | 0.0149 | -0.6380 | 0.0015 | 0.0000 | True | -0.0098 | -0.0026 |
| 5g_nr | RAPT-ED-Guard vs Event-Driven | -0.0033 | 0.0769 | -0.3840 | 0.0442 | 0.0000 | True | -0.0064 | -0.0001 |
| 5g_nr | Event-Driven+Reuse vs Event-Driven | -0.0080 | 0.0001 | -0.8461 | 0.0001 | 0.0000 | True | -0.0116 | -0.0045 |
| 5g_nr | Frozen+Periodic vs Frozen | -0.0272 | 0.0000 | -2.4782 | 0.0000 | 0.9994 | False | -0.0312 | -0.0231 |
| 5g_nr | Full Retraining vs Frozen | 0.0011 | 1.0000 | 0.1148 | 0.5345 | 0.0000 | True | -0.0024 | 0.0046 |
| 5g_nr | RAPT vs Full Retraining | -0.0152 | 0.0000 | -0.5623 | 0.0045 | 0.1697 | False | -0.0253 | -0.0051 |
| 5g_nr | RAPT-Guard vs Full Retraining | 0.0027 | 0.4374 | 0.0988 | 0.5924 | 0.0007 | True | -0.0074 | 0.0128 |
| 5g_nr | ADWIN vs Frozen | 0.0003 | 1.0000 | 0.2334 | 0.2113 | 0.0000 | True | -0.0002 | 0.0009 |
| 5g_nr | Page-Hinkley vs Frozen | -0.0065 | 0.0091 | -0.7862 | 0.0002 | 0.0000 | True | -0.0096 | -0.0034 |
| kpiops | RAPT-Guard vs Frozen | 0.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| kpiops | RAPT-Guard vs Event-Driven | -0.0351 | 0.0000 | -5.8773 | 0.0000 | 1.0000 | False | -0.0373 | -0.0328 |
| kpiops | RAPT vs Frozen | 0.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| kpiops | RAPT vs Event-Driven | -0.0351 | 0.0000 | -5.8773 | 0.0000 | 1.0000 | False | -0.0373 | -0.0328 |
| kpiops | RAPT-ED-Guard vs Frozen | 0.1022 | 0.0000 | 5.3876 | 0.0000 | 1.0000 | False | 0.0951 | 0.1093 |
| kpiops | RAPT-ED-Guard vs Event-Driven | 0.0671 | 0.0000 | 3.5323 | 0.0000 | 1.0000 | False | 0.0600 | 0.0742 |
| kpiops | Event-Driven+Reuse vs Event-Driven | 0.0012 | 0.0008 | 0.5576 | 0.0048 | 0.0000 | True | 0.0004 | 0.0020 |
| kpiops | Frozen+Periodic vs Frozen | 0.2131 | 0.0000 | 49.5478 | 0.0000 | 1.0000 | False | 0.2115 | 0.2147 |
| kpiops | Full Retraining vs Frozen | 0.2263 | 0.0000 | 73.1152 | 0.0000 | 1.0000 | False | 0.2251 | 0.2274 |
| kpiops | RAPT vs Full Retraining | -0.2263 | 0.0000 | -73.1152 | 0.0000 | 1.0000 | False | -0.2274 | -0.2251 |
| kpiops | RAPT-Guard vs Full Retraining | -0.2263 | 0.0000 | -73.1152 | 0.0000 | 1.0000 | False | -0.2274 | -0.2251 |
| kpiops | ADWIN vs Frozen | 0.2127 | 0.0000 | 37.1365 | 0.0000 | 1.0000 | False | 0.2105 | 0.2148 |
| kpiops | Page-Hinkley vs Frozen | 0.2064 | 0.0000 | 52.5948 | 0.0000 | 1.0000 | False | 0.2050 | 0.2079 |


## 5. RAPT-ED-Guard gate accuracy

| dataset | returns | allows | blocks | bad_allows | bad_blocks | bad_allow_rate | missed_reuse |
|---|---|---|---|---|---|---|---|
| 5G Campus QoS | 12.8333 | 0.0000 | 12.8333 | 0.0000 | 0.0000 | nan | 0.0000 |
| 5G NR Latency | 14.6667 | 3.5000 | 11.1667 | 1.4000 | 0.0000 | 0.4000 | 0.0000 |
| Network operator KPIs | 173.0000 | 40.4000 | 132.6000 | 0.6333 | 1.1000 | 0.0160 | 1.1000 |
| UGR'16 | 133.0000 | 1.4333 | 131.5667 | 0.0000 | 1.3000 | 0.0000 | 1.3000 |


Overall bad-allow rate: 0.045; missed reuses: 2 of 288 blocks.


## 6. Long-window vs short-window gate

| stream | policy | macro_f1 | refits |
|---|---|---|---|
| 5g_nr | long/every-window (RAPT-ED-Guard) | 0.8946 | 38 |
| 5g_nr | short/boundary (RAPT) | 0.9088 | 15 |
| campus | long/every-window (RAPT-ED-Guard) | 0.9728 | 54 |
| campus | short/boundary (RAPT) | 0.9277 | 10 |
| concept_drift | long/every-window (RAPT-ED-Guard) | 0.8980 | 8 |
| concept_drift | short/boundary (RAPT) | 0.8902 | 0 |
| kpiops | long/every-window (RAPT-ED-Guard) | 0.7069 | 337 |
| kpiops | short/boundary (RAPT) | 0.5089 | 25 |
| ugr16 | long/every-window (RAPT-ED-Guard) | 0.9358 | 421 |
| ugr16 | short/boundary (RAPT) | 0.9227 | 55 |


## 7. Detector dev sweep (ADWIN delta / Page-Hinkley threshold)

| stream | method | retrain_min | retrain_mean | retrain_max |
|---|---|---|---|---|
| 5g_nr | ADWIN | 0 | 0.2222 | 1 |
| 5g_nr | Page-Hinkley | 10 | 11.0000 | 12 |
| campus | ADWIN | 0 | 0.0000 | 0 |
| campus | Page-Hinkley | 5 | 6.0000 | 7 |
| kpiops | ADWIN | 6 | 6.8889 | 9 |
| kpiops | Page-Hinkley | 8 | 8.3333 | 9 |
| ugr16 | ADWIN | 6 | 7.5556 | 9 |
| ugr16 | Page-Hinkley | 13 | 13.0000 | 13 |


Frozen: `adwin_delta=0.2`, `ph_threshold=0.002`.


## 8. Concept-drift stress panel (floor trigger enabled)

| severity | method | macro_f1 | retrains |
|---|---|---|---|
| 0.2000 | ADWIN | 0.9608 | 0 |
| 0.2000 | Event-Driven | 0.9608 | 0 |
| 0.2000 | Frozen | 0.9608 | 0 |
| 0.2000 | Full Retraining | 0.9608 | 0 |
| 0.2000 | Page-Hinkley | 0.9608 | 0 |
| 0.2000 | RAPT | 0.9608 | 0 |
| 0.2000 | RAPT-ED-Guard | 0.9686 | 5 |
| 0.2000 | RAPT-Guard | 0.9608 | 0 |
| 0.5000 | ADWIN | 0.8902 | 0 |
| 0.5000 | Event-Driven | 0.8941 | 9 |
| 0.5000 | Frozen | 0.8941 | 9 |
| 0.5000 | Full Retraining | 0.8941 | 9 |
| 0.5000 | Page-Hinkley | 0.8902 | 0 |
| 0.5000 | RAPT | 0.8941 | 7 |
| 0.5000 | RAPT-ED-Guard | 0.9020 | 9 |
| 0.5000 | RAPT-Guard | 0.8902 | 0 |
| 1.0000 | ADWIN | 0.8627 | 0 |
| 1.0000 | Event-Driven | 0.8627 | 10 |
| 1.0000 | Frozen | 0.8627 | 10 |
| 1.0000 | Full Retraining | 0.8627 | 10 |
| 1.0000 | Page-Hinkley | 0.8627 | 0 |
| 1.0000 | RAPT | 0.8627 | 10 |
| 1.0000 | RAPT-ED-Guard | 0.8627 | 10 |
| 1.0000 | RAPT-Guard | 0.8627 | 0 |


### Trajectory spread (mean max-min macro-F1 across methods)

| severity | spread |
|---|---|
| 0.2000 | 0.0078 |
| 0.5000 | 0.0196 |
| 1.0000 | 0.0000 |


Spread stays <=0.02: the panel is **degenerate even after the fix**; recommend dropping it from the paper.


## 9. Class-weight / leakage diagnosis (all streams)

| stream | check | value | detail |
|---|---|---|---|
| campus | class_balance_full | 0.3307 | 0=0.331, 1=0.331, 2=0.338 |
| campus | class_balance_prefix | 0.3250 | 0=0.325, 1=0.325, 2=0.350 |
| campus | max_abs_feature_target_corr | 0.9424 | next_p90_delay=0.942, mean_delay=0.848, median_delay=0.842 |
| campus | frozen_macro_f1_class_weight=None | 0.9357 | std=0.0024, 3 dev seeds |
| campus | frozen_macro_f1_class_weight=balanced | 0.9379 | std=0.0004, 3 dev seeds |
| ugr16 | class_balance_full | 0.0316 | 0=0.968, 1=0.032 |
| ugr16 | class_balance_prefix | 0.0806 | 0=0.919, 1=0.081 |
| ugr16 | max_abs_feature_target_corr | 0.6344 | dportirc=0.634, sportirc=0.434, tcpflagsRST=0.320 |
| ugr16 | frozen_macro_f1_class_weight=None | 0.9681 | std=0.0012, 3 dev seeds |
| ugr16 | frozen_macro_f1_class_weight=balanced | 0.9265 | std=0.0207, 3 dev seeds |
| 5g_nr | class_balance_full | 0.1984 | 0=0.543, 1=0.259, 2=0.198 |
| 5g_nr | class_balance_prefix | 0.3300 | 0=0.340, 1=0.330, 2=0.330 |
| 5g_nr | max_abs_feature_target_corr | 0.8267 | p95_latency=0.827, median_latency=0.826, p90_latency=0.826 |
| 5g_nr | frozen_macro_f1_class_weight=None | 0.9089 | std=0.0063, 3 dev seeds |
| 5g_nr | frozen_macro_f1_class_weight=balanced | 0.9114 | std=0.0012, 3 dev seeds |
| kpiops | class_balance_full | 0.2387 | 0=0.761, 1=0.239 |
| kpiops | class_balance_prefix | 1.0000 | 0=1.000 |
| kpiops | max_abs_feature_target_corr | 0.8375 | kpi_r12=0.837, kpi_r3=0.074, kpi_r2=0.058 |
| kpiops | frozen_macro_f1_class_weight=None | 0.6797 | std=0.0000, 3 dev seeds |
| kpiops | frozen_macro_f1_class_weight=balanced | 0.6797 | std=0.0000, 3 dev seeds |


## 10. What changed, what got worse, claims softened

Scope: **experiments only**. No paper file (`paper/main.tex`, `paper/tables/*`)
was edited. All new numbers live under `results/final2/` and are reviewed in one
place in `results/final2/RESULTS_REVIEW.md`. This documents what changed, what
got worse, and which claims the new numbers do and do not support.

## New dataset (Task C)

NordicDat (hard, ~0.28 macro-F1, suspected label issue) is **replaced by
"Network operator KPIs time series"** (Zenodo 8147768, CC BY-NC 4.0). 14 real
aligned KPI series, 5-minute cadence, 37 days, 223 windows, 6 recurring
time-of-day regimes, producer-annotated binary anomaly target (positive rate
0.239). Details in `results/final2/new_dataset.md`. Loader: `final2_load.py`.

## New method: RAPT-ED-Guard (Task A)

`RAPTEDGuard` in `final2_methods.py`. The Event-Driven error trigger OR an
every-window check arms a cheap validation gate: score the stored regime policy
and the prefix policy on the last 50 labelled rows; reuse only if the stored
policy beats the prefix policy by `edguard_margin`. Margin chosen on dev
(5 seeds, first 60%) — 0.05 wins overall (0.8775 vs 0.8197 at margin 0.0),
frozen in `configs/final2.yaml`. No prefix-policy pretraining was added to keep
the run inside the time cap; this is a documented limitation, not a hidden one.

**Result (30 seeds, full stream, macro-F1 / total CPU s):**

| stream | Frozen | Event-Driven | RAPT | RAPT-Guard | **RAPT-ED-Guard** |
|---|---|---|---|---|---|
| Campus | 0.937 | 0.964 | 0.939 | 0.939 | **0.984** |
| 5G NR  | 0.911 | 0.908 | 0.897 | 0.915 | **0.905** |
| KPI-ops| 0.680 | 0.715 | 0.680 | 0.680 | **0.782** |
| UGR'16 | 0.969 | 0.898 | 0.914 | 0.936 | **0.952** |

RAPT-ED-Guard beats Frozen and Event-Driven on Campus and KPI-ops (Holm
$p<0.001$), and beats Event-Driven on UGR'16 (Holm $p<0.001$). It is
**significantly below Frozen on UGR'16** ($-0.017$, Holm $p<0.001$) and below
Event-Driven on 5G NR ($-0.003$). Its adaptation CPU is high because its gate
refits often (it matches the adapters' accuracy, not RAPT's cost).

## Reuse is sometimes harmful (honest negative)

On KPI-ops, base RAPT reuses a stale regime policy **173 times** and collapses
exactly to Frozen (0.680; window diff 0.0). On UGR'16 RAPT (0.914) and its
non-gated variants are **below even Frozen** (0.969). This is the paper's
central "identity-only reuse can be unsafe" point, now with KPI-ops as a second
instance. RAPT-Anchor's lookback oracle barely helps on either (0.680 / 0.925),
i.e. recent labelled rows are a weak signal when the regime's class prior
shifted.

## Gate accuracy (Task A reporting)

From `results/final2/gate_accuracy.csv` (30 seeds):

| stream | returns | allows | blocks | bad allows | bad-allow rate | missed reuse |
|---|---|---|---|---|---|---|
| Campus | 12.8 | 0.0 | 12.8 | 0.0 | — (all blocked) | 0.0 |
| 5G NR | 14.7 | 3.5 | 11.2 | 1.4 | 0.400 | 0.0 |
| KPI-ops | 173.0 | 40.4 | 132.6 | 0.6 | 0.016 | 1.1 |
| UGR'16 | 133.0 | 1.4 | 131.6 | 0.0 | 0.000 | 1.3 |

Overall bad-allow rate 4.5%; 2 of 288 blocks were missed reuses. The gate is
**very conservative** — it blocks almost everything on UGR'16 (1.4 allows of
133 returns), which is why RAPT-ED-Guard on UGR'16 lands near Frozen rather
than exploiting reuse.

## Long vs short window gate (Task D2)

`results/final2/gate_cadence.csv` + `figures/fig_gate_cadence.png`. Measured,
not asserted. On the constructed within-regime concept-drift stream the
boundary/short gate fires **0** refits and the every-window/long gate fires 8;
on the natural streams the long gate refits far more (UGR'16 421 vs 55,
KPI-ops 337 vs 25). Accuracy: Campus 0.973 vs 0.928, KPI-ops 0.707 vs 0.509,
UGR'16 0.936 vs 0.923, but 5G NR 0.895 vs 0.909 (the long gate is *worse*
there). So the feasibility result is **stream-dependent**, not uniformly
positive.

## Detector fixes (Task D3)

ADWIN and Page-Hinkley default thresholds never fired (== Frozen). Swept on dev
(`results/final2/dev_detector_sweep.csv`): froze `adwin_delta=0.2`,
`ph_threshold=0.002`. **ADWIN was added to the final method set.** Page-Hinkley
now fires 6–20 times/seed; ADWIN fires on UGR'16/KPI-ops (~9–11) but still near
zero on Campus/5G NR. Neither beats the best adapter anywhere
(ADWIN 0.89–0.95; Page-Hinkley 0.89–0.98). The old defaults are retained only in
the sweep table, not used.

## Concept-drift stress panel (Task D4)

Enabled the in-regime floor trigger for all methods. The four trajectories
**still coincide**: mean max−min spread across methods is 0.008 / 0.020 / 0.000
at severities 0.2 / 0.5 / 1.0 (`results/final2/concept_spread.csv`). Conclusion
to report: the panel is **degenerate even after the fix**, because the stream's
per-window macro-F1 is effectively determined by the class prior, not by which
method adapts. Recommendation: drop the panel from the paper and say why.

## Total CPU (Task E)

`total_cpu_sec = adaptation + prediction` is recorded for every method. The
doubled inference of RAPT-Anchor is visible (e.g. UGR'16 prediction 3.66 s vs
~1.27 s for single-inference methods). Frozen+Periodic has the *lowest* total
CPU while scoring well on several streams, so "low cost" alone does not
separate reuse from a refit schedule.

## Reuse-vs-trigger baselines (Task G)

`Event-Driven+Reuse` and `Frozen+Periodic` are in the final set. On KPI-ops
Frozen+Periodic (0.893) nearly matches Full Retraining (0.906) at ~1/10 the
adaptation CPU, and on UGR'16 it is the *worst* method (0.837). Event-Driven+Reuse
beats Event-Driven on UGR'16 (0.964 vs 0.898) but not on KPI-ops. The answer to
"is reuse or the trigger doing the work" is **both, and stream-dependent**.

## Class-weight / leakage diagnosis (Task F)

`results/final2/nordicdat_diagnosis.csv`. Class-weighting does not create the
KPI-ops signal: Frozen is 0.6797 both with and without class weights. The
strongest KPI-ops feature-target correlation is 0.837 (`kpi_r12`); top_feature
importance is not concentrated on `kpi_r9-r11` (the components of the target),
so no direct label leak. KPI-ops prefix is single-class (all normal), so Frozen
starts with a weak policy — the same hard, late-first-anomaly difficulty that
sank NordicDat. UGR'16 is strongly imbalanced (3.2% positives full, 8.1%
prefix).

## What got worse / claims to soften

1. **RAPT's efficiency headline shrinks on total CPU.** RAPT adaptation CPU is
   64–91% below full retraining, but its *total* CPU is only 1–7% below
   (prediction dominates). Report both; do not claim a large total-cost win.
2. **RAPT-ED-Guard is not a free lunch.** It recovers accuracy by refitting
   often, so it is an accuracy method, not a cheap one. On 5G NR it is below
   Event-Driven; on UGR'16 below Frozen.
3. **"Reuse is safe with a gate" is false on UGR'16** for both RAPT-ED-Guard
   ($-0.017$ vs Frozen) and RAPT-Guard ($-0.034$ vs Frozen).
4. **Frozen+Periodic** shows that much of RAPT's accuracy on KPI-ops is not
   attributable to reuse at all.
5. **The concept panel adds no evidence** — drop it.
6. RAPT-Anchor's small gain over RAPT (UGR'16 +0.011, 5G NR +0.007) plus its
   doubled inference makes the oracle look weak, not strong.

## Claims the new numbers do support

* Contrast UGR'16 (identity-only reuse is **actively harmful**, RAPT below
  Frozen) with Campus/KPI-ops (harmful in a different way — RAPT collapses to
  Frozen without a working gate).
* A cheap validation gate recovers **most** of the loss but not all of it,
  and is stream-dependent.
* Total CPU, not adaptation CPU, is the deployment-relevant cost, and on that
  metric reuse's advantage is small.

## Reproduction

See `experiments/final2/README.md` for the exact run order. Dev tuning uses 5
seeds (42–46), first 60% of post-prefix windows; the final run uses 30 seeds
(42–71) on the full streams once with `configs/final2.yaml`. Total compute for
this job ≈ 2.4 h wall (final run 54 min on 4 workers).

## 11. How to reproduce

```
python experiments/final2/fetch_kpiops.py
python experiments/final2/final2_dev_margin.py
python experiments/final2/final2_dev_detectors.py
python experiments/final2/final2_runner.py --mode final --streams all --tag final --n-jobs 4
python experiments/final2/final2_stats.py --tag final
python experiments/final2/final2_gate.py
python experiments/final2/final2_gate_cadence.py
python experiments/final2/final2_concept.py
python experiments/final2/final2_nordicdat.py
python experiments/final2/final2_figures.py
python experiments/final2/make_review.py
python experiments/final2/make_report.py
```


Figures: `results/final2/figures/fig_main_macro_f1.png`, `fig_adapt_cpu.png`, `fig_cost.png`, `fig_gate_cadence.png`.


Machine-readable: `results/final2/summary.json`, `results/final2/tables/*.csv`, `results/final2/paper_values.md`.
