# Final-2 results review

All numbers below are read from `results/final2/` files. Nothing here is hand-typed or taken from the paper.

Streams: `5g_campus` = 5G Campus QoS, `ugr16` = UGR'16, `5g_nr` = 5G NR Latency, `kpiops` = Network operator KPIs


## 1. Main table (30 seeds, full stream)


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


## 2. Statistics (Wilcoxon+Holm, Cohen's dz, TOST +/-0.02)

| stream | comparison | window_n | window_mean_diff | window_p | window_p_holm | seed_dz | seed_t_p | tost_p | tost_equiv | ci95_low | ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5g_campus | RAPT-Guard vs Frozen | 4290 | 0.0020 | 0.0000 | 0.0000 | 1.0275 | 0.0000 | 0.0000 | True | 0.0013 | 0.0027 |
| 5g_campus | RAPT-Guard vs Event-Driven | 4290 | -0.0247 | 0.0000 | 0.0000 | -8.1491 | 0.0000 | 1.0000 | False | -0.0259 | -0.0236 |
| 5g_campus | RAPT vs Frozen | 4290 | 0.0014 | 0.0000 | 0.0000 | 3.6106 | 0.0000 | 0.0000 | True | 0.0013 | 0.0016 |
| 5g_campus | RAPT vs Event-Driven | 4290 | -0.0253 | 0.0000 | 0.0000 | -12.9658 | 0.0000 | 1.0000 | False | -0.0260 | -0.0246 |
| 5g_campus | RAPT-ED-Guard vs Frozen | 4290 | 0.0467 | 0.0000 | 0.0000 | 13.9775 | 0.0000 | 1.0000 | False | 0.0454 | 0.0479 |
| 5g_campus | RAPT-ED-Guard vs Event-Driven | 4290 | 0.0199 | 0.0000 | 0.0000 | 8.9255 | 0.0000 | 0.4412 | False | 0.0191 | 0.0208 |
| 5g_campus | Event-Driven+Reuse vs Event-Driven | 4290 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| 5g_campus | Frozen+Periodic vs Frozen | 4290 | 0.0407 | 0.0000 | 0.0000 | 8.8063 | 0.0000 | 1.0000 | False | 0.0390 | 0.0425 |
| 5g_campus | Full Retraining vs Frozen | 4290 | 0.0466 | 0.0000 | 0.0000 | 12.6422 | 0.0000 | 1.0000 | False | 0.0452 | 0.0479 |
| 5g_campus | RAPT vs Full Retraining | 4290 | -0.0451 | 0.0000 | 0.0000 | -12.4805 | 0.0000 | 1.0000 | False | -0.0465 | -0.0438 |
| 5g_campus | RAPT-Guard vs Full Retraining | 4290 | -0.0445 | 0.0000 | 0.0000 | -10.1527 | 0.0000 | 1.0000 | False | -0.0462 | -0.0429 |
| 5g_campus | ADWIN vs Frozen | 4290 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| 5g_campus | Page-Hinkley vs Frozen | 4290 | 0.0472 | 0.0000 | 0.0000 | 18.5605 | 0.0000 | 1.0000 | False | 0.0463 | 0.0482 |
| ugr16 | RAPT-Guard vs Frozen | 4320 | -0.0336 | 0.0000 | 0.0000 | -2.4291 | 0.0000 | 1.0000 | False | -0.0388 | -0.0284 |
| ugr16 | RAPT-Guard vs Event-Driven | 4320 | 0.0380 | 0.0000 | 0.0000 | 1.2536 | 0.0000 | 0.9985 | False | 0.0266 | 0.0493 |
| ugr16 | RAPT vs Frozen | 4320 | -0.0556 | 0.0000 | 0.0000 | -3.6063 | 0.0000 | 1.0000 | False | -0.0614 | -0.0499 |
| ugr16 | RAPT vs Event-Driven | 4320 | 0.0159 | 0.3619 | 0.7238 | 0.4980 | 0.0107 | 0.2436 | False | 0.0040 | 0.0278 |
| ugr16 | RAPT-ED-Guard vs Frozen | 4320 | -0.0171 | 0.0000 | 0.0000 | -1.5317 | 0.0000 | 0.0862 | False | -0.0213 | -0.0130 |
| ugr16 | RAPT-ED-Guard vs Event-Driven | 4320 | 0.0544 | 0.0000 | 0.0000 | 1.7680 | 0.0000 | 1.0000 | False | 0.0429 | 0.0659 |
| ugr16 | Event-Driven+Reuse vs Event-Driven | 4320 | 0.0661 | 0.0000 | 0.0000 | 1.9862 | 0.0000 | 1.0000 | False | 0.0537 | 0.0785 |
| ugr16 | Frozen+Periodic vs Frozen | 4320 | -0.1326 | 0.0000 | 0.0000 | -3.2465 | 0.0000 | 1.0000 | False | -0.1479 | -0.1174 |
| ugr16 | Full Retraining vs Frozen | 4320 | -0.0182 | 0.0000 | 0.0000 | -2.2847 | 0.0000 | 0.1120 | False | -0.0212 | -0.0152 |
| ugr16 | RAPT vs Full Retraining | 4320 | -0.0375 | 0.0000 | 0.0000 | -3.0236 | 0.0000 | 1.0000 | False | -0.0421 | -0.0328 |
| ugr16 | RAPT-Guard vs Full Retraining | 4320 | -0.0154 | 0.0003 | 0.0010 | -1.3676 | 0.0000 | 0.0166 | True | -0.0196 | -0.0112 |
| ugr16 | ADWIN vs Frozen | 4320 | -0.0195 | 0.5695 | 0.7238 | -1.1410 | 0.0000 | 0.4408 | False | -0.0259 | -0.0131 |
| ugr16 | Page-Hinkley vs Frozen | 4320 | -0.0096 | 0.2320 | 0.6960 | -1.7584 | 0.0000 | 0.0000 | True | -0.0116 | -0.0075 |
| 5g_nr | RAPT-Guard vs Frozen | 11970 | 0.0038 | 0.0275 | 0.1102 | 0.1340 | 0.4687 | 0.0018 | True | -0.0067 | 0.0142 |
| 5g_nr | RAPT-Guard vs Event-Driven | 11970 | 0.0067 | 0.0006 | 0.0046 | 0.2323 | 0.2133 | 0.0084 | True | -0.0041 | 0.0174 |
| 5g_nr | RAPT vs Frozen | 11970 | -0.0141 | 0.0000 | 0.0000 | -0.5017 | 0.0102 | 0.1309 | False | -0.0246 | -0.0036 |
| 5g_nr | RAPT vs Event-Driven | 11970 | -0.0112 | 0.0000 | 0.0000 | -0.4032 | 0.0353 | 0.0465 | True | -0.0216 | -0.0008 |
| 5g_nr | RAPT-ED-Guard vs Frozen | 11970 | -0.0062 | 0.0025 | 0.0149 | -0.6380 | 0.0015 | 0.0000 | True | -0.0098 | -0.0026 |
| 5g_nr | RAPT-ED-Guard vs Event-Driven | 11970 | -0.0033 | 0.0154 | 0.0769 | -0.3840 | 0.0442 | 0.0000 | True | -0.0064 | -0.0001 |
| 5g_nr | Event-Driven+Reuse vs Event-Driven | 11970 | -0.0080 | 0.0000 | 0.0001 | -0.8461 | 0.0001 | 0.0000 | True | -0.0116 | -0.0045 |
| 5g_nr | Frozen+Periodic vs Frozen | 11970 | -0.0272 | 0.0000 | 0.0000 | -2.4782 | 0.0000 | 0.9994 | False | -0.0312 | -0.0231 |
| 5g_nr | Full Retraining vs Frozen | 11970 | 0.0011 | 0.5762 | 1.0000 | 0.1148 | 0.5345 | 0.0000 | True | -0.0024 | 0.0046 |
| 5g_nr | RAPT vs Full Retraining | 11970 | -0.0152 | 0.0000 | 0.0000 | -0.5623 | 0.0045 | 0.1697 | False | -0.0253 | -0.0051 |
| 5g_nr | RAPT-Guard vs Full Retraining | 11970 | 0.0027 | 0.1458 | 0.4374 | 0.0988 | 0.5924 | 0.0007 | True | -0.0074 | 0.0128 |
| 5g_nr | ADWIN vs Frozen | 11970 | 0.0003 | 0.5271 | 1.0000 | 0.2334 | 0.2113 | 0.0000 | True | -0.0002 | 0.0009 |
| 5g_nr | Page-Hinkley vs Frozen | 11970 | -0.0065 | 0.0013 | 0.0091 | -0.7862 | 0.0002 | 0.0000 | True | -0.0096 | -0.0034 |
| kpiops | RAPT-Guard vs Frozen | 5340 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| kpiops | RAPT-Guard vs Event-Driven | 5340 | -0.0351 | 0.0000 | 0.0000 | -5.8773 | 0.0000 | 1.0000 | False | -0.0373 | -0.0328 |
| kpiops | RAPT vs Frozen | 5340 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.0000 | True | 0.0000 | 0.0000 |
| kpiops | RAPT vs Event-Driven | 5340 | -0.0351 | 0.0000 | 0.0000 | -5.8773 | 0.0000 | 1.0000 | False | -0.0373 | -0.0328 |
| kpiops | RAPT-ED-Guard vs Frozen | 5340 | 0.1022 | 0.0000 | 0.0000 | 5.3876 | 0.0000 | 1.0000 | False | 0.0951 | 0.1093 |
| kpiops | RAPT-ED-Guard vs Event-Driven | 5340 | 0.0671 | 0.0000 | 0.0000 | 3.5323 | 0.0000 | 1.0000 | False | 0.0600 | 0.0742 |
| kpiops | Event-Driven+Reuse vs Event-Driven | 5340 | 0.0012 | 0.0003 | 0.0008 | 0.5576 | 0.0048 | 0.0000 | True | 0.0004 | 0.0020 |
| kpiops | Frozen+Periodic vs Frozen | 5340 | 0.2131 | 0.0000 | 0.0000 | 49.5478 | 0.0000 | 1.0000 | False | 0.2115 | 0.2147 |
| kpiops | Full Retraining vs Frozen | 5340 | 0.2263 | 0.0000 | 0.0000 | 73.1152 | 0.0000 | 1.0000 | False | 0.2251 | 0.2274 |
| kpiops | RAPT vs Full Retraining | 5340 | -0.2263 | 0.0000 | 0.0000 | -73.1152 | 0.0000 | 1.0000 | False | -0.2274 | -0.2251 |
| kpiops | RAPT-Guard vs Full Retraining | 5340 | -0.2263 | 0.0000 | 0.0000 | -73.1152 | 0.0000 | 1.0000 | False | -0.2274 | -0.2251 |
| kpiops | ADWIN vs Frozen | 5340 | 0.2127 | 0.0000 | 0.0000 | 37.1365 | 0.0000 | 1.0000 | False | 0.2105 | 0.2148 |
| kpiops | Page-Hinkley vs Frozen | 5340 | 0.2064 | 0.0000 | 0.0000 | 52.5948 | 0.0000 | 1.0000 | False | 0.2050 | 0.2079 |


## 3. RAPT-ED-Guard gate accuracy

| dataset | returns | allows | blocks | bad_allows | bad_blocks | bad_allow_rate | missed_reuse |
|---|---|---|---|---|---|---|---|
| 5G Campus QoS | 12.8333 | 0.0000 | 12.8333 | 0.0000 | 0.0000 | nan | 0.0000 |
| 5G NR Latency | 14.6667 | 3.5000 | 11.1667 | 1.4000 | 0.0000 | 0.4000 | 0.0000 |
| Network operator KPIs | 173.0000 | 40.4000 | 132.6000 | 0.6333 | 1.1000 | 0.0160 | 1.1000 |
| UGR'16 | 133.0000 | 1.4333 | 131.5667 | 0.0000 | 1.3000 | 0.0000 | 1.3000 |


## 4. Long-window vs short-window gate

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


## 5. Concept-drift stress panel (floor trigger enabled)

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


## 6. Detector dev sweep (ADWIN delta / Page-Hinkley threshold)

| stream | method | retrain_events_min | retrain_events_mean | retrain_events_max |
|---|---|---|---|---|
| 5g_nr | ADWIN | 0 | 0.2222 | 1 |
| 5g_nr | Page-Hinkley | 10 | 11.0000 | 12 |
| campus | ADWIN | 0 | 0.0000 | 0 |
| campus | Page-Hinkley | 5 | 6.0000 | 7 |
| kpiops | ADWIN | 6 | 6.8889 | 9 |
| kpiops | Page-Hinkley | 8 | 8.3333 | 9 |
| ugr16 | ADWIN | 6 | 7.5556 | 9 |
| ugr16 | Page-Hinkley | 13 | 13.0000 | 13 |


## 7. RAPT-ED-Guard margin dev sweep

| stream | margin | macro_f1 | gate_allows |
|---|---|---|---|
| 5g_nr | 0.0000 | 0.9088 | 25 |
| 5g_nr | 0.0200 | 0.8946 | 7 |
| 5g_nr | 0.0500 | 0.8946 | 6 |
| campus | 0.0000 | 0.9277 | 45 |
| campus | 0.0200 | 0.9701 | 11 |
| campus | 0.0500 | 0.9728 | 0 |
| kpiops | 0.0000 | 0.5089 | 510 |
| kpiops | 0.0200 | 0.6849 | 201 |
| kpiops | 0.0500 | 0.7069 | 198 |
| ugr16 | 0.0000 | 0.9333 | 361 |
| ugr16 | 0.0200 | 0.9370 | 16 |
| ugr16 | 0.0500 | 0.9358 | 9 |


## 8. Class-weight / leakage diagnosis (all streams)

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


## 9. Frozen configs used


### final.yaml
```
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


### final2.yaml
```
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


## 10. Dataset
New stream replaces NordicDat: *Network operator KPIs time series* (Zenodo 8147768, CC BY-NC 4.0). See `new_dataset.md`. 14 real KPI series, 5-min cadence, 37 days, 223 windows, 6 time-of-day regimes each recurring ~37 times, producer-annotated binary anomaly target.
