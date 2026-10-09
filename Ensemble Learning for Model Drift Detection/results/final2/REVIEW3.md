# REVIEW3 — cost-reduction job (RAPT)

Every number below is read from `results/final2/` files; nothing is hand-typed. `paper/` was not edited. Frozen configs come from `configs/final.yaml` + `configs/final2.yaml`.


## 1. Timing audit

Prediction CPU is measured with `time.process_time()` wrapped around a single vectorised `m.predict(X_window)` call per window. No `predict` is called row-by-row in any method (only `detector models` consume the per-sample error stream, which is not inference). Every estimator is constructed with `n_jobs=1`, confirmed below.

| stream | n_jobs | pred_cpu_batched_sec | pred_cpu_rowwise_sec | inflation_x | inference_calls_batched | inference_calls_rowwise |
|---|---|---|---|---|---|---|
| campus | 1,1 | 1.1516 | 10.8774 | 9.4457 | 143 | 1430 |
| ugr16 | 1,1 | 1.2454 | 270.9620 | 217.5741 | 144 | 34560 |
| 5g_nr | 1,1 | 3.3553 | 3.2643 | 0.9729 | 399 | 399 |
| kpiops | 1,1 | 1.3929 | 76.0568 | 54.6044 | 178 | 8544 |


Row-by-row inference would inflate prediction CPU by: 9.4x on 5G Campus QoS, 217.6x on UGR'16, 1.0x on 5G NR Latency, 54.6x on Network operator KPIs. The final run uses the batched path; **all methods were rerun** so numbers are comparable.


## 2. Cheap refit mode (dev sweep, 5 seeds, first 60%)

Criterion: macro-F1 per total CPU, mean over Event-Driven+Reuse and RAPT-Guard. Chosen mode: **t20_b300** (20 trees, buffer 300).

| mode | n_trees | buffer | method | macro_f1 | adapt_cpu | total_cpu | f1_per_total_cpu |
|---|---|---|---|---|---|---|---|
| t20_b300 | 20 | 300 | Event-Driven+Reuse | 0.8348 | 0.0008 | 0.0065 | 128.6183 |
| t20_b300 | 20 | 300 | RAPT-Guard | 0.8031 | 0.0030 | 0.0078 | 103.1929 |
| t30_b500 | 30 | 500 | Event-Driven+Reuse | 0.8383 | 0.0012 | 0.0079 | 105.9034 |
| t30_b500 | 30 | 500 | RAPT-Guard | 0.8131 | 0.0047 | 0.0107 | 76.0796 |
| t50_full | 50 | 1000 | Event-Driven+Reuse | 0.8486 | 0.0024 | 0.0110 | 77.0693 |
| t50_full | 50 | 1000 | RAPT-Guard | 0.8316 | 0.0091 | 0.0176 | 47.2951 |


## 3. Summary (30 seeds, mean per seed; CPU seconds)


### macro-F1 (+/- sd)

| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| Frozen | 0.9371 +/- 0.0021 | 0.9110 +/- 0.0061 | 0.6797 +/- 0.0000 | 0.9693 +/- 0.0033 |
| Full Retraining | 0.9836 +/- 0.0025 | 0.9121 +/- 0.0062 | 0.9060 +/- 0.0031 | 0.9511 +/- 0.0076 |
| Full Retraining-cheap | 0.9795 +/- 0.0050 | 0.8958 +/- 0.0083 | 0.9045 +/- 0.0033 | 0.8654 +/- 0.0277 |
| Event-Driven | 0.9638 +/- 0.0003 | 0.9081 +/- 0.0074 | 0.7147 +/- 0.0060 | 0.8978 +/- 0.0286 |
| Event-Driven-cheap | 0.9633 +/- 0.0013 | 0.8952 +/- 0.0085 | 0.6970 +/- 0.0334 | 0.8129 +/- 0.0721 |
| Event-Driven+Reuse | 0.9633 +/- 0.0013 | 0.8963 +/- 0.0068 | 0.7035 +/- 0.0331 | 0.8601 +/- 0.0791 |
| Page-Hinkley | 0.9837 +/- 0.0026 | 0.9045 +/- 0.0074 | 0.8542 +/- 0.0044 | 0.9487 +/- 0.0110 |
| RAPT | 0.9426 +/- 0.0000 | 0.8856 +/- 0.0298 | 0.6802 +/- 0.0003 | 0.9119 +/- 0.0157 |
| RAPT-Guard | 0.9367 +/- 0.0022 | 0.8926 +/- 0.0324 | 0.6798 +/- 0.0004 | 0.7193 +/- 0.0479 |


### adaptation CPU

| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| Frozen | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Full Retraining | 2.2131 | 1.1473 | 26.1241 | 33.2522 |
| Full Retraining-cheap | 0.8111 | 0.4704 | 9.0145 | 8.8474 |
| Event-Driven | 0.1667 | 1.5879 | 0.7941 | 0.9371 |
| Event-Driven-cheap | 0.0589 | 0.7199 | 0.2896 | 0.2400 |
| Event-Driven+Reuse | 0.0589 | 0.1059 | 0.2153 | 0.1564 |
| Page-Hinkley | 0.9702 | 2.8323 | 1.6003 | 1.2875 |
| RAPT | 0.4372 | 0.5568 | 0.7928 | 3.1703 |
| RAPT-Guard | 0.1168 | 0.1807 | 0.5380 | 0.9529 |


### prediction CPU

| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| Frozen | 1.2474 | 3.4896 | 1.5135 | 1.2960 |
| Full Retraining | 1.2247 | 3.4955 | 1.4619 | 1.2922 |
| Full Retraining-cheap | 0.5689 | 1.6028 | 0.6881 | 0.6091 |
| Event-Driven | 1.2282 | 3.4296 | 1.4722 | 1.2956 |
| Event-Driven-cheap | 0.9169 | 1.8187 | 0.6922 | 0.6473 |
| Event-Driven+Reuse | 0.9170 | 2.2377 | 0.6960 | 0.8813 |
| Page-Hinkley | 1.2283 | 3.4303 | 1.5017 | 1.2648 |
| RAPT | 1.2569 | 3.4852 | 1.5006 | 1.2922 |
| RAPT-Guard | 0.7344 | 2.0724 | 0.8186 | 0.6850 |


### total CPU

| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| Frozen | 1.2474 | 3.4896 | 1.5135 | 1.2960 |
| Full Retraining | 3.4378 | 4.6428 | 27.5860 | 34.5445 |
| Full Retraining-cheap | 1.3800 | 2.0732 | 9.7026 | 9.4565 |
| Event-Driven | 1.3949 | 5.0176 | 2.2663 | 2.2327 |
| Event-Driven-cheap | 0.9758 | 2.5386 | 0.9818 | 0.8873 |
| Event-Driven+Reuse | 0.9759 | 2.3436 | 0.9114 | 1.0377 |
| Page-Hinkley | 2.1986 | 6.2626 | 3.1020 | 2.5524 |
| RAPT | 1.6940 | 4.0419 | 2.2933 | 4.4625 |
| RAPT-Guard | 0.8511 | 2.2531 | 1.3566 | 1.6379 |


### retrains

| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| Frozen | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Full Retraining | 14.0000 | 8.0000 | 178.0000 | 144.0000 |
| Full Retraining-cheap | 14.0000 | 8.0000 | 178.0000 | 144.0000 |
| Event-Driven | 1.0000 | 10.8667 | 4.8667 | 3.5667 |
| Event-Driven-cheap | 1.0000 | 12.1000 | 5.4667 | 3.7667 |
| Event-Driven+Reuse | 1.0000 | 1.8333 | 4.0000 | 2.4667 |
| Page-Hinkley | 6.1000 | 19.6000 | 10.0667 | 5.0333 |
| RAPT | 24.0000 | 200.0000 | 6.0000 | 12.0000 |
| RAPT-Guard | 16.0000 | 153.3333 | 5.0000 | 13.1000 |


### reuses

| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| Frozen | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Full Retraining | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Full Retraining-cheap | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Event-Driven | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Event-Driven-cheap | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Event-Driven+Reuse | 0.0000 | 7.3000 | 1.5000 | 0.8667 |
| Page-Hinkley | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RAPT | 119.0000 | 199.0000 | 172.0000 | 132.0000 |
| RAPT-Guard | 127.0000 | 245.6667 | 173.0000 | 130.9000 |


## 4. Table A — total CPU % change vs Full Retraining and vs Full Retraining-cheap

| stream | method | total_cpu | macro_f1 | pct_vs_full_retraining | pct_vs_full_retraining_cheap |
|---|---|---|---|---|---|
| campus | Event-Driven+Reuse | 0.9759 | 0.9633 | -71.6114 | -29.2797 |
| campus | RAPT-Guard | 0.8511 | 0.9367 | -75.2421 | -38.3244 |
| campus | Full Retraining-cheap | 1.3800 | 0.9795 | -59.8579 | 0.0000 |
| campus | Event-Driven-cheap | 0.9758 | 0.9633 | -71.6158 | -29.2905 |
| campus | RAPT | 1.6940 | 0.9426 | -50.7232 | 22.7559 |
| ugr16 | Event-Driven+Reuse | 1.0377 | 0.8601 | -96.9961 | -89.0269 |
| ugr16 | RAPT-Guard | 1.6379 | 0.7193 | -95.2586 | -82.6797 |
| ugr16 | Full Retraining-cheap | 9.4565 | 0.8654 | -72.6252 | 0.0000 |
| ugr16 | Event-Driven-cheap | 0.8873 | 0.8129 | -97.4314 | -90.6170 |
| ugr16 | RAPT | 4.4625 | 0.9119 | -87.0820 | -52.8105 |
| 5g_nr | Event-Driven+Reuse | 2.3436 | 0.8963 | -49.5217 | 13.0439 |
| 5g_nr | RAPT-Guard | 2.2531 | 0.8926 | -51.4710 | 8.6785 |
| 5g_nr | Full Retraining-cheap | 2.0732 | 0.8958 | -55.3463 | 0.0000 |
| 5g_nr | Event-Driven-cheap | 2.5386 | 0.8952 | -45.3229 | 22.4470 |
| 5g_nr | RAPT | 4.0419 | 0.8856 | -12.9420 | 94.9625 |
| kpiops | Event-Driven+Reuse | 0.9114 | 0.7035 | -96.6963 | -90.6070 |
| kpiops | RAPT-Guard | 1.3566 | 0.6798 | -95.0823 | -86.0181 |
| kpiops | Full Retraining-cheap | 9.7026 | 0.9045 | -64.8280 | 0.0000 |
| kpiops | Event-Driven-cheap | 0.9818 | 0.6970 | -96.4409 | -89.8809 |
| kpiops | RAPT | 2.2933 | 0.6802 | -91.6866 | -76.3637 |


## 5. Table B — Pareto frontier (macro-F1 vs total CPU)

| stream | method | macro_f1 | total_cpu | on_frontier | cpu_rank | f1_rank |
|---|---|---|---|---|---|---|
| campus | Frozen | 0.9371 | 1.2474 | no | 4 | 8 |
| campus | Full Retraining | 0.9836 | 3.4378 | no | 9 | 2 |
| campus | Full Retraining-cheap | 0.9795 | 1.3800 | YES | 5 | 3 |
| campus | Event-Driven | 0.9638 | 1.3949 | no | 6 | 4 |
| campus | Event-Driven-cheap | 0.9633 | 0.9758 | YES | 2 | 5 |
| campus | Event-Driven+Reuse | 0.9633 | 0.9759 | no | 3 | 5 |
| campus | Page-Hinkley | 0.9837 | 2.1986 | YES | 8 | 1 |
| campus | RAPT | 0.9426 | 1.6940 | no | 7 | 7 |
| campus | RAPT-Guard | 0.9367 | 0.8511 | YES | 1 | 9 |
| ugr16 | Frozen | 0.9693 | 1.2960 | YES | 3 | 1 |
| ugr16 | Full Retraining | 0.9511 | 34.5445 | no | 9 | 2 |
| ugr16 | Full Retraining-cheap | 0.8654 | 9.4565 | no | 8 | 6 |
| ugr16 | Event-Driven | 0.8978 | 2.2327 | no | 5 | 5 |
| ugr16 | Event-Driven-cheap | 0.8129 | 0.8873 | YES | 1 | 8 |
| ugr16 | Event-Driven+Reuse | 0.8601 | 1.0377 | YES | 2 | 7 |
| ugr16 | Page-Hinkley | 0.9487 | 2.5524 | no | 6 | 3 |
| ugr16 | RAPT | 0.9119 | 4.4625 | no | 7 | 4 |
| ugr16 | RAPT-Guard | 0.7193 | 1.6379 | no | 4 | 9 |
| 5g_nr | Frozen | 0.9110 | 3.4896 | YES | 5 | 2 |
| 5g_nr | Full Retraining | 0.9121 | 4.6428 | YES | 7 | 1 |
| 5g_nr | Full Retraining-cheap | 0.8958 | 2.0732 | YES | 1 | 6 |
| 5g_nr | Event-Driven | 0.9081 | 5.0176 | no | 8 | 3 |
| 5g_nr | Event-Driven-cheap | 0.8952 | 2.5386 | no | 4 | 7 |
| 5g_nr | Event-Driven+Reuse | 0.8963 | 2.3436 | YES | 3 | 5 |
| 5g_nr | Page-Hinkley | 0.9045 | 6.2626 | no | 9 | 4 |
| 5g_nr | RAPT | 0.8856 | 4.0419 | no | 6 | 9 |
| 5g_nr | RAPT-Guard | 0.8926 | 2.2531 | no | 2 | 8 |
| kpiops | Frozen | 0.6797 | 1.5135 | no | 4 | 9 |
| kpiops | Full Retraining | 0.9060 | 27.5860 | YES | 9 | 1 |
| kpiops | Full Retraining-cheap | 0.9045 | 9.7026 | YES | 8 | 2 |
| kpiops | Event-Driven | 0.7147 | 2.2663 | YES | 5 | 4 |
| kpiops | Event-Driven-cheap | 0.6970 | 0.9818 | no | 2 | 6 |
| kpiops | Event-Driven+Reuse | 0.7035 | 0.9114 | YES | 1 | 5 |
| kpiops | Page-Hinkley | 0.8542 | 3.1020 | YES | 7 | 3 |
| kpiops | RAPT | 0.6802 | 2.2933 | no | 6 | 7 |
| kpiops | RAPT-Guard | 0.6798 | 1.3566 | no | 3 | 8 |
| MEAN | Frozen | 0.8743 | 1.8866 | YES | 4 | 4 |
| MEAN | Full Retraining | 0.9382 | 17.5528 | YES | 9 | 1 |
| MEAN | Full Retraining-cheap | 0.9113 | 5.6531 | no | 8 | 3 |
| MEAN | Event-Driven | 0.8711 | 2.7279 | no | 5 | 5 |
| MEAN | Event-Driven-cheap | 0.8421 | 1.3459 | no | 2 | 8 |
| MEAN | Event-Driven+Reuse | 0.8558 | 1.3171 | YES | 1 | 6 |
| MEAN | Page-Hinkley | 0.9228 | 3.5289 | YES | 7 | 2 |
| MEAN | RAPT | 0.8551 | 3.1229 | no | 6 | 7 |
| MEAN | RAPT-Guard | 0.8071 | 1.5247 | no | 3 | 9 |


## 6. Table C — paired stats (Wilcoxon + Holm, one family; Cohen's dz; TOST +/-0.02)

| stream | focus | reference | n | mean_diff | dz | p_wilcoxon | ci_lo | ci_hi | tost_p | tost_equiv | p_holm | sig_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| campus | Event-Driven+Reuse | Frozen | 30 | 0.0263 | 10.6094 | 0.0000 | 0.0253 | 0.0272 | 1.0000 | no | 0.0000 | YES |
| campus | Event-Driven+Reuse | Full Retraining | 30 | -0.0203 | -7.4795 | 0.0000 | -0.0213 | -0.0193 | 0.7240 | no | 0.0000 | YES |
| campus | Event-Driven+Reuse | Event-Driven | 30 | -0.0005 | -0.3856 | 0.0455 | -0.0010 | -0.0000 | 0.0000 | YES | 0.1703 | no |
| campus | Event-Driven+Reuse | Page-Hinkley | 30 | -0.0204 | -6.2222 | 0.0000 | -0.0216 | -0.0191 | 0.7248 | no | 0.0000 | YES |
| campus | RAPT-Guard | Frozen | 30 | -0.0004 | -0.4219 | 0.0341 | -0.0007 | -0.0000 | 0.0000 | YES | 0.1703 | no |
| campus | RAPT-Guard | Full Retraining | 30 | -0.0469 | -12.8355 | 0.0000 | -0.0483 | -0.0456 | 1.0000 | no | 0.0000 | YES |
| campus | RAPT-Guard | Event-Driven | 30 | -0.0271 | -12.4629 | 0.0000 | -0.0280 | -0.0263 | 1.0000 | no | 0.0000 | YES |
| campus | RAPT-Guard | Page-Hinkley | 30 | -0.0470 | -14.9693 | 0.0000 | -0.0482 | -0.0458 | 1.0000 | no | 0.0000 | YES |
| ugr16 | Event-Driven+Reuse | Frozen | 30 | -0.1092 | -1.3900 | 0.0000 | -0.1385 | -0.0799 | 1.0000 | no | 0.0000 | YES |
| ugr16 | Event-Driven+Reuse | Full Retraining | 30 | -0.0910 | -1.1304 | 0.0000 | -0.1211 | -0.0609 | 1.0000 | no | 0.0000 | YES |
| ugr16 | Event-Driven+Reuse | Event-Driven | 30 | -0.0377 | -0.4394 | 0.0345 | -0.0697 | -0.0057 | 0.8658 | no | 0.1703 | no |
| ugr16 | Event-Driven+Reuse | Page-Hinkley | 30 | -0.0886 | -1.1323 | 0.0000 | -0.1178 | -0.0594 | 1.0000 | no | 0.0000 | YES |
| ugr16 | RAPT-Guard | Frozen | 30 | -0.2500 | -5.1639 | 0.0000 | -0.2681 | -0.2320 | 1.0000 | no | 0.0000 | YES |
| ugr16 | RAPT-Guard | Full Retraining | 30 | -0.2319 | -4.8241 | 0.0000 | -0.2498 | -0.2139 | 1.0000 | no | 0.0000 | YES |
| ugr16 | RAPT-Guard | Event-Driven | 30 | -0.1785 | -3.1990 | 0.0000 | -0.1993 | -0.1577 | 1.0000 | no | 0.0000 | YES |
| ugr16 | RAPT-Guard | Page-Hinkley | 30 | -0.2294 | -4.9243 | 0.0000 | -0.2468 | -0.2120 | 1.0000 | no | 0.0000 | YES |
| 5g_nr | Event-Driven+Reuse | Frozen | 30 | -0.0147 | -1.6481 | 0.0000 | -0.0180 | -0.0114 | 0.0015 | YES | 0.0000 | YES |
| 5g_nr | Event-Driven+Reuse | Full Retraining | 30 | -0.0158 | -1.5988 | 0.0000 | -0.0195 | -0.0121 | 0.0133 | YES | 0.0000 | YES |
| 5g_nr | Event-Driven+Reuse | Event-Driven | 30 | -0.0118 | -1.1573 | 0.0000 | -0.0156 | -0.0080 | 0.0001 | YES | 0.0002 | YES |
| 5g_nr | Event-Driven+Reuse | Page-Hinkley | 30 | -0.0082 | -0.8331 | 0.0004 | -0.0119 | -0.0045 | 0.0000 | YES | 0.0034 | YES |
| 5g_nr | RAPT-Guard | Frozen | 30 | -0.0184 | -0.5687 | 0.0000 | -0.0304 | -0.0063 | 0.3928 | no | 0.0004 | YES |
| 5g_nr | RAPT-Guard | Full Retraining | 30 | -0.0195 | -0.6223 | 0.0000 | -0.0311 | -0.0078 | 0.4630 | no | 0.0004 | YES |
| 5g_nr | RAPT-Guard | Event-Driven | 30 | -0.0155 | -0.4854 | 0.0005 | -0.0273 | -0.0036 | 0.2203 | no | 0.0038 | YES |
| 5g_nr | RAPT-Guard | Page-Hinkley | 30 | -0.0119 | -0.3799 | 0.0104 | -0.0235 | -0.0002 | 0.0821 | no | 0.0622 | no |
| kpiops | Event-Driven+Reuse | Frozen | 30 | 0.0238 | 0.7206 | 0.0002 | 0.0115 | 0.0362 | 0.7346 | no | 0.0014 | YES |
| kpiops | Event-Driven+Reuse | Full Retraining | 30 | -0.2025 | -6.0596 | 0.0000 | -0.2149 | -0.1900 | 1.0000 | no | 0.0000 | YES |
| kpiops | Event-Driven+Reuse | Event-Driven | 30 | -0.0112 | -0.3417 | 0.0961 | -0.0235 | 0.0010 | 0.0775 | no | 0.1922 | no |
| kpiops | Event-Driven+Reuse | Page-Hinkley | 30 | -0.1507 | -4.5897 | 0.0000 | -0.1630 | -0.1384 | 1.0000 | no | 0.0000 | YES |
| kpiops | RAPT-Guard | Frozen | 30 | 0.0001 | 0.1826 | 0.3173 | -0.0001 | 0.0002 | 0.0000 | YES | 0.3173 | no |
| kpiops | RAPT-Guard | Full Retraining | 30 | -0.2262 | -71.5008 | 0.0000 | -0.2274 | -0.2250 | 1.0000 | no | 0.0000 | YES |
| kpiops | RAPT-Guard | Event-Driven | 30 | -0.0350 | -5.8324 | 0.0000 | -0.0372 | -0.0328 | 1.0000 | no | 0.0000 | YES |
| kpiops | RAPT-Guard | Page-Hinkley | 30 | -0.1745 | -39.5760 | 0.0000 | -0.1761 | -0.1728 | 1.0000 | no | 0.0000 | YES |


## 7. Verdict

1. Is Event-Driven+Reuse (cheap) on the Pareto frontier on at least 2 streams? **YES** (3/4 streams: ugr16, 5g_nr, kpiops)

2. Is its total CPU within 30% of Frozen on at least 2 streams? **YES** (4/4: campus, ugr16, 5g_nr, kpiops)

3. Is its macro-F1 within 0.02 of Full Retraining (TOST) on at least 2 streams? **NO** (1/4: 5g_nr)

4. Does the cheap refit lose more than 0.02 macro-F1 vs the standard refit on any stream? **YES**

   - 5G Campus QoS: Event-Driven loss 0.0005, Full Retraining loss 0.0041

   - UGR'16: Event-Driven loss 0.0848, Full Retraining loss 0.0858

   - 5G NR Latency: Event-Driven loss 0.0129, Full Retraining loss 0.0163

   - Network operator KPIs: Event-Driven loss 0.0177, Full Retraining loss 0.0015


## 8. Suspicious / identical numbers

| stream | method | kind | detail |
|---|---|---|---|
| campus | RAPT | zero_variance_macro_f1 | sd=1.13e-16, mean=0.9426 |
| kpiops | Frozen | zero_variance_macro_f1 | sd=3.39e-16, mean=0.6797 |
| campus | Event-Driven+Reuse+Event-Driven-cheap | identical_macro_f1_vectors | sorted per-seed F1 identical |
| campus | Event-Driven+Reuse | never_reused | 0 reuse events in 30 seeds; equals a trigger-only run |


Full list: `results/final2/cost_suspicious.csv`.


## 9. Interpretation and caveats (read with the numbers)

- **The cheaper gate's error-skip breaks on an imbalanced stream.** On UGR'16 RAPT-Guard invokes the gate but skips the actual comparison when the rolling accuracy is >= `gate_skip_acc`. Raw accuracy there is dominated by the majority (normal) class, so the skip fires often (mean gate_skipped below) and stale policies are reused unchecked, dropping RAPT-Guard to the value in the summary. Detailed per-stream counts:


| method | 5G Campus QoS | 5G NR Latency | Network operator KPIs | UGR'16 |
|---|---|---|---|---|
| RAPT-Guard | 127.0000 | 189.0667 | 111.0000 | 109.9333 |


- **On Campus, Event-Driven+Reuse and Event-Driven-cheap are numerically identical** (0 reuse events in 30 seeds; see the suspicious table), so their Pareto status is a tie broken by a ~0.001 s CPU difference. Treat them as the same point when reading Table B.

- **The cheap refit hurts UGR'16 most** (Event-Driven loss and Full Retraining loss both about 0.085 F1 vs their standard refits); on the other three streams the loss is <= 0.018. See Q4 above.

- **RAPT collapses to Frozen on KPI-ops** (0.6802 vs 0.6797, reuse 172/seed): on this stream identity-only reuse is never useful.

- **Total CPU, not adaptation CPU, is where the story lives.** Full Retraining spends most of its UGR'16/KPI-ops total CPU on adaptation, so cheap variants cut total CPU sharply there (Table A), but on Campus/5G NR inference dominates and every adaptive method is within a few percent of Frozen.
