# paper_values_final2_v4 — flat values (read from results/final2/)


## Batch-B macro-F1 (30 seeds)

| method | campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 0.9371 | 0.9693 | 0.9110 | 0.6797 |
| Frozen-cheap | 0.9371 | 0.9684 | 0.9032 | 0.6797 |
| Full Retraining | 0.9836 | 0.9511 | 0.9121 | 0.9060 |
| Full Retraining-cheap | 0.9795 | 0.8654 | 0.8958 | 0.9045 |
| Event-Driven | 0.9638 | 0.8978 | 0.9081 | 0.7147 |
| Event-Driven-cheap | 0.9633 | 0.8129 | 0.8952 | 0.6970 |
| Event-Driven+Reuse | 0.9633 | 0.8601 | 0.8963 | 0.7035 |
| Page-Hinkley | 0.9837 | 0.9487 | 0.9045 | 0.8542 |
| ADWIN | n/a | n/a | n/a | n/a |
| RAPT | 0.9385 | 0.9137 | 0.8969 | 0.6797 |
| RAPT-Guard | 0.9391 | 0.9357 | 0.9148 | 0.6797 |
| RAPT-Enhanced | 0.9385 | 0.9137 | 0.9045 | 0.6797 |
| RAPT-Anchor | 0.9385 | 0.9247 | 0.9042 | 0.6797 |

## Batch-B total CPU s (30 seeds)

| method | campus | ugr16 | 5g_nr | kpiops |
|---|---|---|---|---|
| Frozen | 1.2474 | 1.2960 | 3.4896 | 1.5135 |
| Frozen-cheap | 0.5730 | 0.6018 | 1.6169 | 0.6957 |
| Full Retraining | 3.4378 | 34.5445 | 4.6428 | 27.5860 |
| Full Retraining-cheap | 1.3800 | 9.4565 | 2.0732 | 9.7026 |
| Event-Driven | 1.3949 | 2.2327 | 5.0176 | 2.2663 |
| Event-Driven-cheap | 0.9758 | 0.8873 | 2.5386 | 0.9818 |
| Event-Driven+Reuse | 0.9759 | 1.0377 | 2.3436 | 0.9114 |
| Page-Hinkley | 2.1986 | 2.5524 | 6.2626 | 3.1020 |
| RAPT | 1.5457 | 4.1654 | 3.9394 | 2.1160 |
| RAPT-Guard | 1.7880 | 7.9081 | 4.2293 | 5.0534 |

## Contrasts (batch B, paired seed-level, Holm within contrast)

```
ED+Reuse vs ED-cheap
stream  n  mean_diff       dz  p_wilcoxon     ci_lo    ci_hi       tost_p  tost_equiv   p_holm  sig_holm
campus 30    0.00000 0.000000    1.000000  0.000000 0.000000 0.000000e+00        True 1.000000     False
 ugr16 30    0.04719 0.508989    0.006210  0.012570 0.081811 9.404802e-01       False 0.018631      True
 5g_nr 30    0.00117 0.109840    0.579991 -0.002806 0.005146 6.769994e-11        True 1.000000     False
kpiops 30    0.00650 1.835846    0.000003  0.005178 0.007822 2.537481e-19        True 0.000010      True

ED-cheap vs Full-cheap
stream  n  mean_diff        dz   p_wilcoxon     ci_lo     ci_hi       tost_p  tost_equiv       p_holm  sig_holm
campus 30  -0.016202 -2.977012 1.728948e-06 -0.018234 -0.014170 3.235577e-04        True 5.186845e-06      True
 ugr16 30  -0.052416 -0.710092 1.373943e-04 -0.079980 -0.024853 9.886184e-01       False 2.747886e-04      True
 5g_nr 30  -0.000668 -0.064489 8.792513e-01 -0.004538  0.003201 2.023492e-11        True 8.792513e-01     False
kpiops 30  -0.207489 -6.188464 1.862645e-09 -0.220009 -0.194970 1.000000e+00       False 7.450581e-09      True
```

## Frozen-cheap vs Frozen
```
stream  frozen_f1    frozen_sd  frozen_cheap_f1  frozen_cheap_sd       df1  frozen_total_cpu  frozen_cheap_total_cpu  cpu_pct_change
campus   0.937087 2.142382e-03         0.937069     2.486376e-03 -0.000018          1.247407                0.572958      -54.068052
 ugr16   0.969330 3.332550e-03         0.968398     4.261590e-03 -0.000932          1.296048                0.601821      -53.564944
 5g_nr   0.911028 6.094810e-03         0.903175     1.010004e-02 -0.007853          3.489583                1.616905      -53.664814
kpiops   0.679683 3.387608e-16         0.679683     3.387608e-16  0.000000          1.513540                0.695703      -54.034733
```

## RAPT reproducibility (final2 vs corrected cost)
```
stream  final2_f1  costfix_f1  max_abs_seed_diff  differing_windows  costfix_retrains  costfix_reuses  buggy_cost_retrains  buggy_cost_reuses
campus   0.938516    0.938516                0.0                  0                 2              12                   24                119
 ugr16   0.913682    0.913682                0.0                  0                11             133                   12                132
 5g_nr   0.896909    0.896909                0.0                  0                 3               5                  200                199
kpiops   0.679683    0.679683                0.0                  0                 5             173                    6                172
```

## RAPT-Guard standard vs cheap
```
stream  final2_guard_std  costfix_guard_std  costfix_guard_cheap
campus          0.939104           0.939104             0.936695
 ugr16          0.935735           0.935735             0.755170
 5g_nr          0.914787           0.914787             0.909858
kpiops          0.679683           0.679683             0.679755
```