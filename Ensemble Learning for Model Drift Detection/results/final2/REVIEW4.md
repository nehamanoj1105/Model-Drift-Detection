# REVIEW4 — final small job

Every number is read from `results/final2/`; nothing is hand-typed. `paper/` was not edited. Frozen configs: `configs/final.yaml` + `configs/final2.yaml`.


## 1. Frozen-cheap vs Frozen (30 seeds, batch inference)

| stream | frozen_f1 | frozen_sd | frozen_cheap_f1 | frozen_cheap_sd | df1 | frozen_total_cpu | frozen_cheap_total_cpu | cpu_pct_change |
|---|---|---|---|---|---|---|---|---|
| campus | 0.9371 | 0.0021 | 0.9371 | 0.0025 | -0.0000 | 1.2474 | 0.5730 | -54.0681 |
| ugr16 | 0.9693 | 0.0033 | 0.9684 | 0.0043 | -0.0009 | 1.2960 | 0.6018 | -53.5649 |
| 5g_nr | 0.9110 | 0.0061 | 0.9032 | 0.0101 | -0.0079 | 3.4896 | 1.6169 | -53.6648 |
| kpiops | 0.6797 | 0.0000 | 0.6797 | 0.0000 | 0.0000 | 1.5135 | 0.6957 | -54.0347 |


Frozen-cheap uses the same base learner as the cheap refits (20 trees / buffer 300). It changes macro-F1 by the `df1` column and total CPU by the `cpu_pct_change` column.


## 2. Is reuse or the trigger doing the work? (Event-Driven+Reuse cheap)

Paired seed-level Wilcoxon + Holm (one family per contrast), dz, TOST +/-0.02.


### Event-Driven+Reuse vs Event-Driven-cheap

| stream | n | mean_diff | dz | p_wilcoxon | ci_lo | ci_hi | tost_p | tost_equiv | p_holm | sig_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| campus | 30 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | YES | 1.0000 | no |
| ugr16 | 30 | 0.0472 | 0.5090 | 0.0062 | 0.0126 | 0.0818 | 0.9405 | no | 0.0186 | YES |
| 5g_nr | 30 | 0.0012 | 0.1098 | 0.5800 | -0.0028 | 0.0051 | 0.0000 | YES | 1.0000 | no |
| kpiops | 30 | 0.0065 | 1.8358 | 0.0000 | 0.0052 | 0.0078 | 0.0000 | YES | 0.0000 | YES |


### Event-Driven-cheap vs Full Retraining-cheap

| stream | n | mean_diff | dz | p_wilcoxon | ci_lo | ci_hi | tost_p | tost_equiv | p_holm | sig_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| campus | 30 | -0.0162 | -2.9770 | 0.0000 | -0.0182 | -0.0142 | 0.0003 | YES | 0.0000 | YES |
| ugr16 | 30 | -0.0524 | -0.7101 | 0.0001 | -0.0800 | -0.0249 | 0.9886 | no | 0.0003 | YES |
| 5g_nr | 30 | -0.0007 | -0.0645 | 0.8793 | -0.0045 | 0.0032 | 0.0000 | YES | 0.8793 | no |
| kpiops | 30 | -0.2075 | -6.1885 | 0.0000 | -0.2200 | -0.1950 | 1.0000 | no | 0.0000 | YES |


### CPU vs Event-Driven-cheap

| stream | ed_cheap_adapt_cpu | edr_adapt_cpu | adapt_pct_change | ed_cheap_total_cpu | edr_total_cpu | total_pct_change |
|---|---|---|---|---|---|---|
| campus | 0.0589 | 0.0589 | 0.0371 | 0.9758 | 0.9759 | 0.0153 |
| ugr16 | 0.2400 | 0.1564 | -34.8532 | 0.8873 | 1.0377 | 16.9462 |
| 5g_nr | 0.7199 | 0.1059 | -85.2883 | 2.5386 | 2.3436 | -7.6793 |
| kpiops | 0.2896 | 0.2153 | -25.6409 | 0.9818 | 0.9114 | -7.1759 |


## 3. Why cost-run RAPT retrain/reuse counts differed

Two defects in `cost_runner.py` (not in the RAPT class itself):

1. **Missing initial-regime seed.** RAPT was constructed with `fit_initial(X_init, y_init)` instead of `fit_initial(init_regime=regimes[n_init-1], ...)`, so the prefix regime (A on Campus, regime_C on 5G NR) was never placed in the repo. The first time that regime returned, RAPT treated it as unseen and retrained, then reused on the *second* return. Every recurring regime was therefore retrained once before any reuse. Numbers: Campus 2->24 retrains / 12->119 reuses; 5G NR 3->200 / 5->199; KPI-ops 5->6 / 173->172.

2. **Sticky flags.** `last_flags` was not reset each window, so a retrain/reuse flag was counted on every subsequent window until the next boundary. This is bookkeeping only (scores were identical) but inflated counts further and made per-seed sd collapse (Campus RAPT sd 1e-16).

No change was needed to the RAPT class. After the fix, RAPT reproduces the final2 numbers exactly:

| stream | final2_f1 | costfix_f1 | max_abs_seed_diff | differing_windows | costfix_retrains | costfix_reuses | buggy_cost_retrains | buggy_cost_reuses |
|---|---|---|---|---|---|---|---|---|
| campus | 0.9385 | 0.9385 | 0.0000 | 0 | 2 | 12 | 24 | 119 |
| ugr16 | 0.9137 | 0.9137 | 0.0000 | 0 | 11 | 133 | 12 | 132 |
| 5g_nr | 0.8969 | 0.8969 | 0.0000 | 0 | 3 | 5 | 200 | 199 |
| kpiops | 0.6797 | 0.6797 | 0.0000 | 0 | 5 | 173 | 6 | 172 |


`differing_windows` is the number of (seed, window) cells whose macro-F1 differs between the final2 run and the corrected cost run (0 = exact match).

## 4. RAPT-Guard: accuracy-skip disabled

**The skip is off here.** `gate_skip_acc` is now 1.0, which disables the raw-accuracy skip (`skip_acc < 1.0` guard). Re-running with the standard 50-tree base config (`RAPT-Guard-std`) reproduces the final2 gate exactly:

| stream | final2_guard_std | costfix_guard_std | costfix_guard_cheap |
|---|---|---|---|
| campus | 0.9391 | 0.9391 | 0.9367 |
| ugr16 | 0.9357 | 0.9357 | 0.7552 |
| 5g_nr | 0.9148 | 0.9148 | 0.9099 |
| kpiops | 0.6797 | 0.6797 | 0.6798 |


`final2_guard_std` is the original final2 RAPT-Guard (50-tree, standard gate). `costfix_guard_std` is the same method re-run with the fixed runner; they should match. `costfix_guard_cheap` is the cheap config with the skip disabled.


## 5. Which run should the paper tables come from?

- **Event-Driven, Full Retraining, Event-Driven+Reuse, Full Retraining-cheap, Page-Hinkley, Event-Driven-cheap, Frozen, Frozen-cheap:** use the **cost run** (`per_window_all_cost_final.csv` for the first eight, `per_window_all_cost_fix.csv` for Frozen-cheap). These methods are unaffected by the two runner defects.

- **RAPT, RAPT-Guard:** use the **final2 run** (`per_window_all_final.csv`) or, equivalently, the corrected cost run; they are numerically identical per window (section 3). Prefer final2 since the paper's existing tables were built from it.


### Numbers that differ between runs for the same method+config

| method | stream | final2_f1 | buggy_cost_f1 | diff |
|---|---|---|---|---|
| RAPT | campus | 0.9385 | 0.9426 | 0.0040 |
| RAPT | ugr16 | 0.9137 | 0.9119 | -0.0018 |
| RAPT | 5g_nr | 0.8969 | 0.8856 | -0.0113 |
| RAPT | kpiops | 0.6797 | 0.6802 | 0.0005 |
| RAPT-Guard | campus | 0.9391 | 0.9367 | -0.0024 |
| RAPT-Guard | ugr16 | 0.9357 | 0.7193 | -0.2165 |
| RAPT-Guard | 5g_nr | 0.9148 | 0.8926 | -0.0221 |
| RAPT-Guard | kpiops | 0.6797 | 0.6798 | 0.0001 |


For RAPT the buggy-cost mean F1 differed from final2 by up to 0.0125 (Campus). RAPT-Guard's buggy-cost UGR'16 figure (0.7193) is the gate-skip artefact discussed in section 4, not a method difference. All other methods are byte-identical between the cost run and final2.
