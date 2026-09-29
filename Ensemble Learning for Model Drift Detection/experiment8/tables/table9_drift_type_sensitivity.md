# Table 9: Performance on Recurring Concept Drift (SEA Benchmark)

| method                |   mean_f1 |   total_adaptation_cpu |   efficiency |
|:----------------------|----------:|-----------------------:|-------------:|
| baseline_continuous   |  0.755579 |            14.6872     |     0.595078 |
| baseline_event_driven |  0.752307 |             9.68865    |     0.896282 |
| baseline_frozen       |  0.665354 |             0.00014264 | 33900.8      |
| equal_budget          |  0.797513 |             7.76407    |     2.15631  |
| oracle_allocation     |  0.798905 |             0.950102   |     6.86688  |
| proposed_value_based  |  0.793674 |             4.50067    |     2.00472  |
| random_selective      |  0.767755 |             4.22045    |     2.38112  |
| weakest_selective     |  0.780513 |            17.9909     |     1.89392  |
