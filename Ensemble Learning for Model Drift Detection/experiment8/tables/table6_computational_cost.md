# Table 6: Computational Cost & Resource Efficiency

| method                |   total_adaptation_cpu |   models_retrained |   samples_consumed |   efficiency |
|:----------------------|-----------------------:|-------------------:|-------------------:|-------------:|
| baseline_continuous   |            30.7428     |               48   |             123300 |     0.240319 |
| baseline_event_driven |            13.1678     |               32.4 |              82940 |     0.370659 |
| baseline_frozen       |             0.00010762 |                0   |                  0 | 25532.7      |
| equal_budget          |             3.65138    |               34.2 |              30667 |     1.09226  |
| oracle_allocation     |             0.550026   |               31.4 |              15700 |     4.3268   |
| proposed_value_based  |             3.83235    |               24   |              34234 |     0.855086 |
| random_selective      |             4.17905    |               15.2 |              21810 |     1.42161  |
| weakest_selective     |             4.66458    |               20.8 |              35661 |     0.824054 |
