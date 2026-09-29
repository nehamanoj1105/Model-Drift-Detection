# Table 4: Overall Predictive Performance (Main Synthetic Stream)

| method                |   mean_f1 |   mean_accuracy |   mean_precision |   mean_recall |
|:----------------------|----------:|----------------:|-----------------:|--------------:|
| baseline_continuous   |  0.779635 |        0.830925 |         0.796234 |      0.782056 |
| baseline_event_driven |  0.779221 |        0.8303   |         0.797006 |      0.781381 |
| baseline_frozen       |  0.763353 |        0.823775 |         0.781543 |      0.767446 |
| equal_budget          |  0.761947 |        0.81935  |         0.787548 |      0.767241 |
| oracle_allocation     |  0.766617 |        0.82245  |         0.788636 |      0.771667 |
| proposed_value_based  |  0.772936 |        0.825575 |         0.79253  |      0.775488 |
| random_selective      |  0.774902 |        0.825975 |         0.793036 |      0.779444 |
| weakest_selective     |  0.773288 |        0.830125 |         0.789843 |      0.775457 |
