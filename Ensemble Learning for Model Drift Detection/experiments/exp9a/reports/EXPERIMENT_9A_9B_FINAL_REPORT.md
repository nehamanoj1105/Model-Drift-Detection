# EXPERIMENTS 9A & 9B — FINAL CROSS-DATASET REPORT

_Auto-generated. 9A is now evaluated on three independent telecom datasets; 9B is the existing 5G latency QoS drift-severity evaluation. The proposed model is referred to simply as **RAPT**; RAPT-Enhanced applies the existing Enhanced-Hybrid-RAPT mechanisms and adds no new algorithm variant._

## 9A — Three telecom datasets

Datasets: 5G Campus Network QoS, UGR'16, NordicDat. Models: Frozen, Event-Driven, Full Retraining, RAPT, RAPT-Enhanced; historical drift detectors ADWIN, EDD, Page-Hinkley, EDMA. Protocol: prequential Test-Then-Train, no leakage, identical class-anchored buffer for all adaptive methods, seeds [42,43,44,45,46].

| Dataset       | Model           | Accuracy          | Macro-F1          | Precision         | Recall            | Adapt CPU          | Runtime            | Retrains            | Reuses              |
|:--------------|:----------------|:------------------|:------------------|:------------------|:------------------|:-------------------|:-------------------|:--------------------|:--------------------|
| 5G Campus QoS | Frozen          | 0.9822 +/- 0.0006 | 0.9364 +/- 0.0023 | 0.9379 +/- 0.0025 | 0.9377 +/- 0.0017 | 0.0000 +/- 0.0000  | 1.2482 +/- 0.0126  | 0.0000 +/- 0.0000   | 0.0000 +/- 0.0000   |
| 5G Campus QoS | Event-Driven    | 0.9908 +/- 0.0003 | 0.9636 +/- 0.0006 | 0.9637 +/- 0.0008 | 0.9646 +/- 0.0002 | 0.1080 +/- 0.0062  | 1.3926 +/- 0.0187  | 1.0000 +/- 0.0000   | 0.0000 +/- 0.0000   |
| 5G Campus QoS | Full Retraining | 0.9951 +/- 0.0000 | 0.9836 +/- 0.0013 | 0.9838 +/- 0.0015 | 0.9843 +/- 0.0012 | 1.4008 +/- 0.0103  | 2.6677 +/- 0.0158  | 14.0000 +/- 0.0000  | 0.0000 +/- 0.0000   |
| 5G Campus QoS | RAPT            | 0.9838 +/- 0.0003 | 0.9381 +/- 0.0016 | 0.9393 +/- 0.0017 | 0.9392 +/- 0.0016 | 0.2292 +/- 0.0057  | 1.4931 +/- 0.0181  | 2.0000 +/- 0.0000   | 12.0000 +/- 0.0000  |
| 5G Campus QoS | RAPT-Enhanced   | 0.9838 +/- 0.0003 | 0.9381 +/- 0.0016 | 0.9393 +/- 0.0017 | 0.9392 +/- 0.0016 | 0.2337 +/- 0.0147  | 2.1046 +/- 0.0468  | 2.0000 +/- 0.0000   | 12.0000 +/- 0.0000  |
| UGR'16        | Frozen          | 0.9899 +/- 0.0007 | 0.9690 +/- 0.0029 | 0.9795 +/- 0.0030 | 0.9662 +/- 0.0028 | 0.0000 +/- 0.0000  | 2.4914 +/- 0.0201  | 0.0000 +/- 0.0000   | 0.0000 +/- 0.0000   |
| UGR'16        | Event-Driven    | 0.9898 +/- 0.0013 | 0.8972 +/- 0.0277 | 0.9042 +/- 0.0269 | 0.8955 +/- 0.0280 | 0.5754 +/- 0.0066  | 3.1303 +/- 0.0293  | 3.0000 +/- 0.0000   | 0.0000 +/- 0.0000   |
| UGR'16        | Full Retraining | 0.9873 +/- 0.0007 | 0.9595 +/- 0.0012 | 0.9726 +/- 0.0014 | 0.9580 +/- 0.0015 | 23.6318 +/- 0.3942 | 26.2627 +/- 0.4072 | 144.0000 +/- 0.0000 | 0.0000 +/- 0.0000   |
| UGR'16        | RAPT            | 0.9635 +/- 0.0182 | 0.8360 +/- 0.0262 | 0.8527 +/- 0.0205 | 0.8336 +/- 0.0294 | 1.9711 +/- 0.0198  | 4.4990 +/- 0.0165  | 11.0000 +/- 0.0000  | 133.0000 +/- 0.0000 |
| UGR'16        | RAPT-Enhanced   | 0.9880 +/- 0.0017 | 0.9276 +/- 0.0123 | 0.9375 +/- 0.0117 | 0.9251 +/- 0.0123 | 2.7182 +/- 0.0243  | 5.9329 +/- 0.0465  | 11.0000 +/- 0.0000  | 133.0000 +/- 0.0000 |
| NordicDat     | Frozen          | 0.4770 +/- 0.0123 | 0.2779 +/- 0.0224 | 0.4215 +/- 0.0226 | 0.2487 +/- 0.0229 | 0.0000 +/- 0.0000  | 2.5488 +/- 0.0128  | 0.0000 +/- 0.0000   | 0.0000 +/- 0.0000   |
| NordicDat     | Event-Driven    | 0.4583 +/- 0.0367 | 0.2547 +/- 0.0177 | 0.3909 +/- 0.0208 | 0.2274 +/- 0.0151 | 0.2533 +/- 0.0474  | 2.8435 +/- 0.0490  | 2.2000 +/- 0.4472   | 0.0000 +/- 0.0000   |
| NordicDat     | Full Retraining | 0.5694 +/- 0.0075 | 0.4227 +/- 0.0299 | 0.4938 +/- 0.0233 | 0.4085 +/- 0.0293 | 2.2034 +/- 0.0100  | 4.7593 +/- 0.0193  | 18.0000 +/- 0.0000  | 0.0000 +/- 0.0000   |
| NordicDat     | RAPT            | 0.6172 +/- 0.0127 | 0.3818 +/- 0.0184 | 0.4703 +/- 0.0185 | 0.3582 +/- 0.0192 | 0.2594 +/- 0.0021  | 2.8257 +/- 0.0331  | 2.0000 +/- 0.0000   | 16.0000 +/- 0.0000  |
| NordicDat     | RAPT-Enhanced   | 0.5677 +/- 0.0222 | 0.3783 +/- 0.0321 | 0.4600 +/- 0.0307 | 0.3614 +/- 0.0318 | 2.6492 +/- 0.2561  | 5.8812 +/- 0.2327  | 2.0000 +/- 0.0000   | 16.0000 +/- 0.0000  |

## 9A selection for cross-dataset comparison

UGR'16 is used for the head-to-head comparison with 9B (5G latency QoS): both are independent recurring-regime telecom streams, UGR'16 provides real temporal/concept drift and strong natural recurrence (168 recurrence events), and it is a genuine intrusion-detection task distinct from 9B's QoS-regression-style task.

| Dataset                     | Model           | Macro-F1          | Accuracy          |   Adapt CPU (s) |   Retrains |
|:----------------------------|:----------------|:------------------|:------------------|----------------:|-----------:|
| 9A UGR'16 (binary anomaly)  | Frozen          | 0.9690 +/- 0.0029 | 0.9899 +/- 0.0007 |           0     |          0 |
| 9A UGR'16 (binary anomaly)  | Event-Driven    | 0.8972 +/- 0.0277 | 0.9898 +/- 0.0013 |           0.575 |          3 |
| 9A UGR'16 (binary anomaly)  | Full Retraining | 0.9595 +/- 0.0012 | 0.9873 +/- 0.0007 |          23.632 |        144 |
| 9A UGR'16 (binary anomaly)  | RAPT            | 0.8360 +/- 0.0262 | 0.9635 +/- 0.0182 |           1.971 |         11 |
| 9A UGR'16 (binary anomaly)  | RAPT-Enhanced   | 0.9276 +/- 0.0123 | 0.9880 +/- 0.0017 |           2.718 |         11 |
| 9B 5G latency QoS (3-class) | Frozen          | 0.8961 +/- 0.0099 | 0.9075 +/- 0.0061 |           0     |          0 |
| 9B 5G latency QoS (3-class) | Event-Driven    | 0.8903 +/- 0.0107 | 0.9055 +/- 0.0060 |           1.041 |         11 |
| 9B 5G latency QoS (3-class) | Full Retraining | 0.9027 +/- 0.0059 | 0.9145 +/- 0.0045 |           0.764 |          9 |
| 9B 5G latency QoS (3-class) | RAPT            | 0.8894 +/- 0.0110 | 0.9025 +/- 0.0068 |           0.396 |          3 |
| 9B 5G latency QoS (3-class) | RAPT-Enhanced   | 0.8915 +/- 0.0117 | 0.9055 +/- 0.0076 |           0.861 |          3 |

## 9B — 5G latency QoS drift-severity evaluation

Existing experiment, unchanged: natural drift plus controlled covariate, concept and recurring-concept drift at severities [0.10, 0.20, 0.30, 0.50, 1.00]. Full detail, figures and tables are in `results/experiment_9b/` and `EXPERIMENT_9B_FINAL_REPORT.md`.

## Cross-dataset conclusion (observed)

RAPT consistently reduces adaptation cost relative to Full Retraining across the three 9A datasets and 9B while matching Frozen on stable streams. Under concept change coupled with regime recurrence (UGR'16) base RAPT's blind policy reuse is harmful and the parity-refit enhancement recovers performance. These are observed results; statistical significance is reported in the per-experiment tables and must not be assumed where p >= 0.05.
