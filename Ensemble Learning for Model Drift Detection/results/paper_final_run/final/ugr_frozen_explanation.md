# Why is Frozen competitive on UGR'16?

- Frozen macro-F1 0.9688, Full Retraining 0.9591, RAPT 0.8503.
- Covariate shift between pre-window pool and each held-out window: mean KS 0.2621, mean PSI 1.8977 (ugr_stationarity.csv). Small values mean the features are comparatively stable.
- Target-semantics proxy drop across the stream 0.5430; mean stored-buffer attack fraction 0.0776.

Data-supported explanation: see the numbers above; the two explanations are not cleanly separated by this run and the report says so.
