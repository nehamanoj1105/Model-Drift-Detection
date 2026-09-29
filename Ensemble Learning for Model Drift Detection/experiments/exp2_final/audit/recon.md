# Stage 0 Reconnaissance & Architectural Protocol Mapping

## 1. Vendored Code & File Paths
All core baseline modules are vendored unchanged from Exp1 into experiments/exp2_final/vendor/exp1/:
- heterogeneous_ensemble.py (HeterogeneousEnsemble, create_base_ensemble): Base Random Forest (50 trees, max_depth=7) + Extra Trees (50 trees, max_depth=7) ensemble with soft-voting.
- streaming_preprocessor.py (StreamingPreprocessor, get_feature_names): Online standardizer fit on X_init warmup prefix.
- event_driven.py (EventDrivenEnsemble): Drift-detector ensemble using rolling error threshold (mu + k*sigma).
- 
apt_e.py (HeterogeneousEnsembleV2Fixed, RegimeCheckpointV2Fixed, RAPTv2SystemFixed): RAPT-E baseline controller with EWMA model weighting and fallback escalation.
- 
apt.py (RegimeCheckpoint, RAPTSystem): Original RAPT baseline.

## 2. Dataset S2 (9B 5G NR Latency) Exp1 Protocol Specification
- Stream file: experiments/exp9b/data/processed_exp9b_stream.csv (499 rows, 24 columns).
- Warmup prefix: 
_init = 99 warmup windows (1,980 instances).
- Preprocessor: StreamingPreprocessor fit strictly on X_init (windows 0..98).
- Evaluation windows: Windows 99 through 498 (400 evaluation windows).
- Baseline reference target under Exp1 protocol:
  - Frozen Macro F1: ~0.8839 - 0.8961
  - Event-Driven Macro F1: ~0.8903 - 0.9396
  - Local Retraining Macro F1: ~0.9506

## 3. Dataset 9A Material Confirmation
- Dataset 9A (NTNU Campus QoS) exhibits extreme ceiling saturation (Frozen Macro F1 = 0.9945).
- Excluded from primary candidate transfer claims per Phase 0 audit findings.

## 4. Exp2 Corrected Audit & Fixed Bugs Summary
- Outcome timing fixed: Outcomes are recorded ONLINE during streaming, not post-run.
- Oracle fixed: Oracle evaluates candidates dynamically per decision interval rather than staying locked to Frozen.
- Similarity-Weighted fixed: Applies soft temperature softmax weighting across candidates instead of single-candidate argmax.
- Gate ranges: Replaced static upper-capped gates with strict 2-sided tolerance |Macro F1 - Exp1_Ref| <= 0.005.
- Labels & features: Online, leak-proof calculation using only information available prior to interval t.
