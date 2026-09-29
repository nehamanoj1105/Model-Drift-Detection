# Project Overview: Model Drift Detection & Regime-Aware Policy Transfer (RAPT)

This project explores streaming machine learning under concept drift on real-world IoT sensor streams (ToN_IoT Weather dataset). The research evolved from initial adaptive ensemble weighting to explicit regime identification, culminating in **Regime-Aware Policy Transfer (RAPT)** and **Enhanced Hybrid RAPT**.

---

## Evolution of the Research Track

1. **[Experiments 1–3: Ensemble Foundations & Drift Detection](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/01_experiments_1to3.md)**
   Established baseline soft-voting ensembles, evaluated multi-armed bandit model weighting (Q-Learning, UCB, EXP3), and tested statistical drift detection (ADWIN/Wasserstein) on synthetic and real streams.

2. **[Experiment 4: Event-Driven Retraining Baseline](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/02_experiment4_baseline.md)**
   Benchmarked traditional event-driven retraining (Dual-Trigger Wasserstein distance + performance drop) on the 19,000-sample ToN_IoT streaming benchmark across 38 prequential windows.

3. **[RAPT Stage 1: Feasibility & Go/No-Go Verification](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/03_rapt_stage1_gonogo.md)**
   Validated the core concept of Regime-Aware Policy Transfer, proving that sub-millisecond quantile-fingerprint matching ($\mathbf{f} \in \mathbb{R}^{D \times N_q}$) can successfully recognize recurring regimes and reuse pre-trained models with zero retraining overhead.

4. **[RAPT Stage 2: Autonomous Regime Repository](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/04_rapt_stage2_repository.md)**
   Implemented a bounded-capacity repository ($K=8$) with LRU eviction and distance-weighted softmax synthesis ($s_k = \exp(-\gamma d_k)$), reducing adaptation CPU overhead by over $69\%$ compared to full retraining.

5. **[RAPT Stage 3: Two-Tier Hybrid RAPT Architecture](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/05_rapt_stage3_hybrid.md)**
   Introduced a two-tier hybrid framework combining the macro-regime repository with an online micro-learner via dynamic similarity blending weight $\beta_t = (1 - s_{\max})^\alpha$, achieving $29.60\%$ adaptation CPU savings and $+1.69\text{ } F_1$ point onset recovery gain.

6. **[Enhanced Hybrid RAPT: Parity & Recall Optimization](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/06_enhanced_hybrid.md)**
   Resolved RAPT's primary recall bottleneck by introducing Dynamic Threshold Tuning ($\tau_t$), Buffer-Blended Refitting, and Selective Parity Triggers—boosting $F_1$ to $0.7676$ (+5.04 points) while maintaining sub-millisecond fingerprint query latency and $48.4\%$ adaptation CPU savings.

7. **[Consolidated Corrections Log & Audit Trail](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiments/CORRECTIONS_LOG.md)**
   Provides a single, transparent audit trail documenting all intermediate draft numbers, OpenMP CPU timing variability findings, retrain buffer size rationale ($1,500$ vs $500$ samples), and metric identity reconciliations.
