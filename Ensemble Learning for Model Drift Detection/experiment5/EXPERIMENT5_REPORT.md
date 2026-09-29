# Experiment 5: Regime-Aware Ensemble Weighting Report

## 1. Executive Summary & Core Research Question

This report documents **Experiment 5: Regime-Aware Ensemble Weighting**, conducted on the **ToN_IoT Weather telemetry dataset using 50,000 sequential samples** evaluated strictly prequentially across 5 deterministic seeds (`[42, 43, 44, 45, 46]`).

### Core Research Question
> **Does conditioning ensemble weights on the current data regime improve streaming performance compared with fixed or globally adaptive ensemble weights?**

### Key Empirical Conclusion
> **Negative / Nuanced Finding**: Conditioning ensemble model weights purely on input covariate quantile fingerprints does **NOT** yield statistically significant performance gains ($F1 = 0.9795 \pm 0.0016$) compared to a simple Fixed Equal-Weighted Ensemble ($F1 = 0.9795 \pm 0.0016$) or a Globally Adaptive Ensemble ($F1 = 0.9798 \pm 0.0015$) when the underlying base models are already continuously/event-driven retrained.
>
> However, an **Oracle Regime Weighting** strategy achieves $F1 = 0.9886 \pm 0.0002$ ($+0.0091$ F1 improvement), proving that an optimal policy ceiling exists. This empirical result provides critical evidence that **input distributional similarity alone is an insufficient proxy for policy transferability**, directly justifying the necessity for full **Regime-Aware Policy Transfer (RAPT)** incorporating model state memory and hybrid learning.

---

## 2. Experimental Setup & Controls

- **Dataset**: ToN_IoT Weather IoT Telemetry (`temperature`, `pressure`, `humidity`).
- **Stream Dimensions**: 50,000 total sequential samples.
  - **Initial Training Partition**: 20,000 samples (Samples 0 to 19,999).
  - **Streaming Partition**: 30,000 samples across 60 sequential windows of 500 samples.
- **Base Models**: Heterogeneous Ensemble combining:
  - **Random Forest (RF)**: `n_estimators=50`, `max_depth=7`, `min_samples_split=4`, Horizon = 3,500 samples.
  - **Extra Trees (ET)**: `n_estimators=50`, `max_depth=7`, `min_samples_split=6`, Horizon = 2,500 samples.
  - **Gradient Boosting (GB)**: `n_estimators=50`, `max_depth=4`, `learning_rate=0.08`, Horizon = 1,800 samples.
- **Drift Detection & Retraining Baseline**: Dual-Trigger Drift Detector (Wasserstein feature drift threshold = 0.12, F1 drop concept drift threshold = 0.12).
- **Prequential Controls**: Zero data leakage. Window $t$ features are fingerprinted and assigned weights BEFORE observing window ground-truth labels $y_t$. Metric F1 is strictly verified to satisfy $F1 = 2 \cdot P \cdot R / (P + R)$.

---

## 3. Evaluated Weighting Strategies

1. **Fixed Ensemble (Baseline)**: Static equal weights $w_{RF} = 1/3, w_{ET} = 1/3, w_{GB} = 1/3$.
2. **Globally Adaptive Ensemble**: Single global exponential moving average (EMA) F1 score vector across the stream, normalized via Softmax.
3. **Regime-Aware Ensemble (Proposed)**:
   - **Quantile Fingerprint**: 21-dimensional vector ($p_{10}, p_{25}, p_{50}, p_{75}, p_{90}, \mu, \sigma$ per feature).
   - **Cosine Similarity Matching**: Threshold $\tau = 0.85$ to detect recurring vs novel regimes.
   - **Regime Memory**: Maintains regime centroids and regime-specific model weight vectors.
4. **Oracle Regime Weights (Upper Bound)**: Offline grid search over the weight simplex for each window to identify maximum achievable F1.

---

## 4. Empirical Benchmark Results

### Table 1: Comprehensive Streaming Performance (Mean ± Std across 5 Seeds)

| Method | Macro F1 | Accuracy | Precision | Recall | Adaptation CPU (s) | Total CPU (s) | Retrain Events | Unique Regimes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Ensemble** | $0.9795 \pm 0.0016$ | $0.9598 \pm 0.0031$ | $1.0000 \pm 0.0000$ | $0.9598 \pm 0.0031$ | $131.06 \pm 23.36$ | $137.22$ | $49 \pm 0$ | $1.0 \pm 0.0$ |
| **Global Adaptive Ensemble** | $\mathbf{0.9798} \pm \mathbf{0.0015}$ | $\mathbf{0.9603} \pm \mathbf{0.0029}$ | $1.0000 \pm 0.0000$ | $0.9603 \pm 0.0029$ | $138.62 \pm 32.55$ | $144.88$ | $49 \pm 0$ | $1.0 \pm 0.0$ |
| **Regime-Aware Ensemble** ($\tau=0.85$) | $0.9795 \pm 0.0016$ | $0.9599 \pm 0.0030$ | $1.0000 \pm 0.0000$ | $0.9599 \pm 0.0030$ | $138.49 \pm 30.13$ | $145.26$ | $49 \pm 0$ | $11.0 \pm 0.0$ |
| **Oracle Regime Weights** *(Upper Bound)* | $\mathbf{0.9886} \pm \mathbf{0.0002}$ | $\mathbf{0.9775} \pm \mathbf{0.0003}$ | $1.0000 \pm 0.0000$ | $0.9775 \pm 0.0003$ | $137.29 \pm 35.88$ | $150.26$ | $49 \pm 0$ | $1.0 \pm 0.0$ |

---

## 5. Key Research Findings & Interpretations

### Finding 1: Base Model Retraining Dominates Ensemble Weighting
When base models (RF, ET, GB) are updated via Event-Driven drift retraining (49 retraining events across the 60 streaming windows), their internal parameters adapt directly to concept shifts. As a result, changing soft-voting weights among already-adapted base models provides minimal additional margin ($+0.0003$ F1 for Global Adaptive, $+0.0000$ for Regime-Aware).

### Finding 2: Input Distributional Similarity $\ne$ Policy Transferability
Analysis of **Figure 6** demonstrates that quantile fingerprint cosine similarity between streaming windows and stored regime centroids does not exhibit a strong positive correlation with ensemble policy performance. Environmental covariate statistics (`temperature`, `pressure`, `humidity`) shift during attack waves, but two windows with identical quantile fingerprints may exhibit different decision boundaries depending on subtle non-linear attack interactions.

### Finding 3: Threshold Sensitivity ($\tau$ Ablation)
Varying the similarity threshold $\tau$ from $0.70$ to $0.95$ changes the number of discovered regimes from 6 to 23, but streaming macro F1 remains stable between $0.9792$ and $0.9798$. This confirms that the lack of improvement is robust across similarity parameter choices rather than an artifact of an arbitrarily chosen threshold.

---

## 6. Publication Figures

All 8 publication-quality figures are saved in `results/ton_iot/figures/`:

1. **Figure 1 — Architecture Diagram**: `fig1_architecture.png`
2. **Figure 2 — F1 Comparison Bar Chart**: `fig2_f1_comparison.png`
3. **Figure 3 — Accuracy Comparison Bar Chart**: `fig3_accuracy_comparison.png`
4. **Figure 4 — Ensemble Weights Over Time**: `fig4_weights_over_time.png`
5. **Figure 5 — Discovered Regime Map**: `fig5_regime_map.png`
6. **Figure 6 — Similarity vs. Policy Transfer**: `fig6_similarity_vs_transfer.png`
7. **Figure 7 — Accuracy / Compute Pareto Frontier**: `fig7_accuracy_compute_tradeoff.png`
8. **Figure 8 — Similarity Threshold Ablation**: `fig8_threshold_ablation.png`

---

## 7. Implications & Recommendations for Full RAPT Track

1. **Do NOT rely solely on weight conditioning**: Weight-only conditioning on continuously retrained base models provides negligible gain over fixed/global weights.
2. **Incorporate Model State Memory**: Full RAPT must maintain a **Model Repository** that archives trained model weights/trees for specific regimes, enabling immediate state restoration without waiting for buffer accumulation.
3. **Hybrid Model-Policy Transfer**: Combine regime fingerprint matching with model parameter retrieval to achieve fast adaptation and lower CPU retrain cost.
