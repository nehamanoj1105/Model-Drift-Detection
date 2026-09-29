# Experiment Report: SEA Recurring-Concept Policy Transfer Benchmark

**Benchmark Dataset**: Synthetic SEA Concepts Stream (Street & Kim, 2001)  
**Concept Schedule**: $A \rightarrow B \rightarrow C \rightarrow A \rightarrow B \rightarrow C \rightarrow A \rightarrow B \rightarrow C$ (9 Concept Blocks, 10 Windows/Block)  
**Stream Parameters**: 500 samples per window, 90 windows per seed = 45,000 samples per seed  
**Seeds Evaluated**: 5 Seeds (`[42, 43, 44, 45, 46]`)  
**Total Evaluated Streaming Windows**: 450 Window Observations ($90 \times 5$)  

---

## 1. Executive Summary & Primary Research Question

### Core Research Question
> **When a previously observed concept returns in a data stream, can a previously learned ensemble policy or model state be transferred to the recurring concept more effectively than cold-start adaptation, and can observable distributional similarity reliably identify when such transfer is safe?**

### Final Decision: **CASE C — TRANSFER LIMITATION UNDER RAPID BASE-MODEL ADAPTATION**

```
+---------------------------------------------------------------------------------------------------------------------+
|                                            SEA BENCHMARK SUMMARY RESULTS                                            |
+------------------------------------+------------------+------------------+----------------+-------------+-----------+
| Method                             | Macro F1         | Accuracy         | Recovery W+1   | CPU Time(s) | Retrains  |
+------------------------------------+------------------+------------------+----------------+-------------+-----------+
| Global Adaptive Ensemble           | 0.7954 ± 0.1126  | 0.8158 ± 0.1126  | 0.6151         | 51.63s      | 0.0       |
| Fixed Equal Ensemble               | 0.7953 ± 0.1126  | 0.8157 ± 0.1126  | 0.6151         | 50.75s      | 0.0       |
| Similarity Model-State Transfer    | 0.7952 ± 0.1126  | 0.8157 ± 0.1126  | 0.6151 ± 0.002 | 51.41s      | 0.0       |
| Oracle Recurrence Transfer         | 0.7946 ± 0.1132  | 0.8151 ± 0.1132  | 0.6154 ± 0.002 | 130.59s     | 2.0       |
| Similarity Policy Transfer         | 0.7944 ± 0.1140  | 0.8149 ± 0.1140  | 0.6130 ± 0.000 | 59.30s      | 8.0       |
| Cold Adaptation Baseline           | 0.7943 ± 0.1140  | 0.8148 ± 0.1140  | 0.6130 ± 0.000 | 57.88s      | 8.0       |
+------------------------------------+------------------+------------------+----------------+-------------+-----------+
```

### Key Findings & Insights

1. **Insignificant Model Transfer Headroom**:
   - Model-State Transfer provided a minor $+0.0009$ Macro F1 gain over Cold Adaptation ($p = 0.1862$, statistically insignificant).
   - Ground-truth Oracle Recurrence Transfer provided a minor $+0.0003$ Macro F1 gain ($p = 0.7356$, statistically insignificant).
2. **Rapid Base-Model Adaptation**:
   - On decision tree ensembles (Random Forest, Extra Trees, Gradient Boosting), prequential sliding-window adaptation adapts to new SEA decision thresholds ($\theta_A=7 \to \theta_B=10 \to \theta_C=13$) within **1 to 2 streaming windows** (500–1000 samples).
   - Because online adaptation is extremely fast, the cold-start penalty is transient (lasting only 1 window per 10-window block), capping stream-wide transfer headroom below 0.1% Macro F1.
3. **Fingerprint Audit (Covariate Shift vs. Concept Drift)**:
   - In standard SEA, input features $X = (f_1, f_2, f_3)$ are uniformly distributed $U(0, 10)^3$ across all concepts.
   - A purely feature-space fingerprint $F(X)$ produces $Sim(F_A, F_B) \approx 0.999$, causing severe **false transfers**.
   - Implementing **Joint Distribution Fingerprinting** $F(X, Y)$ (incorporating label positive rates $P(Y=1)$ and conditional feature quantiles) successfully resolved concept identities, achieving zero false transfers.

---

## 2. Benchmark Specification & Synthetic Generator

The SEA Concepts streaming benchmark (Street & Kim, 2001) was generated with controlled recurring concept boundaries:

* **Input Features**: $f_1, f_2, f_3 \sim U(0, 10)$. Features $f_1, f_2$ are decision attributes; $f_3$ is an irrelevant noise attribute.
* **Labeling Rule**: $y = 1 \iff f_1 + f_2 \le \theta$, else $0$, with 10% label noise.
* **Concept Thresholds**:
  - **Concept A**: $\theta_A = 7.0$ (Positive Class Rate $\approx 30.1\%$)
  - **Concept B**: $\theta_B = 10.0$ (Positive Class Rate $\approx 50.0\%$)
  - **Concept C**: $\theta_C = 13.0$ (Positive Class Rate $\approx 70.8\%$)
* **Stream Schedule**: 9 Concept Blocks ($A \rightarrow B \rightarrow C \rightarrow A \rightarrow B \rightarrow C \rightarrow A \rightarrow B \rightarrow C$).
* **Window Size**: 500 samples/window, 10 windows/block $\implies 90$ windows (45,000 samples) per seed.

---

## 3. Method Descriptions & Ablation Spectrum

We evaluated 6 methods across 5 seeds:

1. **Fixed Equal Ensemble**: Static $[1/3, 1/3, 1/3]$ ensemble weights.
2. **Global Adaptive Ensemble**: Global policy updated continuously based on exponential inverse-loss model performance.
3. **Cold Adaptation Baseline**: Upon concept shift, base models retrain on recent data and policy weights reset to $[1/3, 1/3, 1/3]$ (no memory).
4. **Similarity Policy Transfer**: Stores fingerprint $F_k(X, Y)$ and policy weights $\pi_k$. Upon drift, if $Sim(F_t, F_k) \ge 0.97$, transfers policy weights $\pi_k$.
5. **Similarity Model-State Transfer**: Stores fingerprint $F_k(X, Y)$ and complete base-model checkpoints $(RF_k, ET_k, GB_k)$. Upon drift, if $Sim \ge 0.97$, restores complete base-model checkpoints.
6. **Oracle Recurrence Transfer**: Uses ground-truth `concept_id` to retrieve previous model states for recurring concepts $A, B, C$.

---

## 4. Recurrence Recovery Curve Analysis

For each recurrence event (Blocks 4, 5, 6, 7, 8, 9), we measured recovery performance across windows $+1, +2, +3, +5$ following concept transition:

```
+-----------------------------------------------------------------------------------------+
|                               POST-RECURRENCE RECOVERY F1                               |
+------------------------------------+------------+------------+------------+-------------+
| Method                             | Window +1  | Window +2  | Window +3  | Window +5   |
+------------------------------------+------------+------------+------------+-------------+
| Cold Adaptation Baseline           | 0.6130     | 0.8421     | 0.8510     | 0.8548      |
| Similarity Policy Transfer         | 0.6130     | 0.8421     | 0.8510     | 0.8548      |
| Similarity Model-State Transfer    | 0.6151     | 0.8425     | 0.8512     | 0.8550      |
| Oracle Recurrence Transfer         | 0.6154     | 0.8428     | 0.8514     | 0.8551      |
+------------------------------------+------------+------------+------------+-------------+
```

**Observation**: Recovery curves demonstrate that cold-start models recover over 97% of steady-state performance by **Window +2** (sample 1,000). Consequently, transferring a historical checkpoint improves Window +1 Macro F1 by only $+0.0021$ to $+0.0024$ F1.

---

## 5. Statistical Hypothesis Testing

Paired window-level comparisons were conducted across all 450 window observations ($N=450$). 95% Confidence Intervals were computed using 1,000 bootstrap resamples, with significance assessed via the Wilcoxon signed-rank test.

```
+-----------------------------------------------------------------------------------------+
|                                PAIRED STATISTICAL TESTS                                 |
+------------------------------------+------------+------------------+----------+---------+
| Comparison                         | Mean Diff  | 95% CI           | p-value  | Effect  |
+------------------------------------+------------+------------------+----------+---------+
| Cold vs. Similarity Policy Transfer| +0.000064  | [-0.0004, 0.0005]| 0.6462   | 0.0132  |
| Cold vs. Similarity Model Transfer | +0.000914  | [+0.0001, 0.0018]| 0.1862   | 0.1002  |
| Cold vs. Oracle Recurrence Transfer| +0.000320  | [-0.0005, 0.0011]| 0.7356   | 0.0379  |
| Policy vs. Model State Transfer    | +0.000850  | [-0.0003, 0.0018]| 0.5102   | 0.0935  |
+------------------------------------+------------+------------------+----------+---------+
```

### Evaluation of Pre-Registered Hypotheses

* **H1 (Model-State Transfer Accelerates Recovery)**: **REJECTED**. $p = 0.1862 > 0.05$, gain $< +0.01$ F1.
* **H2 (Policy Transfer Outperforms Cold Adaptation)**: **REJECTED**. Policy transfer gain is $+0.000064$ F1 ($p = 0.6462$).
* **H3 (Fingerprint Similarity Tracks Oracle)**: **SUPPORTED**. Joint fingerprinting $F(X, Y)$ matched Oracle behavior ($|F1_{\text{model}} - F1_{\text{oracle}}| = 0.0006 < 0.01$).
* **H4 (Transfer Benefit Decays Over Time)**: **SUPPORTED**. The small gain was concentrated in Window +1 (+0.0021 F1) and vanished by Window +3 (+0.0002 F1).

---

## 6. Generated Publication Figures

All publication figures were generated and saved to:  
`c:\Users\emhaenn\Downloads\Model-Drift-Detection\Ensemble Learning for Model Drift Detection\experiment5\results\sea_transfer\figures\`

1. **Figure 1 — Concept Timeline**: [fig1_concept_timeline.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig1_concept_timeline.png)  
   Illustrates the 9-block SEA concept schedule ($A \to B \to C \to A \to B \to C \to A \to B \to C$) and recurrence transition boundaries.
2. **Figure 2 — Streaming Macro F1 Over Time**: [fig2_streaming_f1_over_time.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig2_streaming_f1_over_time.png)  
   Time-series of Macro F1 across windows 0 to 89 for all 6 evaluated methods.
3. **Figure 3 — Post-Recurrence F1 Recovery Curves**: [fig3_post_recurrence_recovery_curves.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig3_post_recurrence_recovery_curves.png)  
   Post-drift recovery curves showing F1 at windows $+1, +2, +3, +5$.
4. **Figure 4 — Recovery Time Comparison**: [fig4_recovery_time_comparison.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig4_recovery_time_comparison.png)  
   Bar plot comparing number of windows needed to reach 95% steady-state F1.
5. **Figure 5 — Similarity vs. Transfer Quality**: [fig5_similarity_vs_transfer_quality.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig5_similarity_vs_transfer_quality.png)  
   Scatter plot analyzing correlation between fingerprint similarity and transfer F1 gain.
6. **Figure 6 — False Transfer Confusion Analysis**: [fig6_false_transfer_analysis.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig6_false_transfer_analysis.png)  
   4-quadrant confusion matrix breakdown (Successful, False, Missed, Correct Rejection).
7. **Figure 7 — CPU Compute Cost vs. Recovery**: [fig7_cpu_vs_recovery_tradeoff.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig7_cpu_vs_recovery_tradeoff.png)  
   Scatter plot comparing total CPU runtime vs Window +1 recovery performance.
8. **Figure 8 — Ensemble Policy Weights Dynamics**: [fig8_policy_weights_dynamics.png](file:///c:/Users/emhaenn/Downloads/Model-Drift-Detection/Ensemble%20Learning%20for%20Model%20Drift%20Detection/experiment5/results/sea_transfer/figures/fig8_policy_weights_dynamics.png)  
   Stacked line plot of active ensemble weights ($w_{RF}, w_{ET}, w_{GB}$) across concept shifts.

---

## 7. Conclusions & Architectural Guidance for RAPT

### Research Conclusion
On lightweight decision tree streams where base models adapt rapidly within 1–2 windows (~500–1000 samples), **historical model-state retrieval provides negligible stream-wide benefit (< 0.1% F1)**.

### Architectural Takeaway for RAPT Design
RAPT model-repository transfer is beneficial under specific structural conditions:
1. **Expensive Base Models**: Neural networks or complex models where full retraining takes thousands of steps.
2. **Severe Cold-Start Drop**: Concept drifts where cold adaptation causes catastrophic performance drops spanning many streaming windows.
3. **High Data Velocity**: Streams where incoming window size is small relative to the complexity of the concept shift.
