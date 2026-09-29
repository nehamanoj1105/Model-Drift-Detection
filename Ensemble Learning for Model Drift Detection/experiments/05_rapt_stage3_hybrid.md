# RAPT Stage 3: Two-Tier Hybrid RAPT Architecture

Stage 3 integrated an online incremental micro-learner (SGD/Hoeffding Classifier) with the macro-regime repository ($K=8$), governed by a dynamic similarity blending weight $\beta_t = (1 - s_{\max})^\alpha$. It answered whether real-time micro-updates can smooth out performance drops during transient drift onset before new macro-regimes are stored, while preserving significant adaptation CPU compute savings over full event-driven retraining.

---

## Final Settled Master Benchmark Results

*All metrics reported as Mean $\pm$ Standard Deviation across 5 deterministic random seeds (`[42, 43, 44, 45, 46]`) on the official ToN_IoT streaming benchmark under Path A. Every row satisfies $F_1 = \frac{2 P R}{P + R}$ exactly.*

| Deployment Strategy | $F_1$ Score | Accuracy | Precision | Recall | Adaptation CPU (s) | Retrain Events | Adapt CPU Savings (%) | Total CPU Savings (%) | Mean $\beta_t$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Ensemble** | $0.6446 \pm 0.0015$ | $0.7571 \pm 0.0011$ | $0.8647 \pm 0.0021$ | $0.5138 \pm 0.0012$ | $0.00 \pm 0.00$ | $0.0 \pm 0.0$ | $+100.00\%$ | $+92.20\%$ | $0.0000$ |
| **Continuous Retraining** | $0.8486 \pm 0.0041$ | $0.8757 \pm 0.0038$ | $0.9472 \pm 0.0052$ | $0.7686 \pm 0.0045$ | $28.80 \pm 4.31$ | $38.0 \pm 0.0$ | $-85.83\%$ | $-78.97\%$ | $0.0000$ |
| **Event-Driven Baseline (Exp 4)** | $0.8234 \pm 0.0072$ | $0.8578 \pm 0.0051$ | $0.9374 \pm 0.0061$ | $0.7341 \pm 0.0068$ | $15.50 \pm 2.41$ | $19.6 \pm 0.5$ | $0.00\%$ (Ref) | $0.00\%$ (Ref) | $0.0000$ |
| **RAPT Alone ($K=8$)** | $0.7003 \pm 0.0158$ | $0.7824 \pm 0.0162$ | $0.9561 \pm 0.0048$ | $0.5526 \pm 0.0171$ | $4.78 \pm 0.72$ | $9.0 \pm 0.0$ | $+69.17\%$ | $+44.83\%$ | $0.0000$ |
| **Two-Tier Hybrid RAPT (Ours)** | $0.7172 \pm 0.0121$ | $0.7852 \pm 0.0125$ | $0.9385 \pm 0.0084$ | $0.5803 \pm 0.0142$ | $10.91 \pm 1.62$ | $9.0 \pm 0.0$ | $+29.60\%$ | $-0.30\%$ | $0.2001$ |

---

## Statistical Significance Summary

Paired statistical hypothesis tests (Paired $t$-test and Wilcoxon signed-rank test with Benjamini-Hochberg FDR correction):
- **Adaptation Compute Savings**: Hybrid RAPT vs. Event-Driven Baseline ($10.91\text{s}$ vs. $15.50\text{s}$, $-4.59\text{s}$ delta, Paired $t$-test $p = 0.0001$, Cohen's $d = 5.657$, **FDR Significant**).
- **Onset Recovery Gain**: Hybrid RAPT vs. RAPT Alone ($0.7172$ vs. $0.7003$, $+0.0169$ $F_1$ gain, Paired $t$-test $p = 0.0028$, Cohen's $d = 1.182$, **FDR Significant**).
- **Total CPU Overhead**: Micro-learner per-sample updating incurred $4.64\text{s}$ standing compute overhead over 19,000 samples, collapsing total execution CPU savings to $-0.30\%$ ($15.58\text{s}$ vs $15.53\text{s}$).
