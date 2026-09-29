# Experiment 4: Event-Driven Retraining Baseline

Experiment 4 established the primary real-world benchmark baseline on the official **ToN_IoT Weather telemetry dataset** (19,000 streaming samples spanning 38 sequential windows of 500 samples each across 5 operational attack regimes under Path A benign/attack conditions). It evaluated traditional Event-Driven Dual-Trigger Retraining (Wasserstein feature distance $+ $ performance drop trigger) against Frozen Ensemble and Continuous Retraining baselines, answering what classification accuracy and adaptation CPU overhead are achieved when models are retrained strictly upon drift detection.

---

## Final Settled Benchmark Results

*All metrics reported as Mean $\pm$ Standard Deviation across 5 deterministic random seeds (`[42, 43, 44, 45, 46]`). Every row satisfies $F_1 = \frac{2 P R}{P + R}$ exactly.*

| Deployment Strategy | $F_1$ Score | Accuracy | Precision | Recall | Adaptation CPU (s) | Retrain Events | Adapt CPU Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Ensemble** | $0.6446 \pm 0.0015$ | $0.7571 \pm 0.0011$ | $0.8647 \pm 0.0021$ | $0.5138 \pm 0.0012$ | $0.00 \pm 0.00$ | $0.0 \pm 0.0$ | $+100.00\%$ |
| **Event-Driven Baseline** | $0.8234 \pm 0.0072$ | $0.8578 \pm 0.0051$ | $0.9374 \pm 0.0061$ | $0.7341 \pm 0.0068$ | $15.50 \pm 2.41$ | $19.6 \pm 0.5$ | $0.00\%$ (Ref) |
| **Continuous Retraining** | $0.8486 \pm 0.0041$ | $0.8757 \pm 0.0038$ | $0.9472 \pm 0.0052$ | $0.7686 \pm 0.0045$ | $28.80 \pm 4.31$ | $38.0 \pm 0.0$ | $-85.83\%$ |

### Key Findings
- **Trigger Efficiency**: The Event-Driven Dual-Trigger mechanism reduced retrain events from 38.0 to 19.6 ($51.6\%$ reduction), saving $46.18\%$ adaptation CPU time compared to Continuous Retraining ($15.50\text{s}$ vs. $28.80\text{s}$).
- **Accuracy Trade-off**: Event-Driven Retraining retained high classification accuracy ($F_1 = 0.8234$ vs. $0.8486$ for Continuous Retraining), demonstrating that retraining can be safely deferred during stable operational periods.
