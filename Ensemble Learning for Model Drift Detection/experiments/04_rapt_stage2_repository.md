# RAPT Stage 2: Autonomous Regime Repository

Stage 2 designed and benchmarked the Autonomous Regime Repository ($K=8$), implementing LRU memory eviction and distance-weighted softmax synthesis ($s_k = \exp(-\gamma d_k)$) across continuous streaming regimes. It answered how much adaptation CPU compute can be saved by replacing event-driven model retraining with autonomous regime storage, retrieval, and multi-model synthesis during recurring drift.

---

## Final Settled Benchmark Results

*All metrics reported as Mean $\pm$ Standard Deviation across 5 deterministic random seeds (`[42, 43, 44, 45, 46]`). Every row satisfies $F_1 = \frac{2 P R}{P + R}$ exactly.*

| Strategy | $F_1$ Score | Accuracy | Precision | Recall | Adaptation CPU (s) | Retrain Events | Adapt CPU Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Event-Driven Baseline (Exp 4)** | $0.8234 \pm 0.0072$ | $0.8578 \pm 0.0051$ | $0.9374 \pm 0.0061$ | $0.7341 \pm 0.0068$ | $15.50 \pm 2.41$ | $19.6 \pm 0.5$ | $0.00\%$ (Ref) |
| **RAPT Alone ($K=8$)** | $0.7003 \pm 0.0158$ | $0.7824 \pm 0.0162$ | $0.9561 \pm 0.0048$ | $0.5526 \pm 0.0171$ | $4.78 \pm 0.72$ | $9.0 \pm 0.0$ | $+69.17\%$ |

### Key Findings
- **Adaptation CPU Savings**: RAPT Alone ($K=8$) reduced adaptation CPU time from $15.50\text{s}$ to $4.78\text{s}$, achieving **$69.17\%$ adaptation compute savings** over the Event-Driven Baseline.
- **Retrain Event Reduction**: Retrain events were cut from 19.6 to 9.0 ($54.1\%$ reduction), as the repository stored unique operational regimes and reused them during recurring drift windows.
- **Precision vs. Recall Trade-Off**: RAPT Alone achieved exceptional precision ($0.9561$), but suffered a drop in recall ($0.5526$) during transient shift windows prior to regime identification—motivating the hybrid micro-learner architecture in Stage 3.
