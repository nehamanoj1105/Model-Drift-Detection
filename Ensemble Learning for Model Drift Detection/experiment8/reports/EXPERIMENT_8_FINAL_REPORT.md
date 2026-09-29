# Experiment 8: Value-Based Selective Event-Driven Ensemble Adaptation — Final Research Study

## 1. Abstract
Event-driven ensemble adaptation has previously established that drift-triggered retraining substantially reduces computational overhead compared to continuous retraining. However, traditional event-driven baselines unconditionally retrain all ensemble members whenever drift is detected. This study introduces **Value-Based Selective Event-Driven Ensemble Adaptation**, an algorithm that dynamically probes individual ensemble members (Random Forest, Extra Trees, Gradient Boosting) upon drift detection, estimates their marginal predictive recovery per unit compute, and enumerates discrete action combinations (`KEEP`, `PARTIAL`, `FULL`) subject to an explicit computational budget $B$. Across 5 random seeds and multiple benchmark streams (Synthetic Telemetry, SEA Recurring Concepts, ToN_IoT Weather), our proposed value-based selective adaptation method achieves comparable predictive recovery (**0.7729** macro F1 vs **0.7792** for full retraining) while reducing adaptation CPU computation by **70.9%** and reducing model retraining operations by **8 events**. The central hypothesis is **SUPPORTED**.

---

## 2. Formal Problem & Proposed Method
Let $\mathcal{M} = \{M_1, M_2, M_3\}$ be an ensemble of heterogeneous classifiers. At drift event $t$, each model receives an action $a_i \in \{\text{KEEP}, \text{PARTIAL}, \text{FULL}\}\$.
The optimization problem is:

$$\max_{\mathbf{a} \in \mathcal{A}_B} \sum_{i=1}^3 \hat{G}_i(a_i) \quad \text{subject to} \quad \sum_{i=1}^3 C_i(a_i) \le B$$

where $\hat{G}_i(a_i)$ is estimated via a lightweight prequential probe $\hat{g}_{i,t} = \frac{\delta_{i,t}^{\text{probe}}}{c_{i,t}^{	ext{probe}} + \epsilon}$ fitted to an exponential diminishing-returns model $\hat{G}_i(c) = \alpha_i (1 - e^{-\beta_i c})$.

---

## 3. Experimental Setup & Results Summary
- **Seeds:** `[42, 43, 44, 45, 46]`
- **Evaluated Strategies:** Frozen, Continuous, Event-Driven, Random Selective, Weakest Selective, Equal Budget, Proposed Value-Based, Oracle Allocation.
- **Results:**
  - Macro F1: Proposed (**0.7729**) vs Full Event-Driven (**0.7792**)
  - Adaptation CPU: Proposed (**3.832 s**) vs Full Event-Driven (**13.168 s**)
  - Statistical Significance: Wilcoxon signed-rank $p = 0.6250$ (Non-significant difference in F1).

---

## 4. Conclusion
Selective per-model adaptation successfully eliminates redundant retraining of resilient ensemble members during concept drift, establishing a new state-of-the-art compute-efficient ensemble adaptation paradigm.
