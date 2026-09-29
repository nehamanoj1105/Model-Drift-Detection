# Theoretical Analysis: KKT Optimality & Estimation Error Bound

## 1. Problem Formulation
Consider an ensemble of $M$ models $\mathcal{M} = \{M_1, \dots, M_M\}$.
At an adaptation event, we allocate adaptation compute resources $c = (c_1, \dots, c_M)^T$ subject to a maximum adaptation budget $B > 0$.

The continuous adaptation value allocation optimization problem is:

$$\max_{\mathbf{c}} \sum_{i=1}^M G_i(c_i) \quad \text{subject to} \quad \sum_{i=1}^M c_i \le B, \quad c_i \ge 0 \quad \forall i=1,\dots,M$$

where $G_i(c_i)$ represents the expected predictive recovery (e.g. macro F1 improvement) of model $M_i$ given compute allocation $c_i$.

---

## 2. Mathematical Assumptions

1. **Monotonicity:** $G_i'(c_i) > 0$ for all $c_i \ge 0$. More adaptation compute yields non-decreasing predictive recovery.
2. **Concavity (Diminishing Returns):** $G_i''(c_i) \le 0$ for all $c_i \ge 0$. The marginal predictive gain per additional compute unit is non-increasing.
3. **Additivity:** Total ensemble recovery is additive over individual model recovery gains, $\sum_i G_i(c_i)$.
4. **Bounded Estimation Error:** The pre-adaptation gain estimator $\hat{G}_i(a_i)$ satisfies $|\hat{G}_i(a_i) - G_i(a_i)| \le \epsilon$ for all candidate actions $a_i$.

---

## 3. Theorem 1: KKT Optimality Conditions
Under Assumptions 1-3, an optimal allocation $\mathbf{c}^* = (c_1^*, \dots, c_M^*)^T$ satisfies the Karush-Kuhn-Tucker (KKT) conditions:

$$G_i'(c_i^*) = \lambda \quad \forall i \text{ such that } c_i^* > 0$$

$$G_i'(c_i^*) \le \lambda \quad \forall i \text{ such that } c_i^* = 0$$

where $\lambda \ge 0$ is the Lagrange multiplier corresponding to the total budget constraint $\sum_i c_i^* \le B$.

### Proof:
The Lagrangian function is:

$$\mathcal{L}(\mathbf{c}, \lambda, \boldsymbol{\mu}) = \sum_{i=1}^M G_i(c_i) - \lambda \left( \sum_{i=1}^M c_i - B \right) + \sum_{i=1}^M \mu_i c_i$$

Taking the partial derivative with respect to $c_i$:

$$\frac{\partial \mathcal{L}}{\partial c_i} = G_i'(c_i) - \lambda + \mu_i = 0 \implies G_i'(c_i) = \lambda - \mu_i$$

By complementary slackness ($\mu_i c_i = 0, \mu_i \ge 0$):
- If $c_i^* > 0$, then $\mu_i = 0 \implies G_i'(c_i^*) = \lambda$.
- If $c_i^* = 0$, then $\mu_i \ge 0 \implies G_i'(c_i^*) = \lambda - \mu_i \le \lambda$. $\blacksquare$

### Biological / Engineering Interpretation:
At optimal adaptation budget allocation, the **marginal predictive recovery per unit compute is equalized** across all actively adapted ensemble members.

---

## 4. Theorem 2: Suboptimality Bound under Estimation Error
Let $\mathbf{a}^* = \arg\max_{\mathbf{a} \in \mathcal{A}_B} \sum_{i=1}^M G_i(a_i)$ be the true oracle action combination, and let $\hat{\mathbf{a}}^* = \arg\max_{\mathbf{a} \in \mathcal{A}_B} \sum_{i=1}^M \hat{G}_i(a_i)$ be the action selected by the proposed estimated value function. Under Assumption 4, the suboptimality gap is bounded by:

$$G(\mathbf{a}^*) - G(\hat{\mathbf{a}}^*) \le 2 M \epsilon$$

### Proof:
By definition of $\hat{\mathbf{a}}^*$:

$$\sum_{i=1}^M \hat{G}_i(\hat{a}_i^*) \ge \sum_{i=1}^M \hat{G}_i(a_i^*)$$

Using Assumption 4 ($G_i(a_i) \ge \hat{G}_i(a_i) - \epsilon$ and $\hat{G}_i(a_i) \ge G_i(a_i) - \epsilon$):

$$G(\hat{\mathbf{a}}^*) = \sum_{i=1}^M G_i(\hat{a}_i^*) \ge \sum_{i=1}^M \hat{G}_i(\hat{a}_i^*) - M \epsilon$$

$$\ge \sum_{i=1}^M \hat{G}_i(a_i^*) - M \epsilon \ge \sum_{i=1}^M (G_i(a_i^*) - \epsilon) - M \epsilon = G(\mathbf{a}^*) - 2 M \epsilon$$

Rearranging yields:

$$G(\mathbf{a}^*) - G(\hat{\mathbf{a}}^*) \le 2 M \epsilon \quad \blacksquare$$

---

## 5. Empirical Alignment & Limitations
- For $M=3$, $2 M \epsilon = 6 \epsilon$. With probe error $\epsilon \approx 0.02$, maximum expected degradation relative to oracle is $\le 0.12$.
- In practical streaming environments, tree-based models exhibit discrete step-function gains rather than smooth continuous functions; however, the discrete enumeration over the 27 action combinations exacts the KKT principle over discrete choices.
