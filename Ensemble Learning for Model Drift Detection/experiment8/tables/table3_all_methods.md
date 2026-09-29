# Table 3: Summary of Evaluated Adaptation Strategies

| Strategy Name | Description | Budget Enforced | Prequential Probe |
|---|---|---|---|
| Baseline Frozen | Never retrains | No | No |
| Baseline Continuous | Retrains all models every window | No | No |
| Baseline Event-Driven | Retrains all models on drift | No | No |
| Random Selective | Random feasible action under B | Yes | No |
| Weakest Selective | Retrains lowest F1 model under B | Yes | No |
| Equal Budget | Allocates B/3 per model | Yes | No |
| Proposed Value-Based | Maximizes gain per unit compute | Yes | Yes |
| Oracle Allocation | Realized optimal action | Yes | No |
