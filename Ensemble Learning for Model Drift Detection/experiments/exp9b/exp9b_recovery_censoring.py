"""
Recovery-metric validity under permanent concept drift.

The 95%-of-pre-drift recovery rule can only be satisfied when a stream returns to
its pre-drift conditional distribution. Under 9B-C concept drift the post-drift
block has a permanently reversed P(Y|X), so no model can regain 95% of its
pre-drift correctness and every (model, severity) recovery value is censored at
the 40-window horizon. This is a property of the construction, not of any model,
and it is the reason the concept-drift recovery figures carry no signal.

This script quantifies the censoring and writes the evidence to
results/experiment_9b/concept_drift/recovery_censoring.csv.
"""

import os
import pandas as pd

RAW = os.path.join("results", "experiment_9b", "raw")
OUT = os.path.join("results", "experiment_9b", "concept_drift",
                   "recovery_censoring.csv")


def censoring(scenario):
    df = pd.read_csv(os.path.join(RAW, f"{scenario}_recovery.csv"))
    per = (df.groupby(["method", "drift_level"])
             .agg(recovered_rate=("recovered", "mean"),
                  recovery_windows_mean=("recovery_windows", "mean"),
                  n=("recovered", "size"))
             .reset_index())
    per.insert(0, "scenario", scenario)
    return per


def main():
    frames = [censoring(s) for s in ("covariate", "concept", "recurring")]
    out = pd.concat(frames, ignore_index=True)
    out = out.round(4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
