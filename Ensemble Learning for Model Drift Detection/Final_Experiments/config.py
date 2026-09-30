"""
Final_Experiments configuration.

Dataset choice is SCIENTIFIC, not score-based. The 5G Campus Network QoS
stream is used because it is the only dataset in this repository that is:

  * unambiguously a telecom QoS dataset (5G campus testbed telemetry);
  * moderate-dimensional (19 features), so a fingerprint/similarity gate is
    meaningful rather than degenerate (UGR'16 has 134 sparse netflow features,
    where an RBF gate on raw fingerprints saturates);
  * natively windowed with semantic regime labels and genuine recurrence
    (3 regimes, 14 recurrence events);
  * cheap enough to run the full mechanism ladder across 5 seeds.

No dataset is selected to maximise any model's score. Every mechanism is
reported as measured, including negative results.
"""

import os

EXPERIMENT = "Final_Experiments"
SEEDS = [42, 43, 44, 45, 46]
WINDOW_SIZE = 10            # matches the existing 5G Campus processed stream
INITIAL_FRAC = 0.20
BUFFER_CAPACITY = 1000

# Predictive models compared.
MODELS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT"]

# RAPT mechanism ladder (ablation). Each rung adds ONE mechanism to the
# previous rung, so the incremental contribution is isolated.
#   RAPT_T2        : Tier-2 policy repository only (the current 9A RAPT)
#   RAPT_T1        : + Tier-1 online micro-learner with beta blending
#   RAPT_T1_REFIT  : + absolute-threshold parity refit (reference form, 0.5 acc)
#   RAPT_FULL      : + fingerprint similarity gate on reuse
#   RAPT_REL_REFIT : RAPT_FULL + RELATIVE-drop refit trigger (accuracy drop vs
#                    pre-drift baseline), which the absolute trigger cannot fire
#                    on a 3-class stream where degraded policies still score ~0.95
#   RAPT_REFRESH_W{5,10} : RAPT_FULL + bounded periodic refresh every 5 / 10
#                    windows, which stops a reused policy staying frozen for the
#                    whole duration of a recurring regime.
#   RAPT_EVIDENCE  : RAPT_FULL + evidence-triggered refresh (only when the
#                    policy's rolling accuracy drops below its own decayed
#                    baseline), so refreshes are paid only when needed.
#   RAPT_CHEAP     : RAPT_REFRESH_W5 with a cheap refresh (20 trees, 300-sample
#                    buffer) instead of a full 100-tree / 1000-sample refit.
#   RAPT_COMBO     : RAPT_EVIDENCE + cheap refresh (the recommended combination).
#   RAPT_FLOOR     : cheap refresh triggered by an ABSOLUTE accuracy floor
#                    (rolling acc < 0.97), which catches a reused policy that
#                    was always bad -- something the relative evidence trigger
#                    cannot see because its baseline decays down to match.
RAPT_VARIANTS = ["RAPT_T2", "RAPT_T1", "RAPT_T1_REFIT", "RAPT_FULL",
                 "RAPT_REL_REFIT", "RAPT_REFRESH_W5", "RAPT_REFRESH_W10",
                 "RAPT_EVIDENCE", "RAPT_CHEAP", "RAPT_COMBO", "RAPT_FLOOR"]

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP9A_DIR = os.path.join(PROJECT_DIR, "experiments", "exp9a")
# Reuse the existing exp9a preprocessing/loader unchanged (same window size,
# imputation and target definition) so results stay comparable.
CAMPUS_WINDOW_CSV = os.path.join(PROJECT_DIR, "experiments", "exp9", "data",
                                 "processed_exp9_stream.csv")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "results", "raw")
FIGURES_DIR = os.path.join(HERE, "results", "figures")
TABLES_DIR = os.path.join(HERE, "results", "tables")


def ensure_dirs():
    for d in (RAW_DIR, FIGURES_DIR, TABLES_DIR):
        os.makedirs(d, exist_ok=True)
