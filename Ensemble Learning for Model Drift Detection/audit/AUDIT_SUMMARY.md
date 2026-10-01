# Audit summary — RAPT COMSNETS submission

Independent re-validation and claims audit of
"When Is Reuse Safe? Regime-Aware Policy Transfer for Recurring Drift in Network
Telemetry", on branch `paper-audit`.

## What was checked

`verify_paper_numbers.py` re-derives every quantitative claim in `main.tex` from
committed repository artifacts rather than from the aggregates printed in the paper.
It reads `main.tex` directly, so the checks fail if the prose and the artifacts
diverge.

Result: **205/205 checks pass.**

| Group | Source of truth |
|-------|-----------------|
| Table II (primary comparison) | `experiments/exp9a/tables/table9a_main.csv` |
| Table III (ablation ladder) | `results/experiment_9b/.../summary.csv` |
| Statistical tests (p, d) | `experiments/exp9a/tables/statistics_9a.csv` |
| Abstract cost range | derived from Table II CPU columns |
| 5G NR natural-drift metrics | `results/experiment_9b/natural_drift/summary.csv` |
| Equivalence test (TOST) | `results/revalidation/A7_tost.csv` |
| Label-delay protocol | `results/revalidation/A8_label_delay.csv` |

An independent harness (`results/revalidation/A1_summary_all_streams.csv`) re-runs the
same prequential protocol from raw per-window records and reproduces every predictive
metric to four decimal places.

## Known non-reproducible quantities

- **Adaptation CPU (wall-clock).** A re-run reproduced every F1/accuracy/precision/
  recall exactly but shifted adaptation CPU by up to 0.14 s. The ordering across models
  and the relative savings (48–92%) are stable; the absolute seconds are indicative.
- **5G NR aggregation.** The paper aggregates the 5G NR stream differently from the A1
  harness; this is a reporting convention, not a discrepancy in the raw numbers.

## Corrections applied

1. Fig. 2 caption: ticks are **novelty** refits, not parity refits.
2. Ablation discussion: the RAPT-Incremental sentence contradicted its own table (it
   claimed a lower cost than RAPT-Full where the table shows higher). Rewritten.
3. Trigger section: zero refreshes belong to the **relative** (RAPT-Evidence) rule and
   7.6 refreshes belong to RAPT-Floor; the attribution was reversed.
4. Detector comparison: "less accurate than EDD in none" contradicted the reported
   per-stream F1. Replaced with the measured cost/F1 trade-off.
5. Controls paragraph: the label-delay sentence attributed the $0.06$ drop to RAPT on
   UGR'16, but the A8 run never covered UGR'16 and the $0.06$ belongs to Full Retraining
   on NordicDat. Corrected to RAPT $0.13$ / Full Retraining $0.06$ on NordicDat.
6. Added an Independent Re-validation section.
7. Dropped two figures that duplicated tables (`primary_performance`,
   `cost_tradeoff`) and tightened prose to hold technical content to six pages.

## Files

- `verify_paper_numbers.py` — the audit harness.
- `VALIDATION.md`, `PHASE2_FINDINGS.md` — earlier validation passes.
- `FIGURE_PROVENANCE.md` — generator and data source for every figure.
- `RAPT_DEFAULTS.md` — default hyperparameters and where they are set.
- `RELATED_WORK.md`, `CHANGES.diff` — supporting material.
- `results/revalidation/` — the A1–A10 re-validation run (also on the
  `revalidation-20261001` branch).
