# RAPT Manuscript — Submission Notes

## Files

| File | Purpose |
|------|---------|
| `manuscript.tex` | Final IEEE conference manuscript (two-column, IEEEtran) |
| `references.bib` | BibTeX source of the 29 references (manuscript uses an embedded `thebibliography`) |
| `figures/` | Figure files referenced by the manuscript |
| `manuscript.pdf` | Compiled PDF |

## Build

```bash
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Requires `IEEEtran.cls` (`texlive-publishers`) and `booktabs`, `multirow`.

## Length

Technical content ends on page 6; the reference list begins on page 7. This
satisfies the COMSNETS 6-page limit for technical content, with references in the
standard additional space.

To reach six pages the following were condensed without removing any result:
preprocessing and leakage control were merged into one subsection; the nine
method subsections were consolidated into three; the historical-detector and
ablation figures were dropped because Table (detector) and Table (ablation) carry
the same numbers; the architecture placeholder figure was removed; and prose was
tightened throughout. All measured values, statistical tests and claims are
unchanged from the 8-page version.

## Number provenance

Every quantitative claim in the manuscript is drawn from committed repository
artifacts, not invented:

- Cross-dataset primary metrics (5G Campus, UGR'16, NordicDat): `experiments/exp9a/`
  tables and the `Results_Final/` summary.
- 5G NR latency natural-drift metrics and detection events: `results/experiment_9b/natural_drift/`.
- Efficiency ablation (RAPT-Full, Evidence, Floor, Cheap, Incremental): the
  9B efficiency-ablation results and `Results_Final/`.
- Drift-detector comparison (ADWIN, Page-Hinkley, EDD, EDMA): the 9A detector tables.
- Statistical tests: `experiments/exp9a/tables/statistics_9a.csv` and
  `results/experiment_9b/natural_drift/statistical_tests.csv`.

## Claims discipline

The manuscript does not claim that RAPT outperforms the baselines in accuracy.
It reports that RAPT recorded lower macro-F1 than Full Retraining on all four
streams, that reuse was harmful on UGR'16 without the parity refit, and that the
cost-aware configuration *matched* Full Retraining within measurement noise
(Δmacro-F1 = +0.0022, p = 0.3125, 95% CI [−0.0036, +0.0080]) at roughly 40%
lower adaptation cost. Non-significant results are labelled as such.

## Figures

The manuscript includes two figures: `primary_performance.png` (macro-F1 across
models) and `computational_cost.png` (adaptation CPU). The methodology-diagram
placeholder has been removed from the 6-page version; no architecture diagram file
exists in the repository, and it should be added as a rendered diagram if the
venue permits a full-width figure.
