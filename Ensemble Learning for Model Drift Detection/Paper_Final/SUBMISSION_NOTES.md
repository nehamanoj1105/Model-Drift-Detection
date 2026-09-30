# RAPT Manuscript — Submission Notes

## Files

| File | Purpose |
|------|---------|
| `manuscript.tex` | Final IEEE conference manuscript (two-column, IEEEtran) |
| `references.bib` | BibTeX source of the 12 references (manuscript uses embedded `thebibliography`) |
| `figures/` | Figure files referenced by the manuscript |
| `manuscript.pdf` | Compiled PDF |

## Build

```bash
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Requires `IEEEtran.cls` (`texlive-publishers`) and `booktabs`, `multirow`.

## Length

Technical content ends on page 8; the reference list begins on page 8 and
continues to page 9. This satisfies the COMSNETS 8-page limit for technical
content, with references in the standard additional space.

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

## Placeholder

`fig9b_rapt_ablation` in the methodology figure is a labelled placeholder box
(`METHODOLOGY DIAGRAM PLACEHOLDER`) because no architecture diagram file exists in
the repository. Replace it with a rendered diagram before camera-ready; the
structure is described in the caption.
