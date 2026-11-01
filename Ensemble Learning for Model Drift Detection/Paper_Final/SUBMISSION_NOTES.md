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
method subsections were consolidated into three; the architecture placeholder
figure was removed; and prose was tightened throughout. All measured values,
statistical tests and claims are unchanged from the 8-page version.

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

The manuscript includes three figures:

| Fig. | File | Content |
|------|------|---------|
| 1 | `primary_performance.png` | Macro-F1 on the two largest cross-dataset streams |
| 2 | `computational_cost.png` | Adaptation CPU on the two largest streams |
| 3 | `detector_comparison.png` | Historical detector comparison |

The architecture-diagram placeholder has been removed; no architecture diagram file
exists in the repository, and it should be added as a rendered diagram if the venue
permits a full-width figure. Two figure environments (`rapt_ablation.png` and
`cost_tradeoff.png`) were dropped to recover the page budget after the detector and
label-delay findings were added; their numbers remain in Tables II and III.

## Page budget

COMSNETS permits eight pages of technical content plus additional reference space.
The revision places the technical body (Sections I--IX) within six pages, with the
bibliography on the following page. All retained numbers were re-checked against
`results/final/v3/T10_paper_table_audit.csv` after trimming.

## Revalidation additions (this revision)

The revision adds four audited results and removes nothing measured:

- **Detector wiring correction** (Sec. VI-G). The published "ADWIN, Page-Hinkley and
  EDMA never fire" is a harness artefact: River's `update()` returns `None` and the
  return value was stored as the flag; EDD's 38--39 is a retrain throttle, not the
  fire count. Corrected wiring fires EDD 114/115/111/344 and EDMA 10/1/4/14, and
  ADWIN/Page-Hinkley fire on the per-sample signal (12/187, 4/115).
- **20-seed cheap-config statistics** (Sec. VI-F). Raises power above the n=5
  Wilcoxon floor of p=0.0625; all eight Holm-corrected comparisons are significant.
- **Label-delay sensitivity** (Sec. VI-H). NordicDat's advantage for RAPT does not
  survive a one-window label delay (0.4618 -> 0.3362).
- **Paper-number audit** (Sec. V-D). All 20 primary-table F1 cells and the 6 ablation
  cells are re-derived from committed CSVs (`T10_paper_table_audit.csv`); the 5G NR
  row is drawn from the 9B harness and agrees to 0.0005 on RAPT.

The controlled covariate/concept drift-severity study (Experiment 9B-B/C/D) is
reported in `EXPERIMENT_9B_FINAL_REPORT.md` and referenced from the Limitations
section as supplementary material; it is not in the six-page technical content.
