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
figure was removed; and prose was tightened throughout. All measured values and
statistical tests are unchanged from the 8-page version; the interpretation
corrections below are the only substantive prose changes since.

## Interpretation corrections (post-audit)

The number audit (`audit/verify_paper_numbers.py`, 205/205 checks) only verifies
that prose matches the committed CSVs; it does not validate the *causal reading*
of a number. A separate review of the E/A artifacts found four places where the
reading, not the number, was wrong. These are corrected in the manuscript, and the
supporting evidence is in `audit/AUDIT_FINDINGS.md`:

- **Abstract, UGR'16 paragraph, Discussion, Conclusion.** The RAPT → RAPT-Enhanced
  gain (0.8360 → 0.9276) was attributed to parity refitting. Parity refits are 0 on
  every stream and seed, so the gain is the larger novelty-refit buffer
  (1500 requested, capped at 1000 rows).
- **UGR'16 paragraph.** "Regime identifiers recur but the feature-to-label mapping
  does not" is not supported: the conditional-vs-covariate test is inconclusive on
  UGR'16, and `A2_hypothesis_verdict.csv` records H2 as INCONCLUSIVE.
- **Re-validation section.** "Within measurement noise" overstated the equivalence
  test, which holds at ±0.010 but not ±0.005.
- **Cost claims.** "48–92% less adaptation CPU" applies to base RAPT; the
  RAPT-Enhanced configuration that recovers accuracy saves less.

`results/revalidation/A2_hypothesis_verdict.csv` is the artifact for the second
item; it is produced by `results/revalidation/a2_hypothesis_verdict.py`.

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

## Verification

`audit/verify_paper_numbers.py` re-derives every quantitative claim in
`paper/main.tex` from committed artifacts (199 checks, 0 failures). The primary
table is sourced from `experiments/exp9a/tables/table9a_main.csv` and the 9B
`natural_drift/summary.csv`; `results/revalidation/A1_summary_all_streams.csv`
is an independent re-run of the same protocol and matches all predictive
metrics. Adaptation CPU is machine-dependent (a re-run shifted it by up to
0.14 s); the ordering and relative savings are stable.

## Corrections applied in the audit pass

- Fig. 2 caption: the vertical ticks are novelty refits, not parity refits.
- The RAPT-Incremental sentence in the ablation discussion previously
  contradicted itself ("lower accuracy ... significantly lower cost" vs. the
  table's higher cost); it now states that RAPT-Incremental is worse than
  RAPT-Full on both axes and is not carried forward.
- The trigger section previously credited the zero-refresh result to
  "RAPT-Floor"; it now correctly attributes zero refreshes to the relative
  (RAPT-Evidence) rule and 7.6 to RAPT-Floor.
- The detector-comparison sentence claiming RAPT was "less accurate than EDD in
  none" contradicted the reported numbers; it now states RAPT's cost advantage
  and that on UGR'16/NordicDat its lower-cost trigger also yields lower F1.
- Added a short Independent Re-validation section (reproduced metrics, cost
  reproducibility, oracle/provenance check, equivalence test, label delay).
- Condensed the Datasets, trigger, ablation, limitations and Conclusion prose,
  and dropped two figures that duplicated tables (`primary_performance`,
  `cost_tradeoff`), to hold technical content to six pages.

## Figures

The manuscript includes six figures:

| Fig. | File | Content |
|------|------|---------|
| 1 | (inline TikZ) | RAPT architecture and repository |
| 2 | `ugr_rolling_accuracy.png` | UGR'16 rolling accuracy, novelty refits marked |
| 3 | `computational_cost.png` | Adaptation CPU on the two largest streams |
| 4 | (inline TikZ) | Accuracy--cost trade-off across all four streams |
| 5 | `rapt_ablation.png` | Efficiency-ablation ladder (5G Campus) |
| 6 | `detector_comparison.png` | Historical detector comparison |

Two figures were dropped in the audit pass. `primary_performance.png` duplicated the
macro-F1 column of Table II, and `cost_tradeoff.png` duplicated the ablation ladder; both
sets of numbers remain in the table. All figures are generated by
`paper/make_paper_figures.py` (300 dpi PNG + vector PDF) except the inline TikZ panels and
`ugr_rolling_accuracy.png`, whose generator is `Paper_Final/make_ugr_rolling_figure.py`.

The architecture-diagram placeholder has been removed; no architecture diagram file
exists in the repository, and it should be added as a rendered diagram if the venue
permits a full-width figure.
