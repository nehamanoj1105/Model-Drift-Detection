# COMSNETS Submission Audit — RAPT

Manuscript: `Paper_Final/manuscript.tex` (compiled `manuscript.pdf`, 7 pages:
technical content ends on page 6, references on page 7 — within the COMSNETS
6-page technical limit).

This audit maps each quantitative claim in the manuscript to the committed
evidence and a verdict, and lists the edits required before submission. Verdicts:
SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED, MIS-ATTRIBUTED.

---

## 1. Claim-by-claim

| # | Manuscript claim (loc.) | Evidence | Verdict | Required edit |
|---|---|---|---|---|
| 1 | RAPT macro-F1 below Full Retraining on all four streams (Table II, Sec. V-A) | `v2/B_statement_checks.csv` | SUPPORTED (both metrics, 4/4) | none |
| 2 | Frozen highest macro-F1 on UGR'16 (Sec. V-A) | per-window 0.9690 > Event-Driven 0.8972 | SUPPORTED on per-window; NOT on pooled (Event-Driven 0.8413) | keep, but state the metric; pooled Event-Driven is higher |
| 3 | RAPT lowest macro-F1 on UGR'16 | 0.8360 per-window / 0.6629 pooled | SUPPORTED | none |
| 4 | UGR'16 RAPT-Enhanced recovers RAPT by 0.0917 | `T2_mechanism_isolation.csv` | SUPPORTED as a number | none |
| 5 | "RAPT-Enhanced, which parity-refits a degraded reused policy, recovered…" (Sec. V-A) | parity_refits = 0 at every threshold; parity on/off Δ = 0.0000; buffer 500→1000 gives the +0.0917 | **MIS-ATTRIBUTED** | attribute the recovery to the novelty-refit buffer (500→1000 rows), not the parity refit |
| 6 | "the parity refit earns its place, converting 0.8360 into 0.9276" (Discussion) | as #5 | **NOT SUPPORTED** | remove/rewrite; the parity branch is inert on UGR'16 |
| 7 | Table II is one comparable table | `v2/C_metric_reconciliation.csv`: Campus/UGR'16/NordicDat per-window, 5G NR pooled | **PARTIALLY SUPPORTED** (mixed scales) | pick one scale; corrected tables in `latex_tables/C_primary_perwindow.tex`, `C_primary_pooled.tex` |
| 8 | "ADWIN, Page-Hinkley and EDMA recorded zero adaptation events… reproduce the frozen baseline by construction" (Sec. V-C) | `v2/A10_detector_corrected.csv` | **PARTIALLY SUPPORTED** | ADWIN/Page-Hinkley zero for a wiring reason (river `update()` returns None); EDMA fires 2.8–16.2 once fixed |
| 9 | EDD fired 38–39 times per stream and reached highest macro-F1 on Campus/NordicDat | fixed EDD 38–39; F1 0.9847 / 0.4184 | SUPPORTED | none (counts 114/115/117 in the old pipeline were inflated) |
| 10 | RAPT-Cheap matches Full Retraining at much lower cost (Table III, abstract) | Campus only (`T3_cost_matched.csv`) | **PARTIALLY SUPPORTED** | scope the claim to 5G Campus; UGR'16/NordicDat lower-F1, 5G NR slower and lower-F1 |
| 11 | 5G NR cheap refresh cost anomaly | `A3_5g_nr_table_ii.csv`: RAPT-Cheap 3.26 s vs Full Retraining 0.72 s (+351%), 0.8830 vs 0.9110 | finding | state it as a limitation |
| 12 | RAPT-Incremental dominated on both axes | Table III | SUPPORTED | none |
| 13 | RAPT-Evidence zero refreshes is structural | Section V-B mechanism argument | SUPPORTED | none |
| 14 | Five-seed comparisons capped at p = 0.0625 | `T5_statistics.csv` | SUPPORTED | none |
| 15 | Window-level UGR'16 tests significant (p = 2.6e-11, 5.4e-9) | reproduced | SUPPORTED (with the independence caveat already stated) | none |

---

## 2. Metric reconciliation (the open inconsistency)

The manuscript's Table II reports, for each stream, a "Macro-F1". Re-running all
four streams under both definitions:

- 5G Campus, UGR'16, NordicDat: the Table II values equal the **per-window mean**
  exactly (all 15 cells).
- 5G NR: the Table II values equal the **pooled** value (Frozen 0.8961 vs pooled
  0.896124; per-window is 0.9075).

So the 5G NR block is on a different scale from the other three. On UGR'16 the
gap between the two scales reaches 0.199 for RAPT (0.8360 vs 0.6629). The
corrected tables put every stream on one scale; the manuscript should adopt
either `C_primary_perwindow.tex` (recommended — matches three of four published
streams) or `C_primary_pooled.tex`.

---

## 3. Detector section rewrite (required)

Current text: "Three of the four detectors did not fire: ADWIN, Page-Hinkley and
EDMA recorded zero adaptation events on all three streams and so reproduce the
frozen baseline by construction."

Corrected facts (`A10_detector_corrected.csv`):

| Detector | Original events | Fixed events | Reason |
|---|---|---|---|
| ADWIN | 0 | 0 | river `ADWIN.update()` returns `None`; `bool(...)` is always False |
| Page-Hinkley | 0 | 0 | same wiring issue |
| EDMA | 0 (default) | 2.8–16.2 | comparison order: sample absorbed into its own EWMA before the test |
| EDD | 114/115/117/344 | 38/39/39/115 | first-window spurious fire + harness count difference |

Required edits:
1. State that ADWIN and Page-Hinkley never fire because the harness reads the
   wrong return value from river 0.26.1 — a wiring artefact, not a property of
   the stream.
2. Report EDMA as a firing detector under corrected wiring; it does not
   reproduce the frozen baseline (NordicDat 0.2701 vs Frozen 0.2779; UGR'16
   0.9474 vs 0.9690; Campus 0.9835 vs 0.9364).
3. Correct the EDD event counts and re-state "most fires / highest F1 / highest
   cost" against the fixed counts.
4. Regenerate `detector_comparison` from the corrected counts if the figure is
   kept.

---

## 4. Length / 6-page status

`manuscript.pdf` = 7 pages; technical content ends on page 6, references on
page 7. The required edits above are mostly re-wording of existing sentences
(detector paragraph, the parity sentence, Table II caption) plus one table
swap, so they fit without adding a page. Adding the 5G NR cost-anomaly sentence
(Section 1 #11) is one sentence in Limitations.

---

## 5. What survives the audit

- RAPT is below Full Retraining on all four streams (both metrics).
- RAPT is worst on UGR'16 (both metrics).
- The UGR'16 RAPT-Enhanced recovery is real as a number (+0.0917).
- The cost advantage is 5G Campus-specific.
- EDD is the strongest historical detector on Campus/NordicDat at the highest
  cost.
- The five-seed power ceiling and window-independence caveat are correctly
  stated.
