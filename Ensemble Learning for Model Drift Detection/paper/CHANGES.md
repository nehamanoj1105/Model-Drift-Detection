# CHANGES

## What changed

### Manuscript length and structure
- Trimmed the manuscript from 9 pages to 6 pages including references (MINDS @
  COMSNETS 2027 limit). The overview figure was reduced from a full-width
  `figure*` to a single column; two result figures (cost, detector events) were
  dropped and their content is carried by Table II and Table III.
- Rebuilt `main.tex` deterministically from the committed baseline plus the
  clean `results_section.tex` / `tail_section.tex` fragments, after repeated
  regular-expression edits had corrupted the file. Section order is now:
  Introduction, Related Work, Problem Formulation, Proposed Method, Experimental
  Methodology, Results, Discussion, Independent Re-validation, Limitations,
  Conclusion.
- Switched the bibliography style from `plain` to `IEEEtran` and fixed the
  undefined citation keys, so every `\cite` now resolves (22 references, no
  undefined citations or references).

### Numbers and claims
- The title/abstract/contributions were rewritten to claim only what the 30-seed
  final results support: reuse cuts adaptation CPU a lot (64--91% below full
  retraining); base RAPT macro-F1 is below full retraining on all four streams;
  identity-only reuse fails silently on UGR'16 (0.9137 against Frozen 0.9693);
  RAPT-Guard recovers most of that loss (0.9357) but still does not beat Frozen
  on UGR'16.
- Removed the stale "two streams where reuse is safe" phrasing; Guard matches or
  beats Frozen on three of four streams.
- Replaced the per-method "Table I Samples/window" ambiguity: the row now
  correctly reports window size, with a separate "Total samples" row.
- Added a buffer-definition paragraph to the Problem Formulation (class-anchored
  buffer, stratified prefix slice), replacing the vague "last few instances"
  description.
- Fixed the non-significant NordicDat/5G NR claims so the Results text matches
  the saved statistics (e.g. 5G NR RAPT-Guard vs Full Retraining `p=0.15`,
  NordicDat RAPT vs Full Retraining `p=6.1e-4`).

### Results consistency
- Added `paper/make_paper_values.py`, which reads `results/final/summary.json`,
  `results/final/tables/*.csv`, the ablation table and the revalidation CSVs and
  writes `paper/paper_values.md`. Every number in the paper can be re-derived
  from that file.
- Added `paper/make_paper_tables.py` table generation from the saved CSVs.
- Regenerated Table III (detectors) into a compact single-column layout.
- Regenerated `statistics.csv` with the added RAPT/RAPT-Guard/RAPT-Anchor vs
  Full Retraining comparisons.

## What got worse
- The overview figure is now single-column and smaller, so its internal text is
  tighter.
- Two result figures (cost, detector events) were removed to hit the page limit;
  the cost and detector numbers now appear only in Tables II and III.
- The mechanism ablation is reported in prose only (the table was removed); it
  still uses the legacy five-seed harness and is power-limited.
- Base RAPT is still below Full Retraining on all four streams, and RAPT-Guard
  still does not beat Frozen on UGR'16. The abstract now says so plainly.

## Claims softened
- "RAPT improves accuracy" is not claimed anywhere; the abstract and Results say
  the opposite.
- The long-window gate is claimed as a detection mechanism only, with no large
  accuracy effect.
- Guard is not claimed to beat Frozen or Event-Driven on every stream: on UGR'16
  it remains 0.0336 below Frozen.
