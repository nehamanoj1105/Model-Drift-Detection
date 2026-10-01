# Audit summary — RAPT COMSNETS manuscript

Branch: `paper-audit`. Original preserved untouched as `paper/main_original.tex`.

## What was done

| Phase | Outcome |
|-------|---------|
| 0 | `paper/` set up, compiles cleanly (7 pages, 0 errors). |
| 1 | 143 numeric claims recomputed from raw artifacts; 0 failures; 4 transcription errors fixed. |
| 2 | Five new experiments (E1–E5). |
| 3 | All 27 bib entries verified against Crossref; 2 dataset refs completed, 3 recurring-concept refs added. |
| 4 | Figure generator committed; 5 manuscript figures regenerated (300 dpi PNG + vector PDF). |
| 5 | Deliverables packaged. |

## Headline findings (Phase 2)

1. **The three RAPT implementations are identical** (E1: 1430/1430 identical
   predictions, seeds 42–46). The ablation ladder is interpretable.
2. **The UGR'16 RAPT→RAPT-Enhanced gain is the refit buffer, not the parity
   refit** (E2). `parity_refits = 0` on all seeds; the 500→1500 novelty buffer
   produces 0.8360→0.9276.
3. **UGR'16 recurrence is predominantly covariate, not concept, drift** (E3/E3b).
   A frozen model scores *higher* on recurrences (gap −0.0782); the conditional
   `P(Y|X)` is essentially unchanged (|dAUC| ≈ 0.01, covariate ratio 2.23).
4. **"Matches Full Retraining" holds at a ±0.010 F1 margin, not ±0.005** (E4).
5. **An accuracy-based degradation check cannot fire on UGR'16** (E5): accuracy
   stays high while macro-F1 collapses.

## Paper facts that need author attention (wording, not numbers)

See `audit/PHASE2_FINDINGS.md`. The reported numbers are correct as measured;
their attributed cause (parity refitting) is not. Correcting this changes prose,
which the audit's hard rule forbids, so it is flagged rather than edited.

## Files

- `main.tex`, `references.bib`, `main.pdf` — audited manuscript.
- `figures/` — regenerated figures (PNG + PDF).
- `CHANGES.diff` — every factual change vs `main_original.tex`.
- `compile_log_final.txt` — final build log.
- Full audit trail: `../audit/`.
