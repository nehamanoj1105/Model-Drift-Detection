# Audit & Configuration Changelog

**Location:** `experiments/exp2_final/audit/changelog.md`

All configuration edits, gate threshold modifications, and code changes made after initial audit findings are logged chronologically in this file.

---

## Change Entry 001: Post-Run Modification of GATE_9B Threshold in exp2_corrected
- **Date:** 2026-09-26 (Post-run)
- **Target File:** `experiments/exp2_corrected/evaluation.py` (Line 278)
- **Previous Value:** `GATE_9B = {'RAPT-E': (0.8500, 0.9300), 'Event-Driven': (0.8500, 0.9300), 'Frozen': (0.8500, 0.9300)}`
- **New Value:** `GATE_9B = {'RAPT-E': (0.8500, 0.9500), 'Event-Driven': (0.8500, 0.9500), 'Frozen': (0.8500, 0.9500)}`
- **Context & Rationale:** When `run_exp2_corrected.py` was executed, RAPT-E achieved a Macro F1 of `0.9350` on `9B_Original`. Because `0.9350 > 0.9300`, line 228 of `evaluation.py` evaluated `passed = False` and printed `FAIL FAIL` with range `(0.85, 0.93)` into `EXP2_CORRECTED_REPORT.md`. Following the run, line 278 of `evaluation.py` was edited in the Python source code to `(0.8500, 0.9500)`, but `run_exp2_corrected.py` was never re-executed to update the printed report.
- **Audit Decision for exp2_final:** Fixed upper caps are scientifically invalid. In `exp2_final`, baseline reproduction gates enforce a symmetric tolerance `$|\text{Macro F1}_{\text{run}} - \text{Macro F1}_{\text{Exp1\_Ref}}| \le 0.005$` without arbitrary upper caps.

---

## Change Entry 002: Invalidation and Superseding of EXP2_CORRECTED_REPORT.md
- **Date:** 2026-09-28
- **Target File:** `experiments/exp2_corrected/EXP2_CORRECTED_REPORT.md`
- **Change:** Added prominent SUPERSEDED banner warning that all results in `EXP2_CORRECTED_REPORT.md` are invalidated due to post-run outcome recording, static Oracle models, and collapsed probability models.

---

## Change Entry 003: Stream Admission Gate Revision (Phase 0 Round 4)
- **Date:** 2026-09-28
- **Target File:** `experiments/exp2_final/gates.py` & `experiments/exp2_final/audit/phase0_findings.md`
- **Disclosure:** This revision was enacted **after** seeing baseline-only numbers (Frozen, Event-Driven, Local Retraining) and **before** executing any proposed-method runs (Probability-Guided, Oracle, Similarity-Weighted).
- **Old Rule:** Frozen Macro F1 $\ge 0.70$.
- **New Rule:** 
  1. Policy-B Event-Driven Macro F1 $\ge 0.70$
  2. Event-Driven beats Frozen by $\ge 0.020$ ($\text{ED}_{\text{F1}} - \text{Frozen}_{\text{F1}} \ge 0.020$)
  3. Frozen Macro F1 $\le 0.95$
- **Rationale:** The static 0.70 Frozen floor was inadequate for streams like Elec2 (where Frozen is 0.5181, but concept drift occurs across 45 blocks), and did not enforce that candidate transfer offers a non-trivial improvement over Frozen baseline before admitting a stream to candidate transfer evaluation.
- **Admission Outcome:** Admitted stream: **S2 (9B_Original)** under Exp1 protocol ($\text{ED}=0.9396, \text{Frozen}=0.8839, \Delta=+0.0557$). S1 Main, N1 Noise, N2 Fresh Concepts, and S4 Elec2 are excluded from candidate transfer admission due to zero Event-Driven gain over Frozen ($\Delta = 0.0000 < 0.0200$).
## [2026-09-28 19:34:04 UTC] Vendor Import Fix
- **File:** endor/exp1/event_driven.py
- **Modification:** Changed rom models_9b import create_base_ensemble to rom heterogeneous_ensemble import create_base_ensemble.
- **Reason:** Legacy file models_9b.py was renamed to heterogeneous_ensemble.py during vendoring; import path required updating to prevent runtime import failure.
