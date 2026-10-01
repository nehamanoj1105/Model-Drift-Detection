# Phase 1 — Result-consistency validation

Every numeric claim in `paper/main.tex` is recomputed from committed raw artifacts.
Harness: `audit/validate_phase1_full.py` (`python3 validate_phase1_full.py` from
`audit/`). Final result: **0 failing claims** out of 143 checks.

All means are over seeds `[42,43,44,45,46]`; all standard deviations use `ddof=1`
(sample SD). Statistical tests are window-paired Wilcoxon signed-rank with paired
Cohen's $d$, as the paper states.

## Sources of truth

| Quantity | Artifact |
|---|---|
| Table II (Campus, UGR'16, Nordic) | `results/experiment_9a_three/raw/summary_{5g_campus,ugr16,nordicdat}_full.csv` |
| Table II (5G NR) | `results/experiment_9b/natural_drift/summary.csv` |
| Section V-F statistics | `results/experiment_9a_three/raw/per_window_{...}_full.csv` |
| Ablation ladder (Table III) | `Final_Experiments/results/raw/{summary_full,per_window_full}.csv` |
| Detector comparison | `results/experiment_9a_three/raw/summary_{...}_full.csv` |

## Verified without change

- Table II F1/Acc/Prec/Rec for all 5 models x 4 streams, including standard deviations.
- Table II adaptation CPU, runtime, retrain and reuse counts.
- Section V-F: Campus $\Delta$F1 $=-0.0455$, $p=2.6\times10^{-4}$, $d=-0.32$;
  UGR'16 $\Delta$F1 $=-0.1236$, $p=2.6\times10^{-11}$, $d=-0.67$;
  UGR'16 Enhanced vs RAPT $\Delta$F1 $=+0.0917$, $p=5.4\times10^{-9}$, $d=0.56$;
  Nordic $\Delta$F1 $=-0.0409$, $p=0.255$, $d=-0.12$.
- Section V-G cost reductions: 83.6%, 91.7%, 88.2%, 48.2%.
- Ablation ladder values, refreshes, and the RAPT-Cheap equivalence test
  ($\Delta=+0.0022$, $p=0.3125$, $d=0.47$, 95% CI $[-0.0036,+0.0080]$).
- Detector behaviour: ADWIN/Page-Hinkley/EDMA fire 0 events on all three streams;
  EDD fires 38/39/39 retrains at CPU 3.98/6.10/4.69 s.
- Section V-I refit timing (reproduced independently): 100 trees 103.9 ms,
  20 trees 21.8 ms, 200-observation buffer 78.2 ms, 1000-observation buffer 103.4 ms
  — matching the paper's 104 / 22 / 80 / 104 ms.

## Corrections applied (4)

Each is a transcription error corrected against the raw artifact; no prose was
reworded.

1. **Initial training prefix (Section V-B).** Paper said "ninety-six for the three
   cross-dataset streams"; the harness uses 36 (`initial_train_windows=36` in every
   `stream_def_*.json`; `three_dataset_config.py: INITIAL_TRAIN_FRACTION = 0.20`,
   and 0.20 x 179/180/182 = 36). Changed 96 -> 36. The 5G NR value (99) is correct.
2. **UGR'16 recurrences (Section V-A).** Paper said "1,680 unique recurrences";
   the stream has 180 regime segments over 12 distinct regimes, so 168 recurrences
   (this is also the value in Table I and in the repository convention
   recurrences = segments - distinct). Changed 1,680 -> 168.
3. **5G NR recurrences (Table I).** Paper said 10; the 5G NR stream has 10 regime
   segments over 4 distinct regimes, so 6 recurrences under the same convention as
   the other columns. Changed 10 -> 6.
4. **Nordic Event-Driven retrains (Table II).** Paper said 2; the per-seed counts are
   `[3,2,2,2,2]`, mean 2.2. Changed 2 -> 2.2.

## Known problems (a-l)

| # | Problem | Status |
|---|---|---|
| a | Figure files have no committed generator script | **resolved (Phase 4)** — `paper/make_paper_figures.py` |
| b | RAPT code is duplicated across three trees (`experiments/exp9a`, `experiments/exp9b`, `Final_Experiments`) | resolved for the 9A/ladder pair (E1: identical); 9B copy still separate |
| c | `Final_Experiments` runs on 5G Campus only; ablation is single-stream | open — documented limitation |
| d | Zero-variance ablation rows: `RAPT_FULL`/`RAPT_EVIDENCE`/`RAPT_REL_REFIT` have identical F1 on all 5 seeds | **explained (E2/E5)** — those variants share the same reuse path and never fire their trigger, so they coincide by construction |
| e | Nordic Event-Driven retrain mean is 2.2, not an integer | resolved (fix 4) |
| f | Detector baselines ran on 3 streams only, not 4 | open — 5G NR excluded |
| g | Prefix claim 96 vs 36 | resolved (fix 1) |
| h | UGR'16 recurrence claim 1680 vs 168 | resolved (fix 2) |
| i | 5G NR recurrence 10 vs 6 | resolved (fix 3) |
| j | Figure provenance: 5 of 5 PNGs have no source script | **resolved (Phase 4)** |
| k | No documented RAPT default constants (floor, similarity threshold, buffer) | **resolved (Phase 5)** — see `RAPT_DEFAULTS.md` |
| l | `references.bib` verified entries vs paper | **resolved (Phase 3)** — all 27 verified, 3 recurring-concept refs added |

## Compile status

7 pages, 0 errors, 0 undefined references, 0 overfull boxes, 0 TODO placeholders.
