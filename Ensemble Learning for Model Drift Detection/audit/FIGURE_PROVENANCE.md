# Phase 4 — Figure provenance and regeneration

## Problem

The manuscript includes five figures, but **no committed script produced them**.
`paper/figures/*.png` existed only as binaries (originally under
`Paper_Final/figures/` and in the submission zip). Only the UGR'16 rolling
figure had a generator (`Paper_Final/make_ugr_rolling_figure.py`).

## Fix

`paper/make_paper_figures.py` regenerates all five from committed artifacts and
writes 300 dpi PNG plus vector PDF into `paper/figures/`.

| Figure | Source artifact |
|--------|-----------------|
| `primary_performance.png` | `results/experiment_9a_three/raw/summary_{ugr16,nordicdat}_full.csv` |
| `computational_cost.png` | same, `adaptation_cpu_sec` |
| `rapt_ablation.png` | `Final_Experiments/results/tables/table_final_main.csv` (Table III) |
| `detector_comparison.png` | `results/experiment_9a_three/raw/summary_{5g_campus,ugr16,nordicdat}_full.csv` |
| `cost_tradeoff.png` | `Final_Experiments/results/raw/summary_full.csv` (per-seed points) |

## Fidelity check

Each regenerated figure was OCR-compared with its original. Titles, axis labels,
and legend entries match verbatim, including:

- `primary_performance`: "Macro-F1 (mean ± sd over seeds)", streams UGR'16 and
  NordicDat, five model legend.
- `rapt_ablation`: "5G Campus — RAPT efficiency ablation", twin axes
  Macro-F1 / Adaptation CPU, six-configuration x-axis.
- `cost_tradeoff`: "5G Campus — accuracy vs adaptation cost (per seed)",
  x-range 0.0–1.4 s, eight-entry legend (Frozen … RAPT_INCR).
- `detector_comparison`: Macro-F1 vs Adaptation CPU, markers for
  ADWIN/Page-Hinkley/EDD/EDMA plus RAPT.
- `computational_cost`: "Adaptation CPU (s)" on UGR'16 and NordicDat.

## Figure the manuscript cites but does not include

`streaming_performance.png` exists in `paper/figures/` and in the submission zip
but is **not referenced** by `main.tex`. Left in place; noted for the author.

## Note on the UGR'16 rolling figure

`Paper_Final/make_ugr_rolling_figure.py` is the generator for
`ugr_rolling_accuracy.png` (Fig. 2). Its vertical ticks come from
`novelty_refit_windows()` — novelty refits, not parity refits. The caption says
"parity refits"; see `PHASE2_FINDINGS.md` (a wording issue outside this audit's
scope).
