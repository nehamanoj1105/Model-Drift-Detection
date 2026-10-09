# RAPT paper fix-up — plan

Working root: `Ensemble Learning for Model Drift Detection/`.
All paths below are relative to that directory unless noted.

## Important pre-flight finding (read first)

The task's STEP 1/3 references several named artifacts that do **not exist
verbatim** anywhere in the repo, in `git log --all`, or on any remote branch:

| referenced | status in repo |
|---|---|
| `RAPT-Guard` | does not exist — new method, to be implemented |
| `RAPT-Anchor` | does not exist — new method, ambiguous definition, to be defined |
| "long-window gate vs short-window gate" experiment | no such experiment or claim |
| "concept-drift stress panel" | the 9B concept-drift experiments exist (`experiments/exp9b/`), no panel named this |
| Section V.E vs VI.B window-level tests | actual `main.tex` has no such numbering |
| Table I "Samples/window" row | actual Table I (`tab:datasets`) has a "Samples" and a "Window size" row |
| "two ways / three ways reuse" | not present in any .tex |
| conclusion sentence on Event-Driven vs RAPT-Anchor | not present |

`paper/main.tex` is the paper of record (identical md5 to
`Paper_Final/manuscript.tex`; the two share one source). It is a 9-section
manuscript, not the VII-section structure the task describes. We therefore apply
the **substance** of each STEP 3 item to the real manuscript and record the
mapping in `CHANGES.md`.

`paper/main.tex` currently also **violates anonymity** (author "Neha Manoj,
Ericsson Research") — fixed in STEP 3.

## Data

`experiments/exp9a/data/ugr16/UGR16v1.mat` and `.../nordicdat/nordicdat.csv`
are git-ignored / absent; re-downloaded (UGR16 mirror + Zenodo 10964584). Done.

## Time estimates (compute)

Measured one-seed times on this box: Campus 18.6 s, UGR'16 60.8 s,
NordicDat 34.5 s; 5G NR (9B harness) ~15 s. ~130 s/seed for the base methods.

| step | estimate |
|---|---|
| 1a-c dev sweeps (guard margin/window, detector params, gate window) | 25–40 min |
| 1d concept panel with in-regime floor trigger | 10–15 min |
| 1e total-CPU accounting | free (already recorded) |
| 1f NordicDat diagnosis + class-weighted forest | 10 min |
| 1g new baselines (ED+reuse, Frozen+periodic) | folded into runs |
| STEP 2 final 30 seeds × 4 streams × ~15 methods, 4-way parallel | 60–90 min |
| STEP 3 paper text + compile | 20 min |
| STEP 4 make_paper_values + CHANGES | 15 min |

Cap any single experiment at 20 min; parallelise seeds with joblib (4 cores).

## Deliverables

- `configs/final.yaml`
- `results/final/{raw/,summary.json,*.csv}`
- `experiments/final/*` unified harness
- `make_paper_values.py` -> `paper_values.md`
- updated `paper/main.tex` + 6-page `paper/main.pdf`
- `CHANGES.md`
