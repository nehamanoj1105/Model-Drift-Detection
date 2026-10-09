# fast-final2 job (2026-10)

Second, minimal pass on top of `experiments/final/`. Goal: the four STEP-1
fixes, a 30-seed final run, and the Task-C dataset swap. **No paper files are
edited by this job**; the single review artefact is
`results/final2/RESULTS_REVIEW.md`.

## What it adds

* **Task A — RAPT-ED-Guard** (`final2_methods.py`): the Event-Driven error
  trigger OR a regime return arms a cheap validation gate. The gate scores the
  stored regime policy and the prefix policy on the last 50 labelled rows and
  reuses the stored model only if it beats the prefix model by a margin chosen
  on dev (`edguard_margin = 0.05`). Otherwise it refits the regime checkpoint.
* **Task B/E — total CPU**: `total_cpu_sec = adaptation + prediction` recorded
  for every method, including the doubled inference cost of RAPT-Anchor.
* **Task C — dataset swap**: NordicDat replaced by *Network operator KPIs*
  (`final2_load.py`, Zenodo 8147768). See `results/final2/new_dataset.md`.
* **Task D2 — long vs short gate** (`final2_gate_cadence.py`): measured table +
  figure instead of an unsupported claim.
* **Task D3 — detector fixes** (`final2_dev_detectors.py`): ADWIN/PH swept on dev
  to fire at a sensible cadence; **ADWIN added** to the final method set.
* **Task D4 — concept panel** (`final2_concept.py`): floor trigger enabled for
  all methods; the spread is reported (and it stays tiny -> the panel is
  degenerate even after the fix).
* **Task F — class weights** (`final2_nordicdat.py`): balance / leakage checks
  and class-weighted forests on all streams.
* **Task G — reuse-vs-trigger** (`Event-Driven+Reuse`, `Frozen+Periodic`,
  `RAPT-Enhanced`): already present, extended to the new stream.

## Run order

```
python experiments/final2/fetch_kpiops.py                 # download data (once)
python experiments/final2/final2_dev_margin.py            # choose edguard_margin -> configs/final2.yaml
python experiments/final2/final2_dev_detectors.py         # choose adwin_delta / ph_threshold
python experiments/final2/final2_runner.py --mode dev  --streams all --tag dev
python experiments/final2/final2_concept.py
python experiments/final2/final2_gate_cadence.py
python experiments/final2/final2_nordicdat.py
python experiments/final2/final2_runner.py --mode final --streams all --tag final --n-jobs 4
python experiments/final2/final2_stats.py --tag final
python experiments/final2/final2_gate.py
python experiments/final2/make_review.py
```

All tuning uses 5 dev seeds (42-46) and the first 60% of post-prefix windows.
The final run uses all 30 seeds (42-71), full streams, once, with the frozen
`configs/final2.yaml`.

## Method set (12)

Frozen, Frozen+Periodic, Event-Driven, Event-Driven+Reuse, ADWIN, Page-Hinkley,
Full Retraining, RAPT, RAPT-Guard, RAPT-Enhanced, RAPT-Anchor, RAPT-ED-Guard.
The base model (`rapt_9b.py`, the shared RandomForest+ExtraTrees vote) and the
existing RAPT/RAPT-Guard/RAPT-Anchor implementations are **not modified**.
