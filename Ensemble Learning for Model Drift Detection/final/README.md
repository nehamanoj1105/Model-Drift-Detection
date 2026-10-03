# RAPT — final experiment (single run, single config)

Everything in the paper — tables, figures, headline numbers and the claim audit —
is produced by **one entry point from one config**:

```bash
python run_all.py --config final.yaml
```

`run_all.py` performs the whole-stream run (all models, all seeds, all datasets),
writes raw per-window parquet, then invokes the reporting modules. No number in a
table or figure is typed by hand and none is reused from an earlier run.

## Environment

```
python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18,
river 0.26.1, matplotlib 3.11, pyarrow 25, pypsutil 7.2
```

`results/paper_final_run/final/env.json` records the exact versions, CPU model,
OS, git commit and seeds for the run that produced the artifacts.

## What each artifact is

| File | Produced by | Contents |
|---|---|---|
| `raw/per_window_*.parquet` | `run_all.py` | per-dataset, per-model, per-seed, per-window: y_true, y_pred, window_acc/f1, adapt_cpu_s, event_type, max_window_in_buffer |
| `raw/per_seed_*.parquet` | `run_all.py` | per-seed aggregates |
| `dataset_stats.csv` | `run_all.py` | samples, features, classes, balance, windows, regimes, transitions, recurrences, prefix, evaluated windows |
| `ugr_regime_class_balance.csv` | `run_all.py` | per-regime class balance for UGR'16 |
| `leakage_check.{csv,txt}` | `run_all.py` | automated no-future-leak assertion |
| `cpu_breakdown.csv` | `run_all.py` | adaptation CPU per model/seed |
| `cpu_breakdown_components.csv` | `run_all.py` | RAPT fingerprint / gate / recalibrate / refit / checkpoint / refresh CPU |
| `refit_timing.csv` | `run_all.py` | refit micro-benchmark over (trees x buffer), 30 repeats |
| `parity_ablation.csv`, `novelty_attribution.csv` | `diagnostics.py` | UGR'16 parity grid + novelty-buffer attribution |
| `ugr_stationarity.csv` | `make_stationarity.py` | per-window KS/PSI and target-semantics proxy |
| `ugr_diagnostics.csv`, `ugr_buffer_counts.csv`, `ugr_frozen_explanation.md` | `ugr_diagnostics.py` | staleness, buffer class counts, Frozen explanation |
| `table_primary.csv`, `table_ablation.csv`, `table_detectors.csv` | `make_report.py` | mean ± std over seeds |
| `stats.csv` | `make_report.py` | seed-level Wilcoxon (A) + moving-block bootstrap (B) |
| `cost_savings.csv` | `make_report.py` | CPU reduction with 95% CI |
| `numbers.tex` | `make_report.py` | `\newcommand` macros for every quoted number |
| `latex/tab_*.tex` | `make_latex.py` | ready-to-`\input` tables |
| `figures/*.pdf`, `figures/*.png` | `make_figures.py` | vector PDF + 300 dpi PNG |
| `figure_consistency_check.txt` | `make_figures.py` | asserts figure values equal tables |
| `FINAL_RESULTS.md` | `make_report_text.py` | every table assembled from CSV |
| `claims_audit.md` | `make_claims_audit.py` | TRUE/FALSE/PARTLY per manuscript claim |

## Reproduce step by step (optional)

```bash
python run_all.py --config final.yaml                       # everything
# or, if raw/*.parquet already exist and you only want the report:
python make_report.py --config final.yaml
python make_latex.py
python make_figures.py --config final.yaml
python make_report_text.py
python make_claims_audit.py
python make_stationarity.py --config final.yaml
python ugr_diagnostics.py --config final.yaml
python diagnostics.py
```

Set `RAPT_SKIP_DIAGNOSTICS=1` to skip the parity grid and staleness re-runs when
iterating on the report layer only.

## Protocol guarantees

* **Prequential**: predict → score → adapt, one window at a time; the test label
  is used for adaptation only after the window has been scored.
* **No future leakage**: an in-loop assertion plus a post-run check on the raw
  parquet (`max_window_in_buffer < window_idx`). See `leakage_check.txt`.
* **Prefix only for preprocessing**: medians, scaling and target thresholds are
  fit on the initial 20% prefix (rounding: `round_half_up`), never on test data.
* **Identical harness for every model**: same RF+ExtraTrees base, same class
  anchored buffer, same capacity, same prefix; only the adaptation differs.
* **Determinism**: numpy/RandomState, sklearn `random_state` and River `seed` are
  all set. Any model with zero seed variance is reported in
  `determinism_check.txt`, never silently printed as `0.0000`.

## Detector provenance (names corrected for the paper)

| Paper label | Implementation |
|---|---|
| ADWIN | `river.drift.ADWIN(delta=0.002)` |
| Page-Hinkley | `river.drift.PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)` |
| EDDM (paper: "EDD") | `river.drift.binary.EDDM(warm_start=30, alpha=0.95, beta=0.9)` |
| ECDD-EWMA (paper: "EDMA") | custom EWMA control chart (Ross et al. 2012); River 0.26 has no ECDD |
