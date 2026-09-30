# Experiment 9A — third-party dataset provenance

The raw third-party datasets are **not committed** (size/licensing). They are
placed under `experiments/exp9a/data/` and re-downloaded on demand.

## UGR'16  ->  data/ugr16/UGR16v1.mat

- Source (feature-data mirror): https://github.com/josecamachop/UGR16_FeatureData
  (file `matlab/UGR16v1.mat`). Original dataset: UGR'16, Univ. of Granada
  (Maciá-Fernández et al., 2018), ISP netflow traces.
- File: 26.6 MB, MATLAB v5. Variables used: `test` (30-day labelled block,
  43,200 minutes x 134 features), `Yt` (per-minute attack-type labels),
  `classDt/classMt/classWDt` (day, minute-of-day, workday/weekend), `obs_lt`,
  `var_l` (feature names).
- The **calibration** split (`X`/`Y`) is not used: it contains only blacklist
  traffic and no labelled attacks, so it cannot support a supervised stream.
- Target: binary anomaly = any non-blacklist attack minute (`Yt[:, 0,1,2,3,5,6,7]`).
- Regimes: workday/weekend x 4-hour period; window = 240 minutes.

## NordicDat  ->  data/nordicdat/nordicdat.csv

- Source: Zenodo record 10.5281/zenodo.10964584, "NordicDat" cross-border
  LTE/5G predictive-QoS traces, licensed CC-BY-4.0.
- File: 13.6 MB CSV, 91,455 rows, 23 columns (timestamp, GPS, operator, band,
  RAN type, radio/QoS KPIs, throughput, delay).
- Target: 3-class next-window median-delay QoS using training-prefix
  thresholds (median and 80th percentile).
- Regimes: mobile network operator; window = 500 seconds.

## 5G Campus Network QoS

- Reuses the repository's already-processed per-window QoS stream
  (`experiments/exp9/data/*_windowed.csv`); no new download.

## Reproducing

    cd "Ensemble Learning for Model Drift Detection"
    python experiments/exp9a/three_dataset_profile.py
    python experiments/exp9a/three_dataset_run.py full
    python experiments/exp9a/three_dataset_figures.py full
    python experiments/exp9a/three_dataset_tables.py full
    python experiments/exp9a/three_dataset_cross.py
    python experiments/exp9a/three_dataset_report.py
