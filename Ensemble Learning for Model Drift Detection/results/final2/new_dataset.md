# New dataset selection (Task C) — written BEFORE running any method

Job: replace NordicDat with one public network-telemetry dataset on which
regime-recurrence results can be meaningful.

## Selection criteria

1. Downloadable in under 10 minutes, no login/paywall.
2. Timestamps at a fixed cadence (so windows are chronological and regimes can
   be labelled from the calendar).
3. A labelled target (supervised stream), preferably an anomaly/incident label
   defined by the data producers rather than invented by us.
4. At least 3 recurring regimes. Preferred form: a calendar segment that
   repeats (time-of-day / day-of-week) or multiple sites/KPIs that repeat.
5. At least 100 windows at the chosen window size.
6. Same preprocessing/leakage rules as the other streams: chronological order,
   prefix-only imputation/thresholds, no look-ahead, regime id and target not
   used as model inputs.

## Candidates considered

| Candidate | Download | Timestamps | Label | Recurring regimes | Verdict |
|---|---|---|---|---|---|
| CIC-IDS2017 (ML CSVs) | GitHub/HF, fast | **No** in the ML CSVs (flow starts only); one attack per day, no within-week repeat | BENIGN/attack | weak (days don't repeat) | rejected |
| CIC-IDS2017 per-packet parquet (rdpahalavan) | 10s of GB | yes | yes | yes | rejected (too big for the time cap) |
| UNSW-NB15 | GitHub/HF, fast | Stime/Ltime present but dropped in the common CSV; single capture, grouped attacks, not recurring | attack_cat/label | no | rejected |
| CESNET-TimeSeries24 | Zenodo, many GB | yes | weak (only two anomalies) | yes | rejected (size/labels) |
| IEEE DataPort UE LTE/5G (Tsourdinis) | requires DataPort login | yes | no explicit label | yes | rejected (login) |
| **Network operator KPIs time series (Zenodo 8147768)** | 6 MB zip, ~50 s | yes, 5-min cadence, 37.3 days | **15 real incidents** with start/end sample ids, from the producers; + labelled synthetic anomaly series | yes (daily + weekly cycle; 14 sites, 4 KPI types) | **SELECTED** |

## Selected dataset

* Title: *Network operator KPIs time series dataset*
* DOI: 10.5281/zenodo.8147768 (Zenodo record 8147768)
* Licence: CC BY-NC 4.0
* Cadence: one sample every 5 minutes (300 s), 10,738 samples per series,
  span 37.3 days. Values scaled to [0, 1000]; timestamps shifted to start at 0.
* The release ships **synthetic** series (`data_series/s1..s50.txt`) and
  **real** series (`data_real/r1..r14.txt`) with the producers' incident
  annotations in `data_real_incidents.txt` (series id, start sample, end
  sample). We use the real series and their real incident labels.

### How we build the supervised stream (fixed before running methods)

* Pool the 14 real series into one chronological stream. Window = 4 h
  (48 samples of 5 min). Each window is identified by (series id, window
  index); the series switch is a regime boundary.
* **Regimes** (the recurrence signal): the **same site/KPI at the same
  time-of-day band** returns every day. Concretely `regime_id =
  f"{data_series_info[kpi]}_d{day_of_year:03d % 7}_{hour//4}"` — a site/KPI
  paired with a recurring 4-h time-of-day slot (0..5) and the day-of-week
  phase; the exact string is emitted by the loader for auditability. Each site
  revisits its slots day after day, so previously-seen regimes recur many
  times (verified in the profile output).
* **Target**: sample-level binary label; a sample is positive (`1`) iff its
  5-min bucket lies inside a producer-annotated incident interval for that
  series. Union of incident windows; negative otherwise. (r8 has an open-ended
  incident `-1`; it is treated as ending at the last sample, which makes the
  tail of r8 positive — documented, not hidden.)
* **Model inputs**: the four KPI values plus compact rolling context
  (previous-window mean and delta for the series) — no regime id, no target,
  no raw sample index, no day/hour fields.
* Prefix = first 20 % of windows (same protocol as the other streams).

### Why this is better than NordicDat

NordicDat keyed regimes on operator and gave macro-F1 ~0.33–0.47 because the
target was a prefix-threshold QoS class with no producer label. Here the
target is a producer-annotated incident, the cadence is regular, and regimes
recur on a calendar cycle — so regime-identity reuse and the safety gate can
be told apart. If, after running, the dataset still proves degenerate we will
report that rather than switch datasets.
